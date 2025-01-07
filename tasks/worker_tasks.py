# tasks/worker_tasks.py

import os
import time
import mimetypes
import asyncio
import supabase

from .celery_app import celery_app
from solar_api import process_solar_data  # from your solar_api.py

@celery_app.task
def process_solar_task(property_id: str):
    """
    Celery task that:
      1) Fetches a property row from Supabase (lat, lng, etc.).
      2) Runs process_solar_data(...) -> saves images locally as .png/.gif
      3) Uploads them to 'property-images/{property_id}/filename.png'
      4) Updates the property row with the resulting public URLs and building insights
      5) Marks status='completed'
    """

    # 1) Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")

    sb_client = supabase.create_client(supabase_url, supabase_key)

    # 2) Retrieve property
    resp = sb_client \
        .from_("properties") \
        .select("id, latitude, longitude, status") \
        .eq("id", property_id) \
        .single() \
        .execute()

    if not resp.data:
        print(f"[ERROR] Property {property_id} not found in Supabase.")
        return

    lat = resp.data.get("latitude")
    lng = resp.data.get("longitude")
    if lat is None or lng is None:
        print("[ERROR] Missing lat/lng. Marking 'failed'.")
        sb_client.from_("properties").update({"status": "failed"}).eq("id", property_id).execute()
        return

    # Mark property as 'processing'
    sb_client.from_("properties").update({"status": "processing"}).eq("id", property_id).execute()

    # 3) Call the solar API logic
    google_solar_key = os.getenv("API_KEY", "")
    local_subdir = f"./solar_output/{property_id}"
    try:
        results = asyncio.run(process_solar_data(lat, lng, local_subdir, google_solar_key))
    except Exception as e:
        print("[ERROR] process_solar_data failed:", e)
        sb_client.from_("properties").update({"status": "failed"}).eq("id", property_id).execute()
        return

    # 4) Upload to Supabase
    bucket_name = "property-images"
    # No timestamp in subdir -> just "property_id"
    remote_subdir = property_id  

    single_file_map = {
        "dsm_path": "DSM",
        "rgb_path": "RGB",
        "mask_path": "Mask",
        "flux_path": "AnnualFlux",
        "annual_flux_composite_path": "FluxOverRGB",
        "monthly_flux_composite_gif": "MonthlyFluxCompositeGIF"
    }

    list_file_map = {
        "monthly_flux_paths": "MonthlyFlux12",
        "monthly_flux_composite_paths": "MonthlyFluxComposites"
    }

    db_updates = {
        "DSM": None,
        "RGB": None,
        "Mask": None,
        "AnnualFlux": None,
        "FluxOverRGB": None,
        "MonthlyFluxCompositeGIF": None,
        "MonthlyFlux12": [],
        "MonthlyFluxComposites": []
    }

    # Helper to upload a file with correct content type
    def upload_file(local_path: str):
        filename = os.path.basename(local_path)
        remote_path = f"{remote_subdir}/{filename}"

        # Attempt to guess the MIME type
        # If the file is a .gif, ensure we pass 'image/gif'
        # Otherwise for .png, we'll get 'image/png'
        mime_type, _ = mimetypes.guess_type(local_path)
        if not mime_type:
            # fallback
            mime_type = "application/octet-stream"

        with open(local_path, "rb") as f:
            # The official supabase-py approach
            # Upsert ensures we overwrite if the file already exists
            sb_client.storage.from_(bucket_name).upload(
                remote_path,
                f,
                options={"upsert": "true", "contentType": mime_type}
            )

        # Build a public URL for the newly uploaded path
        url_info = sb_client.storage.from_(bucket_name).get_public_url(remote_path)
        if url_info and "publicURL" in url_info:
            return url_info["publicURL"]
        return None

    # Single-file uploads
    for key, column in single_file_map.items():
        local_path = results.get(key)
        if local_path and os.path.exists(local_path):
            public_url = upload_file(local_path)
            if public_url:
                db_updates[column] = public_url

    # Multi-file uploads
    for key, column in list_file_map.items():
        file_list = results.get(key, [])
        if isinstance(file_list, list):
            urls = []
            for local_path in file_list:
                if local_path and os.path.exists(local_path):
                    url = upload_file(local_path)
                    if url:
                        urls.append(url)
            db_updates[column] = urls

    # 5) Final DB update
    building_insights = results.get("building_insights", {})
    final_update = {
        "status": "completed",
        "building_insights_jsonb": building_insights,
        "DSM": db_updates["DSM"],
        "RGB": db_updates["RGB"],
        "Mask": db_updates["Mask"],
        "AnnualFlux": db_updates["AnnualFlux"],
        "FluxOverRGB": db_updates["FluxOverRGB"],
        "MonthlyFluxCompositeGIF": db_updates["MonthlyFluxCompositeGIF"],
        "MonthlyFlux12": db_updates["MonthlyFlux12"],
        "MonthlyFluxComposites": db_updates["MonthlyFluxComposites"],
    }

    sb_client.from_("properties").update(final_update).eq("id", property_id).execute()

    print(f"[INFO] Done. property_id={property_id}")
    print("[INFO] Updated columns:")
    for k, v in final_update.items():
        if k != "building_insights_jsonb":
            print(f"  {k}: {v}")
        else:
            print("  building_insights_jsonb: [JSON data]")
