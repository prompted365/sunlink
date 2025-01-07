# tasks/worker_tasks.py

import os
import time
import asyncio
import supabase

from .celery_app import celery_app
from solar_api import process_solar_data

@celery_app.task
def process_solar_task(property_id: str):
    """
    1) Fetches property from Supabase (id, lat, lng).
    2) Calls process_solar_data() -> returns file paths (including GIF).
    3) Uploads all files to 'property-images' bucket.
    4) Updates property row with each file's URL and building_insights_jsonb.
    5) Marks status='completed'.
    """

    # -------------------------------------------------------------------------
    # 1) Supabase client
    # -------------------------------------------------------------------------
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY.")

    sb_client = supabase.create_client(supabase_url, supabase_key)

    # -------------------------------------------------------------------------
    # 2) Fetch property row
    # -------------------------------------------------------------------------
    resp = sb_client \
        .from_("properties") \
        .select("id, latitude, longitude") \
        .eq("id", property_id) \
        .single() \
        .execute()

    if not resp.data:
        print(f"[ERROR] Property not found: {property_id}")
        return

    lat = resp.data.get("latitude")
    lng = resp.data.get("longitude")
    if lat is None or lng is None:
        print("[ERROR] Missing lat/lng. Marking 'failed'.")
        sb_client.from_("properties").update({"status": "failed"}).eq("id", property_id).execute()
        return

    # Mark property as 'processing'
    sb_client.from_("properties").update({"status": "processing"}).eq("id", property_id).execute()

    # -------------------------------------------------------------------------
    # 3) Process solar data (async -> sync)
    # -------------------------------------------------------------------------
    google_solar_key = os.getenv("API_KEY", "")
    sub_dir = f"./solar_output/{property_id}"
    try:
        results = asyncio.run(process_solar_data(lat, lng, sub_dir, google_solar_key))
    except Exception as e:
        print(f"[ERROR] {e}")
        sb_client.from_("properties").update({"status": "failed"}).eq("id", property_id).execute()
        return

    # e.g. results includes:
    # {
    #   "building_insights": {...},
    #   "dsm_path": "...",
    #   "rgb_path": "...",
    #   "mask_path": "...",
    #   "flux_path": "...",
    #   "annual_flux_composite_path": "...",
    #   "monthly_flux_paths": [...],
    #   "monthly_flux_composite_paths": [...],
    #   "monthly_flux_composite_gif": "...",
    # }

    # -------------------------------------------------------------------------
    # 4) Upload all files (Simplified approach, no .status_code checks)
    # -------------------------------------------------------------------------
    bucket_name = "property-images"
    now_ms = int(time.time() * 1000)
    remote_subdir = f"{property_id}_{now_ms}"

    # Single-file -> DB column
    single_file_map = {
        "dsm_path": "DSM",
        "rgb_path": "RGB",
        "mask_path": "Mask",
        "flux_path": "AnnualFlux",
        "annual_flux_composite_path": "FluxOverRGB",
        "monthly_flux_composite_gif": "MonthlyFluxCompositeGIF"
    }

    # Multi-file -> DB array column
    list_file_map = {
        "monthly_flux_paths": "MonthlyFlux12",
        "monthly_flux_composite_paths": "MonthlyFluxComposites"
    }

    db_updates = {
        # single
        "DSM": None,
        "RGB": None,
        "Mask": None,
        "AnnualFlux": None,
        "FluxOverRGB": None,
        "MonthlyFluxCompositeGIF": None,
        # arrays
        "MonthlyFlux12": [],
        "MonthlyFluxComposites": []
    }

    # --- Upload single files ---
    for result_key, col_name in single_file_map.items():
        local_path = results.get(result_key)
        if local_path and os.path.exists(local_path):
            filename = os.path.basename(local_path)
            remote_path = f"{remote_subdir}/{filename}"

            # We simply do the recommended doc approach
            upload_resp = sb_client.storage.from_(bucket_name).upload(remote_path, open(local_path, "rb"), {
                "upsert": "true"
            })
            # No .status_code check; we trust "upload_resp" might contain info but we won't parse it

            # Get public URL
            url_resp = sb_client.storage.from_(bucket_name).get_public_url(remote_path)
            if url_resp and "publicURL" in url_resp:
                db_updates[col_name] = url_resp["publicURL"]

    # --- Upload multi-file arrays ---
    for result_key, col_name in list_file_map.items():
        path_list = results.get(result_key, [])
        if isinstance(path_list, list):
            final_urls = []
            for local_path in path_list:
                if local_path and os.path.exists(local_path):
                    filename = os.path.basename(local_path)
                    remote_path = f"{remote_subdir}/{filename}"

                    sb_client.storage.from_(bucket_name).upload(remote_path, open(local_path, "rb"), {
                        "upsert": "true"
                    })
                    url_resp = sb_client.storage.from_(bucket_name).get_public_url(remote_path)
                    if url_resp and "publicURL" in url_resp:
                        final_urls.append(url_resp["publicURL"])

            db_updates[col_name] = final_urls

    # -------------------------------------------------------------------------
    # 5) Final DB update (status, building_insights, file columns)
    # -------------------------------------------------------------------------
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
        "MonthlyFluxComposites": db_updates["MonthlyFluxComposites"]
    }

    sb_client \
        .from_("properties") \
        .update(final_update) \
        .eq("id", property_id) \
        .execute()

    print("[INFO] Completed solar processing for property:", property_id)
    print("[INFO] Updated columns:", list(final_update.keys()))
