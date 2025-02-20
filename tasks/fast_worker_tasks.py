import os
import io
import math
import time
import asyncio
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import reproject
from pyproj import Transformer
from dotenv import load_dotenv
import logging
import aiohttp
import mimetypes
import threading

from supabase import create_client
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# ------------------------------------------------------------------------------
# Environment and Logger Setup
# ------------------------------------------------------------------------------
load_dotenv()

logger = logging.getLogger("solar_processing")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console Handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
# File Handler
fh = logging.FileHandler("solar_processing.log")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

app = FastAPI(title="Solar Fast API")

# ------------------------------------------------------------------------------
# Background Runner Helpers
# ------------------------------------------------------------------------------
def run_in_background(func, *args, **kwargs):
    """Run a function in a background daemon thread."""
    thread = threading.Thread(target=func, args=args, kwargs=kwargs)
    thread.daemon = True
    thread.start()

def upload_image_bytes_background_sync(sb_client, supabase_url: str, bucket_name: str, remote_path: str, image_bytes: bytes, mime_type: str):
    """
    Synchronous wrapper that creates its own event loop to run the async
    upload function. This avoids "no running event loop" errors when called
    from a background thread.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(upload_image_bytes_background(sb_client, supabase_url, bucket_name, remote_path, image_bytes, mime_type))
    finally:
        loop.close()

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def pil_image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL image to bytes."""
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    buffer.seek(0)
    return buffer.getvalue()

def get_expected_public_url(supabase_url: str, bucket_name: str, remote_path: str) -> str:
    """Compute the expected public URL given the Supabase URL, bucket name, and remote path."""
    supabase_url = supabase_url.rstrip("/")
    return f"{supabase_url}/storage/v1/object/public/{bucket_name}/{remote_path}"

async def upload_image_and_get_url(sb_client, supabase_url: str, bucket_name: str, remote_path: str, image_bytes: bytes, mime_type: str) -> str:
    """
    Asynchronously upload image bytes to Supabase Storage using asyncio.to_thread
    and return the expected public URL.
    """
    try:
        await asyncio.to_thread(sb_client.storage.from_(bucket_name).upload, remote_path, image_bytes)
        logger.info(f"Uploaded: {remote_path}")
    except Exception as e:
        logger.error(f"Upload failed for {remote_path}: {e}")
    return get_expected_public_url(supabase_url, bucket_name, remote_path)

def reproject_to_match(src_array, src_profile, target_profile, resampling_method):
    """
    Reproject/resample a raster array to match a target profile.
    """
    start_time = time.time()
    dst_height = target_profile['height']
    dst_width = target_profile['width']
    dst_count = src_profile['count']
    dst_dtype = src_profile['dtype']
    reprojected_array = np.zeros((dst_count, dst_height, dst_width), dtype=dst_dtype)
    for band in range(dst_count):
        reproject(
            source=src_array[band],
            destination=reprojected_array[band],
            src_transform=src_profile['transform'],
            src_crs=src_profile['crs'],
            dst_transform=target_profile['transform'],
            dst_crs=target_profile['crs'],
            resampling=resampling_method
        )
    reproj_profile = target_profile.copy()
    reproj_profile.update({'count': dst_count, 'dtype': dst_dtype})
    elapsed = time.time() - start_time
    logger.info(f"Reprojected raster in {elapsed:.2f}s.")
    return reprojected_array, reproj_profile

def composite_flux_on_rgb(rgb_array, flux_rgba_array, mask_array):
    """
    Composite the flux RGBA array over an RGB array using a building mask.
    """
    start_time = time.time()
    r = rgb_array[0].astype(float)
    g = rgb_array[1].astype(float)
    b = rgb_array[2].astype(float)
    alpha_full = np.full_like(r, 255, dtype=np.uint8).astype(float)
    rgb_rgba = np.stack([r, g, b, alpha_full], axis=-1)
    flux_rgba = flux_rgba_array.transpose(1, 2, 0).astype(float)
    flux_rgba[..., 3] *= mask_array
    fa = flux_rgba[..., 3] / 255.0
    ra = rgb_rgba[..., 3] / 255.0
    out = np.zeros_like(rgb_rgba)
    for c in range(3):
        out[..., c] = flux_rgba[..., c] * fa + rgb_rgba[..., c] * (1.0 - fa)
    out[..., 3] = (fa + ra * (1.0 - fa)) * 255.0
    final = np.clip(out, 0, 255).astype(np.uint8)
    elapsed = time.time() - start_time
    logger.info(f"Composite flux completed in {elapsed:.2f}s.")
    return final

async def download_geotiff_subset(url: str, api_key: str, center_lat: float, center_lng: float, extent_m: float = 20) -> dict:
    """
    Download a GeoTIFF file asynchronously, subset it based on the center point and extent, and return its data.
    """
    start_time = time.time()
    if url and 'solar.googleapis.com' in url and 'key=' not in url:
        url += f'&key={api_key}'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            raw_data = await resp.read()
    buffer_data = io.BytesIO(raw_data)
    with rasterio.open(buffer_data) as ds:
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        bx, by = transformer.transform(center_lng, center_lat)
        left, right = bx - extent_m, bx + extent_m
        bottom, top = by - extent_m, by + extent_m
        window = from_bounds(left, bottom, right, top, ds.transform)
        subset = ds.read(window=window)
        transform = ds.window_transform(window)
        profile = ds.profile.copy()
        profile.update({'height': subset.shape[1], 'width': subset.shape[2], 'transform': transform, 'count': ds.count})
    elapsed = time.time() - start_time
    logger.info(f"Downloaded subset from {url} in {elapsed:.2f}s.")
    return {'rasters': subset, 'width': subset.shape[2], 'height': subset.shape[1], 'num_bands': subset.shape[0], 'profile': profile}

def apply_color_ramp(raster_data, color_stops=None) -> Image.Image:
    """
    Apply a color ramp to a single-band raster and return a PIL image.
    """
    start_time = time.time()
    if color_stops is None:
        color_stops = [
            (0.0,  (0, 0, 128)),
            (0.25, (0, 128, 128)),
            (0.5,  (0, 255, 0)),
            (0.75, (255, 255, 0)),
            (1.0,  (255, 0, 0))
        ]
    arr = np.array(raster_data, dtype=float)
    arr[arr == -9999] = np.nan
    arr[np.isinf(arr)] = np.nan
    valid = ~np.isnan(arr)
    h, w = arr.shape
    if not valid.any():
        logger.warning("No valid pixels; returning blank image.")
        return Image.new('RGBA', (w, h), (0, 0, 0, 0))
    mn, mx = np.nanmin(arr[valid]), np.nanmax(arr[valid])
    if math.isclose(mn, mx):
        logger.info("Uniform raster; returning gray image.")
        return Image.new('RGBA', (w, h), (128, 128, 128, 255))
    norm = (arr - mn) / (mx - mn)
    norm[~valid] = 0.0
    out = np.zeros((h, w, 4), dtype=np.uint8)
    norm_flat = norm.flatten()
    out_flat = out.reshape(-1, 4)
    for i in range(len(color_stops) - 1):
        lower = color_stops[i][0]
        upper = color_stops[i+1][0]
        mask = (norm_flat >= lower) & (norm_flat <= upper)
        frac = (norm_flat[mask] - lower) / (upper - lower)
        frac = np.clip(frac, 0, 1)
        r_val = color_stops[i][1][0] + frac * (color_stops[i+1][1][0] - color_stops[i][1][0])
        g_val = color_stops[i][1][1] + frac * (color_stops[i+1][1][1] - color_stops[i][1][1])
        b_val = color_stops[i][1][2] + frac * (color_stops[i+1][1][2] - color_stops[i][1][2])
        out_flat[mask, 0] = r_val.astype(np.uint8)
        out_flat[mask, 1] = g_val.astype(np.uint8)
        out_flat[mask, 2] = b_val.astype(np.uint8)
        out_flat[mask, 3] = 255
    out = out_flat.reshape(h, w, 4)
    elapsed = time.time() - start_time
    logger.info(f"Applied color ramp in {elapsed:.2f}s.")
    return Image.fromarray(out, mode='RGBA')

def create_rgb_image(rasters) -> Image.Image:
    """
    Create an RGB image from a 3-band raster. If fewer than 3 bands, apply a color ramp on band 0.
    """
    start_time = time.time()
    bands, h, w = rasters.shape
    if bands < 3:
        img = apply_color_ramp(rasters[0])
        logger.info(f"RGB fallback image in {time.time()-start_time:.2f}s.")
        return img
    r = rasters[0].astype(np.uint8)
    g = rasters[1].astype(np.uint8)
    b = rasters[2].astype(np.uint8)
    a = np.full((h, w), 255, dtype=np.uint8)
    img = Image.fromarray(np.stack([r, g, b, a], axis=-1), mode='RGBA')
    elapsed = time.time() - start_time
    logger.info(f"Created RGB image in {elapsed:.2f}s.")
    return img

def create_mask_image(raster_data) -> Image.Image:
    """Create a building mask image from a 1-bit raster."""
    start_time = time.time()
    arr = np.array(raster_data, dtype=float)
    arr[arr == -9999] = 0.0
    arr = np.clip(arr, 0, 1) * 255
    img = Image.fromarray(np.stack([arr, arr, arr, np.full(arr.shape, 255, dtype=np.uint8)], axis=-1), mode='RGBA')
    elapsed = time.time() - start_time
    logger.info(f"Created mask image in {elapsed:.2f}s.")
    return img

def create_monthly_flux_images(rasters) -> [Image.Image]:
    """Create color-ramped images for each band in a monthly flux raster."""
    start_time = time.time()
    imgs = [apply_color_ramp(rasters[i]) for i in range(rasters.shape[0])]
    elapsed = time.time() - start_time
    logger.info(f"Created {rasters.shape[0]} monthly flux images in {elapsed:.2f}s.")
    return imgs

async def compose_annual_composite_async(rgb_rasters, flux_reproj, mask_reproj, sb_client, bucket_name, remote_subdir, supabase_url) -> str:
    """
    Compose an annual flux composite over the RGB image, upload it asynchronously,
    and return its public URL.
    """
    flux_img = apply_color_ramp(flux_reproj[0])
    flux_arr = np.array(flux_img).transpose(2, 0, 1)
    mask = np.where(mask_reproj[0] == 1, 1.0, 0.0)
    comp = composite_flux_on_rgb(rgb_rasters, flux_arr, mask)
    comp_img = Image.fromarray(comp, mode='RGBA')
    img_bytes = pil_image_to_bytes(comp_img, fmt="PNG")
    filename = f"FluxOverRGB_{int(time.time()*1000)}.png"
    remote_path = f"{remote_subdir}/{filename}"
    url = await upload_image_and_get_url(sb_client, supabase_url, bucket_name, remote_path, img_bytes, "image/png")
    return url

async def process_single_monthly_composite_async(rgb_rasters, monthly_band, mask_reproj_band, sb_client, bucket_name, remote_subdir, index, supabase_url) -> str:
    """
    Compose a composite image for a single monthly flux band, upload it asynchronously,
    and return its public URL.
    """
    img = apply_color_ramp(monthly_band)
    arr = np.array(img).transpose(2, 0, 1)
    mask = np.where(mask_reproj_band == 1, 1.0, 0.0)
    comp = composite_flux_on_rgb(rgb_rasters, arr, mask)
    comp_img = Image.fromarray(comp, mode='RGBA')
    img_bytes = pil_image_to_bytes(comp_img, fmt="PNG")
    filename = f"MonthlyFluxComposite_{index}_{int(time.time()*1000)}.png"
    remote_path = f"{remote_subdir}/{filename}"
    url = await upload_image_and_get_url(sb_client, supabase_url, bucket_name, remote_path, img_bytes, "image/png")
    return url

async def process_and_upload_image(process_func, sb_client, supabase_url, bucket_name, remote_subdir, prefix: str, *args, **kwargs) -> str:
    """
    Run a blocking image processing function in a thread, then upload the resulting image asynchronously
    and return its public URL.
    """
    image = await asyncio.to_thread(process_func, *args, **kwargs)
    img_bytes = pil_image_to_bytes(image, fmt="PNG")
    filename = f"{prefix}_{int(time.time()*1000)}.png"
    remote_path = f"{remote_subdir}/{filename}"
    url = await upload_image_and_get_url(sb_client, supabase_url, bucket_name, remote_path, img_bytes, "image/png")
    return url

# New Helper: Process Monthly Composite In Memory
def process_single_monthly_composite_in_memory(rgb_rasters, monthly_band, mask_reproj_band) -> Image.Image:
    """
    Compose a composite image for a single monthly flux band and return the PIL image.
    """
    band_img = apply_color_ramp(monthly_band)
    band_arr = np.array(band_img).transpose(2, 0, 1)
    mask = np.where(mask_reproj_band == 1, 1.0, 0.0)
    comp = composite_flux_on_rgb(rgb_rasters, band_arr, mask)
    return Image.fromarray(comp, mode='RGBA')

def create_gif_in_memory(images: [Image.Image]) -> bytes:
    """
    Create an animated GIF from a list of PIL images and return the GIF as bytes.
    """
    buffer = io.BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=250, loop=0)
    buffer.seek(0)
    return buffer.getvalue()
# -------------------------------------------------------------------------------
# Main Processing Function (Asynchronous) – Refactored for Reduced Wait Time
# -------------------------------------------------------------------------------
async def process_solar_data(lat: float, lng: float, api_key: str, property_id: str) -> dict:
    """
    Main function that:
      1. Fetches building insights and data layers concurrently.
      2. Downloads DSM, RGB, Mask, Annual Flux, and Monthly Flux datasets concurrently.
      3. Processes and uploads each image as soon as its download completes.
      4. Composes annual and monthly composites (and a GIF from the monthly composites).
    Returns a dictionary with building insights and public URLs.
    """
    overall_start = time.time()
    logger.info("Starting solar data processing.")

    # Initialize Supabase client.
    supabase_url_val = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url_val or not supabase_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")
    sb_client = create_client(supabase_url_val, supabase_key)
    bucket_name = os.getenv("STORAGE_BUCKET_NAME", "property-images")
    remote_subdir = property_id

    # 1) Fetch building insights and data layers concurrently.
    url_building = (
        f"https://solar.googleapis.com/v1/buildingInsights:findClosest?key={api_key}"
        f"&location.latitude={lat:.5f}&location.longitude={lng:.5f}"
    )
    url_layers = (
        f"https://solar.googleapis.com/v1/dataLayers:get?key={api_key}"
        f"&location.latitude={lat:.5f}&location.longitude={lng:.5f}"
        "&radius_meters=100&view=FULL_LAYERS&required_quality=HIGH"
    )
    async with aiohttp.ClientSession() as session:
        logger.info("Fetching building insights and data layers...")
        bld_resp = await session.get(url_building)
        bld_resp.raise_for_status()
        building_insights = await bld_resp.json()

        layers_resp = await session.get(url_layers)
        layers_resp.raise_for_status()
        data_layers = await layers_resp.json()
    logger.info("Fetched building insights and data layers.")

    # Extract dataset URLs.
    dsm_url = data_layers.get("dsmUrl")
    rgb_url = data_layers.get("rgbUrl")
    mask_url = data_layers.get("maskUrl")
    annual_flux_url = data_layers.get("annualFluxUrl")
    monthly_flux_url = data_layers.get("monthlyFluxUrl")

    # 2) Start downloading datasets concurrently.
    dsm_future = asyncio.create_task(download_geotiff_subset(dsm_url, api_key, lat, lng))
    rgb_future = asyncio.create_task(download_geotiff_subset(rgb_url, api_key, lat, lng))
    mask_future = asyncio.create_task(download_geotiff_subset(mask_url, api_key, lat, lng))
    flux_future = asyncio.create_task(download_geotiff_subset(annual_flux_url, api_key, lat, lng))
    
    monthly_future = None
    if monthly_flux_url:
        monthly_future = asyncio.create_task(download_geotiff_subset(monthly_flux_url, api_key, lat, lng))
        logger.info("Started monthly flux data download.")

    # 3) As soon as each download completes, process and upload the images.
    async def process_dsm():
        dsm_data = await dsm_future
        return await process_and_upload_image(
            apply_color_ramp, sb_client, supabase_url_val, bucket_name,
            remote_subdir, "dsm", dsm_data['rasters'][0]
        )

    async def process_rgb():
        rgb_data = await rgb_future
        return await process_and_upload_image(
            create_rgb_image, sb_client, supabase_url_val, bucket_name,
            remote_subdir, "rgb", rgb_data['rasters']
        )

    async def process_mask():
        mask_data = await mask_future
        return await process_and_upload_image(
            create_mask_image, sb_client, supabase_url_val, bucket_name,
            remote_subdir, "mask", mask_data['rasters'][0]
        )

    async def process_flux():
        flux_data = await flux_future
        return await process_and_upload_image(
            apply_color_ramp, sb_client, supabase_url_val, bucket_name,
            remote_subdir, "flux", flux_data['rasters'][0]
        )

    # Launch processing tasks immediately.
    dsm_task  = asyncio.create_task(process_dsm())
    rgb_task  = asyncio.create_task(process_rgb())
    mask_task = asyncio.create_task(process_mask())
    flux_task = asyncio.create_task(process_flux())

    # Wait for the image processing uploads.
    processed_results = await asyncio.gather(dsm_task, rgb_task, mask_task, flux_task, return_exceptions=True)
    result_dict = dict(zip(
        ["dsm_public_url", "rgb_public_url", "mask_public_url", "flux_public_url"],
        processed_results
    ))

        # 4) Process monthly flux composites (if available).
    if monthly_future:
        monthly_data = await monthly_future  # Await monthly download when needed.
        logger.info("Monthly flux data downloaded.")

        # Process and upload raw monthly flux images in parallel.
        monthly_imgs = create_monthly_flux_images(monthly_data['rasters'])
        raw_monthly_upload_tasks = [
            upload_image_and_get_url(
                sb_client, supabase_url_val, bucket_name,
                f"{remote_subdir}/MonthlyFlux_{i+1}_{int(time.time()*1000)}.png",
                pil_image_to_bytes(img, fmt="PNG"),
                "image/png"
            )
            for i, img in enumerate(monthly_imgs)
        ]
        # Reproject mask and monthly flux concurrently.
        mask_data = mask_future.result()
        rgb_data = rgb_future.result()
        mask_reproj_task = asyncio.to_thread(
            reproject_to_match,
            mask_data['rasters'], mask_data['profile'], rgb_data['profile'], Resampling.nearest
        )
        monthly_reproj_task = asyncio.to_thread(
            reproject_to_match,
            monthly_data['rasters'], monthly_data['profile'], rgb_data['profile'], Resampling.cubic
        )
        (mask_reproj, _), (monthly_reproj, _) = await asyncio.gather(mask_reproj_task, monthly_reproj_task)

        # Process each monthly composite concurrently.
        composite_tasks = [
            asyncio.to_thread(
                process_single_monthly_composite_in_memory,
                rgb_data['rasters'][:3],
                monthly_reproj[i],
                mask_reproj[0]
            )
            for i in range(monthly_reproj.shape[0])
        ]
        monthly_flux_composite_images = await asyncio.gather(*composite_tasks)

        # Upload each monthly composite image concurrently.
        composite_upload_tasks = [
            upload_image_and_get_url(
                sb_client, supabase_url_val, bucket_name,
                f"{remote_subdir}/MonthlyFluxComposite_{i+1}_{int(time.time()*1000)}.png",
                pil_image_to_bytes(img, fmt="PNG"),
                "image/png"
            )
            for i, img in enumerate(monthly_flux_composite_images)
        ]


        # Create a GIF from the monthly composites.
        if monthly_flux_composite_images:
            gif_bytes = create_gif_in_memory(monthly_flux_composite_images)
            gif_filename = f"MonthlyFluxComposite_{int(time.time()*1000)}.gif"
            remote_path = f"{remote_subdir}/{gif_filename}"
            monthly_flux_composite_gif_public_url = await upload_image_and_get_url(
                sb_client, supabase_url_val, bucket_name, remote_path, gif_bytes, "image/gif"
            )
            result_dict["monthly_flux_composite_gif_public_url"] = monthly_flux_composite_gif_public_url

        
        monthly_flux_public_urls = await asyncio.gather(*raw_monthly_upload_tasks)
        result_dict["monthly_flux_public_urls"] = monthly_flux_public_urls
        
        monthly_flux_composite_public_urls = await asyncio.gather(*composite_upload_tasks)
        result_dict["monthly_flux_composite_public_urls"] = monthly_flux_composite_public_urls

        # 6) Process annual composite.
        # Retrieve the original downloaded data.
    flux_data = flux_future.result()
    mask_data = mask_future.result()
    rgb_data = rgb_future.result()

    # Reproject annual flux and mask concurrently.
    annual_flux_task = asyncio.to_thread(
        reproject_to_match,
        flux_data['rasters'], flux_data['profile'], rgb_data['profile'], Resampling.cubic
    )
    mask_reproj_task = asyncio.to_thread(
        reproject_to_match,
        mask_data['rasters'], mask_data['profile'], rgb_data['profile'], Resampling.nearest
    )
    (annual_flux_reproj, _), (mask_reproj, _) = await asyncio.gather(annual_flux_task, mask_reproj_task)

    # Compose and upload the annual composite.
    annual_composite_public_url = await compose_annual_composite_async(
        rgb_data['rasters'][:3], annual_flux_reproj, mask_reproj,
        sb_client, bucket_name, remote_subdir, supabase_url_val
    )
    result_dict["annual_flux_composite_public_url"] = annual_composite_public_url

    total_elapsed = time.time() - overall_start
    logger.info(f"Solar data processing completed in {total_elapsed:.2f}s.")

    return {"building_insights": building_insights, **result_dict}

# ------------------------------------------------------------------------------
# New Async Helpers for Composites
# ------------------------------------------------------------------------------
async def compose_annual_composite_async(rgb_rasters, flux_reproj, mask_reproj, sb_client, bucket_name, remote_subdir, supabase_url) -> str:
    """
    Compose an annual flux composite over the RGB image, upload it asynchronously,
    and return its public URL.
    """
    flux_img = apply_color_ramp(flux_reproj[0])
    flux_arr = np.array(flux_img).transpose(2, 0, 1)
    mask = np.where(mask_reproj[0] == 1, 1.0, 0.0)
    comp = composite_flux_on_rgb(rgb_rasters, flux_arr, mask)
    comp_img = Image.fromarray(comp, mode='RGBA')
    img_bytes = pil_image_to_bytes(comp_img, fmt="PNG")
    filename = f"FluxOverRGB_{int(time.time()*1000)}.png"
    remote_path = f"{remote_subdir}/{filename}"
    url = await upload_image_and_get_url(sb_client, supabase_url, bucket_name, remote_path, img_bytes, "image/png")
    return url

async def process_single_monthly_composite_async(rgb_rasters, monthly_band, mask_reproj_band, sb_client, bucket_name, remote_subdir, index, supabase_url) -> str:
    """
    Compose a composite image for a single monthly flux band, upload it asynchronously,
    and return its public URL.
    """
    img = apply_color_ramp(monthly_band)
    arr = np.array(img).transpose(2, 0, 1)
    mask = np.where(mask_reproj_band == 1, 1.0, 0.0)
    comp = composite_flux_on_rgb(rgb_rasters, arr, mask)
    comp_img = Image.fromarray(comp, mode='RGBA')
    img_bytes = pil_image_to_bytes(comp_img, fmt="PNG")
    filename = f"MonthlyFluxComposite_{index}_{int(time.time()*1000)}.png"
    remote_path = f"{remote_subdir}/{filename}"
    url = await upload_image_and_get_url(sb_client, supabase_url, bucket_name, remote_path, img_bytes, "image/png")
    return url

async def process_and_upload_image(process_func, sb_client, supabase_url, bucket_name, remote_subdir, prefix: str, *args, **kwargs) -> str:
    """
    Run a blocking image processing function in a thread, then upload the resulting image asynchronously
    and return its public URL.
    """
    image = await asyncio.to_thread(process_func, *args, **kwargs)
    img_bytes = pil_image_to_bytes(image, fmt="PNG")
    filename = f"{prefix}_{int(time.time()*1000)}.png"
    remote_path = f"{remote_subdir}/{filename}"
    url = await upload_image_and_get_url(sb_client, supabase_url, bucket_name, remote_path, img_bytes, "image/png")
    return url

# ------------------------------------------------------------------------------
# Celery Task Definition
# ------------------------------------------------------------------------------
from celery import Celery
celery_app = Celery("SunlinkCelery",
                    broker=os.getenv("UPSTASH_BROKER_URL"),
                    backend=os.getenv("UPSTASH_BROKER_URL"))
celery_app.conf.update(
    task_track_started=True,
    broker_use_ssl={"ssl_cert_reqs": False},
    redis_backend_use_ssl={"ssl_cert_reqs": False},
    include=["__main__"],
    broker_connection_retry_on_startup=True
)

@celery_app.task(name="process_solar_task")
def process_solar_task(property_id: str):
    """
    Celery task that:
      1. Fetches the property from Supabase.
      2. Processes solar data and waits for all uploads to complete.
      3. Updates the property record with the public URLs.
    """
    overall_start = time.time()
    logger.info(f"Starting process_solar_task for property_id={property_id}")

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        msg = "Missing SUPABASE_URL or SUPABASE_KEY"
        logger.error(msg)
        raise RuntimeError(msg)
    sb_client = create_client(supabase_url, supabase_key)

    resp = sb_client.from_("properties").select("id, latitude, longitude, status")\
                     .eq("id", property_id).single().execute()
    if not resp.data:
        logger.error(f"Property {property_id} not found in Supabase. Exiting.")
        return
    lat = resp.data.get("latitude")
    lng = resp.data.get("longitude")
    if lat is None or lng is None:
        logger.error(f"Property {property_id} missing lat/lng. Marking 'error'.")
        sb_client.from_("properties").update({"status": "error"}).eq("id", property_id).execute()
        return
    logger.info(f"Retrieved property for {property_id}.")

    sb_client.from_("properties").update({"status": "processing"}).eq("id", property_id).execute()

    api_key = os.getenv("API_KEY", "")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(process_solar_data(lat, lng, api_key, property_id))
    except Exception as e:
        logger.error(f"process_solar_data failed: {e}")
        sb_client.from_("properties").update({"status": "error"}).eq("id", property_id).execute()
        return
    finally:
        loop.close()
    logger.info("Solar data processed.")

    final_update = {
        "status": "completed",
        "building_insights_jsonb": results.get("building_insights", {}),
        "DSM": results.get("dsm_public_url"),
        "RGB": results.get("rgb_public_url"),
        "Mask": results.get("mask_public_url"),
        "AnnualFlux": results.get("flux_public_url"),
        "FluxOverRGB": results.get("annual_flux_composite_public_url"),
        "MonthlyFluxCompositeGIF": results.get("monthly_flux_composite_gif_public_url"),
        "MonthlyFlux12": results.get("monthly_flux_public_urls"),
        "MonthlyFluxComposites": results.get("monthly_flux_composite_public_urls"),
    }
    logger.info("Final update (shortened): %s", {k: v for k, v in final_update.items() if k != "building_insights_jsonb"})
    
    sb_client.from_("properties").update(final_update).eq("id", property_id).execute()
    logger.info("Property record updated.")
    total_elapsed = time.time() - overall_start
    logger.info(f"Completed process_solar_task for {property_id} in {total_elapsed:.2f} seconds")
