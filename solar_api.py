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

# We'll use aiohttp for async network requests
import aiohttp

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

load_dotenv()

app = FastAPI(title="Solar Fast API")

# -------------------------------------------------------------------
# Utility: Reproject/Resample one raster to match another
# -------------------------------------------------------------------
def reproject_to_match(src_array, src_profile, target_profile, resampling_method):
    """
    Reproject/resample src_array (shape [bands, H, W]) to align with target_profile's
    resolution, transform, CRS, etc.
    """
    dst_height = target_profile['height']
    dst_width = target_profile['width']
    dst_count = src_profile['count']
    dst_dtype = src_profile['dtype']

    # Prepare an empty array for the reprojected data
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
    reproj_profile.update({
        'count': dst_count,
        'dtype': dst_dtype
    })

    return reprojected_array, reproj_profile


# -------------------------------------------------------------------
# Utility: Composite flux (RGBA) over RGB using a building mask
# -------------------------------------------------------------------
def composite_flux_on_rgb(rgb_array, flux_rgba_array, mask_array):
    """
    Composite the flux RGBA array over the RGB array using the mask.
    """
    r = rgb_array[0].astype(float)
    g = rgb_array[1].astype(float)
    b = rgb_array[2].astype(float)
    alpha_full = np.full_like(r, 255, dtype=np.uint8).astype(float)
    rgb_rgba = np.stack([r, g, b, alpha_full], axis=-1)  # (H, W, 4)

    flux_rgba = flux_rgba_array.transpose(1, 2, 0).astype(float)  # (H, W, 4)

    # Apply building mask to flux alpha
    flux_rgba[..., 3] *= mask_array

    fa = flux_rgba[..., 3] / 255.0
    ra = rgb_rgba[..., 3] / 255.0
    out = np.zeros_like(rgb_rgba)

    for c in range(3):
        out[..., c] = flux_rgba[..., c] * fa + rgb_rgba[..., c] * (1.0 - fa)

    out_alpha = fa + ra * (1.0 - fa)
    out[..., 3] = out_alpha * 255.0

    final_rgba = np.clip(out, 0, 255).astype(np.uint8)
    return final_rgba


# -------------------------------------------------------------------
# Utility: Download and subset GeoTIFF data (async network calls)
# -------------------------------------------------------------------
async def download_geotiff_subset(url, api_key, center_lat, center_lng, extent_m=20):
    """
    Download a GeoTIFF asynchronously, crop it, and return relevant data.
    """
    if url and 'solar.googleapis.com' in url and 'key=' not in url:
        url += f'&key={api_key}'

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            raw_data = await resp.read()

    buffer_data = io.BytesIO(raw_data)
    with rasterio.open(buffer_data) as ds:
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        building_x, building_y = transformer.transform(center_lng, center_lat)

        left = building_x - extent_m
        right = building_x + extent_m
        top = building_y + extent_m
        bottom = building_y - extent_m
        window = from_bounds(left, bottom, right, top, ds.transform)

        subset = ds.read(window=window)
        transform = ds.window_transform(window)
        sub_profile = ds.profile.copy()
        sub_profile.update({
            'height': subset.shape[1],
            'width': subset.shape[2],
            'transform': transform,
            'count': ds.count
        })

    return {
        'rasters': subset,
        'width': subset.shape[2],
        'height': subset.shape[1],
        'num_bands': subset.shape[0],
        'profile': sub_profile
    }


# -------------------------------------------------------------------
# Utility: Save PIL image to specified sub-directory with unique name
# -------------------------------------------------------------------
def save_pil_image(img_pil, sub_dir, prefix='geoTiffImage'):
    """
    Save a PIL image into the specified sub_dir with a unique name.
    """
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    timestamp_ms = int(round(time.time() * 1000))
    filename = f"{prefix}_{timestamp_ms}.png"
    filepath = os.path.join(sub_dir, filename)
    img_pil.save(filepath, format='PNG')
    return filepath


# -------------------------------------------------------------------
# 1. Single-band Color Ramp (e.g., DSM, Flux)
# -------------------------------------------------------------------
def apply_color_ramp(raster_data, color_stops=None):
    """
    Convert single-band float array to a color using a ramp.
    """
    if color_stops is None:
        color_stops = [
            (0.0,  (0,   0,   128)),   # darkest
            (0.25, (0, 128,   128)),
            (0.5,  (0, 255,    0)),
            (0.75, (255, 255,  0)),
            (1.0,  (255,   0,  0))     # brightest
        ]

    arr = np.array(raster_data, dtype=float)
    arr[arr == -9999] = np.nan
    arr[np.isinf(arr)] = np.nan
    valid_mask = ~np.isnan(arr)

    h, w = arr.shape
    if not np.any(valid_mask):
        return Image.new('RGBA', (w, h), (0, 0, 0, 0))

    min_val = np.nanmin(arr[valid_mask])
    max_val = np.nanmax(arr[valid_mask])
    if math.isclose(min_val, max_val):
        return Image.new('RGBA', (w, h), (128, 128, 128, 255))

    rng = max_val - min_val
    norm = (arr - min_val) / rng
    norm[~valid_mask] = 0.0

    out_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    norm_flat = norm.flatten()
    out_rgba_flat = out_rgba.reshape(-1, 4)

    for i in range(len(color_stops) - 1):
        lower = color_stops[i][0]
        upper = color_stops[i + 1][0]
        mask = (norm_flat >= lower) & (norm_flat <= upper)
        frac = (norm_flat[mask] - lower) / (upper - lower)
        frac = np.clip(frac, 0, 1)
        r = color_stops[i][1][0] + frac * (color_stops[i + 1][1][0] - color_stops[i][1][0])
        g = color_stops[i][1][1] + frac * (color_stops[i + 1][1][1] - color_stops[i][1][1])
        b = color_stops[i][1][2] + frac * (color_stops[i + 1][1][2] - color_stops[i][1][2])
        out_rgba_flat[mask, 0] = r.astype(np.uint8)
        out_rgba_flat[mask, 1] = g.astype(np.uint8)
        out_rgba_flat[mask, 2] = b.astype(np.uint8)
        out_rgba_flat[mask, 3] = 255

    out_rgba = out_rgba_flat.reshape(h, w, 4)
    return Image.fromarray(out_rgba, mode='RGBA')


# -------------------------------------------------------------------
# 2. Create RGB Image from 3-band Raster
# -------------------------------------------------------------------
def create_rgb_image(rasters):
    """
    For a 3-band raster, build an RGBA image. 
    If <3 bands, fallback to color ramp on band 0.
    """
    bands, h, w = rasters.shape
    if bands < 3:
        return apply_color_ramp(rasters[0])

    r = rasters[0].astype(np.uint8)
    g = rasters[1].astype(np.uint8)
    b = rasters[2].astype(np.uint8)
    a = np.full((h, w), 255, dtype=np.uint8)
    rgba = np.stack([r, g, b, a], axis=-1)
    return Image.fromarray(rgba, mode='RGBA')


# -------------------------------------------------------------------
# 3. Create Building Mask Image
# -------------------------------------------------------------------
def create_mask_image(raster_data):
    """
    For a 1-bit building mask, treat '1' as white and '0' as black.
    -9999 => 0.
    """
    arr = np.array(raster_data, dtype=float)
    arr[arr == -9999] = 0.0
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)

    h, w = arr.shape
    rgba = np.stack([arr, arr, arr, np.full((h, w), 255, dtype=np.uint8)], axis=-1)
    return Image.fromarray(rgba, mode='RGBA')


# -------------------------------------------------------------------
# 4. Create Monthly Flux Images
# -------------------------------------------------------------------
def create_monthly_flux_images(rasters):
    """
    For shape (12, height, width). Return color-ramped PIL images for each band.
    """
    out_images = []
    band_count = rasters.shape[0]
    for i in range(band_count):
        band_data = rasters[i]
        img = apply_color_ramp(band_data)
        out_images.append(img)
    return out_images


# -------------------------------------------------------------------
# Main processing function (async)
# -------------------------------------------------------------------
async def process_solar_data(lat, lng, sub_dir, api_key):
    """
    Demonstrate how to:
      1) Fetch building insights & data layers
      2) Download DSM, RGB, Mask, annual flux, monthly flux
      3) Create color-ramped images, composites, and a monthly flux GIF
      4) Skip hourly shade logic

    Args:
        lat (float): Latitude
        lng (float): Longitude
        sub_dir (str): Local path for saving images
        api_key (str): Your Google Solar API key
    """
    # 1) Fetch building insights (JSON)
    url_building = (
        "https://solar.googleapis.com/v1/buildingInsights:findClosest?"
        f"key={api_key}&location.latitude={lat:.5f}&location.longitude={lng:.5f}"
    )
    # 2) Fetch data layers (JSON)
    url_layers = (
        "https://solar.googleapis.com/v1/dataLayers:get?"
        f"key={api_key}&location.latitude={lat:.5f}&location.longitude={lng:.5f}"
        "&radius_meters=100&view=FULL_LAYERS&required_quality=HIGH"
    )

    async with aiohttp.ClientSession() as session:
        async with session.get(url_building) as r_b:
            r_b.raise_for_status()
            building_insights = await r_b.json()

        async with session.get(url_layers) as r_l:
            r_l.raise_for_status()
            data_layers = await r_l.json()

    # Extract relevant URLs
    dsm_url = data_layers.get("dsmUrl")
    rgb_url = data_layers.get("rgbUrl")
    mask_url = data_layers.get("maskUrl")
    annual_flux_url = data_layers.get("annualFluxUrl")
    monthly_flux_url = data_layers.get("monthlyFluxUrl")

    # 3) Download data subsets (async)
    dsm_data, rgb_data, mask_data, flux_data = await asyncio.gather(
        download_geotiff_subset(dsm_url, api_key, lat, lng),
        download_geotiff_subset(rgb_url, api_key, lat, lng),
        download_geotiff_subset(mask_url, api_key, lat, lng),
        download_geotiff_subset(annual_flux_url, api_key, lat, lng)
    )

    monthly_data = None
    if monthly_flux_url:
        monthly_data = await download_geotiff_subset(monthly_flux_url, api_key, lat, lng)

    # 4) Convert to images & save locally

    # a) DSM => color ramp
    dsm_img = apply_color_ramp(dsm_data['rasters'][0])
    dsm_path = save_pil_image(dsm_img, sub_dir, prefix='DSM')

    # b) RGB => 3-band
    rgb_img = create_rgb_image(rgb_data['rasters'])
    rgb_path = save_pil_image(rgb_img, sub_dir, prefix='RGB')

    # c) Building Mask => black/white
    mask_img = create_mask_image(mask_data['rasters'][0])
    mask_path = save_pil_image(mask_img, sub_dir, prefix='Mask')

    # d) Annual Flux => color ramp (unmasked)
    flux_img = apply_color_ramp(flux_data['rasters'][0])
    flux_path = save_pil_image(flux_img, sub_dir, prefix='AnnualFlux')

    # e) Monthly Flux => 12 color-ramped images
    monthly_flux_paths = []
    if monthly_data is not None:
        monthly_imgs = create_monthly_flux_images(monthly_data['rasters'])
        for i, m_img in enumerate(monthly_imgs):
            out_path = save_pil_image(m_img, sub_dir, prefix=f'MonthlyFlux_{i+1}')
            monthly_flux_paths.append(out_path)

    # Reproject & composite annual flux on RGB
    re_flux_array, _flux_profile = reproject_to_match(
        flux_data['rasters'],
        flux_data['profile'],
        rgb_data['profile'],
        resampling_method=Resampling.cubic
    )
    re_mask_array, _ = reproject_to_match(
        mask_data['rasters'],
        mask_data['profile'],
        rgb_data['profile'],
        resampling_method=Resampling.nearest
    )

    flux_color_img = apply_color_ramp(re_flux_array[0])
    flux_color_arr = np.array(flux_color_img).transpose(2, 0, 1)

    # Convert mask to 0..1
    mask_float = np.where(re_mask_array[0] == 1, 1.0, 0.0)

    rgb_array = rgb_data['rasters'][:3]  # shape (3, H, W)
    comp_rgba = composite_flux_on_rgb(rgb_array, flux_color_arr, mask_float)
    final_composite_img = Image.fromarray(comp_rgba, mode='RGBA')
    composite_path = save_pil_image(final_composite_img, sub_dir, prefix='FluxOverRGB')

    # Optional: Reproject & composite monthly flux + create GIF
    monthly_flux_composite_paths = []
    monthly_comp_pil_images = []
    if monthly_data is not None:
        re_monthly_flux, _ = reproject_to_match(
            monthly_data['rasters'],
            monthly_data['profile'],
            rgb_data['profile'],
            resampling_method=Resampling.cubic
        )
        band_count = re_monthly_flux.shape[0]
        for i in range(band_count):
            band_color_img = apply_color_ramp(re_monthly_flux[i])
            band_color_arr = np.array(band_color_img).transpose(2, 0, 1)
            monthly_comp_rgba = composite_flux_on_rgb(rgb_array, band_color_arr, mask_float)
            monthly_comp_img = Image.fromarray(monthly_comp_rgba, mode='RGBA')
            out_path = save_pil_image(monthly_comp_img, sub_dir, prefix=f'MonthlyFluxComposite_{i+1}')
            monthly_flux_composite_paths.append(out_path)
            monthly_comp_pil_images.append(monthly_comp_img)

        # Create GIF
        if monthly_comp_pil_images:
            gif_filename = f"MonthlyFluxComposite_{int(time.time())}.gif"
            gif_filepath = os.path.join(sub_dir, gif_filename)
            monthly_comp_pil_images[0].save(
                gif_filepath,
                save_all=True,
                append_images=monthly_comp_pil_images[1:],
                duration=250,
                loop=0
            )

    # Return whatever details you need
    return {
        "building_insights": building_insights,
        "dsm_path": dsm_path,
        "rgb_path": rgb_path,
        "mask_path": mask_path,
        "flux_path": flux_path,
        "monthly_flux_paths": monthly_flux_paths,
        "annual_flux_composite_path": composite_path,
        "monthly_flux_composite_paths": monthly_flux_composite_paths,
        "monthly_flux_composite_gif": gif_filepath
    }


# ----------------------------------------------------------------------------
# FastAPI Endpoint
# ----------------------------------------------------------------------------

class SolarRequest(BaseModel):
    lat: float
    lng: float
    sub_dir: Optional[str] = "./solar_output"
    api_key: Optional[str] = None

@app.post("/process-solar")
async def solar_process_endpoint(req: SolarRequest):
    """
    Simple endpoint that wraps the process_solar_data() call.

    Usage:
      POST /process-solar
      {
        "lat": 40.4240,
        "lng": -86.9290,
        "sub_dir": "./solar_output",
        "api_key": "YOUR_GOOGLE_API_KEY"
      }
    """
    # If no key in request, attempt from environment
    final_key = req.api_key or os.getenv('API_KEY', '')

    # Invoke the async processing function
    results = await process_solar_data(req.lat, req.lng, req.sub_dir, final_key)
    return {"results": results}
