import os
import io
import math
import cv2
import base64
import requests
import numpy as np
import hashlib
import uuid
import time
import json
import logging
import threading
from PIL import Image, ImageOps
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, send_from_directory, url_for, send_file, abort
from flask_cors import CORS
from dotenv import load_dotenv
from functools import wraps
from functools import wraps as functools_wraps

# In-memory cache for processed base images: {room_id: (url, processed_img)}
_processed_base_cache = {}
_processed_base_cache_lock = threading.Lock()

# Cache of final rendered output: {output_cache_key: (final_image_url, rendered_layer_stack)}
# The stack snapshot carries post-render write-backs (e.g. the wall auto repeat), so cache hits return the same appliedHotspots as the original render.
_output_cache: dict[tuple, tuple] = {}
_output_cache_lock = threading.Lock()

# Per-room metric depth for the wall pipeline: {(room_id, canvas_hw): depth|None}
# Computed once per room canvas at reduced resolution (metric values are
# resolution-independent; the wall renderer resizes the map to its canvas).
_room_depth_cache: dict[tuple, object] = {}
_room_depth_cache_lock = threading.Lock()
_DEPTH_PROC_MAX_DIM = 1280

# Per-output-key render locks — prevent concurrent threads from rendering the same output
_render_locks: dict[tuple, threading.Lock] = {}
_render_locks_mutex = threading.Lock()

# Background executor for overlapping prefetch with base image download
_bg_executor = ThreadPoolExecutor(max_workers=4)

# Per-room locks so only one thread upscales/preprocesses a given room at a time
_room_process_locks: dict[str, threading.Lock] = {}
_room_process_locks_mutex = threading.Lock()

# Per-URL locks to prevent thundering herd on download_image
_download_locks: dict[str, threading.Lock] = {}
_download_locks_mutex = threading.Lock()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Timing decorator
def log_time(func):
    @functools_wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        msg = f"[TIMER] {func.__name__} took {end - start:.3f} seconds"
        print(msg)
        logging.info(msg)
        # Write to timings.log
        try:
            with open("timings.log", "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        except Exception as e:
            print(f"[TIMER-LOGGING-ERROR] {e}")
        return result
    return wrapper

from utils.curtain import apply_pattern as apply_curtain_pattern
from utils.rugs import _detect_floor_quad, _floor_quad_from_depth, _room_dims_from_depth, _estimate_floor_masks, _rug_masks_combine, b64_to_cv2, encode_shadow_map_b64, extract_shadow_map
from utils.floor import apply_pattern as apply_floor_pattern
# from utils.wall import apply_pattern as apply_wall_pattern
from utils.wall_depth import apply_pattern as apply_wall_pattern
from utils.pdf_generator import generate_report_pdf
from utils.segmentation import process_scene_pipeline, get_metric_depth
from utils.qr_generator import generate_catalogue_qr

load_dotenv()
app = Flask(__name__)
# CORS(app, origins="*")

ALLOWED_ORIGINS = [
    "https://ai.homexperia.com",
    "https://precarnival-ernesto-unbiting.ngrok-free.dev",
    "https://dev.homexperia.com",
    "http://localhost:5173",
    "http://localhost:5174"
]

CORS(app, resources={
    r"/api/*": {"origins": ALLOWED_ORIGINS},
    r"/uploads/*": {"origins": "*"},
    r"/generated/*": {"origins": "*"},
    r"/masks/*": {"origins": "*"}   
})

UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated'
MASK_FOLDER = 'masks'
OUTPUT_FOLDER = 'outputs'
CACHE_FOLDER = "image_cache"
DEBUG_FOLDER = "Debugs"
DEBUG_IMAGES = False
DATA_FILE = os.path.join('data', 'rooms_data.json')
IMAGE_FOLDER = os.path.join('static', 'room-images')
API_KEY = os.getenv("APP_API_KEY")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

os.makedirs(CACHE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

app.config['GENERATED_FOLDER'] = GENERATED_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
LIGHT_PURPLE = "\033[94m"
PURPLE = "\033[95m"
CYAN = "\033[96m"
LIGHT_GRAY = "\033[97m"
BLACK = "\033[90m"
RESET = "\033[00m"

@app.before_request
def enforce_strict_origins():
    # Bypass Static Assets
    public_paths = [f"/{UPLOAD_FOLDER}/", f"/{GENERATED_FOLDER}/", f"/{MASK_FOLDER}/"]
    if any(request.path.startswith(path) for path in public_paths):
        return  # Skip strict validation for images

    # Evaluate Origin and Referer headers
    origin = request.headers.get("Origin")
    referer = request.headers.get("Referer")

    # If an origin is present (all valid frontend web browsers), check it strictly
    if origin and origin not in ALLOWED_ORIGINS:
        abort(403)
        
    # If a user is directly opening the link or using Postman without an Origin header,
    # you can fallback to checking if the traffic originated from your domains:
    if not origin and referer:
        if not any(domain in referer for domain in ALLOWED_ORIGINS):
            abort(403)
    
    if not origin and not referer: abort(403)

def require_api_key(f):
    @wraps(f)
    def protected_function(*args, **kwargs):
        if request.headers.get('x-api-key') != API_KEY:
            return jsonify({'error': 'Unauthorized access'}), 401
        return f(*args, **kwargs)
    return protected_function

def require_admin_auth(f):
    @wraps(f)
    def pass_protected_function(*args, **kwargs):
        if request.headers.get('x-api-key') != API_KEY or request.headers.get('Authorization') != AUTH_TOKEN:
            return jsonify({'error': 'Unauthorized access'}), 401
        return f(*args, **kwargs)
    return pass_protected_function

@log_time
def load_room_data():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

@log_time
def download_image(url):
    if not url:
        return None

    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    cache_path = os.path.join(CACHE_FOLDER, f"{url_hash}.jpg")

    # Fast path: already cached, no lock needed
    if os.path.exists(cache_path):
        img = cv2.imread(cache_path, cv2.IMREAD_COLOR)
        if img is not None:
            cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"{LIGHT_GRAY}✅ [CACHE HIT] Loaded from disk: {url_hash}.jpg ({cache_size_mb:.2f}MB){RESET}")
            logging.info(f"[CACHE HIT] Disk cache reused for URL: {url[:60]}...")
            return img
        os.remove(cache_path)

    # Acquire a per-URL lock so only one thread downloads each unique URL
    with _download_locks_mutex:
        if url_hash not in _download_locks:
            _download_locks[url_hash] = threading.Lock()
        url_lock = _download_locks[url_hash]

    with url_lock:
        # Re-check cache after acquiring lock (another thread may have downloaded it)
        if os.path.exists(cache_path):
            img = cv2.imread(cache_path, cv2.IMREAD_COLOR)
            if img is not None:
                cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
                print(f"{LIGHT_GRAY}✅ [CACHE HIT] Loaded from disk (after lock): {url_hash}.jpg ({cache_size_mb:.2f}MB){RESET}")
                logging.info(f"[CACHE HIT] Disk cache reused (post-lock) for URL: {url[:60]}...")
                return img

        try:
            print(f"{CYAN}⬇️ [CACHE MISS] Downloading: {url[:60]}...{RESET}")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image")
            try:
                cv2.imwrite(cache_path, img)
                cached_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
                print(f"{LIGHT_GRAY}💾 [CACHED] Saved to disk: {url_hash}.jpg ({cached_size_mb:.2f}MB){RESET}")
                logging.info(f"[NEW CACHE] Stored image for URL: {url[:60]}...")
            except Exception as cache_err:
                print(f"{YELLOW}⚠️ [WARN] Could not cache image to disk: {cache_err}{RESET}")
            return img
        except Exception as e:
            print(f"{RED}🔴 [ERROR] Failed to download image from {url}: {e}{RESET}")
            return None

def find_category(category):
    if 'curtain' in category: return 'curtain'
    if 'floor' in category: return 'floor'
    if 'wall' in category: return 'wall'
    return 'curtain'

@log_time
def preprocess_image(image, room_id):
    if image is None: return None

    print(f"{CYAN}➡ [INFO] Applying pre-processing (Denoise + Sharpen)...{RESET}")
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    gaussian_blur = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian_blur, -0.5, 0)
    if DEBUG_IMAGES:
        cv2.imwrite(f"Debugs/debug_sharpened_{room_id}.jpg", sharpened)
    return sharpened

@log_time
def upscale_image(image, room_id, target_max_dim=4000):
    if image is None: return None
    
    h, w = image.shape[:2]
    current_max = max(h, w)

    if current_max >= target_max_dim: 
        return image
    
    scale = 4500 / current_max
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    print(f"{CYAN}➡ [INFO] Upscaling image from {w}x{h} to {new_w}x{new_h}{RESET}")

    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    if DEBUG_IMAGES:
        cv2.imwrite(f"Debugs/debug_upscaled_{room_id}.jpg", upscaled)
    return upscaled

def prefetch_layer_assets(layers):
    asset_urls = []

    for layer in layers:
        product_url = layer.get('product', {}).get('productImageUrl')
        mask_url = layer.get('mask_image')

        if product_url:
            asset_urls.append(product_url)
        if mask_url:
            asset_urls.append(mask_url)

    unique_urls = list(dict.fromkeys(asset_urls))
    if not unique_urls:
        return {}

    prefetched_assets = {}
    worker_count = min(8, len(unique_urls))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_url = {
            executor.submit(download_image, url): url
            for url in unique_urls
        }

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                prefetched_assets[url] = future.result()
            except Exception as exc:
                print(f"[WARN] Failed to prefetch asset {url}: {exc}")
                prefetched_assets[url] = None

    return prefetched_assets

@log_time
def get_or_create_mask(room_id, hotspot_id, base_image, coords, mask_url=None, prefetched_mask=None):
    mask_filename = f"mask_{room_id}_{hotspot_id}.png"
    mask_path = os.path.join(MASK_FOLDER, mask_filename)

    if os.path.exists(mask_path):
        print(f"{LIGHT_GRAY}➡ [INFO] Loading cached mask: {mask_filename}{RESET}")
        return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # if prefetched_mask is not None:
    #     print(f"➡ [INFO] Using prefetched mask for {hotspot_id}...")
    #     downloaded_mask = prefetched_mask.copy()
    #     if len(downloaded_mask.shape) == 3:
    #         downloaded_mask = cv2.cvtColor(downloaded_mask, cv2.COLOR_BGR2GRAY)
    #     cv2.imwrite(mask_path, downloaded_mask)
    #     return downloaded_mask

    if mask_url:
        print(f"{CYAN}➡ [INFO] Downloading mask from URL for {hotspot_id}...{RESET}")
        downloaded_mask = download_image(mask_url)
        if downloaded_mask is not None:
            if len(downloaded_mask.shape) == 3:
                downloaded_mask = cv2.cvtColor(downloaded_mask, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(mask_path, downloaded_mask)
            return downloaded_mask
        else:
            print(f"{YELLOW}⚠ [WARN] Failed to download mask from {mask_url}. Falling back to SAM API.{RESET}")

    print(f"{CYAN}➡ [INFO] Generating new mask for {hotspot_id}...{RESET}")

    orig_h, orig_w = base_image.shape[:2]
    MAX_DIM = 2040
    scale_factor = 1.0
    
    processed_image = base_image
    processed_coords = list(coords)

    if max(orig_h, orig_w) > MAX_DIM:
        scale_factor = MAX_DIM / max(orig_h, orig_w)
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        
        processed_image = cv2.resize(base_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        processed_coords[0] = int(coords[0] * scale_factor)
        processed_coords[1] = int(coords[1] * scale_factor)
        
        print(f"{CYAN}➡ [INFO] Downscaling for SAM: {orig_w}x{orig_h} -> {new_w}x{new_h} (Scale: {scale_factor:.4f}){RESET}")

    _, buffer = cv2.imencode('.jpg', processed_image)
    img_b64 = base64.b64encode(buffer).decode("utf-8")
    
    api_url = os.getenv("SAM_API_URL") 
    api_key = os.getenv("SAM_API_KEY")
    
    headers = { "x-api-key": api_key, "Content-Type": "application/json" }
    
    payload = {
        "base64": True,
        "image": img_b64,
        "overlay_mask": False,
        "refine_mask": True,
        "coordinates": str(processed_coords)
    }

    try:
        r = requests.post(api_url, headers=headers, json=payload, timeout=90)
        r.raise_for_status()
        
        response_json = r.json()
        if "image" in response_json:
            mask_b64 = response_json["image"]
        elif "masks" in response_json:
            mask_b64 = response_json["masks"][0]
        else:
            print(f"{RED}🔴 [ERROR] Unexpected API response keys{RESET}")
            return None

        mask_bytes = base64.b64decode(mask_b64)

        received_mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        
        if scale_factor != 1.0:
            final_mask = cv2.resize(received_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            print(f"{CYAN}➡ [INFO] Upscaled mask back to {orig_w}x{orig_h}{RESET}")
        else:
            final_mask = received_mask

        cv2.imwrite(mask_path, final_mask)
            
        return final_mask

    except Exception as e:
        print(f"{RED}🔴 [ERROR] SAM API Failed: {e}{RESET}")
        return None

def get_room_depth(room_id, image_cv2):
    """Metric depth map for a room canvas (Depth-Anything-V2), computed ONCE per (room, canvas size) and cached. 
    Runs at reduced resolution for speed; any failure caches None so the wall renderer degrades to its 2D path."""
    key = (room_id, image_cv2.shape[:2])
    with _room_depth_cache_lock:
        if key in _room_depth_cache:
            return _room_depth_cache[key]

    depth = None
    try:
        h, w = image_cv2.shape[:2]
        scale = min(1.0, _DEPTH_PROC_MAX_DIM / float(max(h, w)))
        proc = cv2.resize(image_cv2, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1.0 else image_cv2

        depth = get_metric_depth(proc)
        
        if depth is not None:
            depth = np.asarray(depth, dtype=np.float32)
            print(f"{GREEN}✅ [DEPTH] Wall depth map ready for room {room_id} ({depth.shape[1]}x{depth.shape[0]}){RESET}")
    
    except Exception as e:
        print(f"{YELLOW}⚠ [DEPTH] Wall depth estimation failed ({e}); walls use 2D detection{RESET}")

    with _room_depth_cache_lock:
        _room_depth_cache[key] = depth
    return depth

@log_time
def process_single_layer(current_image, layer_data, room_id, prefetched_assets=None):
    hotspot_id = layer_data.get('hotspotId')
    product_url = layer_data.get('product', {}).get('productImageUrl')
    mask_url = layer_data.get('mask_image')
    
    coordinates = layer_data.get('coords', {})
    h, w = current_image.shape[:2]

    if isinstance(coordinates, dict):
        raw_x = float(coordinates.get('x', 0))
        raw_y = float(coordinates.get('y', 0))
    elif isinstance(coordinates, list) and len(coordinates) >= 2:
        raw_x = float(coordinates[0])
        raw_y = float(coordinates[1])

    if raw_x >= 1 or raw_y >= 1:
        abs_x = int(raw_x)
        abs_y = int(raw_y)
    else:
        abs_x = int(raw_x * w)
        abs_y = int(raw_y * h)
        
    coords = [abs_x, abs_y]

    settings = layer_data.get('settings', {})

    repeat = int(settings.get('repeat', 12))
    shading = float(settings.get('shading', 0.6))
    rotation = int(settings.get('rotation', 0))
    groutWidth = int(settings.get('groutWidth', 0))
    groutColor = settings.get('groutColor', '#000000')

    prefetched_mask = None
    if prefetched_assets and mask_url in prefetched_assets:
        prefetched_mask = prefetched_assets[mask_url]

    mask = get_or_create_mask(
        room_id,
        hotspot_id,
        current_image,
        coords,
        mask_url,
        prefetched_mask=prefetched_mask,
    )
    
    if mask is None:
        print(f"{YELLOW}⚠ [WARN] Skipping layer {hotspot_id} due to missing mask.{RESET}")
        return current_image

    texture = None
    if prefetched_assets and product_url in prefetched_assets:
        cached_texture = prefetched_assets[product_url]
        if cached_texture is not None:
            texture = cached_texture.copy()

    if texture is None:
        texture = download_image(product_url)

    if texture is None:
        print(f"{YELLOW}⚠ [WARN] Skipping layer {hotspot_id} due to missing texture.{RESET}")
        return current_image

    category = find_category(layer_data.get('category').lower())
    print(f"{PURPLE}➡ [INFO] Applying {category} \nRepeat: {repeat}\nRot: {rotation}°\nShade: {shading}\nGroutWidth: {groutWidth}px\nGroutColor: {groutColor}\n{RESET}")

    orig_h, orig_w = current_image.shape[:2]
    MAX_DIM = 4500
    scale_factor = 1.0

    if max(orig_h, orig_w) > MAX_DIM:
        scale_factor = MAX_DIM / max(orig_h, orig_w)
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        
        processed_image = cv2.resize(current_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        processed_image = preprocess_image(processed_image, room_id = room_id)
        
        print(f"{CYAN}➡ [INFO] Downscaling the Image for Texture Application: {orig_w}x{orig_h} -> {new_w}x{new_h} (Scale: {scale_factor:.4f}){RESET}")

    current_image = processed_image if scale_factor != 1.0 else current_image

    try:
        if category == 'curtain':
            return apply_curtain_pattern(current_image, texture, mask, repeat=repeat, shading_strength=shading)
        elif category == 'floor':
            return apply_floor_pattern(current_image, texture, mask, repeat=repeat, rotation_deg=rotation, grout_width=groutWidth, grout_color=groutColor)
        elif category == 'wall':
            raw_wall_repeat = settings.get('repeat')
            try:
                wall_repeat = float(raw_wall_repeat) if raw_wall_repeat is not None else None
            except (TypeError, ValueError):
                wall_repeat = None

            wall_depth = None
            try:
                wall_depth = get_room_depth(room_id, current_image)
            except Exception as depth_err:
                print(f"[DEPTH] Unavailable for wall layer: {depth_err}")

            processed_image, auto_repeat = apply_wall_pattern(
                current_image,
                texture,
                mask,
                fallback_repeat=wall_repeat,
                depth_map=wall_depth
            )
            if auto_repeat is not None:
                layer_data['settings']['repeat'] = auto_repeat
            return processed_image
        else:
            return apply_curtain_pattern(current_image, texture, mask, repeat=repeat, shading_strength=shading)
            
    except Exception as e:
        print(f"{RED}🔴 [ERROR] Failed to apply pattern for {hotspot_id}: {e}{RESET}")
        return current_image

@app.route('/api/process-room', methods=['POST'])
@require_api_key
@log_time
def process_room():
    data = request.json
    
    room_id = data.get('roomId')
    base_image_url = data.get('baseImageUrl')
    raw_apply_hotspot = data.get('applyHotspot')
    
    incoming_layers = []
    if isinstance(raw_apply_hotspot, list):
        incoming_layers = raw_apply_hotspot
    elif isinstance(raw_apply_hotspot, dict):
        incoming_layers = [raw_apply_hotspot]
    else:
        incoming_layers = []

    global_product = data.get('product', {})
    
    existing_hotspots = data.get('appliedHotspots', [])
    remaining_hotspots = data.get('remainingHotspots', [])

    if not room_id or not base_image_url:
        return jsonify({'success': False, 'error': 'Missing roomId or baseImageUrl'}), 400

    full_layer_stack = list(existing_hotspots) 

    for layer_req in incoming_layers:
        new_hotspot_id = layer_req.get('hotspotId')
        if not new_hotspot_id: continue

        layer_product_data = layer_req.get('productImageUrl') or global_product
        
        active_layer_entry = {
            "category_type": layer_req.get('category_type'),
            "hotspotId": new_hotspot_id,
            "product": layer_product_data,
            "category": layer_req.get('category'),
            "coords": layer_req.get('coords'),
            "settings": layer_req.get('settings', {}),
            "mask_image": layer_req.get('mask_image')
        }

        replaced = False
        for i, existing_item in enumerate(full_layer_stack):
            if existing_item.get('hotspotId') == new_hotspot_id:
                full_layer_stack[i] = active_layer_entry
                replaced = True
                break
        
        if not replaced:
            full_layer_stack.append(active_layer_entry)

    prefetch_future = _bg_executor.submit(prefetch_layer_assets, full_layer_stack)

    with _processed_base_cache_lock:
        cached = _processed_base_cache.get(room_id)
        if cached and cached[0] == base_image_url:
            current_image = cached[1].copy()
        else:
            current_image = None

    if current_image is None:
        with _room_process_locks_mutex:
            if room_id not in _room_process_locks:
                _room_process_locks[room_id] = threading.Lock()
            room_lock = _room_process_locks[room_id]

        with room_lock:
            with _processed_base_cache_lock:
                cached = _processed_base_cache.get(room_id)
                if cached and cached[0] == base_image_url:
                    current_image = cached[1].copy()

            if current_image is None:
                raw_image = download_image(base_image_url)
                if raw_image is None:
                    return jsonify({'success': False, 'error': 'Failed to download base image'}), 400
                upscaled = upscale_image(raw_image, room_id)
                current_image = preprocess_image(upscaled, room_id)
                with _processed_base_cache_lock:
                    _processed_base_cache[room_id] = (base_image_url, current_image.copy())


    print(f"{CYAN}➡ [INFO] Re-rendering stack of {len(full_layer_stack)} layers...{RESET}")

    layer_hash = hashlib.md5(json.dumps(full_layer_stack, sort_keys=True, default=str).encode()).hexdigest()
    output_cache_key = (room_id, base_image_url, layer_hash)

    with _output_cache_lock:
        cached_output = _output_cache.get(output_cache_key)
    if cached_output:
        cached_url, cached_stack = cached_output
        print(f"{LIGHT_GRAY}➡ [INFO] Returning cached output for room {room_id}{RESET}")
        return jsonify({
            "success": True,
            "finalImageUrl": cached_url,
            "appliedHotspots": cached_stack,
            "remainingHotspots": remaining_hotspots
        })

    with _render_locks_mutex:
        if output_cache_key not in _render_locks:
            _render_locks[output_cache_key] = threading.Lock()
        render_lock = _render_locks[output_cache_key]

    with render_lock:
        with _output_cache_lock:
            cached_output = _output_cache.get(output_cache_key)
        if cached_output:
            cached_url, cached_stack = cached_output
            print(f"{LIGHT_GRAY}➡ [INFO] Returning cached output for room {room_id} (post-lock){RESET}")
            return jsonify({
                "success": True,
                "finalImageUrl": cached_url,
                "appliedHotspots": cached_stack,
                "remainingHotspots": remaining_hotspots
            })

        try:
            prefetched_assets = prefetch_future.result()

            for layer in full_layer_stack:
                current_image = process_single_layer(
                    current_image,
                    layer,
                    room_id,
                    prefetched_assets=prefetched_assets,
                )

            filename = f"final_{room_id}_{int(time.time())}.jpg"
            filepath = os.path.join(app.config['GENERATED_FOLDER'], filename)
            cv2.imwrite(filepath, current_image, [cv2.IMWRITE_JPEG_QUALITY, 85])

            final_image_url = f"https://precarnival-ernesto-unbiting.ngrok-free.dev/generated/{filename}"

            with _output_cache_lock:
                # Snapshot the stack (detached from request dicts) so hits
                # return the post-render settings, incl. wall repeat write-back
                _output_cache[output_cache_key] = (
                    final_image_url,
                    json.loads(json.dumps(full_layer_stack, default=str)),
                )

            return jsonify({
                "success": True,
                "finalImageUrl": final_image_url,
                "appliedHotspots": full_layer_stack,
                "remainingHotspots": remaining_hotspots
            })

        except Exception as e:
            print(f"{RED}🔴 [ERROR] Processing failed: {e}{RESET}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "success": False,
                "error": str(e),
                "finalImageUrl": base_image_url,
                "appliedHotspots": existing_hotspots,
                "remainingHotspots": remaining_hotspots
            })

@app.route('/api/curtain-generation', methods=['POST'])
@require_api_key
@log_time
def generate_and_segment_curtains():
    data = request.json
    image_url = data.get('image_url')
    mask_urls = data.get('mask_urls', [])
    curtain_style = data.get('curtain_style', 'pinch pleat')

    if not image_url or not mask_urls:
        return jsonify({'success': False, 'error': 'Missing image_url or mask_urls'}), 400

    try:
        # Trigger OpenAI Pipeline
        from utils.curtain_generation import run_generation_pipeline
        new_filepath, new_filename = run_generation_pipeline(
            image_url = image_url,
            mask_urls = mask_urls,
            curtain_style = curtain_style,
            upload_folder = app.config['UPLOAD_FOLDER'],
            mask_folder = MASK_FOLDER,
            cache_folder = CACHE_FOLDER
        )

        print("➡ [INFO] Re-running segmentation on newly generated room...")
        
        # Load the newly generated image into memory for OneFormer/SAM
        pil_image = Image.open(new_filepath).convert("RGB")
        new_room_id = str(uuid.uuid4()) # Generate a fresh ID for this new canvas state
        server_base_url = "https://precarnival-ernesto-unbiting.ngrok-free.dev"

        segmentation_result = process_scene_pipeline(
            image=pil_image,
            room_id=new_room_id,
            filename=new_filename,
            masks_folder=MASK_FOLDER,
            generated_folder=app.config['OUTPUT_FOLDER'],
            server_base_url=server_base_url
        )

        segmentation_result["original_wo_curtain"] = image_url

        return jsonify(segmentation_result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"🔴 [ERROR] Curtain generation & segmentation failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/reset', methods=['POST'])
@require_api_key
def reset_room():
    data = request.json
    room_id = data.get('roomId')

    if not room_id:
        return jsonify({'success': False, 'error': 'Missing roomId'}), 400

    with _output_cache_lock:
        stale_keys = [k for k in _output_cache if k[0] == room_id]
        for k in stale_keys:
            del _output_cache[k]
    with _processed_base_cache_lock:
        _processed_base_cache.pop(room_id, None)
    with _room_depth_cache_lock:
        stale_depth = [k for k in _room_depth_cache if k[0] == room_id]
        for k in stale_depth:
            del _room_depth_cache[k]

    deleted_count = 0

    target_folders = [app.config['GENERATED_FOLDER'], MASK_FOLDER]

    try:
        for folder in target_folders:
            if not os.path.exists(folder):
                continue
            
            for filename in os.listdir(folder):
                if f"_{room_id}_" in filename:
                    file_path = os.path.join(folder, filename)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"⚠ [WARN] Failed to delete {filename}: {e}")

        print(f"➡ [INFO] Reset complete for room {room_id}. Deleted {deleted_count} files.")
        return jsonify({'success': True})

    except Exception as e:
        print(f"🔴 [ERROR] Reset failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/mask-generation', methods=['POST'])
@require_api_key
def generate_masks_only():
    data = request.json
    room_id = data.get('roomId')
    hotspot_id = 'hotspotId'
    current_image = download_image(data.get('baseImageUrl'))
    coordinates = data.get('coords', {})

    h, w = current_image.shape[:2]

    if isinstance(coordinates, dict):
        raw_x = float(coordinates.get('x', 0))
        raw_y = float(coordinates.get('y', 0))
    elif isinstance(coordinates, list) and len(coordinates) >= 2:
        raw_x = float(coordinates[0])
        raw_y = float(coordinates[1])

    if raw_x >= 1 or raw_y >= 1:
        abs_x = int(raw_x)
        abs_y = int(raw_y)
    else:
        abs_x = int(raw_x * w)
        abs_y = int(raw_y * h)
        
    coords = [abs_x, abs_y]

    if not room_id or not coords:
        return jsonify({'success': False, 'error': 'Missing roomId or coordinates details'}), 400

    if current_image is None:
        return jsonify({'success': False, 'error': 'Could not get the base image for this URL'}), 404

    try:
        mask_filename = f"mask_{room_id}_{hotspot_id}.png"
        mask_path = os.path.join(MASK_FOLDER, mask_filename)
        
        if os.path.exists(mask_path):
            try:
                os.remove(mask_path)
                print(f"➡ [INFO] Mask Gen Call: Deleted cached mask {mask_filename} to force regeneration.")
            except Exception as del_err:
                print(f"⚠ [WARN] Mask Gen Call: Could not delete cached mask: {del_err}")

        mask_img = get_or_create_mask(room_id, hotspot_id, current_image, coords)

        if mask_img is not None:
            _, buffer = cv2.imencode('.png', mask_img)
            # mask_b64 = base64.b64encode(buffer).decode('utf-8')
            # mask_image = io.BytesIO(buffer)
            # return send_file(
            #     mask_image,
            #     mimetype='image/png',
            #     as_attachment=False, 
            #     download_name=f"mask_{room_id}.png"
            # )
            
            mask_url = f"https://precarnival-ernesto-unbiting.ngrok-free.dev/masks/mask_{room_id}_{hotspot_id}.png"
            return jsonify ({
                "success": True,
                "roomId": room_id,
                "maskImageUrl": mask_url
            })
        else:
            return jsonify ({
                "success": False,
                "roomId": room_id,
                "error": "SAM API returned no mask"
            }), 500

    except Exception as e:
        print(f"🔴 [ERROR] Mask generation failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate-pdf', methods=['POST'])
@require_api_key
def generate_pdf_report():
    try:
        data = request.json
        room_id = data.get('roomID', 'Unknown')
        
        pdf_output = generate_report_pdf(data)
        
        return send_file(
            pdf_output,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"HomeXperia_Design_{room_id}.pdf"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

def _encode_png_b64(image):
    ok, buf = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('Could not encode PNG image')
    return base64.b64encode(buf).decode('utf-8')

def _normalize_quad(quad, width, height):
    return [
        [round(float(x) / float(width), 6), round(float(y) / float(height), 6)]
        for x, y in quad.astype(np.float32)
    ]

@app.route('/api/rug-visualizer-scene', methods=['POST'])
@require_api_key
def rug_visualizer_scene():
    data = request.json or {}
    room_url = data.get('room_url')
    room_b64 = data.get('room_b64')
    floor_mask_urls = data.get('floor_mask_urls') or []

    try:
        room_img = download_image(room_url) if room_url else b64_to_cv2(room_b64)
        if room_img is None:
            return jsonify({'error': 'Could not load room image'}), 400

        combined_mask = None
        if floor_mask_urls:
            combined_mask = _rug_masks_combine(floor_mask_urls)

        floor_quad, floor_top_y = _detect_floor_quad(room_img, floor_mask=combined_mask)

        if combined_mask is not None:
            height_r, width_r = room_img.shape[:2]
            visible_floor = cv2.resize(combined_mask, (width_r, height_r), interpolation=cv2.INTER_AREA)
        else:
            visible_floor, _, _ = _estimate_floor_masks(room_img, floor_quad)

        height, width = room_img.shape[:2]
        
        # --- Metric depth: computed ONCE here, reused for perspective + room dimensions ---
        depth_map = None
        try:
            depth_map = get_metric_depth(room_img)
        except Exception as e:
            print(f"{YELLOW}⚠ [DEPTH] Metric depth estimation failed ({e}){RESET}")
            import traceback
            traceback.print_exc()

        focal_px = 0.8 * max(width, height)

        # --- Perspective Correction (rug floor quad) ---
        # Falls back to the geometric quad if depth can't fit a plane.
        persp_quad, persp_top_y = floor_quad, floor_top_y
        if depth_map is not None:
            try:
                depth_quad = _floor_quad_from_depth(depth_map, visible_floor, focal_px, room_img.shape)
                if depth_quad is not None:
                    persp_quad, persp_top_y = depth_quad
                    print(f"{GREEN}✅ [PERSPECTIVE] Using depth-based floor quad{RESET}")
                else:
                    print(f"{YELLOW}⚠ [PERSPECTIVE] Depth quad fit failed; keeping geometric quad{RESET}")
            except Exception as e:
                print(f"{YELLOW}⚠ [PERSPECTIVE] Depth perspective failed ({e}); keeping geometric quad{RESET}")
                import traceback
                traceback.print_exc()

        # --- Room Dimensions (depth-based) ---
        if depth_map is not None:
            try:
                dims = _room_dims_from_depth(depth_map, visible_floor, focal_px)
                if dims is not None:
                    room_width_ft = dims['width_ft']
                    room_length_ft = dims['length_ft']
                    room_area_sqft = dims['area_sqft']
                    print(f"{CYAN}➡ [DIMENSIONS] {room_width_ft} ft (W) x {room_length_ft} ft (L) | "
                          f"{room_area_sqft} sqft | floor depth {dims['median_depth_m']} m{RESET}")
                else:
                    print(f"{YELLOW}⚠ [DIMENSIONS] Could not fit a floor plane for dimensions{RESET}")
            except Exception as e:
                print(f"{YELLOW}⚠ [DIMENSIONS] Depth dimension estimation failed ({e}){RESET}")
                import traceback
                traceback.print_exc()

        # Final safety net (Default Values for room dimensions)
        if room_width_ft is None or room_length_ft is None:
            print(f"{YELLOW}⚠ [DIMENSIONS] No depth-based measurement -> using default 15.0 x 15.0 ft{RESET}")
            room_width_ft = 15.0
            room_length_ft = 15.0
        if room_area_sqft is None:
            room_area_sqft = round(room_width_ft * room_length_ft, 2)

        # Extract the shadow map using the original image and the visible floor mask
        shadow_map_float = extract_shadow_map(room_img, visible_floor)
        
        # Resize floor mask to max 1920px to keep payload small
        MAX_W = 3600
        if width > MAX_W:
            scale_f = MAX_W / width
            ow = MAX_W
            oh = int(round(height * scale_f))
            visible_floor = cv2.resize(visible_floor, (ow, oh), interpolation=cv2.INTER_AREA)
            shadow_map_float = cv2.resize(shadow_map_float, (ow, oh), interpolation=cv2.INTER_AREA)

        return jsonify({
            'room_width': width,
            'room_height': height,
            'floor_top_norm': round(float(persp_top_y) / float(height), 6),
            'floor_quad_norm': _normalize_quad(persp_quad, width, height),
            'floor_mask_b64': _encode_png_b64(visible_floor),
            'shadow_map_b64': encode_shadow_map_b64(shadow_map_float),
            'room_area_sqft': room_area_sqft,
            'room_width_ft': room_width_ft,
            'room_length_ft': room_length_ft
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/generated/<filename>')
def serve_generated(filename):
    return send_from_directory(app.config['GENERATED_FOLDER'], filename)

@app.route('/outputs/<filename>')
@require_admin_auth
def serve_outputs(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/masks/<filename>')
def serve_masks(filename):
    return send_from_directory(MASK_FOLDER, filename)

@app.route('/api/admin/cleanup/<target_folder>', methods=['DELETE'])
@require_admin_auth
def admin_cleanup(target_folder):
    ALLOWED_FOLDERS = {
        'uploads': app.config['UPLOAD_FOLDER'],
        'generated': app.config['GENERATED_FOLDER'],
        'masks': MASK_FOLDER
    }

    if target_folder not in ALLOWED_FOLDERS:
        return jsonify({
            'success': False, 
            'error': f'Invalid folder. Allowed: {list(ALLOWED_FOLDERS.keys())}'
        }), 400

    folder_path = ALLOWED_FOLDERS[target_folder]
    
    if not os.path.exists(folder_path):
        return jsonify({'success': False, 'error': 'Folder does not exist'}), 404

    deleted_count = 0
    errors = []

    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if os.path.isfile(file_path) or os.path.islink(file_path):
                try:
                    os.unlink(file_path)
                    deleted_count += 1
                except Exception as e:
                    errors.append(f"Failed to delete {filename}: {str(e)}")
        
        return jsonify({
            'success': True,
            'folder': target_folder,
            'deleted_files': deleted_count,
            'errors': errors if errors else None
        })

    except Exception as e:
        print(f"{RED}🔴 [CRITICAL] Cleanup failed for {target_folder}: {e}{RESET}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rooms', methods=['GET'])
@require_api_key
def get_all_rooms():
    data = load_room_data()
    summary = []
    
    for room_id, details in data.items():
        image_url = url_for('static', filename=f'room-images/{details["filename"]}', _external=True)
        
        summary.append({
            "roomId": room_id,
            "imageUrl": image_url
        })
    
    return jsonify({"rooms": summary})

@app.route('/api/room/<room_id>', methods=['GET'])
@require_api_key
def get_room_details(room_id):
    data = load_room_data()
    room = data.get(room_id)
    
    if not room:
        return jsonify({"error": "Room not found"}), 404
    
    room['imageUrl'] = url_for('static', filename=f'room-images/{room["filename"]}', _external=True)
    
    return jsonify(room)

@app.route('/api/cache/stats', methods=['GET'])
@require_admin_auth
def get_cache_stats():
    try:
        cache_files = []
        total_size_bytes = 0
        
        if os.path.exists(CACHE_FOLDER):
            for filename in os.listdir(CACHE_FOLDER):
                file_path = os.path.join(CACHE_FOLDER, filename)
                if os.path.isfile(file_path):
                    size_bytes = os.path.getsize(file_path)
                    total_size_bytes += size_bytes
                    cache_files.append({
                        'name': filename,
                        'size_mb': round(size_bytes / (1024 * 1024), 2)
                    })
        
        return jsonify({
            'cache_dir': CACHE_FOLDER,
            'total_files': len(cache_files),
            'total_size_mb': round(total_size_bytes / (1024 * 1024), 2),
            'total_size_gb': round(total_size_bytes / (1024 * 1024 * 1024), 3),
            'files': sorted(cache_files, key=lambda x: x['size_mb'], reverse=True)[:20]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/clear', methods=['POST'])
@require_admin_auth
def clear_cache():
    try:
        import shutil
        if os.path.exists(CACHE_FOLDER):
            shutil.rmtree(CACHE_FOLDER)
            os.makedirs(CACHE_FOLDER, exist_ok=True)
        return jsonify({'success': True, 'message': 'Image cache cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
@require_api_key
@log_time
def analyze_scene():
    room_id = str(uuid.uuid4())
    ext = "jpg"
    image = None

    try:
        if request.is_json and 'imageBase64' in request.json:
            base64_data = request.json['imageBase64']
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            img_data = base64.b64decode(base64_data)
            image = Image.open(BytesIO(img_data)).convert("RGB")
            image = ImageOps.exif_transpose(image)

        elif 'image' in request.files:
            file = request.files['image']
            image = Image.open(file).convert("RGB")
            image = ImageOps.exif_transpose(image)
            if '.' in file.filename:
                ext = file.filename.rsplit('.', 1)[1].lower()

        elif request.is_json and 'imageUrl' in request.json:
            response = requests.get(request.json['imageUrl'], timeout=15)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
        else:
            return jsonify({"success": False, "error": "No image provided (base64, file, or url)"}), 400

        filename = f"upload_{room_id}.{ext}"
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(img_path)

        server_base_url = "https://precarnival-ernesto-unbiting.ngrok-free.dev"

        print(f"{CYAN}➡ [INFO] Starting automatic scene analysis for Room: {room_id}{RESET}")
        result = process_scene_pipeline(
            image=image,
            room_id=room_id,
            filename=filename,
            masks_folder=MASK_FOLDER,
            generated_folder=app.config['OUTPUT_FOLDER'],
            server_base_url=server_base_url
        )

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"{RED}🔴 [ERROR] Scene analysis failed: {e}{RESET}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/catalogue-qr-generation', methods=['POST'])
@require_api_key
@log_time
def catalogue_qr_generation():
    data = request.json
    
    if not data:
        return jsonify({'success': False, 'error': 'Invalid or missing JSON payload'}), 400
        
    customer_code = data.get('customer_code')
    brand_logo = data.get('brand_logo')
    filter_value = data.get('filter_value')
    
    if not customer_code or not filter_value:
        return jsonify({'success': False, 'error': 'Missing customer_code or filter_value'}), 400
        
    try:
        print(f"{CYAN}➡ [INFO] Generating {len(filter_value)} QRs for {customer_code}...{RESET}")
        
        base_url = "https://precarnival-ernesto-unbiting.ngrok-free.dev"
        
        generated_data = generate_catalogue_qr(
            customer_code=customer_code,
            brand_logo_url=brand_logo,
            filter_value=filter_value,
            generated_folder=app.config['GENERATED_FOLDER'],
            base_url=base_url
        )
        
        print(f"{GREEN}✅ [SUCCESS] QR Generation complete.{RESET}")
        return jsonify({
            'success': True,
            'customerCode': customer_code,
            'qrCode': generated_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"{RED}🔴 [ERROR] QR Generation failed: {e}{RESET}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=3000)