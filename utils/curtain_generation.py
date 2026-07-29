import os
import uuid
import base64
import requests
import cv2
import numpy as np
from openai import OpenAI
from PIL import Image, ImageOps

client = OpenAI()

# def download_asset(url: str, cache_dir: str, prefix: str) -> str:
#     if not url: return None
    
#     filename = f"{prefix}_{uuid.uuid4().hex}.png"
#     filepath = os.path.join(cache_dir, filename)
    
#     resp = requests.get(url, timeout=30)
#     resp.raise_for_status()
    
#     with open(filepath, 'wb') as f:
#         f.write(resp.content)
        
#     return filepath

def fetch_download_asset(url: str, uploads_dir: str, masks_dir: str) -> str:
    if not url: return None
    
    parts = url.split('/')
    filename = parts[-1]
    parent_dir = parts[-2] if len(parts) >= 2 else ""

    if parent_dir == 'uploads':
        target_dir = uploads_dir
    elif parent_dir == 'masks':
        target_dir = masks_dir
    else:
        target_dir = uploads_dir

    local_path = os.path.join(target_dir, filename)

    # Use local file if it exists
    if os.path.exists(local_path):
        print(f"➡ [INFO] Found local asset: {filename}")
        return local_path

    # Else, download directly to the target folder
    print(f"⬇️ [INFO] Downloading missing asset: {filename}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    
    with open(local_path, 'wb') as f:
        f.write(resp.content)
        
    return local_path

def combine_masks(mask_paths: list, cache_dir: str) -> str:
    if not mask_paths: return None
    if len(mask_paths) == 1: return mask_paths[0]
    
    combined_mask = None
    for path in mask_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if combined_mask is None:
            combined_mask = img
        else:
            combined_mask = cv2.bitwise_or(combined_mask, img)
            
    combined_path = os.path.join(cache_dir, f"combined_mask_{uuid.uuid4().hex}.png")
    cv2.imwrite(combined_path, combined_mask)
    return combined_path

def prepare_images_for_openai(room_path: str, mask_path: str, output_room_path: str, output_mask_path: str):
    target_size = (1024, 1024)

    with Image.open(room_path) as room_img:
        room_img = room_img.convert("RGBA")
        room_square = ImageOps.pad(room_img, target_size, color=(0, 0, 0, 0))
        room_square.save(output_room_path, format="PNG")

    with Image.open(mask_path) as mask_img:
        mask_img = mask_img.convert("RGBA")
        mask_square = ImageOps.pad(mask_img, target_size, color=(0, 0, 0, 255))
        
        data = mask_square.getdata()
        new_data = [(255, 255, 255, 0) if item[0] > 200 else (0, 0, 0, 255) for item in data]
                
        mask_square.putdata(new_data)
        mask_square.save(output_mask_path, format="PNG")

def remove_padding(generated_path: str, original_path: str, output_path: str):
    with Image.open(original_path) as orig_img:
        orig_w, orig_h = orig_img.size

    target_size = 1024
    
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    
    with Image.open(generated_path) as gen_img:
        cropped_img = gen_img.crop((pad_x, pad_y, pad_x + new_w, pad_y + new_h)) # Crop out the padded bars
        
        final_img = cropped_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS) # Resize back to exact original dimensions using high-quality resampling
        
        final_img.save(output_path, format="PNG")

def run_generation_pipeline(image_url: str, mask_urls: list, curtain_style: str, upload_folder: str, mask_folder: str, cache_folder: str) -> tuple:
    
    print(f"➡ [INFO] Starting GenAI Curtain Pipeline for style: {curtain_style}")
    
    # Download Assets and combine masks
    local_room_path = fetch_download_asset(image_url, upload_folder, mask_folder)
    local_mask_paths = [fetch_download_asset(url, upload_folder, mask_folder) for url in mask_urls]
    combined_mask_path = combine_masks(local_mask_paths, cache_folder)
    
    if not combined_mask_path:
        raise ValueError("Failed to process masks for generation.")

    # Prepare formatting for OpenAI
    random_suffix = uuid.uuid4().hex
    ready_room_path = os.path.join(cache_folder, f"ready_room_{random_suffix}.png")
    ready_mask_path = os.path.join(cache_folder, f"ready_mask_{random_suffix}.png")
    
    prepare_images_for_openai(local_room_path, combined_mask_path, ready_room_path, ready_mask_path)
    
    # Call OpenAI
    prompt = f"A realistic, high-quality window dressing featuring {curtain_style} style curtains. Natural lighting matching the room. Use the masked area as the window location for the curtains. Don't alter the rest of the room."
    
    response = client.images.edit(
        model="gpt-image-2",
        image=open(ready_room_path, "rb"),
        mask=open(ready_mask_path, "rb"),
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    
    # Save raw OpenAI output to cache temporarily
    raw_gen_path = os.path.join(cache_folder, f"raw_gen_{uuid.uuid4().hex}.png")
    with open(raw_gen_path, "wb") as f:
        f.write(base64.b64decode(response.data[0].b64_json))

    # Remove padding and save the final hi-res image to UPLOAD folder
    print("➡ [INFO] Removing padding and restoring original resolution...")
    new_filename = f"upload_gen_{uuid.uuid4().hex}.png"
    new_filepath = os.path.join(upload_folder, new_filename)
    
    remove_padding(
        generated_path = raw_gen_path, 
        original_path = local_room_path, 
        output_path = new_filepath
    )
        
    print(f"✅ [SUCCESS] Final hi-res image saved to {new_filepath}")
            
    return new_filepath, new_filename