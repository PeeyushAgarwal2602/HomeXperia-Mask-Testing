import cv2
import numpy as np
import math

def get_lighting_map(img, blur_k=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    return gray.astype(np.float32) / 255.0

def tile_texture(pattern, area_w, area_h, tile_size_w):
    ph, pw = pattern.shape[:2]
    scale = tile_size_w / float(pw)
    tile_size_h = int(ph * scale)
    tile = cv2.resize(pattern, (int(tile_size_w), int(tile_size_h)), interpolation=cv2.INTER_LANCZOS4)
    
    th, tw = tile.shape[:2]
    if th == 0 or tw == 0: return np.zeros((area_h, area_w, 3), dtype=np.uint8)

    grid_h, grid_w = area_h + th, area_w + tw
    grid_out = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for y in range(0, grid_h, th):
        for x in range(0, grid_w, tw):
            h_slice = min(th, grid_h - y)
            w_slice = min(tw, grid_w - x)
            grid_out[y:y+h_slice, x:x+w_slice] = tile[:h_slice, :w_slice]

    return grid_out[:area_h, :area_w]

def blend_realism(original, texture, mask_gray, opacity=1.0, shadow_strength=0.7):
    orig_f = original.astype(np.float32) / 255.0
    tex_f = texture.astype(np.float32) / 255.0
    
    lighting_map = get_lighting_map(original)
    lighting_3ch = cv2.merge([lighting_map, lighting_map, lighting_map])
    
    shaded_texture = tex_f * (lighting_3ch ** shadow_strength)
    
    mask_f = mask_gray.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (3, 3), 0)
    mask_3ch = cv2.merge([mask_f, mask_f, mask_f])
    
    result = (orig_f * (1.0 - mask_3ch)) + (shaded_texture * mask_3ch * opacity + orig_f * mask_3ch * (1-opacity))
    return np.clip(result * 255, 0, 255).astype(np.uint8)

# --- MAIN EXPORTED FUNCTION ---

def apply_pattern(room_img, curtain_tex, mask_img, repeat=12, shading_strength=0.7):
    print("[INFO] Processing Curtain (Accepting HD Input)...")
    H, W = room_img.shape[:2]
    
    if len(mask_img.shape) == 3: mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else: mask_gray = mask_img
    
    mask_gray = cv2.resize(mask_gray, (W, H), interpolation=cv2.INTER_NEAREST)
    
    try:
        tile_size = W / max(1, repeat)
        tiled_clean = tile_texture(curtain_tex, W, H, tile_size)
        
        fold_strength = 15 * (2.0 / 1.5) 
        
        gray = cv2.cvtColor(room_img, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask_gray) # Mask out everything except the curtain before blurring
        gray_blur = cv2.GaussianBlur(gray_masked, (31, 101), 0)
        
        disp_map = (gray_blur.astype(np.float32) - 127.5) / 127.5 
        
        map_x, map_y = np.meshgrid(np.arange(W), np.arange(H))
        map_x = map_x.astype(np.float32) + (disp_map * fold_strength)
        map_y = map_y.astype(np.float32)
        
        displaced_tex = cv2.remap(tiled_clean, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return blend_realism(room_img, displaced_tex, mask_gray, shadow_strength=shading_strength)

    except Exception as e:
        print(f"Error in apply_curtain_pattern: {e}")
        return room_img