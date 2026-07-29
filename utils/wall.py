import cv2
import numpy as np
import math

def get_lighting_map(img, blur_k=51):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur_k % 2 == 0: blur_k += 1
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

def create_super_texture(pattern, target_w, target_h, tile_size_w):
    pad_w = target_w
    pad_h = target_h
    total_w = target_w + 2 * pad_w
    total_h = target_h + 2 * pad_h
    super_tex = tile_texture(pattern, total_w, total_h, tile_size_w)
    src_points = np.array([
        [pad_w, pad_h],               
        [pad_w + target_w, pad_h],    
        [pad_w + target_w, pad_h + target_h], 
        [pad_w, pad_h + target_h]     
    ], dtype="float32")
    return super_tex, src_points

def blend_hard_replace(original, texture, mask_gray, shadow_strength=0.6):
    orig_f = original.astype(np.float32) / 255.0
    tex_f = texture.astype(np.float32) / 255.0
    lighting_map = get_lighting_map(original, blur_k=51)
    lighting_3ch = cv2.merge([lighting_map, lighting_map, lighting_map])
    shaded_texture = tex_f * (lighting_3ch ** shadow_strength)
    mask_f = mask_gray.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (3, 3), 0)
    mask_3ch = cv2.merge([mask_f, mask_f, mask_f])
    result = (orig_f * (1.0 - mask_3ch)) + (shaded_texture * mask_3ch)
    return np.clip(result * 255, 0, 255).astype(np.uint8)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_global_corners(contours):
    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    if len(approx) == 4: pts_dst = np.squeeze(approx).astype(np.float32)
    else:
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        pts_dst = box
    return order_points(pts_dst)

def apply_pattern(room_img, wall_tex, mask_img, repeat=3):
    print("[INFO] Processing Wall (Accepting HD Input)...")
    H, W = room_img.shape[:2]
    if len(mask_img.shape) == 3: mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else: mask_gray = mask_img
    mask_gray = cv2.resize(mask_gray, (W, H), interpolation=cv2.INTER_NEAREST)
    _, thresh = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return room_img

    try:
        pts_dst = get_global_corners(contours)
        tile_size = W / max(1, repeat)
        super_tex, pts_src = create_super_texture(wall_tex, W, H, tile_size)
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped_tex = cv2.warpPerspective(super_tex, M, (W, H), flags=cv2.INTER_LINEAR)
        return blend_hard_replace(room_img, warped_tex, mask_gray, shadow_strength=0.4)
    except Exception as e:
        print(f"Error in apply_wall_pattern: {e}")
        return room_img