import cv2
import numpy as np
import math

from utils.mask_alpha import resize_mask_alpha

def get_lighting_map(img, blur_k=51):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur_k % 2 == 0: blur_k += 1
    gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    return gray.astype(np.float32) / 255.0

def blend_hard_replace(original, texture, mask_gray, shadow_strength=0.15):
    orig_f = original.astype(np.float32) / 255.0
    tex_f = texture.astype(np.float32) / 255.0

    lighting_map = get_lighting_map(original, blur_k=51)
    lighting_3ch = cv2.merge([lighting_map, lighting_map, lighting_map])

    shaded_texture = tex_f * (lighting_3ch ** shadow_strength)
    # mask_gray arrives pre-feathered from resize_mask_alpha, with the ramp width
    # scaled to the canvas. The (3, 3) blur that used to stand in for it here was
    # a no-op at 4500px and left the boundary hard.
    mask_f = mask_gray.astype(np.float32) / 255.0
    mask_3ch = cv2.merge([mask_f, mask_f, mask_f])
    
    result = (orig_f * (1.0 - mask_3ch)) + (shaded_texture * mask_3ch)
    return np.clip(result * 255, 0, 255).astype(np.uint8)

def _detect_floor_quad(room_img):
    """
    Analyzes the Base Image to calculate the true 3D perspective quad of the room floor.
    Returns: quad (TL, TR, BR, BL) and floor_top_y
    """
    H_orig, W_orig = room_img.shape[:2]

    MAX_PROC_W = 1536
    if W_orig > MAX_PROC_W:
        proc_scale = MAX_PROC_W / W_orig
        proc_img   = cv2.resize(room_img, (MAX_PROC_W, int(H_orig * proc_scale)))
    else:
        proc_scale = 1.0
        proc_img   = room_img

    H, W = proc_img.shape[:2]
    gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)

    def _weighted_median(values, weights):
        if not values:
            return None
        order = np.argsort(np.asarray(values))
        vals  = np.asarray(values,  dtype=np.float64)[order]
        wts   = np.asarray(weights, dtype=np.float64)[order]
        cum   = np.cumsum(wts)
        idx   = int(np.searchsorted(cum, cum[-1] * 0.5))
        return float(vals[min(idx, len(vals) - 1)])

    lower_y0   = max(0, int(H * 0.10))
    edges_full = cv2.Canny(gray[lower_y0:H, :], 35, 115)
    lines_full = cv2.HoughLinesP(
        edges_full, 1, np.pi / 180,
        threshold=max(22, W // 24),
        minLineLength=max(26, W // 9),
        maxLineGap=max(20, W // 24),
    )

    left_segs, right_segs = [], []
    if lines_full is not None:
        for x1_, y1_, x2_, y2_ in lines_full[:, 0]:
            y1g = y1_ + lower_y0;  y2g = y2_ + lower_y0
            dx  = float(x2_ - x1_); dy = float(y2g - y1g)
            if math.hypot(dx, dy) < max(24.0, W * 0.03): continue
            if abs(dy) < 12.0: continue
            slope = dy / (dx + 1e-6)
            if abs(slope) < 0.18 or abs(slope) > 8.0: continue
            xm = (x1_ + x2_) * 0.5
            if slope < 0 and xm < W * 0.62:
                left_segs.append((x1_, y1g, x2_, y2g))
            elif slope > 0 and xm > W * 0.38:
                right_segs.append((x1_, y1g, x2_, y2g))

    MAX_SEGS = 15
    if len(left_segs) > MAX_SEGS:
        left_segs  = sorted(left_segs,  key=lambda s: math.hypot(s[2]-s[0], s[3]-s[1]), reverse=True)[:MAX_SEGS]
    if len(right_segs) > MAX_SEGS:
        right_segs = sorted(right_segs, key=lambda s: math.hypot(s[2]-s[0], s[3]-s[1]), reverse=True)[:MAX_SEGS]

    edges_all = cv2.Canny(gray, 28, 90)
    x0s, x1s  = int(W * 0.08), int(W * 0.92)
    row_dens  = np.mean(edges_all[:, x0s:x1s].astype(np.float32), axis=1)

    smooth_k = max(10, H // 55)
    row_sm   = np.convolve(row_dens, np.ones(smooth_k) / smooth_k, mode='same')

    y_scan_lo = int(H * 0.30)
    y_scan_hi = int(H * 0.82)
    zone      = row_sm[y_scan_lo:y_scan_hi]
    threshold = max(float(np.std(zone)) * 0.50, 3.0)

    win = max(20, H // 30)
    floor_top_y = int(H * 0.62)
    for y in range(y_scan_hi, y_scan_lo, -1):
        above = float(np.mean(row_sm[max(0, y - win) : y]))
        below = float(np.mean(row_sm[y: min(H, y + win)]))
        if above - below >= threshold:
            floor_top_y = y
            break

    ref_lo = max(int(H * 0.30), floor_top_y - int(H * 0.08))
    ref_hi = min(int(H * 0.84), floor_top_y + int(H * 0.08))
    roi_ref  = gray[ref_lo:ref_hi, int(W * 0.04):int(W * 0.96)]
    edges_ref = cv2.Canny(roi_ref, 28, 95)
    lines_ref = cv2.HoughLinesP(
        edges_ref, 1, np.pi / 180,
        threshold=max(22, W // 22),
        minLineLength=max(28, W // 8),
        maxLineGap=max(20, W // 22),
    )
    if lines_ref is not None:
        floor_top_y_init = floor_top_y
        best_score, best_y = 0.0, floor_top_y
        for x1_, y1_, x2_, y2_ in lines_ref[:, 0]:
            if abs(y2_ - y1_) > 14: continue
            length = math.hypot(x2_ - x1_, y2_ - y1_)
            gy   = int((y1_ + y2_) * 0.5) + ref_lo
            dist = abs(gy - floor_top_y_init)
            score = (length / W) * math.exp(-dist / (H * 0.04))
            if score > best_score:
                best_score = score
                best_y     = gy
        if best_score > 0.10:
            floor_top_y = best_y

    floor_top_y = max(int(H * 0.33), min(int(H * 0.82), floor_top_y))

    left_top_hits,  left_top_w  = [], []
    left_bot_hits,  left_bot_w  = [], []
    right_top_hits, right_top_w = [], []
    right_bot_hits, right_bot_w = [], []

    for (x1_, y1g, x2_, y2g) in left_segs + right_segs:
        dx = float(x2_ - x1_); dy = float(y2g - y1g)
        seg_len = math.hypot(dx, dy)
        slope = dy / (dx + 1e-6)
        inv = dx / dy
        x_at_top = x1_ + (floor_top_y - y1g) * inv
        x_at_bottom = x1_ + (H - 1 - y1g) * inv
        if not (-0.35*W <= x_at_top <= 1.35*W and -0.45*W <= x_at_bottom <= 1.45*W):
            continue
        xm = (x1_ + x2_) * 0.5
        sw = seg_len * (1.0 + min(1.0, abs(slope) / 2.5))
        if slope < 0 and xm < W * 0.62:
            left_top_hits.append(x_at_top); left_top_w.append(sw)
            left_bot_hits.append(x_at_bottom); left_bot_w.append(sw)
        if slope > 0 and xm > W * 0.38:
            right_top_hits.append(x_at_top); right_top_w.append(sw)
            right_bot_hits.append(x_at_bottom); right_bot_w.append(sw)

    left_top_x     = _weighted_median(left_top_hits,  left_top_w)
    left_bottom_x  = _weighted_median(left_bot_hits,  left_bot_w)
    right_top_x    = _weighted_median(right_top_hits, right_top_w)
    right_bottom_x = _weighted_median(right_bot_hits, right_bot_w)

    left_conf  = len(left_top_hits)
    right_conf = len(right_top_hits)

    if left_conf < 3 or right_conf < 3:
        left_top_x    = W * 0.08;  right_top_x    = W * 0.92
        left_bottom_x = W * -0.02; right_bottom_x = W * 1.02

    left_top_x     = float(np.clip(left_top_x     or W*0.08,  -0.10*W, 0.55*W))
    right_top_x    = float(np.clip(right_top_x    or W*0.92,   0.45*W, 1.10*W))
    left_bottom_x  = float(np.clip(left_bottom_x  or W*-0.02, -0.22*W, 0.42*W))
    right_bottom_x = float(np.clip(right_bottom_x or W*1.02,   0.58*W, 1.22*W))

    min_top_width = max(18.0, W * 0.58)
    if right_top_x - left_top_x < min_top_width:
        cx = 0.5 * (left_top_x + right_top_x)
        left_top_x  = cx - min_top_width * 0.5
        right_top_x = cx + min_top_width * 0.5

    min_bottom_width = max(24.0, W * 0.42)
    if right_bottom_x - left_bottom_x < min_bottom_width:
        cx = 0.5 * (left_bottom_x + right_bottom_x)
        left_bottom_x  = cx - min_bottom_width * 0.5
        right_bottom_x = cx + min_bottom_width * 0.5

    MAX_TAPER    = 0.80
    actual_top_w = right_top_x - left_top_x
    actual_bot_w = right_bottom_x - left_bottom_x
    if actual_bot_w > 0 and actual_top_w / actual_bot_w > MAX_TAPER:
        target_bot_w   = actual_top_w / 0.62
        cx_bot         = 0.5 * (left_bottom_x + right_bottom_x)
        left_bottom_x  = cx_bot - target_bot_w * 0.5
        right_bottom_x = cx_bot + target_bot_w * 0.5

    left_bottom_x  -= W * 0.04
    right_bottom_x += W * 0.04
    left_bottom_x   = max(-0.30 * W, left_bottom_x)
    right_bottom_x  = min( 1.30 * W, right_bottom_x)

    quad = np.array([
        [left_top_x, floor_top_y],    # Top-Left
        [right_top_x, floor_top_y],   # Top-Right
        [right_bottom_x, H - 1],      # Bottom-Right
        [left_bottom_x, H - 1],       # Bottom-Left
    ], dtype=np.float32)

    if proc_scale != 1.0:
        quad[:, 0] /= proc_scale
        quad[:, 1] /= proc_scale
        quad[2, 1]  = H_orig - 1
        quad[3, 1]  = H_orig - 1
        floor_top_y = int(round(floor_top_y / proc_scale))

    return quad, floor_top_y

def tile_texture(pattern, area_w, area_h, total_stride_w, grout_width=0, grout_color=(180, 180, 180)):
    """
    Tiles the pattern so that the total width of one tile + its grout perfectly matches `total_stride_w`.
    Returns the tiled image and the exact integer tile dimensions (tw, th).
    """
    if isinstance(grout_color, str):
        try:
            hex_c = grout_color.lstrip('#')
            if len(hex_c) == 6:
                r = int(hex_c[0:2], 16)
                g = int(hex_c[2:4], 16)
                b = int(hex_c[4:6], 16)
                grout_color = (b, g, r)
            else:
                grout_color = (180, 180, 180) 
        except ValueError:
            grout_color = (180, 180, 180)

    # Calculate actual inner image width by subtracting grout
    pattern_w = max(1, int(total_stride_w) - grout_width)
    
    ph, pw = pattern.shape[:2]
    scale = pattern_w / float(pw)
    pattern_h = max(1, int(ph * scale))

    tile = cv2.resize(pattern, (pattern_w, pattern_h), interpolation=cv2.INTER_LANCZOS4)

    if grout_width > 0:
        th_grout = pattern_h + grout_width
        tw_grout = pattern_w + grout_width
        tile_with_grout = np.full((th_grout, tw_grout, 3), grout_color, dtype=np.uint8)
        tile_with_grout[0:pattern_h, 0:pattern_w] = tile
        tile = tile_with_grout

    th, tw = tile.shape[:2]
    if th == 0 or tw == 0: 
        return np.zeros((area_h, area_w, 3), dtype=np.uint8), 0, 0

    grid_h, grid_w = area_h + th, area_w + tw
    grid_out = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for y in range(0, grid_h, th):
        for x in range(0, grid_w, tw):
            h_slice = min(th, grid_h - y)
            w_slice = min(tw, grid_w - x)
            grid_out[y:y+h_slice, x:x+w_slice] = tile[:h_slice, :w_slice]

    return grid_out[:area_h, :area_w], tw, th

def apply_pattern(room_img, floor_tex, mask_img, repeat=3, rotation_deg=0, grout_width=0, grout_color=(180, 180, 180)):
    print(f"[INFO] Processing Floor. Rotation: {rotation_deg}, Grout: {grout_width}px, Repeat: {repeat}")
    H, W = room_img.shape[:2]
    
    if len(mask_img.shape) == 3: mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else: mask_gray = mask_img
    
    mask_gray = resize_mask_alpha(mask_gray, W, H)

    try:
        quad, _ = _detect_floor_quad(room_img)

        # Perspective mapping from quad
        dst_w = float(np.linalg.norm(quad[2] - quad[3]))
        dst_h = float(max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2])))
        
        pts_src = np.array([
            [0, 0],
            [dst_w, 0],
            [dst_w, dst_h],
            [0, dst_h]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts_src, quad)
        M_inv = np.linalg.inv(M)

        # Calculate dynamic tile scale based on Mask's physical flat width
        coords = cv2.findNonZero(mask_gray)
        if coords is None:
            return room_img
            
        coords_hom = np.ones((len(coords), 3), dtype=np.float32)
        coords_hom[:, :2] = coords[:, 0, :]
        
        flat_hom_scale = (M_inv @ coords_hom.T).T
        valid_mask_scale = flat_hom_scale[:, 2] > 0.001 
        flat_hom_scale = flat_hom_scale[valid_mask_scale]
        
        if len(flat_hom_scale) < 10:
            return room_img
            
        flat_coords_scale = flat_hom_scale[:, :2] / flat_hom_scale[:, 2:]

        # Use 2nd and 98th percentile to ignore random noise/splatters from the mask edge
        min_flat_x = np.percentile(flat_coords_scale[:, 0], 2)
        max_flat_x = np.percentile(flat_coords_scale[:, 0], 98)
        mask_flat_width = max(10.0, max_flat_x - min_flat_x)
        
        total_stride_w = float(mask_flat_width) / max(1, repeat)

        # Generate the base Tile (Just ONE tile, saves massive memory!)
        _, temp_tw, temp_th = tile_texture(floor_tex, 10, 10, total_stride_w, grout_width, grout_color)
        tw = max(1, int(temp_tw))
        th = max(1, int(temp_th))
        single_tile, _, _ = tile_texture(floor_tex, tw, th, total_stride_w, grout_width, grout_color)

        # Pure Numpy Inverse Mapping (Screen -> Flat Space -> Tile Space)
        # This prevents OpenCV's memory crashes or black voids by calculating pixel-by-pixel.
        x_start, y_start, w_box, h_box = cv2.boundingRect(mask_gray)
        px = np.arange(x_start, x_start + w_box)
        py = np.arange(y_start, y_start + h_box)
        X_screen, Y_screen = np.meshgrid(px, py)
        
        pts_screen = np.vstack((X_screen.ravel(), Y_screen.ravel(), np.ones_like(X_screen.ravel())))
        
        # Apply Inverse Perspective Matrix
        pts_flat_hom = M_inv @ pts_screen
        Z = pts_flat_hom[2, :]
        
        # STRICT horizon filter. Z <= 0 is physically behind the camera or above the horizon line.
        valid = Z > 0.0001
        
        X_flat = pts_flat_hom[0, valid] / Z[valid]
        Y_flat = pts_flat_hom[1, valid] / Z[valid]
        
        # Apply User Rotation
        cx, cy = 0.0, float(dst_h) 
        theta = math.radians(-rotation_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        X_shifted = X_flat - cx
        Y_shifted = Y_flat - cy
        
        X_rot = X_shifted * cos_t - Y_shifted * sin_t + cx
        Y_rot = X_shifted * sin_t + Y_shifted * cos_t + cy
        
        # Map infinite flat plane to the repeating Tile using Modulo Arithmetic
        U = ((X_rot - cx) % tw).astype(np.int32)
        V = ((Y_rot - cy) % th).astype(np.int32)
        
        # Safety clip to perfectly guarantee no index out-of-bounds
        U = np.clip(U, 0, tw - 1)
        V = np.clip(V, 0, th - 1)
        
        # Reconstruct the image vector-style
        warped_tex = np.zeros_like(room_img)
        valid_y = Y_screen.ravel()[valid]
        valid_x = X_screen.ravel()[valid]
        
        warped_tex[valid_y, valid_x] = single_tile[V, U]

        # Only pixels the inverse map actually wrote carry texture; anything the
        # STRICT horizon filter (Z <= 0.0001) rejected is still black. Clamp the
        # alpha to what was written instead of letting the blend mix in black and
        # darken the photo there. Built from the written coordinates rather than
        # from pixel darkness so a legitimately black tile isn't clipped away.
        covered = np.zeros((H, W), np.uint8)
        covered[valid_y, valid_x] = 255
        mask_gray = cv2.bitwise_and(mask_gray, covered)

        return blend_hard_replace(room_img, warped_tex, mask_gray, shadow_strength=0.15)
        
    except Exception as e:
        print(f"Error in apply_pattern: {e}")
        import traceback
        traceback.print_exc()
        return room_img