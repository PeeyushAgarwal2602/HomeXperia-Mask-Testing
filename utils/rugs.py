import cv2
import numpy as np
import math
import base64
import requests

def get_lighting_map(img, blur_k=51):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur_k % 2 == 0: blur_k += 1
    gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    return gray.astype(np.float32) / 255.0

def blend_hard_replace(original, texture, mask_gray, shadow_strength=0.1):
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

def extract_shadow_map(room_img, floor_mask):
    # The 'L' channel (Lightness) separates illumination from color perfectly.
    lab_img = cv2.cvtColor(room_img, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab_img)

    # This preserves sharp shadow edges while smoothing minor noise.
    l_smooth = cv2.bilateralFilter(l_channel, d=15, sigmaColor=75, sigmaSpace=75)
    
    floor_pixels = l_smooth[floor_mask > 0] # Analyze only the visible floor pixels
    
    if len(floor_pixels) == 0:
        return np.ones_like(l_channel, dtype=np.float32)

    base_lightness = np.percentile(floor_pixels, 85)
    shadow_map = l_smooth.astype(np.float32) / (base_lightness + 1e-5)
    
    # Pull the darks down and lower the clip floor.
    shadow_map = np.power(shadow_map, 1.5)       # Deepens the mid-tones
    shadow_map = np.clip(shadow_map, 0.15, 1.0)  # Allows shadows to get much darker (15% vs 40%)

    # Invert the visible floor mask so furniture, beds, and walls become solid white (255)
    inv_floor = cv2.bitwise_not(floor_mask)

    # Expand (dilate) this inverted mask to create a "valid shadow zone" around furniture. (~4% of image width, but at least 21px to handle smaller images)
    radius = max(21, int(room_img.shape[1] * 0.04) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    shadow_zone = cv2.dilate(inv_floor, kernel)

    # Smooth the shadow zone heavily so shadows fade out naturally at the edges
    fade_radius = max(31, int(room_img.shape[1] * 0.08) | 1)
    shadow_zone_float = cv2.GaussianBlur(shadow_zone.astype(np.float32), (fade_radius, fade_radius), 0) / 255.0

    # Keep the shadow map inside the zone, force pure white (1.0) on the open floor
    shadow_map = shadow_map * shadow_zone_float + 1.0 * (1.0 - shadow_zone_float)

    # Set anything completely outside the visible floor to 1.0
    shadow_map[floor_mask == 0] = 1.0

    return shadow_map

def encode_shadow_map_b64(shadow_map_float):
    shadow_map_uint8 = (shadow_map_float * 255).astype(np.uint8)
    
    # Encode as PNG
    success, buffer = cv2.imencode('.png', shadow_map_uint8)
    if not success:
        raise ValueError("Could not encode shadow map")
        
    return base64.b64encode(buffer).decode('utf-8')

def _detect_floor_quad(room_img, floor_mask=None):
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

    # Oblique perspective lines
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

    # Keep only 15 longest per side — kills curtain/rug noise
    MAX_SEGS = 15
    if len(left_segs) > MAX_SEGS:
        left_segs  = sorted(left_segs,  key=lambda s: math.hypot(s[2]-s[0], s[3]-s[1]), reverse=True)[:MAX_SEGS]
    if len(right_segs) > MAX_SEGS:
        right_segs = sorted(right_segs, key=lambda s: math.hypot(s[2]-s[0], s[3]-s[1]), reverse=True)[:MAX_SEGS]

    edges_all = cv2.Canny(gray, 28, 90)
    x0s, x1s  = int(W * 0.08), int(W * 0.92)
    row_dens  = np.mean(edges_all[:, x0s:x1s].astype(np.float32), axis=1)

    # Smooth over ±window rows to reduce per-pixel noise
    smooth_k = max(10, H // 55)
    row_sm   = np.convolve(row_dens, np.ones(smooth_k) / smooth_k, mode='same')

    # Dynamic threshold: half the std of edge density in the search zone
    y_scan_lo = int(H * 0.30)
    y_scan_hi = int(H * 0.82)
    zone      = row_sm[y_scan_lo:y_scan_hi]
    threshold = max(float(np.std(zone)) * 0.50, 3.0)

    win = max(20, H // 30)   # comparison window above/below each candidate y

    floor_top_y = int(H * 0.62)  # fallback

    if floor_mask is not None:
        if proc_scale != 1.0:
            mask_proc = cv2.resize(floor_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            mask_proc = floor_mask
            
        if len(mask_proc.shape) == 3:
            mask_proc = cv2.cvtColor(mask_proc, cv2.COLOR_BGR2GRAY)

        # --- ISOLATE THE LARGEST ISLAND ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_proc, connectivity=8)
        if num_labels > 1:
            # Find the largest component (excluding the black background at index 0)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            clean_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
        else:
            clean_mask = mask_proc

        coords = cv2.findNonZero(clean_mask)

        if coords is not None:
            min_y = int(np.min(coords[:, 0, 1]))
            offset = int(H * 0.02)
            floor_top_y = max(int(H * 0.20), min_y - offset)
    else:
        # Fallback to old edge-detection logic if no mask is provided
        for y in range(y_scan_hi, y_scan_lo, -1):    # bottom → top
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

    # TL/TR/BL/BR from oblique lines 
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

    # Give the bottom corners a slight outward flare to ensure they cover the screen width
    left_bottom_x  = float(np.clip(left_bottom_x  or W*-0.02, -0.20*W, 0.40*W))
    right_bottom_x = float(np.clip(right_bottom_x or W*1.02,   0.60*W, 1.20*W))
    
    bot_w = right_bottom_x - left_bottom_x
    bot_center = (left_bottom_x + right_bottom_x) / 2.0
    
    # Force the top center to perfectly align with the bottom center
    top_center = bot_center
    
    # Enforce a strict realistic taper for rugs
    ideal_taper_ratio = 0.55
    target_top_w = bot_w * ideal_taper_ratio
    
    left_top_x = top_center - (target_top_w / 2.0)
    right_top_x = top_center + (target_top_w / 2.0)

    quad = np.array([
        [left_top_x, floor_top_y],
        [right_top_x, floor_top_y],
        [right_bottom_x, H - 1],
        [left_bottom_x, H - 1],
    ], dtype=np.float32)

    # Scale quad back to original dimensions if we downsampled
    if proc_scale != 1.0:
        quad[:, 0] /= proc_scale
        quad[:, 1] /= proc_scale
        quad[2, 1]  = H_orig - 1
        quad[3, 1]  = H_orig - 1
        floor_top_y = int(round(floor_top_y / proc_scale))

    return quad, floor_top_y

def _floor_quad_from_depth(depth_m, floor_mask, focal_px, image_shape, coverage=0.98):
    if depth_m is None or floor_mask is None:
        return None

    H, W = depth_m.shape[:2]
    f = float(focal_px)
    cx, cy = W / 2.0, H / 2.0

    if floor_mask.shape[:2] != (H, W):
        floor_mask = cv2.resize(floor_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(floor_mask > 127)
    if len(xs) < 200:
        return None

    Z = depth_m[ys, xs].astype(np.float64)
    ok = np.isfinite(Z) & (Z > 0.1) & (Z < 30.0)
    xs, ys, Z = xs[ok], ys[ok], Z[ok]
    if len(xs) < 200:
        return None

    # Back-project floor pixels to 3D camera coordinates.
    X = (xs - cx) * Z / f
    Y = (ys - cy) * Z / f
    P = np.stack([X, Y, Z], axis=1)
    if len(P) > 40000:
        idx = np.linspace(0, len(P) - 1, 40000).astype(np.int64)
        P = P[idx]

    # Robust floor-plane fit: SVD normal + iterative outlier rejection.
    c = P.mean(axis=0)
    n = np.array([0.0, 1.0, 0.0])
    for _ in range(4):
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        n = vt[-1]
        dist = (P - c) @ n
        keep = np.abs(dist) <= (2.5 * float(np.std(dist)) + 1e-9)
        if keep.sum() < 50:
            break
        P = P[keep]
        c = P.mean(axis=0)

    # Two orthonormal axes spanning the floor plane.
    n = n / (np.linalg.norm(n) + 1e-9)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u1 = seed - (seed @ n) * n
    u1 /= (np.linalg.norm(u1) + 1e-9)
    u2 = np.cross(n, u1)

    A = (P - c) @ u1
    B = (P - c) @ u2

    lo = max(0.0, (1.0 - coverage) * 50.0) # coverage 0.98 -> [1, 99]
    a_lo, a_hi = np.percentile(A, [lo, 100 - lo])
    b_lo, b_hi = np.percentile(B, [lo, 100 - lo])
    inb = (A >= a_lo) & (A <= a_hi) & (B >= b_lo) & (B <= b_hi)
    pts2d = np.stack([A[inb], B[inb]], axis=1).astype(np.float32)
    if len(pts2d) < 20:
        return None

    box = cv2.boxPoints(cv2.minAreaRect(np.ascontiguousarray(pts2d)))  # 4 (a, b) corners

    H_img, W_img = image_shape[:2]
    img = []
    for a, b in box:
        P3 = c + a * u1 + b * u2 # back to 3D camera coords
        Zc = float(P3[2])
        if Zc <= 1e-3:
            return None
        img.append([cx + f * float(P3[0]) / Zc, cy + f * float(P3[1]) / Zc])
    img = np.array(img, dtype=np.float32)

    # Order corners TL, TR, BR, BL (sort by y into top/bottom pairs, then by x).
    order = img[np.argsort(img[:, 1])]
    top = order[:2][np.argsort(order[:2, 0])]
    bot = order[2:][np.argsort(order[2:, 0])]
    quad = np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)

    # Sanity: non-degenerate and not wildly off-screen.
    if cv2.contourArea(quad) < 0.02 * W_img * H_img:
        return None
    for x, y in quad:
        if x < -0.7 * W_img or x > 1.7 * W_img or y < -0.7 * H_img or y > 1.7 * H_img:
            return None

    floor_top_y = float(np.min(quad[:, 1]))
    return quad, floor_top_y

def _room_dims_from_depth(depth_m, floor_mask, focal_px):
    """Room width / length / area (ft) from a METRIC depth map + floor mask.

    Back-projects the floor to a 3D point cloud, robustly fits the floor plane,
    and measures the oriented rectangle that bounds it IN-PLANE (metres -> feet).
    Reads real 3D geometry, so it is immune to furniture rotation, occlusion and
    camera pitch -- the things that broke the old furniture-scale + vanishing-
    point estimate. Returns {width_ft, length_ft, area_sqft, median_depth_m} or
    None if a floor plane can't be fit.
    """
    M_TO_FT = 3.280839895
    ROOM_MIN, ROOM_MAX = 4.0, 30.0

    if depth_m is None or floor_mask is None:
        return None

    H, W = depth_m.shape[:2]
    f = float(focal_px)
    cx, cy = W / 2.0, H / 2.0

    if floor_mask.shape[:2] != (H, W):
        floor_mask = cv2.resize(floor_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(floor_mask > 127)
    if len(xs) < 200:
        return None

    Z = depth_m[ys, xs].astype(np.float64)
    ok = np.isfinite(Z) & (Z > 0.1) & (Z < 30.0)
    xs, ys, Z = xs[ok], ys[ok], Z[ok]
    if len(xs) < 200:
        return None

    # Back-project floor pixels to 3D camera coordinates (metres).
    X = (xs - cx) * Z / f
    Y = (ys - cy) * Z / f
    P = np.stack([X, Y, Z], axis=1)
    if len(P) > 40000:
        idx = np.linspace(0, len(P) - 1, 40000).astype(np.int64)
        P = P[idx]

    # Robust floor-plane fit: SVD normal + iterative outlier rejection.
    c = P.mean(axis=0)
    n = np.array([0.0, 1.0, 0.0])
    for _ in range(4):
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        n = vt[-1]
        dist = (P - c) @ n
        keep = np.abs(dist) <= (2.5 * float(np.std(dist)) + 1e-9)
        if keep.sum() < 50:
            break
        P = P[keep]
        c = P.mean(axis=0)

    # Two orthonormal axes spanning the floor plane.
    n = n / (np.linalg.norm(n) + 1e-9)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u1 = seed - (seed @ n) * n
    u1 /= (np.linalg.norm(u1) + 1e-9)
    u2 = np.cross(n, u1)

    if len(P) < 20:
        return None

    forward = np.array([0.0, 0.0, 1.0]) # Camera Optical Axis
    d_depth = forward - (forward @ n) * n # Forward Projected onto the floor plane
    
    if np.linalg.norm(d_depth) < 1e-6: d_depth = u1 # Near top-down view -> degenerate  
    d_depth = d_depth / (np.linalg.norm(d_depth) + 1e-9)
    
    d_lat = np.cross(n, d_depth) # In-Plane axis perpendicular to depth
    d_lat = d_lat / (np.linalg.norm(d_lat) + 1e-9)

    lat = (P - c) @ d_lat
    dep = (P - c) @ d_depth
    
    lat_lo, lat_hi = np.percentile(lat, [1, 99]) # Robust Extents (trim outliers)
    dep_lo, dep_hi = np.percentile(dep, [1, 99])

    width_ft = float(np.clip((lat_hi - lat_lo) * M_TO_FT, ROOM_MIN, ROOM_MAX))
    length_ft = float(np.clip((dep_hi - dep_lo) * M_TO_FT, ROOM_MIN, ROOM_MAX))

    return {
        "width_ft": round(width_ft, 2),
        "length_ft": round(length_ft, 2),
        "area_sqft": round(width_ft * length_ft, 2),
        "median_depth_m": round(float(np.median(Z)), 2),
    }

def _find_nearest_mask_pixel(mask, start_x, start_y, radius=36):
    height, width = mask.shape[:2]
    if 0 <= start_x < width and 0 <= start_y < height and mask[start_y, start_x] > 0:
        return start_x, start_y

    for delta in range(1, radius + 1):
        y0 = max(0, start_y - delta)
        y1 = min(height - 1, start_y + delta)
        x0 = max(0, start_x - delta)
        x1 = min(width - 1, start_x + delta)

        for y in range(y0, y1 + 1):
            if mask[y, x0] > 0: return x0, y
            if mask[y, x1] > 0: return x1, y
        for x in range(x0, x1 + 1):
            if mask[y0, x] > 0: return x, y0
            if mask[y1, x] > 0: return x, y1

    return None

def _estimate_floor_masks(room_img, floor_quad):
    """
    Estimate which parts of the detected floor remain visibly exposed.
    The complement becomes a soft occlusion overlay so rugs can slide under beds,
    sofas, and other furniture in the client-side visualizer.
    """
    height, width = room_img.shape[:2]

    floor_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(floor_mask, [floor_quad.astype(np.int32)], 255)

    blurred = cv2.GaussianBlur(room_img, (9, 9), 0)
    lab_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)

    seed_samples = []
    seed_points = []
    sample_x = (0.08, 0.22, 0.50, 0.78, 0.92)
    sample_y = (0.97, 0.92, 0.87, 0.82)

    for nx in sample_x:
        for ny in sample_y:
            px = int(round(nx * (width - 1)))
            py = int(round(ny * (height - 1)))
            nearest = _find_nearest_mask_pixel(floor_mask, px, py)
            if nearest is None:
                continue

            sx, sy = nearest
            seed_points.append((sx, sy))

            x0 = max(0, sx - 6)
            x1 = min(width, sx + 7)
            y0 = max(0, sy - 6)
            y1 = min(height, sy + 7)
            patch_mask = floor_mask[y0:y1, x0:x1] > 0
            patch_lab = lab_img[y0:y1, x0:x1]
            if np.any(patch_mask):
                seed_samples.append(patch_lab[patch_mask])

    if not seed_samples:
        return floor_mask, np.zeros_like(floor_mask), floor_mask

    sample_matrix = np.concatenate(seed_samples, axis=0)
    base_color = np.median(sample_matrix, axis=0)

    distances = np.linalg.norm(lab_img - base_color, axis=2)
    sample_distances = np.linalg.norm(sample_matrix - base_color, axis=1)
    dist_threshold = float(np.clip(np.percentile(sample_distances, 85) + 14.0, 18.0, 48.0))

    visible_candidates = np.where(
        (floor_mask > 0) & (distances <= dist_threshold),
        255,
        0,
    ).astype(np.uint8)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    visible_candidates = cv2.morphologyEx(visible_candidates, cv2.MORPH_OPEN, kernel_small)
    visible_candidates = cv2.morphologyEx(visible_candidates, cv2.MORPH_CLOSE, kernel_large)

    label_count, labels = cv2.connectedComponents(visible_candidates)
    keep_labels = set()

    for sx, sy in seed_points:
        label_id = int(labels[sy, sx])
        if label_id > 0:
            keep_labels.add(label_id)

    bottom_band = labels[max(0, height - 20): height, :]
    for label_id in np.unique(bottom_band):
        if label_id > 0:
            keep_labels.add(int(label_id))

    visible_floor_mask = np.where(np.isin(labels, list(keep_labels)), 255, 0).astype(np.uint8)
    visible_floor_mask = cv2.morphologyEx(visible_floor_mask, cv2.MORPH_CLOSE, kernel_large)
    visible_floor_mask = cv2.bitwise_and(visible_floor_mask, floor_mask)

    occluder_mask = cv2.subtract(floor_mask, visible_floor_mask)
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(occluder_mask)
    filtered_occluders = np.zeros_like(occluder_mask)
    min_area = max(120, (height * width) // 1800)

    for label_id in range(1, label_count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        filtered_occluders[labels == label_id] = 255

    filtered_occluders = cv2.GaussianBlur(filtered_occluders, (0, 0), sigmaX=2.4, sigmaY=2.4)
    filtered_occluders = cv2.bitwise_and(filtered_occluders, floor_mask)

    # Fill holes inside occluder regions (e.g. bed frame with gaps) and
    # dilate downward so the bed bottom edge fully covers the rug edge.
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    filtered_occluders = cv2.morphologyEx(filtered_occluders, cv2.MORPH_CLOSE, kernel_fill)

    # Anisotropic dilation: expand occluder DOWNWARD to seal the
    # bed-floor boundary so the rug edge is hidden behind the bed.
    down_px = max(10, int(height * 0.03))
    kernel_down = cv2.getStructuringElement(cv2.MORPH_RECT, (1, down_px * 2 + 1))
    dilated = cv2.dilate(filtered_occluders, kernel_down, anchor=(0, 0), iterations=1)
    filtered_occluders = cv2.bitwise_and(dilated, floor_mask)

    # ── Depth cutoff: clear occluder in the NEAR-CAMERA floor zone ──
    # The bottom portion of the floor (carpet, hardwood, etc.) must always show the rug.  Only the upper portion (where bed/furniture sits) should occlude.
    floor_top_y = int(np.min(floor_quad[:, 1]))
    floor_bot_y = int(np.max(floor_quad[:, 1]))
    floor_depth = max(1, floor_bot_y - floor_top_y)
    cutoff_y = int(floor_top_y + floor_depth * 0.50)

    # Above the cutoff: EVERYTHING inside the floor quad is occluder.
    # This ensures the bed, blankets, throw, cushions, bench — all of it fully hides the rug so no rug edge is visible near the bed.
    upper_occluder = np.zeros_like(filtered_occluders)
    cv2.fillPoly(upper_occluder, [floor_quad.astype(np.int32)], 255)
    upper_occluder[cutoff_y:, :] = 0  # only keep the top half

    # Merge: use full-quad occluder above cutoff, nothing below
    filtered_occluders = np.maximum(filtered_occluders, upper_occluder)
    filtered_occluders[cutoff_y:, :] = 0

    # Final soft edge
    filtered_occluders = cv2.GaussianBlur(filtered_occluders, (0, 0), sigmaX=3.0, sigmaY=3.0)

    return visible_floor_mask, filtered_occluders, floor_mask

def b64_to_cv2(b64_str):
    if b64_str and ',' in b64_str:
        b64_str = b64_str.split(',')[1]
    image_bytes = base64.b64decode(b64_str)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

def _rug_masks_combine(mask_urls):
    combined_mask = None
    kernel = np.ones((17, 17), np.uint8)
    for url in mask_urls:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            image_bytes = np.asarray(bytearray(resp.content), dtype=np.uint8)
            mask = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[WARN] Could not decode mask from {url}")
                continue
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            combined_mask = mask if combined_mask is None else cv2.bitwise_or(combined_mask, mask)
        except Exception as e:
            print(f"[WARN] Error fetching mask {url}: {e}")
    if combined_mask is not None:
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    return combined_mask