import cv2
import numpy as np
import math
import base64
import requests

from utils.cvcompat import as_points, as_segments

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
        for x1_, y1_, x2_, y2_ in as_segments(lines_full):
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

        coords = as_points(cv2.findNonZero(clean_mask))

        if len(coords):
            min_y = int(np.min(coords[:, 1]))
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
            for x1_, y1_, x2_, y2_ in as_segments(lines_ref):
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

def _floor_quad_from_depth(depth_m, floor_mask, focal_px, image_shape, coverage=0.99,
                           frame=None, scale_factor=1.0):
    """The floor rectangle the client renders rugs on, PLUS its own size in feet.

    Returns (quad, floor_top_y, u_span_ft, v_span_ft), where u_span_ft is the
    real length of the TL->TR edge and v_span_ft the TL->BL edge.

    The quad is the camera-aligned BOUNDING BOX of the whole floor mask, taken
    in the floor plane and projected back through the pinhole model -- so it is
    still a true floor rectangle (the client's homography stays exact) but it
    now covers every floor pixel instead of hugging the dense near-camera part.
    It used to be cv2.minAreaRect over the plane INLIERS, which is a minimum-
    area box over a point set the sigma-rejection had already eroded; on a real
    room that lands its far edge well short of the visible floor, and a rug
    sized against it can never reach the walls.

    Why the spans come from here and not from a separate measurement: the client
    normalises rug size against them (hu = rugLength/u_span, hv = rugWidth/u_span,
    vAxisScale = u_span/v_span) and then maps the result through THIS quad. The
    two therefore have to describe one rectangle. Measuring the spans off the
    corners actually being returned -- in the same plane frame, under the same
    scale_factor the dimensions use -- is what makes that true. Deriving them
    independently (a second plane fit, a different in-plane basis, or a scale
    correction applied to one and not the other) rescales every rug by exactly
    however much the two estimates disagree.

    frame: a pre-fitted _fit_floor_frame(); pass it to avoid refitting.
    scale_factor: the reference-object depth correction, applied exactly as
    _room_dims_from_depth applies it. Scaling depth uniformly leaves the
    projected pixel corners untouched and scales every in-plane extent, so the
    quad is unchanged and only its measured size moves.
    """
    if frame is None:
        frame = _fit_floor_frame(depth_m, floor_mask, focal_px)
    if frame is None:
        return None

    H, W = depth_m.shape[:2]
    f = float(focal_px)
    cx, cy = W / 2.0, H / 2.0

    c = frame["center"]
    d_lat, d_dep = frame["d_lat"], frame["d_depth"]

    # Measure the extent over EVERY floor-mask point, not the plane inliers.
    # Points far off the plane are still dropped (segmentation bleed, depth
    # garbage) but with a fixed physical tolerance instead of a sigma that
    # tightens each iteration and quietly discards the far floor.
    P = frame.get("points_all")
    if P is None or len(P) < 200:
        P = frame["points"]
    n = frame["normal"]
    resid = np.abs((P - c) @ n)
    tol = max(0.20, 4.0 * float(frame.get("plane_resid_m") or 0.0))
    on_plane = resid <= tol
    if on_plane.sum() >= 200:
        P = P[on_plane]

    lat = (P - c) @ d_lat
    dep = (P - c) @ d_dep

    # Trim only enough to shrug off stragglers. This is a percentile over POINT
    # COUNT, and perspective packs most pixels into the near floor, so a wide
    # trim here costs a lot of real floor depth for very little robustness.
    lo = max(0.0, (1.0 - coverage) * 50.0) # coverage 0.99 -> [0.5, 99.5]
    a_lo, a_hi = np.percentile(lat, [lo, 100 - lo])
    b_lo, b_hi = np.percentile(dep, [lo, 100 - lo])
    if not (a_hi > a_lo and b_hi > b_lo):
        return None

    H_img, W_img = image_shape[:2]

    # A point on the floor plane -> pixel. Zc is linear in (a, b), so a corner
    # that lands at or behind the horizon is walked back toward the cloud centre
    # until it projects, instead of failing the whole fit.
    def _project(a, b, eps=1e-2):
        for _ in range(24):
            P3 = c + a * d_lat + b * d_dep
            Zc = float(P3[2])
            if Zc > eps:
                return [cx + f * float(P3[0]) / Zc, cy + f * float(P3[1]) / Zc], a, b
            a *= 0.85
            b *= 0.85
        return None, a, b

    # Corners of the floor mask's camera-aligned bounding box, in the plane.
    box = [(a_lo, b_hi), (a_hi, b_hi), (a_hi, b_lo), (a_lo, b_lo)]  # far-L, far-R, near-R, near-L
    img, plane = [], []
    for a, b in box:
        pt, a2, b2 = _project(a, b)
        if pt is None:
            return None
        img.append(pt)
        plane.append([a2, b2])
    img = np.array(img, dtype=np.float32)
    plane = np.array(plane, dtype=np.float64)

    # Order corners TL, TR, BR, BL (sort by y into top/bottom pairs, then by x).
    # Sort INDICES rather than the array so `plane` follows `img` into the same
    # order and the spans below are measured on the corners we actually return.
    order = np.argsort(img[:, 1])
    top = order[:2][np.argsort(img[order[:2], 0])]
    bot = order[2:][np.argsort(img[order[2:], 0])]
    idx = [int(top[0]), int(top[1]), int(bot[1]), int(bot[0])]
    quad = img[idx].astype(np.float32)
    plane_quad = plane[idx]

    # Sanity: non-degenerate, finite, and the far edge really is above the near
    # edge. Corners are NOT required to sit near the frame any more -- a quad
    # that covers the whole floor legitimately runs off the bottom and sides,
    # and the client's homography extrapolates there perfectly well.
    if not np.all(np.isfinite(quad)):
        return None
    if cv2.contourArea(quad) < 0.02 * W_img * H_img:
        return None
    if float(np.mean(quad[:2, 1])) >= float(np.mean(quad[2:, 1])):
        return None

    s = float(scale_factor) if scale_factor and scale_factor > 0 else 1.0
    u_span_ft = float(np.linalg.norm(plane_quad[1] - plane_quad[0]) * M_TO_FT * s)
    v_span_ft = float(np.linalg.norm(plane_quad[3] - plane_quad[0]) * M_TO_FT * s)

    floor_top_y = float(np.min(quad[:, 1]))
    return quad, floor_top_y, u_span_ft, v_span_ft

M_TO_FT = 3.280839895


def _backproject(depth_m, mask, focal_px, min_pts=200, max_pts=40000):
    """Mask pixels -> 3D camera-space points (metres) via the pinhole model.
    Returns (points Nx3, all valid depths) or (None, None)."""
    H, W = depth_m.shape[:2]
    f = float(focal_px)
    cx, cy = W / 2.0, H / 2.0

    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask > 127)
    if len(xs) < min_pts:
        return None, None

    Z = depth_m[ys, xs].astype(np.float64)
    ok = np.isfinite(Z) & (Z > 0.1) & (Z < 30.0)
    xs, ys, Z = xs[ok], ys[ok], Z[ok]
    if len(xs) < min_pts:
        return None, None

    X = (xs - cx) * Z / f
    Y = (ys - cy) * Z / f
    P = np.stack([X, Y, Z], axis=1)
    if len(P) > max_pts:
        idx = np.linspace(0, len(P) - 1, max_pts).astype(np.int64)
        P = P[idx]
    return P, Z


def _fit_floor_frame(depth_m, floor_mask, focal_px):
    """Fit the floor plane and build an in-plane coordinate frame.

    Shared by the room-dimension measurement and the reference-object
    calibration so both read the SAME plane, and it is only fitted once per
    request. Returns a dict with the plane (center/normal), an orthonormal
    in-plane basis (u1/u2), the camera-aligned axes the dimensions use
    (d_lat/d_dep), the plane-inlier points, and every valid floor depth --
    or None if a plane can't be fit.
    """
    if depth_m is None or floor_mask is None:
        return None

    P, Z_all = _backproject(depth_m, floor_mask, focal_px)
    if P is None:
        return None

    # Every floor-mask point, kept BEFORE outlier rejection. The rejection below
    # is the right thing for FITTING the plane but the wrong thing for measuring
    # how far the floor reaches: far-floor pixels are sparse and depth-noisy, so
    # four rounds of 2.5-sigma trimming eat them, and any extent measured on the
    # survivors stops short of the floor you can actually see.
    P_all = P.copy()

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

    if len(P) < 20:
        return None

    plane_resid_m = float(np.std((P - c) @ (n / (np.linalg.norm(n) + 1e-9))))

    # Two orthonormal axes spanning the floor plane.
    n = n / (np.linalg.norm(n) + 1e-9)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u1 = seed - (seed @ n) * n
    u1 /= (np.linalg.norm(u1) + 1e-9)
    u2 = np.cross(n, u1)

    forward = np.array([0.0, 0.0, 1.0]) # Camera Optical Axis
    d_depth = forward - (forward @ n) * n # Forward Projected onto the floor plane

    if np.linalg.norm(d_depth) < 1e-6: d_depth = u1 # Near top-down view -> degenerate
    d_depth = d_depth / (np.linalg.norm(d_depth) + 1e-9)

    d_lat = np.cross(n, d_depth) # In-Plane axis perpendicular to depth
    d_lat = d_lat / (np.linalg.norm(d_lat) + 1e-9)

    return {
        "center": c, "normal": n, "u1": u1, "u2": u2,
        "d_lat": d_lat, "d_depth": d_depth,
        "points": P, "points_all": P_all, "floor_depths": Z_all,
        "plane_resid_m": plane_resid_m,
        "camera_height_m": float(abs(c @ n)),
    }


def _room_dims_from_depth(depth_m, floor_mask, focal_px, scale_factor=1.0, frame=None):
    """Room width / length / area (ft) from a METRIC depth map + floor mask.

    Back-projects the floor to a 3D point cloud, robustly fits the floor plane,
    and measures the oriented rectangle that bounds it IN-PLANE (metres -> feet).
    Reads real 3D geometry, so it is immune to furniture rotation, occlusion and
    camera pitch -- the things that broke the old furniture-scale + vanishing-
    point estimate. Returns {width_ft, length_ft, area_sqft, median_depth_m} or
    None if a floor plane can't be fit.

    scale_factor: multiplier from the reference-object calibration (see
    reference_scale_factor). Multiplying BOTH axes is exactly equivalent to
    rescaling the depth map itself -- scaling depth by s scales X, Y and Z
    uniformly, leaving the plane normal unchanged and every in-plane extent
    scaled by s -- which is the right model for a depth-scale error.

    frame: a pre-fitted _fit_floor_frame(); pass it to avoid refitting.
    """
    ROOM_MIN, ROOM_MAX = 4.0, 30.0

    if frame is None:
        frame = _fit_floor_frame(depth_m, floor_mask, focal_px)
    if frame is None:
        return None

    P, c = frame["points"], frame["center"]
    lat = (P - c) @ frame["d_lat"]
    dep = (P - c) @ frame["d_depth"]

    lat_lo, lat_hi = np.percentile(lat, [1, 99]) # Robust Extents (trim outliers)
    dep_lo, dep_hi = np.percentile(dep, [1, 99])

    s = float(scale_factor) if scale_factor and scale_factor > 0 else 1.0

    # Scale in METRES, before the sanity clamp. Clamping first and multiplying
    # after would let a 2x correction push a clamped 30 ft room out to 60 ft.
    width_ft = float(np.clip((lat_hi - lat_lo) * M_TO_FT * s, ROOM_MIN, ROOM_MAX))
    length_ft = float(np.clip((dep_hi - dep_lo) * M_TO_FT * s, ROOM_MIN, ROOM_MAX))

    return {
        "width_ft": round(width_ft, 2),
        "length_ft": round(length_ft, 2),
        "area_sqft": round(width_ft * length_ft, 2),
        "median_depth_m": round(float(np.median(frame["floor_depths"])), 2),
        "applied_scale_factor": round(s, 4),
        "camera_height_m": round(frame["camera_height_m"] * s, 2),
    }


# --------------------------------------------------------------------------- #
# Reference-object scale calibration
#
# Depth-Anything-V2-Metric-Indoor returns metres, but its absolute scale drifts
# on photos unlike its training set. An object of KNOWN real size in the frame
# pins that scale down: measure it in the reconstruction, divide the known size
# by the measured size, and you have the multiplier that corrects the whole map.
# --------------------------------------------------------------------------- #

# known_ft : the object's real-world size, in feet.
# axis     : which side of the object's FLOOR FOOTPRINT known_ft refers to.
#            'short' -> the narrow side (a bed's width, a chair's width).
#            'long'  -> the long side. Used for flat vertical objects: a door
#                       has essentially no footprint, so its floor projection
#                       collapses to a sliver whose LENGTH is the door's width.
# aspect   : plausible range for footprint_short / footprint_long. Not a hard
#            reject — it feeds the instance score, so a segment that does not
#            look like its class loses to one that does. A door's footprint is
#            a sliver (~0), a bed's is a broad rectangle, a chair's is squarish.
REFERENCE_OBJECTS = {
    "door":  {"known_ft": 3.0, "axis": "long",  "aspect": (0.00, 0.40)},
    "chair": {"known_ft": 2.5, "axis": "short", "aspect": (0.55, 1.10)},
    "bed":   {"known_ft": 6.0, "axis": "short", "aspect": (0.45, 1.10)},
}

# Priority order. The FIRST class here with a usable instance sets the scale on
# its own — there is no averaging across classes. A class is only passed over
# when EVERY instance of it fails to measure (clipped, too few depth pixels,
# degenerate footprint, or an implausible result).
REFERENCE_PRIORITY = ("bed", "door", "chair")

# Final backstop on the returned factor.
SCALE_MIN, SCALE_MAX = 0.5, 2.0

# Per-instance gate. Failing it makes the instance fall through to the next
# instance, then the next class. This MUST stay inside [SCALE_MIN, SCALE_MAX]:
# a gate wider than the clamp lets a bad instance be "accepted" and then quietly
# clamped, which is how an 8.3 ft door (3.0/8.3 = x0.36 -> clamped x0.50) ends
# up sizing a room instead of being passed over for a better reference.
# The bound is what the depth model could credibly be wrong by — roughly +-55%.
# Anything further out is a mis-measurement, not model drift, and returning an
# uncorrected 1.0 beats acting on it.
SAMPLE_MIN, SAMPLE_MAX = 0.65, 1.55

# Instance-score weights. With one class deciding alone there is no cross-check,
# so which INSTANCE is picked matters — these rank them. Set any weight to 0 to
# drop that signal.
SCORE_W_COVERAGE = 0.35   # valid depth pixels behind the measurement
SCORE_W_MARGIN   = 0.30   # clearance from the left/right frame edges
SCORE_W_NEARNESS = 0.20   # near objects have denser, more reliable depth
SCORE_W_ASPECT   = 0.15   # does the footprint look like this class should?

SCORE_FULL_COVERAGE = 8000.0   # depth pixels at which coverage scores 1.0
SCORE_FAR_M = 8.0              # metres beyond the near point at which nearness -> 0
# Frame clearance scoring full marks. Actual clipping is already a hard reject,
# so this only has to penalise near-misses — demanding more clearance than this
# would punish perfectly well-framed objects for sitting off-centre.
SCORE_FULL_MARGIN = 0.05


def _mask_from_bbox(bbox, shape):
    """Filled mask from an [x1, y1, x2, y2] box — lets a detector that returns
    only boxes (DETR and friends) feed the same measurement path as OneFormer's
    masks, at the cost of including background inside the box."""
    if not bbox or len(bbox) < 4:
        return None
    H, W = shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    x1, x2 = max(0, min(x1, x2)), min(W, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(H, max(y1, y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    mask = np.zeros((H, W), np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _plane_to_pixel(frame, ab, focal_px, shape):
    """A point given in floor-plane (u1, u2) coordinates -> image pixel.

    The inverse of the back-projection: lift (a, b) back to a 3D point ON the
    floor plane, then apply the pinhole projection. Used only to draw what was
    measured; nothing in the measurement depends on it.
    """
    P3 = frame["center"] + float(ab[0]) * frame["u1"] + float(ab[1]) * frame["u2"]
    Zc = float(P3[2])
    if Zc <= 1e-6:
        return None
    H, W = shape[:2]
    return [round(W / 2.0 + focal_px * float(P3[0]) / Zc, 1),
            round(H / 2.0 + focal_px * float(P3[1]) / Zc, 1)]


def _measurement_geometry(frame, mean2, basis, extents, measured_idx,
                          known_ft, focal_px, shape):
    """Image-space drawing data for EXACTLY what the measurement used.

    Returns the footprint rectangle as it lies on the floor, the span that was
    actually compared against known_ft, a bar of the KNOWN length along the same
    axis (so the two can be eyeballed side by side — their ratio IS the scale
    correction), and a 1-foot tick ruler in CORRECTED feet.
    """
    (lo0, hi0), (lo1, hi1) = extents
    v0, v1 = basis

    def px(ab):
        return _plane_to_pixel(frame, ab, focal_px, shape)

    corners = []
    for s0, s1 in ((lo0, lo1), (hi0, lo1), (hi0, hi1), (lo0, hi1)):
        point = px(mean2 + s0 * v0 + s1 * v1)
        if point is None:
            return None
        corners.append(point)

    # Split into the axis that was measured and the one across it.
    v_m, (lo_m, hi_m) = (v0, (lo0, hi0)) if measured_idx == 0 else (v1, (lo1, hi1))
    v_o, (lo_o, hi_o) = (v1, (lo1, hi1)) if measured_idx == 0 else (v0, (lo0, hi0))

    span_o = max(1e-6, hi_o - lo_o)
    gap = 0.18 * span_o          # keep the measured and known bars apart
    mid_o = 0.5 * (lo_o + hi_o)

    measured = [px(mean2 + lo_m * v_m + (mid_o - gap) * v_o),
                px(mean2 + hi_m * v_m + (mid_o - gap) * v_o)]

    known = None
    ruler = []
    if known_ft:
        known_m = float(known_ft) / M_TO_FT
        known = [px(mean2 + lo_m * v_m + (mid_o + gap) * v_o),
                 px(mean2 + (lo_m + known_m) * v_m + (mid_o + gap) * v_o)]

        # One corrected foot = (measured span) / known_ft raw metres, since that
        # span is DEFINED to be known_ft feet once the correction is applied.
        step = (hi_m - lo_m) / float(known_ft)
        for k in range(-2, int(known_ft) + 4):
            a = lo_m + k * step
            tick_a = px(mean2 + a * v_m + (hi_o + 0.20 * span_o) * v_o)
            tick_b = px(mean2 + a * v_m + (hi_o + 0.55 * span_o) * v_o)
            if tick_a and tick_b:
                ruler.append({"a": tick_a, "b": tick_b, "ft": k})

    if any(p is None for p in measured) or (known and any(p is None for p in known)):
        return None
    return {"footprint_px": corners, "measured_px": measured,
            "known_px": known, "ruler_px": ruler}


def _measure_reference_object(depth_m, focal_px, frame, obj_mask, axis, known_ft=None):
    """Real size of a detected object, in FEET, measured in the floor plane.

    Back-projects the object's mask to 3D, drops its FOOTPRINT onto the fitted
    floor plane (i.e. discards the height component), and takes the robust
    extent along the footprint's own principal axes.

    Measuring in the plane rather than from the 2D bounding box is what makes
    this independent of how the object is turned relative to the camera. A bed
    rotated 40 degrees has a far wider pixel bbox than a bed facing the camera,
    yet the same footprint -- and a bbox spans the diagonal, so it would read a
    5 ft bed as 8 ft. Using the object's own principal axes also means we can
    ask for its width specifically, instead of whichever side happens to face us.

    Returns (size_ft, detail_dict) or (None, reason).
    """
    # A reference running off the LEFT or RIGHT edge of the frame is truncated,
    # so it always reads narrower than it is — and every measurement here is a
    # horizontal one. (Touching the top or bottom edge is fine and very common:
    # a door reaches the ceiling, a near bed runs off the bottom; neither
    # truncates its width.)
    if obj_mask.shape[:2] == depth_m.shape[:2]:
        edge = obj_mask
    else:
        edge = cv2.resize(obj_mask, (depth_m.shape[1], depth_m.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    if edge[:, :2].any() or edge[:, -2:].any():
        return None, "clipped-by-frame-edge"

    P, _ = _backproject(depth_m, obj_mask, focal_px, min_pts=150, max_pts=20000)
    if P is None:
        return None, "too-few-depth-pixels"

    # Drop depth flyers. Segmentation edges bleed onto the background, and a
    # single far pixel would stretch the footprint arbitrarily.
    Zo = P[:, 2]
    z_lo, z_hi = np.percentile(Zo, [5, 95])
    band = (Zo >= z_lo - 0.35) & (Zo <= z_hi + 0.35)
    P = P[band]
    if len(P) < 100:
        return None, "depth-too-noisy"

    # Footprint: coordinates in the floor plane's basis. The component along the
    # plane normal (the object's height) is simply not used.
    rel = P - frame["center"]
    pts = np.stack([rel @ frame["u1"], rel @ frame["u2"]], axis=1)

    # Oriented extents: PCA for the footprint's own axes, percentile trim for
    # robustness (a minAreaRect would be pinned by its worst outlier).
    mean2 = pts.mean(axis=0)
    centered = pts - mean2
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    lo0, hi0 = np.percentile(centered @ vt[0], [2, 98])
    lo1, hi1 = np.percentile(centered @ vt[1], [2, 98])
    s0, s1 = float(hi0 - lo0), float(hi1 - lo1)

    long_m, short_m = max(s0, s1), min(s0, s1)
    chosen_m = long_m if axis == "long" else short_m
    if chosen_m <= 0.05:
        return None, "degenerate-footprint"

    # Which of the two principal axes ended up being the measured one.
    if axis == "long":
        measured_idx = 0 if s0 >= s1 else 1
    else:
        measured_idx = 0 if s0 < s1 else 1

    return chosen_m * M_TO_FT, {
        "footprint_short_ft": round(short_m * M_TO_FT, 2),
        "footprint_long_ft": round(long_m * M_TO_FT, 2),
        "median_depth_m": round(float(np.median(P[:, 2])), 2),
        "points": int(len(P)),
        "geometry": _measurement_geometry(
            frame, mean2, (vt[0], vt[1]), ((lo0, hi0), (lo1, hi1)),
            measured_idx, known_ft, focal_px, depth_m.shape),
    }


def _instance_score(mask, detail, spec, image_w):
    """How much to trust one instance's measurement, in [0, 1].

    With a single class deciding the scale alone there is no cross-check left,
    so WHICH instance gets picked matters. Four signals, all pointing the same
    way -- a big, close, well-framed segment whose footprint looks like its
    class is worth more than a small, distant, half-occluded one.
    """
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return 0.0

    # Clearance from the left/right frame edges. A hard reject already removes
    # anything actually touching an edge; this prefers what sits well inside.
    margin = min(int(cols[0]), image_w - 1 - int(cols[-1])) / float(max(1, image_w))
    margin_score = min(1.0, margin / SCORE_FULL_MARGIN)

    coverage = min(1.0, float(detail.get("points", 0)) / SCORE_FULL_COVERAGE)

    near_m = float(detail.get("median_depth_m", 0.0))
    nearness = float(np.clip(1.0 - (near_m - 1.0) / SCORE_FAR_M, 0.0, 1.0))

    lo, hi = spec.get("aspect", (0.0, 1.0))
    long_ft = float(detail.get("footprint_long_ft", 0.0))
    ratio = float(detail.get("footprint_short_ft", 0.0)) / long_ft if long_ft > 1e-6 else 0.0
    if lo <= ratio <= hi:
        aspect = 1.0
    else:
        off = (lo - ratio) if ratio < lo else (ratio - hi)
        aspect = float(max(0.0, 1.0 - off / 0.35))

    return float(SCORE_W_COVERAGE * coverage + SCORE_W_MARGIN * margin_score +
                 SCORE_W_NEARNESS * nearness + SCORE_W_ASPECT * aspect)


def reference_scale_factor(depth_m, focal_px, frame, detections):
    """ONE reference class sets the depth scale, chosen by priority.

    Walks REFERENCE_PRIORITY (bed -> door -> chair). For the first class that is
    present, every instance is measured and scored, and the best-scoring usable
    one sets scale = known_ft / measured_ft by itself. Nothing is averaged --
    not across classes, and not across instances of a class.

    A class is only passed over when EVERY instance of it fails: clipped by the
    frame edge, too few depth pixels, a degenerate footprint, or a scale outside
    [SAMPLE_MIN, SAMPLE_MAX]. Lower-priority classes are then never measured.

    Returns (scale_factor, samples). Every detection appears in samples with a
    `status` so a wrong number traces to the object that caused it:
        selected     - this one set the scale
        rejected     - measured or gated out (see `reason`)
        not-selected - usable, but another instance of its class scored higher
        skipped      - never measured; a higher-priority class already won
    Returns (1.0, samples) when nothing usable was found, leaving the depth map
    exactly as the model produced it.
    """
    samples = []
    if frame is None or depth_m is None:
        return 1.0, samples

    by_label = {}
    for index, det in enumerate(detections or []):
        if det.get("label") in REFERENCE_OBJECTS:
            by_label.setdefault(det["label"], []).append((index, det))

    def _stub(index, det, status, reason=None):
        entry = {
            "index": index,
            "label": det.get("label"),
            "known_ft": REFERENCE_OBJECTS[det["label"]]["known_ft"],
            "status": status,
        }
        if reason:
            entry["reason"] = reason
        return entry

    selected = None
    for label in REFERENCE_PRIORITY:
        entries = by_label.get(label)
        if not entries:
            continue

        spec = REFERENCE_OBJECTS[label]
        measured_here = []
        for index, det in entries:
            mask = det.get("mask")
            if mask is None:
                mask = _mask_from_bbox(det.get("bbox"), depth_m.shape[:2])
            if mask is None:
                samples.append(_stub(index, det, "rejected", "no-mask-or-bbox"))
                continue

            size_ft, detail = _measure_reference_object(
                depth_m, focal_px, frame, mask, spec["axis"], spec["known_ft"])
            if size_ft is None:
                samples.append(_stub(index, det, "rejected", detail))
                continue

            scale = spec["known_ft"] / size_ft
            entry = _stub(index, det, "candidate")
            entry.update({
                "measured_ft": round(float(size_ft), 2),
                "scale": round(float(scale), 4),
                "score": round(_instance_score(mask, detail, spec, depth_m.shape[1]), 3),
                "footprint_short_ft": detail["footprint_short_ft"],
                "footprint_long_ft": detail["footprint_long_ft"],
                "median_depth_m": detail["median_depth_m"],
                "measured_axis": spec["axis"],
                # Underscore-prefixed: drawing data for the debug overlay only,
                # stripped from the API response by the route.
                "_geom": detail.get("geometry"),
            })
            if not (SAMPLE_MIN <= scale <= SAMPLE_MAX):
                entry["status"] = "rejected"
                entry["reason"] = "implausible-scale"
            samples.append(entry)
            measured_here.append(entry)

        usable = [e for e in measured_here if e["status"] == "candidate"]
        if not usable:
            continue   # whole class unusable -> fall through to the next one

        best = max(usable, key=lambda e: e["score"])
        best["status"] = "selected"
        for entry in usable:
            if entry is not best:
                entry["status"] = "not-selected"
                entry["reason"] = "lower-score-than-{0}".format(best["score"])
        selected = best
        break

    # Anything a winning class made unnecessary is reported, not measured.
    seen = {e["index"] for e in samples}
    for index, det in enumerate(detections or []):
        if index not in seen and det.get("label") in REFERENCE_OBJECTS:
            samples.append(_stub(index, det, "skipped", "higher-priority-class-used"))
    samples.sort(key=lambda e: e["index"])

    if selected is None:
        return 1.0, samples
    return float(np.clip(selected["scale"], SCALE_MIN, SCALE_MAX)), samples


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


# --------------------------------------------------------------------------- #
# Floor-quad visual debugger
#
# floor_quad_norm is not an annotation the client draws — it IS the coordinate
# system every rug lives in. The client feeds it to computeHomography() and each
# rug vertex is applyH(H, u, v). So the only way to see whether a rug is being
# sized correctly is to look at the quad the same way the client does.
#
# Everything below reproduces the client's own math (geometry.js) rather than
# using cv2.getPerspectiveTransform, so what is drawn is what the browser draws.
# --------------------------------------------------------------------------- #

_DBG = {
    "quad":    (0, 255, 255),    # yellow  - the quad edges
    "corner":  (0, 200, 255),    # amber   - corner dots + labels
    "u_axis":  (90, 240, 110),   # green   - u axis (TL->TR), spans u_span_ft
    "v_axis":  (255, 170, 60),   # blue    - v axis (TL->BL), spans v_span_ft
    "ft_grid": (255, 90, 220),   # magenta - the 1-foot grid
    "rug":     (60, 60, 255),    # red     - reference rug footprints
    "floor":   (60, 220, 60),    # green   - visible-floor mask tint
    "warn":    (0, 90, 255),     # orange  - warnings
    "ok":      (120, 255, 120),
    "text":    (245, 245, 245),
}


def _dbg_homography(quad):
    """Port of geometry.js computeHomography(). quad = [TL, TR, BR, BL] pixels."""
    d0, d1, d2, d3 = [np.asarray(p, dtype=np.float64) for p in quad]
    dx1, dy1 = d1[0] - d2[0], d1[1] - d2[1]
    dx2, dy2 = d3[0] - d2[0], d3[1] - d2[1]
    sx = d0[0] - d1[0] + d2[0] - d3[0]
    sy = d0[1] - d1[1] + d2[1] - d3[1]
    det = dx1 * dy2 - dy1 * dx2
    if abs(det) < 1e-12:
        return None
    g = (sx * dy2 - sy * dx2) / det
    h = (dx1 * sy - dy1 * sx) / det
    return (
        d1[0] - d0[0] + g * d1[0], d3[0] - d0[0] + h * d3[0], d0[0],
        d1[1] - d0[1] + g * d1[1], d3[1] - d0[1] + h * d3[1], d0[1],
        g, h, 1.0,
    )


def _dbg_uv(H, u, v):
    """Port of geometry.js applyH(). None when the point is at/behind the
    horizon (w <= 0), which is where the projection blows up."""
    w = H[6] * u + H[7] * v + H[8]
    if w <= 1e-9:
        return None
    x = (H[0] * u + H[1] * v + H[2]) / w
    y = (H[3] * u + H[4] * v + H[5]) / w
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return (x, y)


def quad_floor_coverage(quad, floor_mask):
    """Fraction of floor-mask pixels that fall inside the quad, 0..1.

    The direct answer to "is the quad big enough?". A quad that covers the floor
    scores ~1.0; the old minAreaRect-over-inliers quad scores well under that,
    and the shortfall is exactly the floor a rug can never be sized to reach.
    """
    if quad is None or floor_mask is None:
        return None
    m = floor_mask
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    total = int((m > 127).sum())
    if total == 0:
        return None
    filled = np.zeros(m.shape[:2], np.uint8)
    pts = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    pts = np.clip(pts, -1e5, 1e5).round().astype(np.int32)
    cv2.fillPoly(filled, [pts], 255)
    return float(((filled > 0) & (m > 127)).sum()) / total


def _dbg_pt(p):
    """cv2's anti-aliased rasteriser works in 16.16 fixed point, so coordinates
    far outside the canvas overflow and smear garbage across the buffer. Every
    draw call clips to the canvas first (see _dbg_line); this is the backstop."""
    lim = 12000
    return (int(round(max(-lim, min(lim, float(p[0]))))),
            int(round(max(-lim, min(lim, float(p[1]))))))


def _dbg_line(canvas, p, q, color, thickness=1):
    """Clip to the canvas BEFORE rasterising. Drawing a line whose endpoints sit
    thousands of pixels off-frame is normal here (a floor quad legitimately runs
    off the bottom and sides) and is what tore the earlier overlays apart."""
    if p is None or q is None:
        return
    if not all(np.isfinite(v) for v in (p[0], p[1], q[0], q[1])):
        return
    h, w = canvas.shape[:2]
    ok, a, b = cv2.clipLine((0, 0, w, h), _dbg_pt(p), _dbg_pt(q))
    if not ok:
        return
    cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)


def _dbg_arrow(canvas, p, q, color, thickness=3):
    """Arrow, but only when both ends are close enough to the canvas that the
    head lands somewhere meaningful; otherwise fall back to a clipped line."""
    if p is None or q is None:
        return
    h, w = canvas.shape[:2]
    inside = lambda pt: -w <= pt[0] <= 2 * w and -h <= pt[1] <= 2 * h
    if inside(p) and inside(q):
        cv2.arrowedLine(canvas, _dbg_pt(p), _dbg_pt(q), color, thickness,
                        cv2.LINE_AA, tipLength=0.03)
    else:
        _dbg_line(canvas, p, q, color, thickness)


def _dbg_uv_polyline(canvas, H, pts_uv, color, thickness=1, closed=False):
    """Draw a uv-space polyline through the homography, dropping any segment
    that crosses the horizon instead of letting it whip across the frame."""
    pts = [_dbg_uv(H, u, v) for u, v in pts_uv]
    seq = pts + [pts[0]] if closed else pts
    for a, b in zip(seq, seq[1:]):
        _dbg_line(canvas, a, b, color, thickness)


def _dbg_text_block(canvas, lines, org=(14, 14), scale=0.52, pad=8, line_h=22):
    """Left-aligned text on a dark plate. lines = [(text, color), ...]."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    widths = [cv2.getTextSize(t, font, scale, 1)[0][0] for t, _ in lines]
    box_w = max(widths) + pad * 2
    box_h = len(lines) * line_h + pad * 2
    x0, y0 = org
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), (70, 70, 70), 1)
    for i, (text, color) in enumerate(lines):
        cv2.putText(canvas, text, (x0 + pad, y0 + pad + line_h * (i + 1) - 6),
                    font, scale, color, 1, cv2.LINE_AA)


def _dbg_label(canvas, text, at, color, scale=0.55):
    """Skip labels whose anchor is off-canvas — drawing them is pointless and
    the out-of-range rectangle/putText is another way to corrupt the buffer."""
    if at is None or not all(np.isfinite(v) for v in (at[0], at[1])):
        return
    h, w = canvas.shape[:2]
    x, y = _dbg_pt(at)
    if not (-200 <= x <= w + 200 and 0 <= y <= h + 40):
        return
    x = max(2, min(x, w - 12))
    y = max(16, min(y, h - 4))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    cv2.rectangle(canvas, (max(0, x - 3), max(0, y - th - 5)),
                  (min(w - 1, x + tw + 3), min(h - 1, y + 4)), (16, 16, 16), -1)
    cv2.putText(canvas, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def _dbg_rug_uv(u, v, rot_deg, hu, hv, v_axis_scale):
    """Port of geometry.js rugCornersPx() — the uv corners of a rug, including
    the rotation/vAxisScale composition the client uses."""
    rad = math.radians(rot_deg)
    cos, sin = math.cos(rad), math.sin(rad)
    return [(lx * cos - ly * sin + u, (lx * sin + ly * cos) * v_axis_scale + v)
            for lx, ly in ((-hu, -hv), (hu, -hv), (hu, hv), (-hu, hv))]


def render_floor_quad_debug(room_img, quad, u_span_ft, v_span_ft,
                            visible_floor=None, quad_source="depth",
                            rug_previews=((5.0, 7.0), (9.0, 12.0)),
                            u=0.48, v=0.76, rot=0.0, max_dim=1700, notes=None):
    """Overlay showing exactly what the client will do with floor_quad_norm.

    quad         : [TL, TR, BR, BL] in room_img pixels (the persp_quad sent out).
    u_span_ft    : real length of the TL->TR edge  (becomes room_width_ft).
    v_span_ft    : real length of the TL->BL edge  (becomes room_length_ft).
    rug_previews : (width_ft, length_ft) footprints drawn with the client's math.

    Read it like this:

      * The MAGENTA 1-foot grid is the contract test. Each cell is one square
        foot on the floor. If the quad and the spans agree, the cells read as
        real floor tiles receding into the scene — square-ish and level near the
        camera, evenly compressing toward the far edge. If the cells look
        stretched, sheared, or the tiling is obviously too coarse/fine against
        real furniture, the spans do not describe this quad and every rug is
        being scaled by that same error.

      * The RED rug footprints are drawn with the client's own sizing math. A
        9 x 12 ft rug should look like a 9 x 12 ft rug next to the furniture.

      * The YELLOW quad is the plane itself. Its far edge should sit at the
        floor/wall junction and its near edge at (or below) the frame bottom.
        A far edge floating up the wall means the rug's far end is on the wall.
    """
    canvas = room_img.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    H_img, W_img = canvas.shape[:2]

    # Dim the photo so the overlay reads clearly.
    canvas = (canvas.astype(np.float32) * 0.62).astype(np.uint8)

    # Visible-floor mask tint — this is also the destination-in clip the client
    # applies, so anything outside it is erased from the rug.
    if visible_floor is not None:
        m = visible_floor
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        if m.shape[:2] != (H_img, W_img):
            m = cv2.resize(m, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
        tint = np.zeros_like(canvas)
        tint[:] = _DBG["floor"]
        alpha = (m > 127).astype(np.float32)[..., None] * 0.16
        canvas = (canvas * (1 - alpha) + tint * alpha).astype(np.uint8)

    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    Hm = _dbg_homography(quad)
    if Hm is None:
        _dbg_text_block(canvas, [("DEGENERATE QUAD - homography is singular", _DBG["warn"])])
        return canvas

    u_span_ft = float(u_span_ft) if u_span_ft and u_span_ft > 0 else 0.0
    v_span_ft = float(v_span_ft) if v_span_ft and v_span_ft > 0 else 0.0

    # --- 1-foot physical grid: the contract test -----------------------------
    if u_span_ft > 0 and v_span_ft > 0:
        du, dv = 1.0 / u_span_ft, 1.0 / v_span_ft
        n_u, n_v = int(math.floor(u_span_ft)), int(math.floor(v_span_ft))
        for i in range(1, n_u + 1):
            uu = i * du
            if uu >= 1.0:
                break
            _dbg_uv_polyline(canvas, Hm, [(uu, t / 24.0) for t in range(25)],
                             _DBG["ft_grid"], 1)
        for j in range(1, n_v + 1):
            vv = j * dv
            if vv >= 1.0:
                break
            _dbg_uv_polyline(canvas, Hm, [(t / 24.0, vv) for t in range(25)],
                             _DBG["ft_grid"], 1)
        # Mark every 5 ft along the near edge so the scale is countable.
        for i in range(5, n_u + 1, 5):
            _dbg_label(canvas, f"{i}ft", _dbg_uv(Hm, i * du, 0.985), _DBG["ft_grid"], 0.45)

    # --- The quad itself -----------------------------------------------------
    _dbg_uv_polyline(canvas, Hm, [(0, 0), (1, 0), (1, 1), (0, 1)],
                     _DBG["quad"], 3, closed=True)

    for (uu, vv), name in zip(((0, 0), (1, 0), (1, 1), (0, 1)),
                              ("TL", "TR", "BR", "BL")):
        p = _dbg_uv(Hm, uu, vv)
        if p is None:
            continue
        cv2.circle(canvas, _dbg_pt(p), 8, _DBG["corner"], -1, cv2.LINE_AA)
        _dbg_label(canvas, f"{name} ({uu},{vv})", (p[0] + 12, p[1]), _DBG["corner"], 0.5)

    # --- u / v axes with their measured lengths ------------------------------
    tl, tr, bl = _dbg_uv(Hm, 0, 0), _dbg_uv(Hm, 1, 0), _dbg_uv(Hm, 0, 1)
    if tl and tr:
        _dbg_arrow(canvas, tl, tr, _DBG["u_axis"], 3)
        mid = _dbg_uv(Hm, 0.5, 0.0)
        _dbg_label(canvas, f"u -> floor_quad_width_ft = {u_span_ft:.2f} ft",
                   (mid[0] - 90, mid[1] - 14) if mid else None, _DBG["u_axis"])
    if tl and bl:
        _dbg_arrow(canvas, tl, bl, _DBG["v_axis"], 3)
        mid = _dbg_uv(Hm, 0.0, 0.5)
        _dbg_label(canvas, f"v -> floor_quad_length_ft = {v_span_ft:.2f} ft",
                   (mid[0] + 12, mid[1]) if mid else None, _DBG["v_axis"])

    # --- Reference rugs, drawn with the CLIENT's sizing math -----------------
    v_axis_scale = (u_span_ft / v_span_ft) if v_span_ft > 0 else 1.0
    if u_span_ft > 0:
        for rug_w_ft, rug_l_ft in (rug_previews or ()):
            hu = (rug_l_ft / u_span_ft) / 2.0        # geometry.js: hu from LENGTH
            hv = (rug_w_ft / u_span_ft) / 2.0        # geometry.js: hv from WIDTH
            pts_uv = _dbg_rug_uv(u, v, rot, hu, hv, v_axis_scale)
            _dbg_uv_polyline(canvas, Hm, pts_uv, _DBG["rug"], 3, closed=True)
            # Label on the rug's own near edge (midpoint of BR-BL) so it tracks
            # the footprint instead of floating at a fixed pixel offset.
            near = _dbg_uv(Hm, (pts_uv[2][0] + pts_uv[3][0]) / 2.0,
                               (pts_uv[2][1] + pts_uv[3][1]) / 2.0)
            _dbg_label(canvas, f"{rug_w_ft:g} x {rug_l_ft:g} ft",
                       (near[0] - 40, near[1] - 8) if near else None,
                       _DBG["rug"], 0.6)

    # --- Info panel ----------------------------------------------------------
    src_ok = quad_source == "depth"
    top_w = float(np.linalg.norm(quad[1] - quad[0]))
    bot_w = float(np.linalg.norm(quad[2] - quad[3]))
    taper = (top_w / bot_w) if bot_w > 1e-6 else 0.0

    cov = quad_floor_coverage(quad, visible_floor)

    lines = [
        (f"quad source     : {quad_source}", _DBG["ok"] if src_ok else _DBG["warn"]),
        (f"quad u span     : {u_span_ft:.2f} ft   <- rug sizing basis", _DBG["u_axis"]),
        (f"quad v span     : {v_span_ft:.2f} ft   <- rug sizing basis", _DBG["v_axis"]),
        (f"vAxisScale      : {v_axis_scale:.3f}   (client clamps this!)", _DBG["text"]),
        (f"image           : {W_img} x {H_img} px", _DBG["text"]),
        (f"quad taper      : {taper:.3f}  (top/bottom edge, in px)", _DBG["text"]),
        (f"rug preview rot : {rot:g} deg  at u={u:g} v={v:g}", _DBG["rug"]),
    ]
    if cov is not None:
        lines.insert(1, (f"floor covered   : {cov*100:.1f}% of the floor mask",
                         _DBG["ok"] if cov >= 0.90 else _DBG["warn"]))
        if cov < 0.90:
            lines.append((f"WARNING: quad misses {(1-cov)*100:.0f}% of the floor - a rug",
                          _DBG["warn"]))
            lines.append(("         sized against it cannot reach the walls.",
                          _DBG["warn"]))
    if not src_ok:
        lines.append(("WARNING: 2D fallback quad - its real size is unknown,",
                      _DBG["warn"]))
        lines.append(("         so the spans above do NOT describe it.", _DBG["warn"]))
    if abs(taper - 0.55) < 0.02:
        lines.append(("WARNING: taper == 0.55 -> hard-coded _detect_floor_quad shape",
                      _DBG["warn"]))
    if v_axis_scale > 6.0 or v_axis_scale < 1 / 6.0:
        lines.append((f"WARNING: vAxisScale {v_axis_scale:.2f} exceeds the client clamp",
                      _DBG["warn"]))
    for note in (notes or []):
        lines.append((note, _DBG["text"]))
    _dbg_text_block(canvas, lines)

    if max_dim and max(W_img, H_img) > max_dim:
        s = max_dim / float(max(W_img, H_img))
        canvas = cv2.resize(canvas, (int(W_img * s), int(H_img * s)),
                            interpolation=cv2.INTER_AREA)
    return canvas