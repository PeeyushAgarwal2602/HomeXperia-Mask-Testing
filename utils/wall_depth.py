import cv2
import numpy as np
import math

from utils.mask_alpha import resize_mask_alpha

# Auto-repeat scale anchor: tile width ~ canvas_width / 12 (the old production
# tile size at the frontend's default repeat), as an integer count per wall.
TARGET_TILES_ACROSS_CANVAS = 12.0
WALL_FOCAL_RATIO = 0.8

def _quad_mask_containment(quad, mask_gray):
    poly = np.zeros_like(mask_gray)
    cv2.fillPoly(poly, [quad.astype(np.int32)], 255)
    m = mask_gray > 0
    total = int(m.sum())
    if total == 0: return 0.0
    return int((poly[m] > 0).sum()) / float(total)

def get_lighting_map(img, blur_k=51):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur_k % 2 == 0: blur_k += 1
    gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    return gray.astype(np.float32) / 255.0

def tile_texture(pattern, area_w, area_h, tile_size_w):
    ph, pw = pattern.shape[:2]
    scale = tile_size_w / float(pw)
    tile_size_h = max(1, int(ph * scale))
    tile = cv2.resize(pattern, (max(1, int(tile_size_w)), tile_size_h), interpolation=cv2.INTER_LANCZOS4)
    th, tw = tile.shape[:2]
    if th == 0 or tw == 0: return np.zeros((area_h, area_w, 3), dtype=np.uint8)
    reps_y = -(-area_h // th)
    reps_x = -(-area_w // tw)
    grid_out = np.tile(tile, (reps_y, reps_x, 1))
    return grid_out[:area_h, :area_w]

def create_super_texture(pattern, target_w, target_h, tile_size_w, pad_x_tiles=1, pad_y_tiles=1):
    ph, pw = pattern.shape[:2]
    tw = max(1, int(tile_size_w))
    th = max(1, int(ph * (tile_size_w / float(pw))))
    pad_w = tw * max(1, int(pad_x_tiles))
    pad_h = th * max(1, int(pad_y_tiles))
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

    # Normalize by the mean luminance INSIDE the mask so only the RELATIVE
    # shading (shadow gradients, corner darkening) transfers to the new
    # texture. Without this, a dark or already-wallpapered wall imprints its
    # overall darkness onto the product and the pattern comes out muddy.
    inside = mask_gray > 127
    if np.any(inside):
        mean_l = float(lighting_map[inside].mean())
        if mean_l > 1e-4:
            lighting_map = np.clip(lighting_map / mean_l, 0.0, 1.6)

    lighting_3ch = cv2.merge([lighting_map, lighting_map, lighting_map])
    shaded_texture = tex_f * (lighting_3ch ** shadow_strength)
    # mask_gray arrives pre-feathered from resize_mask_alpha, with the ramp width
    # scaled to the canvas. The (3, 3) blur that used to stand in for it here was
    # a no-op at 4500px and left the boundary hard.
    mask_f = mask_gray.astype(np.float32) / 255.0
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

def _consistent_bottom(mt, mb, cb, xl, xr):
    inconsistent = (mt * mb > 0 and abs(mb) > 0.04) or (abs(mt) <= 0.04 and abs(mb) > 0.08)
    if not inconsistent:
        return mb, cb
    mb_new = -mt
    anchor = xr if mb > 0 else xl
    y_anchor = mb * anchor + cb
    return mb_new, y_anchor - mb_new * anchor

def _detect_wall_quad(mask_gray, debug_img=None):
    H, W = mask_gray.shape
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas)
    if max_area <= 0: return None
    sig_contours = [c for c, a in zip(contours, areas) if a >= max(0.03 * max_area, 400.0)]
    if not sig_contours: sig_contours = [max(contours, key=cv2.contourArea)]
    largest = max(sig_contours, key=cv2.contourArea)

    rects = [cv2.boundingRect(c) for c in sig_contours]
    x = min(r[0] for r in rects)
    y = min(r[1] for r in rects)
    w = max(r[0] + r[2] for r in rects) - x
    h = max(r[1] + r[3] for r in rects) - y

    # Geometric Validation (Angle Matrix Logic)
    def _is_valid_perspective_angles(quad, min_deg=50, max_deg=130):
        for i in range(4):
            p1, p2, p3 = quad[i - 1], quad[i], quad[(i + 1) % 4]
            v1, v2 = p1 - p2, p3 - p2
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7)
            angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            if angle < min_deg or angle > max_deg:
                return False
        return True

    # --- STRATEGY 0: Direct 4-corner fit (captures slanted side edges) ---
    def _polydp_quad():
        cont_area = cv2.contourArea(largest)
        if cont_area <= 0: return None
        peri = cv2.arcLength(largest, True)
        for eps_f in (0.01, 0.02, 0.03):
            approx = cv2.approxPolyDP(largest, eps_f * peri, True)
            if len(approx) != 4: continue
            quad = order_points(np.squeeze(approx).astype(np.float32))
            if not _is_valid_perspective_angles(quad): continue
            # The 4-corner fit must actually cover the mask region
            quad_area = cv2.contourArea(quad)
            if not (0.85 * cont_area <= quad_area <= 1.2 * cont_area): continue
            # Reject inverted geometry (top corners below bottom corners)
            if quad[0][1] >= quad[3][1] - 10 or quad[1][1] >= quad[2][1] - 10: continue
            return quad
        return None

    # --- STRATEGY 1: Hough Transform ---
    def _hough_based_quad():
        clean_boundary = np.zeros_like(mask_gray)
        cv2.drawContours(clean_boundary, sig_contours, -1, 255, 1)
        lines = cv2.HoughLinesP(clean_boundary, 1, np.pi / 180, threshold=20, minLineLength=max(40, w//8), maxLineGap=20)
        top_lines, bot_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if (max(y1, y2) <= 3 or min(y1, y2) >= H - 4 or max(x1, x2) <= 3 or min(x1, x2) >= W - 4): continue
                dx, dy = float(x2 - x1), float(y2 - y1)
                if abs(dx) < 1e-3: continue
                slope = dy / dx
                if abs(slope) > 1.0: continue # Reject pure verticals
                intercept = y1 - slope * x1
                my = (y1 + y2) / 2.0
                length = math.hypot(dx, dy)
                if my < y + h * 0.35: top_lines.append((slope, intercept, length))
                elif my > y + h * 0.65: bot_lines.append((slope, intercept, length))

        def _get_avg(line_list):
            if not line_list: return None, None
            line_list.sort(key=lambda l: l[2], reverse=True)
            best = line_list[:3]
            tot_len = sum(l[2] for l in best)
            if tot_len == 0: return None, None
            return sum(l[0]*l[2] for l in best)/tot_len, sum(l[1]*l[2] for l in best)/tot_len

        mt, ct = _get_avg(top_lines)
        mb, cb = _get_avg(bot_lines)
        if mt is None or mb is None: return None

        # Hough top lines are real (border-excluded) architecture — trust them
        mb, cb = _consistent_bottom(mt, mb, cb, float(x), float(x + w))
        xl, xr = float(x), float(x + w)
        tl_y, tr_y = mt * xl + ct, mt * xr + ct
        bl_y, br_y = mb * xl + cb, mb * xr + cb

        # VALIDATION: Reject inverted geometry
        if tl_y >= bl_y - 10 or tr_y >= br_y - 10: return None
        quad = np.array([[xl, tl_y], [xr, tr_y], [xr, br_y], [xl, bl_y]], dtype=np.float32)

        # VALIDATION: Apply the Internal Angle check
        if not _is_valid_perspective_angles(quad): return None
        return quad

    # --- STRATEGY 2: Profile Fitter (Fallback) ---
    def _profile_based_quad():
        top_raw, bot_raw = [], []
        for col in range(x, x + w):
            col_data = mask_gray[:, col]
            y_indices = np.where(col_data > 0)[0]
            if len(y_indices) > 0:
                top_raw.append([col, y_indices[0]])
                bot_raw.append([col, y_indices[-1]])

        if len(top_raw) < 10 or len(bot_raw) < 10: return None
        top_profile = [p for p in top_raw if p[1] > 2]
        bot_profile = [p for p in bot_raw if p[1] < H - 3]
        top_trusted = len(top_profile) >= 10
        if len(top_profile) < 10: top_profile = top_raw
        if len(bot_profile) < 10: bot_profile = bot_raw

        tp, bp = np.array(top_profile, dtype=np.float32), np.array(bot_profile, dtype=np.float32)

        def _robust_line(pts, is_top):
            spread = np.max(pts[:, 0]) - np.min(pts[:, 0])
            if spread < w * 0.15: return 0.0, float(np.median(pts[:, 1]))
            y_vals = pts[:, 1]
            thresh = np.percentile(y_vals, 50)
            valid_pts = pts[y_vals <= thresh] if is_top else pts[y_vals >= thresh]
            if len(valid_pts) < 10: valid_pts = pts
            vx, vy, cx, cy = cv2.fitLine(valid_pts, cv2.DIST_L1, 0, 0.01, 0.01)
            if abs(vx[0]) < 1e-3: return 0.0, float(np.median(valid_pts[:, 1]))
            m = np.clip(float(vy[0] / vx[0]), -0.25, 0.25)
            c = float(np.median(valid_pts[:, 1]) - m * np.median(valid_pts[:, 0]))
            return m, c

        mt, ct = _robust_line(tp, True)
        mb, cb = _robust_line(bp, False)
        if top_trusted:
            mb, cb = _consistent_bottom(mt, mb, cb, float(x), float(x + w))

        xl, xr = float(x), float(x + w)
        tl_y, tr_y = mt * xl + ct, mt * xr + ct
        bl_y, br_y = mb * xl + cb, mb * xr + cb

        if tl_y >= bl_y or tr_y >= br_y:
            mt, mb = 0.0, 0.0
            ct, cb = float(np.min(tp[:, 1])), float(np.max(bp[:, 1]))
            tl_y, tr_y, bl_y, br_y = ct, ct, cb, cb
        return np.array([[xl, tl_y], [xr, tr_y], [xr, br_y], [xl, bl_y]], dtype=np.float32)

    quad, active_method = None, None
    for name, fn in (("POLY-DP (STRATEGY 0)", _polydp_quad),
                     ("HOUGH (PRIMARY)", _hough_based_quad),
                     ("PROFILE (FALLBACK)", _profile_based_quad)):
        q = fn()
        if q is None: continue
        if q[:, 0].min() > x + 0.04 * w or q[:, 0].max() < x + w - 0.04 * w:
            print(f"   [QUAD REJECT] {name} does not span the mask's width")
            continue
        cov = _quad_mask_containment(q, mask_gray)
        if cov >= 0.88:
            quad, active_method = q, name
            break
        print(f"   [QUAD REJECT] {name} covers only {cov*100:.0f}% of the mask")

    if quad is None:
        active_method = "BBOX (SAFE)"
        quad = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)

    # --- Visual Debugging ---
    if debug_img is not None:
        import os, time
        os.makedirs("Debugs", exist_ok=True)
        dbg = debug_img.copy()
        color = (0, 255, 0) if "POLY" in active_method else (0, 0, 255) if "HOUGH" in active_method else (255, 255, 255) if "BBOX" in active_method else (0, 255, 255)
        cv2.polylines(dbg, [quad.astype(np.int32)], True, color, 3)
        cv2.putText(dbg, active_method, (int(x), max(30, int(y) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imwrite(f"Debugs/wall_quad_debug_{int(time.time()*1000)}.jpg", dbg)

    return quad

def _find_corner_split(mask_gray):
    H, W = mask_gray.shape
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w < max(200, int(0.25 * W)): return None # too narrow to hold two wall faces

    xs, tops = [], []
    step = max(1, w // 300)
    for col in range(x, x + w, step):
        ys = np.where(mask_gray[:, col] > 0)[0]
        if len(ys) and ys[0] > 2:
            xs.append(col)
            tops.append(ys[0])
    if len(xs) < 40: return None
    xs = np.asarray(xs, np.float64)
    tops_raw = np.asarray(tops, np.float64)

    # Gate 0: the top boundary must be CONTINUOUS. A real ceiling line bends
    # at a corner but never jumps; a jump means an occlusion boundary
    # (curtain, furniture) is cutting the mask — untrustworthy for folds.
    if float(np.max(np.abs(np.diff(tops_raw)))) > max(20.0, 0.08 * h): return None
    tops = np.convolve(tops_raw, np.ones(5) / 5.0, mode='same')

    def _fit(px, py):
        m, c = np.polyfit(px, py, 1)
        return m, c, float(np.mean(np.abs(py - (m * px + c))))

    _, _, res_single = _fit(xs, tops)
    best = None
    for frac in np.arange(0.28, 0.73, 0.05):
        sx = x + frac * w
        li = xs < sx
        n_l, n_r = int(li.sum()), int((~li).sum())
        if n_l < 15 or n_r < 15: continue
        if (xs[li].max() - xs[li].min()) < 0.22 * w or (xs[~li].max() - xs[~li].min()) < 0.22 * w: continue
        ml, cl, rl = _fit(xs[li], tops[li])
        mr, cr, rr = _fit(xs[~li], tops[~li])
        res_two = (rl * n_l + rr * n_r) / len(xs)
        if best is None or res_two < best[0]:
            best = (res_two, ml, cl, mr, cr, sx)

    if best is None: return None
    res_two, ml, cl, mr, cr, best_sx = best

    # Gate 1: two segments must explain the top boundary far better than one
    if res_two > 0.55 * res_single: return None
    # Gate 2: the fold must be a real slope change, both slopes architectural
    if abs(ml - mr) < 0.07 or abs(ml) > 0.8 or abs(mr) > 0.8: return None
    # Gate 3: a real room corner bends the ceiling line in a V or Λ — the
    # two slopes point in OPPOSITE directions (one may be near-flat for a
    # frontal wall). A same-direction bend is an occlusion kink: reject.
    ml0, mr0 = ml if abs(ml) >= 0.04 else 0.0, mr if abs(mr) >= 0.04 else 0.0
    if ml0 * mr0 > 0: return None
    # Gate 4: the fold (segment intersection) must sit inside the wall span
    xi = (cr - cl) / (ml - mr)
    if not (x + 0.25 * w <= xi <= x + 0.75 * w): return None
    # Gate 5: the intersection must agree with the best split position —
    # a kink at one end (curtain/furniture cut) fits badly and drifts
    if abs(xi - best_sx) > 0.15 * w: return None
    # Gate 6: the wall must have substantial height at the fold column;
    # real room corners run floor-to-ceiling, occlusion kinks don't
    col = int(np.clip(xi, 0, W - 1))
    ys_at_fold = np.where(mask_gray[:, col] > 0)[0]
    if len(ys_at_fold) == 0 or (ys_at_fold[-1] - ys_at_fold[0]) < 0.35 * h: return None
    return int(xi)

def _backproject_mask(depth_m, mask, focal_px, max_pts=40000):
    H, W = depth_m.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    ys, xs = np.where(mask > 127)
    if len(xs) < 200: return None
    Z = depth_m[ys, xs].astype(np.float64)
    ok = np.isfinite(Z) & (Z > 0.1) & (Z < 30.0)
    xs, ys, Z = xs[ok], ys[ok], Z[ok]
    if len(xs) < 200: return None
    X = (xs - cx) * Z / focal_px
    Y = (ys - cy) * Z / focal_px
    P = np.stack([X, Y, Z], axis=1)
    if len(P) > max_pts:
        idx = np.linspace(0, len(P) - 1, max_pts).astype(np.int64)
        P = P[idx]
    return P

def _fit_plane(P):
    c = P.mean(axis=0)
    n = np.array([0.0, 0.0, 1.0])
    for _ in range(4):
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        n = vt[-1]
        dist = (P - c) @ n
        keep = np.abs(dist) <= (2.5 * float(np.std(dist)) + 1e-9)
        if keep.sum() < 50: break
        P = P[keep]
        c = P.mean(axis=0)
    n = n / (np.linalg.norm(n) + 1e-9)
    return c, n, P

def _wall_quad_from_depth(depth_m, wall_mask, focal_px, image_shape, coverage=0.99):
    if depth_m is None or wall_mask is None: return None
    H, W = depth_m.shape[:2]
    f = float(focal_px)
    cx, cy = W / 2.0, H / 2.0
    if wall_mask.shape[:2] != (H, W):
        wall_mask = cv2.resize(wall_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    P0 = _backproject_mask(depth_m, wall_mask, f)
    if P0 is None: return None
    c, n, P = _fit_plane(P0)
    if len(P) < 50: return None

    up_img = np.array([0.0, -1.0, 0.0])
    u_up = up_img - (up_img @ n) * n
    nu = float(np.linalg.norm(u_up))
    if nu < 1e-6: return None
    u_up /= nu
    u_rt = np.cross(n, u_up)
    u_rt /= (np.linalg.norm(u_rt) + 1e-9)

    A = (P - c) @ u_rt # In-plane horizontal (metres)
    B = (P - c) @ u_up # In-plane vertical (metres)
    lo = max(0.0, (1.0 - coverage) * 50.0)
    a_lo, a_hi = np.percentile(A, [lo, 100 - lo])
    b_lo, b_hi = np.percentile(B, [lo, 100 - lo])
    if a_hi - a_lo < 1e-3 or b_hi - b_lo < 1e-3: return None

    img_pts = []
    for a, b in ((a_lo, b_hi), (a_hi, b_hi), (a_hi, b_lo), (a_lo, b_lo)):
        P3 = c + a * u_rt + b * u_up
        Zc = float(P3[2])
        if Zc <= 1e-3: return None
        img_pts.append([cx + f * float(P3[0]) / Zc, cy + f * float(P3[1]) / Zc])

    quad = order_points(np.array(img_pts, dtype=np.float32))
    H_img, W_img = image_shape[:2]
    if cv2.contourArea(quad) < 0.005 * W_img * H_img: return None

    for x_, y_ in quad:
        if x_ < -0.7 * W_img or x_ > 1.7 * W_img or y_ < -0.7 * H_img or y_ > 1.7 * H_img: return None

    info = {
        "width_m": float(a_hi - a_lo),
        "height_m": float(b_hi - b_lo),
        "normal": [float(v) for v in n],
        "center": [float(v) for v in c],
        "top_y": float(np.min(quad[:, 1])),
    }
    return quad, info

def _wall_folds_from_depth(depth_m, mask_gray, max_folds=2):
    H, W = mask_gray.shape[:2]
    x, y, w, h = cv2.boundingRect(mask_gray)
    if w < max(200, int(0.20 * W)): return []

    step = max(1, w // 300)
    xs, ds = [], []
    for col in range(x, x + w, step):
        ys_c = np.where(mask_gray[:, col] > 0)[0]
        if len(ys_c) < 5: continue
        z = depth_m[ys_c, col]
        z = z[np.isfinite(z) & (z > 0.1) & (z < 30.0)]
        if len(z) < 5: continue
        xs.append(col)
        ds.append(1.0 / float(np.median(z)))

    if len(xs) < 40: return []
    xs, ds = np.asarray(xs, np.float64), np.asarray(ds, np.float64)
    ds = ds / (np.median(ds) + 1e-12) # Dimensionless
    ds = np.convolve(ds, np.ones(5) / 5.0, mode='same') # Denoise

    def _fit(sel):
        m, c2 = np.polyfit(xs[sel], ds[sel], 1)
        return m, float(np.sum(np.abs(ds[sel] - (m * xs[sel] + c2))))

    all_sel = np.ones(len(xs), bool)
    _, r0 = _fit(all_sel)
    res0 = r0 / len(xs)

    MIN_SEG = 0.18 * w
    cands = [x + fr * w for fr in np.arange(0.20, 0.81, 0.04)]

    def _eval(folds):
        bounds = [xs[0] - 1] + list(folds) + [xs[-1] + 1]
        total, slopes = 0.0, []
        for i in range(len(bounds) - 1):
            sel = (xs > bounds[i]) & (xs <= bounds[i + 1])
            if sel.sum() < 12 or xs[sel].max() - xs[sel].min() < MIN_SEG: return None
            m, r = _fit(sel)
            total += r
            slopes.append(m)
        return total / len(xs), slopes

    def _signif(m_a, m_b): return abs(m_a - m_b) * w > 0.20

    best1 = None
    for f1 in cands:
        e = _eval([f1])
        if e and (best1 is None or e[0] < best1[0]): best1 = (e[0], e[1], [f1])

    folds = []
    if best1 and best1[0] < 0.55 * res0 and _signif(best1[1][0], best1[1][1]):
        folds = best1[2]
        if max_folds >= 2:
            best2 = None
            for i, f1 in enumerate(cands):
                for f2 in cands[i + 1:]:
                    if f2 - f1 < max(MIN_SEG, 0.22 * w): continue
                    e = _eval([f1, f2])
                    if e and (best2 is None or e[0] < best2[0]): best2 = (e[0], e[1], [f1, f2])
            if (best2 and best2[0] < 0.6 * best1[0] and _signif(best2[1][0], best2[1][1]) and _signif(best2[1][1], best2[1][2])):
                folds = best2[2]

    out = []
    for fx in folds:
        col = int(np.clip(fx, 0, W - 1))
        ys_c = np.where(mask_gray[:, col] > 0)[0]
        if len(ys_c) and (ys_c[-1] - ys_c[0]) >= 0.35 * h: out.append(int(fx))
    return sorted(out)

def _planes_via_depth(mask_gray, depth_map, focal_px, debug_img=None):
    H, W = mask_gray.shape[:2]
    if depth_map.shape[:2] != (H, W):
        depth_map = cv2.resize(depth_map.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    folds = _wall_folds_from_depth(depth_map, mask_gray)
    edges = [0] + folds + [W]
    planes = []

    for i in range(len(edges) - 1):
        x0, x1 = edges[i], edges[i + 1]
        seg = mask_gray.copy()
        seg[:, :x0] = 0
        seg[:, x1:] = 0
        if cv2.countNonZero(seg) < 500:
            if len(edges) > 2: continue # Empty side band in a multi-plane split
            return None

        res = _wall_quad_from_depth(depth_map, seg, focal_px, (H, W))
        if res is not None:
            quad, info = res
            if _quad_mask_containment(quad, seg) >= 0.85:
                print(f"   [DEPTH PLANE] band {x0}-{x1}: {info['width_m']:.2f}m x {info['height_m']:.2f}m")
                planes.append((quad, (x0, x1), info["width_m"]))
                if debug_img is not None:
                    try:
                        import os, time
                        os.makedirs("Debugs", exist_ok=True)
                        dbg = debug_img.copy()
                        cv2.polylines(dbg, [quad.astype(np.int32)], True, (255, 0, 255), 3)
                        cv2.putText(dbg, "DEPTH-PLANE", (int(quad[0][0]), max(30, int(quad[0][1]) - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                        cv2.imwrite(f"Debugs/wall_quad_debug_{int(time.time()*1000)}.jpg", dbg)
                    except Exception: pass
                continue
            print(f"   [DEPTH PLANE] band {x0}-{x1}: depth quad rejected (low mask coverage)")

        q2 = _detect_wall_quad(seg, debug_img=debug_img)
        if q2 is None: return None
        planes.append((q2, (x0, x1), None))
    return planes if planes else None

def _detect_wall_planes(mask_gray, debug_img=None, depth_map=None, focal_px=None):
    H, W = mask_gray.shape
    if depth_map is not None:
        try:
            f = focal_px if focal_px else WALL_FOCAL_RATIO * max(H, W)
            planes = _planes_via_depth(mask_gray, depth_map, f, debug_img=debug_img)
            if planes:
                return planes
            print("   [DEPTH] Wall plane path yielded nothing; falling back to 2D detection")
        except Exception as e:
            print(f"   [DEPTH] Wall plane path error ({e}); falling back to 2D detection")

    split_x = _find_corner_split(mask_gray)
    if split_x is not None:
        left, right = mask_gray.copy(), mask_gray.copy()
        left[:, split_x:], right[:, :split_x] = 0, 0
        quad_l = _detect_wall_quad(left, debug_img=debug_img)
        quad_r = _detect_wall_quad(right, debug_img=debug_img)
        if quad_l is not None and quad_r is not None:
            return [(quad_l, (0, split_x), None), (quad_r, (split_x, W), None)]
        print("   [CORNER SPLIT] Sub-plane quad detection failed; using single plane")
    quad = _detect_wall_quad(mask_gray, debug_img=debug_img)
    if quad is None: return []
    return [(quad, (0, W), None)]

def _warp_plane(wall_tex, tex_aspect, quad, dst_w, dst_h, n_tiles, out_w, out_h, cover_pts=None):
    tex_h, tex_w = wall_tex.shape[:2]
    # Floor the tile at ~16px in both dimensions so absurd manual repeat
    # values can't degenerate into sub-pixel tiles.
    tile_size_w = max(dst_w / float(n_tiles), 16.0 * max(1.0, tex_aspect))

    # How far past the quad edges must the texture extend (in flat texture
    # space) to cover every masked pixel? Map the mask bounds back through
    # the inverse homography to find out.
    need_x = need_y = 0.0
    if cover_pts is not None and len(cover_pts) > 0:
        src0 = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
        M0 = cv2.getPerspectiveTransform(src0, quad)
        try:
            Minv = np.linalg.inv(M0)
            pts = cv2.perspectiveTransform(np.asarray(cover_pts, dtype=np.float32).reshape(1, -1, 2), Minv)[0]
            if np.all(np.isfinite(pts)):
                # Memory cap; the void-fill net covers pathological leftovers
                need_x = min(max(0.0, float(-pts[:, 0].min()), float(pts[:, 0].max() - dst_w)), 1.0 * dst_w)
                need_y = min(max(0.0, float(-pts[:, 1].min()), float(pts[:, 1].max() - dst_h)), 1.0 * dst_h)
        except np.linalg.LinAlgError: pass

    # Pre-crop the product image to its visible rows/columns when a tile
    # overflows the covered area: identical visible pixels, without
    # allocating an oversized tile in memory.
    tex_eff = wall_tex
    tile_h_px = tile_size_w / tex_aspect
    if tile_h_px > dst_h + need_y:
        crop_h = max(1, min(tex_h, int(round(tex_h * (dst_h + need_y) / tile_h_px))))
        tex_eff = tex_eff[:crop_h, :]
    if tile_size_w > dst_w + need_x:
        crop_w = max(1, min(tex_w, int(round(tex_w * (dst_w + need_x) / tile_size_w))))
        tex_eff = tex_eff[:, :crop_w]
        tile_size_w = dst_w + need_x

    ph_e, pw_e = tex_eff.shape[:2]
    tw = max(1, int(tile_size_w))
    th = max(1, int(ph_e * (tile_size_w / float(pw_e))))
    pad_x_tiles, pad_y_tiles = max(1, int(math.ceil(need_x / tw))), max(1, int(math.ceil(need_y / th)))

    super_tex, pts_src = create_super_texture(tex_eff, int(dst_w), int(dst_h), tile_size_w, pad_x_tiles=pad_x_tiles, pad_y_tiles=pad_y_tiles)
    M = cv2.getPerspectiveTransform(pts_src, quad)
    return cv2.warpPerspective(super_tex, M, (out_w, out_h), flags=cv2.INTER_LINEAR)

def _fill_uncovered(texture, region_mask):
    """Fill masked pixels the plane warps missed with the nearest textured
    pixel. A masked pixel must never render as a black void."""
    empty = (texture.max(axis=2) == 0).astype(np.uint8)
    holes = (empty > 0) & (region_mask > 0)
    if not holes.any() or not (empty == 0).any(): return texture
    try:
        _, labels = cv2.distanceTransformWithLabels(empty, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        src_yx = np.argwhere(empty == 0)
        lab_at_src = labels[empty == 0]
        coord = np.zeros((int(lab_at_src.max()) + 1, 2), np.int32)
        coord[lab_at_src] = src_yx
        hy, hx = np.where(holes)
        nyx = coord[labels[hy, hx]]
        texture[hy, hx] = texture[nyx[:, 0], nyx[:, 1]]
        print(f"   [VOID FILL] Patched {len(hy)} uncovered mask pixels from nearest texture")
    except Exception as e: print(f"   [WARN] Void-fill failed: {e}")
    return texture

def _split_counts(total, widths):
    k = len(widths)
    if k == 1: return [total]
    total = max(total, k)
    wsum = float(sum(widths))
    raw = [total * float(wd) / wsum for wd in widths]
    counts = [max(1, int(r + 0.5)) for r in raw]
    while sum(counts) > total:
        over = [(counts[j] - raw[j], j) for j in range(k) if counts[j] > 1]
        if not over: break
        _, j = max(over)
        counts[j] -= 1
    while sum(counts) < total:
        _, j = min((counts[j] - raw[j], j) for j in range(k))
        counts[j] += 1
    return counts

def apply_pattern(room_img, wall_tex, mask_img, fallback_repeat=None, depth_map=None, shadow_strength=0.6):
    """
    Returns (result_image, auto_repeat) — auto_repeat is the INTEGER total
    repeat chosen by the auto logic (reported back for the frontend's step-1
    slider), or None when a manual repeat was used / the layer was skipped.

    Repeat semantics (auto and manual share them, always integer):
      - repeat = total tiles across the wall's width. On corner walls (two
        detected planes) it is distributed proportionally to plane widths,
        each plane getting at least 1.
      - tile_width = plane_width / N_plane; tile height always follows the
        product image's aspect ratio; rows tile downward and overflow crops
        at the wall edges. The pattern is never stretched or compressed —
        the repeat only changes its scale.
      - Auto reproduces the OLD production tile scale (tile of roughly
        canvas_width / TARGET_TILES_ACROSS_CANVAS), converted to an integer
        count for the detected wall span.

    depth_map: optional METRIC depth (metres, any resolution — resized to the
    canvas). When present, wall planes come from true 3D geometry: disparity
    creases give room-corner folds (up to 2 corners / 3 faces, even in one
    singular mask) and each face gets a depth-fitted perspective quad. Every
    depth failure falls back to the 2D pipeline — depth is advisory only.
    """
    print("[INFO] Processing Wall (Structural Perspective + Corner Split)...")
    H, W = room_img.shape[:2]
    mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY) if len(mask_img.shape) == 3 else mask_img
    mask_gray = resize_mask_alpha(mask_gray, W, H)
    # Plane detection, quad fitting and warp coverage all still work off a binary
    # mask; thresholding the soft alpha at 127 recovers it, now with a sub-pixel
    # accurate boundary instead of one quantised to the upload's pixel grid.
    _, thresh = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    planes = _detect_wall_planes(thresh, debug_img=room_img, depth_map=depth_map, focal_px=WALL_FOCAL_RATIO * max(H, W))
    if not planes: return room_img, None

    try:
        tex_h, tex_w = wall_tex.shape[:2]
        tex_aspect = tex_w / float(tex_h)

        # Flat (pre-warp) size of each architectural plane
        sized = []
        for quad, band, width_m in planes:
            dst_w = max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[3] - quad[2]))
            dst_h = max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2]))
            if dst_w >= 2 and dst_h >= 2: sized.append((quad, band, dst_w, dst_h, width_m))

        if not sized: return room_img, None

        widths = [p[2] for p in sized]
        span_w = float(sum(widths))
        min_total = len(sized)

        # Distribute repeats by REAL metric widths when depth provided them —
        # a foreshortened plane is narrow in pixels but not in metres.
        metric_widths = [p[4] for p in sized]
        split_widths = metric_widths if len(sized) > 1 and all(wm is not None and wm > 0 for wm in metric_widths) else widths

        is_auto = fallback_repeat is None or float(fallback_repeat) == 2.0 or float(fallback_repeat) <= 0

        if is_auto:
            raw = TARGET_TILES_ACROSS_CANVAS * span_w / float(W)
            repeat_total = max(min_total, int(raw + 0.5))
            print(f"\n   [WALL AUTO REPEAT - CANVAS SCALE]")
            print(f"    |- Canvas Width:   {W} px (target tile ~ {W/TARGET_TILES_ACROSS_CANVAS:.0f} px)")
            print(f"    |- Wall Planes:    {len(sized)} (span {span_w:.0f} px = {span_w/W*100:.0f}% of canvas)")
            print(f"    L_ Repeats:        {repeat_total} tiles across the wall (raw {raw:.2f})\n")
        else:
            repeat_total = max(min_total, int(round(float(fallback_repeat))))
            print(f"[INFO] Using provided manual repeat: {repeat_total}")

        per_plane = _split_counts(repeat_total, split_widths)

        warped_total = np.zeros((H, W, 3), dtype=np.uint8)

        def _mask_bounds(region):
            # Bounding corners of the masked pixels the warp must cover
            bx, by, bw2, bh2 = cv2.boundingRect(np.ascontiguousarray(region))
            if bw2 == 0 or bh2 == 0: return None
            return np.array([[bx, by], [bx + bw2, by], [bx + bw2, by + bh2], [bx, by + bh2]], dtype=np.float32)

        if len(sized) > 1:
            # Underlay for corner walls: one single-quad warp over the whole
            # mask, so any pixel the per-plane warps miss falls back to
            # plausible pattern instead of a black void.
            base_quad = _detect_wall_quad(thresh)
            if base_quad is not None:
                b_w = max(np.linalg.norm(base_quad[0] - base_quad[1]), np.linalg.norm(base_quad[3] - base_quad[2]))
                b_h = max(np.linalg.norm(base_quad[0] - base_quad[3]), np.linalg.norm(base_quad[1] - base_quad[2]))
                if b_w >= 2 and b_h >= 2:
                    warped_total = _warp_plane(wall_tex, tex_aspect, base_quad, b_w, b_h, repeat_total, W, H, cover_pts=_mask_bounds(thresh))

        for (quad, band, dst_w, dst_h, _wm), n_tiles in zip(sized, per_plane):
            x0, x1 = band
            cover = _mask_bounds(thresh[:, x0:x1])
            if cover is not None: cover[:, 0] += x0
            warped = _warp_plane(wall_tex, tex_aspect, quad, dst_w, dst_h, n_tiles, W, H, cover_pts=cover)
            plane_slice = warped[:, x0:x1]
            covered = plane_slice.max(axis=2) > 0
            dest = warped_total[:, x0:x1]
            dest[covered] = plane_slice[covered]

        # Safety net: any masked pixel every warp missed gets filled from the
        # nearest textured pixel — a void inside the wall is never acceptable.
        # Driven by the SOFT alpha, not `thresh`: the feather band sits outside
        # the binary mask but still receives texture in the blend, so leaving it
        # black there would darken the original photo along every edge.
        warped_total = _fill_uncovered(warped_total, mask_gray)
        result = blend_hard_replace(room_img, warped_total, mask_gray, shadow_strength=shadow_strength)
        return result, (int(repeat_total) if is_auto else None)
    except Exception as e:
        print(f"Error in apply_pattern: {e}")
        return room_img, None