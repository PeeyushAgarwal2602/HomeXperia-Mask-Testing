import cv2
import numpy as np

"""Structure-aware boundary regularisation for surface masks.

WHY THIS EXISTS
---------------
Measured over 1113 production wall masks: boundary wobble at a 12px scale is only
0.63px, and the final guided-filter pass is roughly neutral on it (+6.7%). The
masks are therefore already LOCALLY smooth, so feathering, blurring and
morphological smoothing cannot help — that is not the defect.

The defect is that the boundary is locally smooth but globally NOT STRAIGHT. On
the median wall, 84% of the perimeter sits on long runs that stay within ~0.5% of
the long side of a straight line yet never actually are one: a ceiling line fitted
over 680 columns of one real mask had 3.1px residual std and 6.8px peak
deviation, which becomes ~17px of visible meander once the canvas is upscaled to
4500px. The eye compares that against the true architectural edge, still visible
in the photo, and reads it as wavy or torn — exactly the wall/ceiling seams,
window-frame strips and soffit bands that get flagged.

Nothing upstream has any notion of "this boundary is a straight line": EoMT
resamples a 640px panoptic grid, MORPH_CLOSE rounds concavities, and
refine_mask_edges snaps to whatever local intensity edge falls inside its radius
(wallpaper pattern and fold shadows included). Straightness has to be imposed
explicitly, and that is all this module does.

APPROACH — coarse-to-fine Douglas-Peucker with a validation gate
----------------------------------------------------------------
Douglas-Peucker gives a polygon whose vertices are real contour points and which
is guaranteed to lie within its tolerance of the original boundary. Run it at TWO
tolerances, because the two failure modes have very different amplitudes:

  COARSE (tau_coarse) catches TORN boundaries. Where EoMT leaves a wide band of
    void along a thin architectural feature, the boundary is bitten inwards by
    far more than a fine tolerance can span, so a single-scale pass shatters the
    edge into many short pieces and restores every one of them — which is why a
    soffit above a window keeps its ragged top edge. A coarse pass spans the
    bites in one edge.

  FINE (tau_fine) catches merely WAVY boundaries, and is the error bound applied
    to everything the coarse pass did not already claim.

A coarse edge is only accepted if a majority of the original arc beneath it lies
within tau_fine of the straight chord. That gate is what protects real corners: a
genuine bend departs from its chord along its whole length, so the majority test
fails and it falls through to the fine pass, whereas a torn straight edge departs
only at localised bites and passes. Without the gate, a coarse tolerance would
happily cut the corner off two adjoining walls.

WHY LENGTH ALONE IS NOT ENOUGH — the straighten_zone
-----------------------------------------------------
A gently curved arc has excellent chord support: every 61px chunk of an oval sits
within a few px of its chord, so a length-plus-support rule straightens each chunk
and turns the oval into a polygon. Measured on real masks: a decorative tree
silhouette collapsed into a diamond and an oval mirror's top arc flattened.

Geometry cannot fix this, and it is worth being precise about why. The obvious
tests were measured on real runs, and the case that MUST be straightened is
numerically indistinguishable from the case that must be preserved:

    soffit torn edge (straighten)   sagitta/length 0.069   one-sidedness 1.00
    oval mirror arc  (preserve)     sagitta/length 0.066   one-sidedness 1.00

A torn architectural edge and a curved object silhouette are the same shape class.
No curvature, sagitta, sign-change or residual-distribution threshold separates
them, so this module does not pretend to.

What separates them is SEMANTIC: what lies on the other side of the boundary. A
wall meeting a ceiling, floor, window frame or another wall shares a genuinely
straight architectural edge. A wall meeting a mirror, a plant, a lamp or curtain
fabric is bounded by that object's organic contour. The caller knows which is
which — it has the panoptic labels — so it passes a `straighten_zone` marking
where straightening is permitted. Outside that zone edges are restored verbatim
however straight they look.

Edges inside the zone are then classified by LENGTH at whichever scale claimed
them:

  long edge  (>= min_len)  -> emit the straight segment: ceiling lines, skirting,
                              window-frame verticals, soffit edges, corner patches.
  short edge (<  min_len)  -> restore the ORIGINAL contour points, so object
                              silhouettes (a sconce, a plant, a curtain hem) keep
                              their shape, optionally with a light arc-length
                              smoothing to take off the pixel staircase.

Error is hard-bounded by the tolerance of the pass that claimed a run, and is
confined to runs already judged near-straight; detail keeps its geometry. Holes
are regularised at the correct polarity, so a window cut out of a wall gets
straight edges just as the wall's own outline does.
"""

# Fine Douglas-Peucker tolerance, as a fraction of the long side (7.7px at 1536).
# Sized from the measured peak deviation of real architectural boundaries (6.8px)
# so a straight-but-wandering line is captured in ONE edge rather than broken into
# several that each follow the wander.
TAU_FINE_FRAC = 0.005

# Coarse tolerance for spanning torn edges (23px at 1536). Only ever applied to
# runs that pass the majority gate below, so this is not a blanket error budget.
TAU_COARSE_FRAC = 0.015

# Share of an arc that must lie within tau_fine of a coarse chord before that
# chord is accepted as the underlying straight structure.
COARSE_ACCEPT_FRAC = 0.60

# Share of an arc that must fall inside straighten_zone before the arc may be
# straightened. Deliberately high: an arc straddling the edge of the zone is
# partly against an object, and half-straightening it produces a corner.
ZONE_ACCEPT_FRAC = 0.90

# Minimum edge length to treat as architecture, as a fraction of the long side
# (61px at 1536). Below this an edge is assumed object detail and is restored.
MIN_LEN_FRAC = 0.04

# Arc-length sigma for smoothing short/detail runs, as a fraction of the long side
# (3.8px at 1536). Takes the staircase and BiRefNet's fringe off object
# silhouettes; set to 0 to leave detail bit-exact.
DETAIL_SIGMA_FRAC = 0.0025

# Largest area change a single contour may suffer before its tolerances are
# retried smaller. The tolerances above are scaled to the IMAGE, but the damage
# they do is scaled to the FEATURE: on a 1400x700 wall a 23px coarse tolerance is
# noise, while on a 48px-wide curtain sliver it is half the width, and DP's chord
# through a gently curved long edge sits inside the curve and shaves the sliver by
# 15%. A fixed ratio cap cannot resolve that — the tolerance a 78px-tall soffit
# needs to span its tears is 29% of its own width, so capping by width would break
# the case this module exists to fix. Bounding the OUTCOME instead of guessing a
# ratio lets each contour keep the largest tolerance it can actually afford.
MAX_AREA_DELTA = 0.06
RETRY_SCALES = (1.0, 0.5, 0.25)


def _arc(pts, i, j):
    """Contour points from i up to (not including) j, walking forward, wrapping."""
    if i < j:
        return pts[i:j]
    return np.vstack([pts[i:], pts[:j]])


def _vertex_indices(pts, poly, start=0, wrap=True):
    """Index of each polygon vertex within pts, monotonically forward.
    approxPolyDP returns actual input points, so these are exact coordinate
    matches; searching forward from the previous hit keeps the walk ordered even
    when a contour revisits a coordinate."""
    n = len(pts)
    idx = []
    for p in poly:
        hit = -1
        span = n if wrap else n - start
        for off in range(max(0, span)):
            j = (start + off) % n if wrap else start + off
            if pts[j][0] == p[0] and pts[j][1] == p[1]:
                hit = j
                break
        if hit < 0:
            return None
        idx.append(hit)
        start = hit
    return idx


def _smooth_open(pts, sigma):
    """Gaussian smoothing along an open arc, replicating the end points so the
    arc's junctions with its neighbours are not pulled away from them."""
    if sigma <= 0 or len(pts) < 5:
        return pts.astype(np.float64)
    k = max(3, int(sigma * 4) | 1)
    if len(pts) <= k:
        return pts.astype(np.float64)
    g = cv2.getGaussianKernel(k, sigma).ravel()
    pad = np.vstack([np.repeat(pts[:1], k, axis=0), pts,
                     np.repeat(pts[-1:], k, axis=0)]).astype(np.float64)
    sm = np.stack([np.convolve(pad[:, i], g, mode="same") for i in (0, 1)], axis=1)
    return sm[k:k + len(pts)]


def _arc_supports_chord(arc, p0, p1, tau_fine, frac_needed=COARSE_ACCEPT_FRAC):
    """Does a majority of `arc` lie within tau_fine of the chord p0->p1?

    True means the deviations are localised (a torn edge) and the chord is the
    real structure. False means the arc departs along its whole length (a genuine
    corner or a curve) and must not be straightened at this scale.
    """
    if len(arc) < 3:
        return True
    v = (p1 - p0).astype(np.float64)
    length = float(np.hypot(*v))
    if length < 1e-6:
        return False
    normal = np.array([-v[1], v[0]], dtype=np.float64) / length
    resid = np.abs((arc.astype(np.float64) - p0.astype(np.float64)) @ normal)
    return float((resid <= tau_fine).mean()) >= frac_needed


def _arc_in_zone(arc, zone, frac_needed=ZONE_ACCEPT_FRAC):
    """Does the arc lie inside the region where straightening is permitted?"""
    if zone is None:
        return True
    h, w = zone.shape[:2]
    xs = np.clip(np.round(arc[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(arc[:, 1]).astype(np.int32), 0, h - 1)
    return float((zone[ys, xs] > 0).mean()) >= frac_needed


def _refine_arc(arc, tau_fine, min_len, detail_sigma, zone=None):
    """Fine pass over one arc the coarse pass did not claim. Emits the arc's
    points from its start up to (not including) its last point — the caller's
    next edge contributes that."""
    if len(arc) < 8:
        return arc[:-1].astype(np.float64) if len(arc) > 1 else arc.astype(np.float64)
    poly = cv2.approxPolyDP(arc.reshape(-1, 1, 2).astype(np.int32), tau_fine, False)
    poly = poly.squeeze()
    if poly.ndim != 2 or len(poly) < 2:
        return _smooth_open(arc[:-1], detail_sigma)
    idx = _vertex_indices(arc, poly, start=0, wrap=False)
    if idx is None:
        return _smooth_open(arc[:-1], detail_sigma)

    out = []
    for a in range(len(poly) - 1):
        sub = arc[idx[a]:idx[a + 1]]
        if (float(np.hypot(*(poly[a + 1] - poly[a]))) >= min_len
                and (len(sub) == 0 or _arc_in_zone(sub, zone))):
            out.append(poly[a].reshape(1, 2).astype(np.float64))
        else:
            out.append(_smooth_open(sub, detail_sigma) if len(sub)
                       else poly[a].reshape(1, 2).astype(np.float64))
    return np.vstack(out) if out else _smooth_open(arc[:-1], detail_sigma)


def regularize_contour(contour, tau_fine, tau_coarse, min_len, detail_sigma=0.0,
                       zone=None):
    """Straighten a contour's architectural runs, preserve its detail runs."""
    pts = contour.squeeze()
    if pts.ndim != 2 or len(pts) < 8:
        return None
    poly = cv2.approxPolyDP(contour, tau_coarse, True).squeeze()
    if poly.ndim != 2 or len(poly) < 3:
        return None
    idx = _vertex_indices(pts, poly)
    if idx is None:
        return None

    k = len(poly)
    out = []
    for a in range(k):
        b = (a + 1) % k
        arc = _arc(pts, idx[a], idx[b])
        long_enough = float(np.hypot(*(poly[b] - poly[a]))) >= min_len
        if (long_enough and _arc_in_zone(arc, zone)
                and _arc_supports_chord(arc, poly[a], poly[b], tau_fine)):
            # Architecture: emit only the start vertex so the fill walks a dead
            # straight line from here to the next vertex.
            out.append(poly[a].reshape(1, 2).astype(np.float64))
        elif len(arc) == 0:
            out.append(poly[a].reshape(1, 2).astype(np.float64))
        else:
            out.append(_refine_arc(np.vstack([arc, pts[idx[b]]]),
                                   tau_fine, min_len, detail_sigma, zone=zone))
    return np.vstack(out)


def _regularize_bounded(contour, tau_fine, tau_coarse, min_len, detail_sigma,
                        zone=None, max_area_delta=MAX_AREA_DELTA):
    """Regularise one contour, backing off the tolerances until the area it costs
    is acceptable. Returns None if even the smallest scale overshoots, so the
    caller keeps the original boundary rather than damaging the shape."""
    a0 = abs(cv2.contourArea(contour))
    for scale in RETRY_SCALES:
        reg = regularize_contour(contour, tau_fine * scale, tau_coarse * scale,
                                 min_len, detail_sigma * scale, zone=zone)
        if reg is None or len(reg) < 3:
            continue
        if a0 <= 0:
            return reg
        a1 = abs(cv2.contourArea(np.round(reg).astype(np.int32)))
        if abs(a1 - a0) / a0 <= max_area_delta:
            return reg
    return None


def regularize_mask(mask_img, straighten_zone=None, tau_fine_frac=TAU_FINE_FRAC,
                    tau_coarse_frac=TAU_COARSE_FRAC, min_len_frac=MIN_LEN_FRAC,
                    detail_sigma_frac=DETAIL_SIGMA_FRAC):
    """Regularise every boundary of a binary mask. Returns a uint8 0/255 mask.

    straighten_zone: uint8 mask, non-zero where straightening is permitted. Build
    it from the panoptic labels — permit boundary against architectural planes
    (ceiling, floor, another wall, a window or door frame) and forbid it against
    objects and curtain fabric. Passing None permits everywhere, which will
    polygonise curved silhouettes; callers with label information should always
    supply it.
    """
    if mask_img is None:
        return mask_img
    binary = (mask_img > 127).astype(np.uint8)
    if not binary.any():
        return mask_img

    h, w = binary.shape[:2]
    d = max(h, w)
    tau_fine = max(1.0, tau_fine_frac * d)
    tau_coarse = max(tau_fine, tau_coarse_frac * d)
    min_len = max(8.0, min_len_frac * d)
    detail_sigma = detail_sigma_frac * d

    cnts, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hier is None or not cnts:
        return mask_img
    hier = hier[0]

    def depth(i):
        dep, parent = 0, hier[i][3]
        while parent != -1:
            dep += 1
            parent = hier[parent][3]
        return dep

    out = np.zeros((h, w), np.uint8)
    # Shallowest first so a nested contour always overwrites its parent: even
    # depth fills, odd depth carves. Handles a component sitting inside a hole,
    # which a flat outer-then-holes pass would erase.
    for i in sorted(range(len(cnts)), key=depth):
        reg = _regularize_bounded(cnts[i], tau_fine, tau_coarse, min_len,
                                  detail_sigma, zone=straighten_zone)
        if reg is None or len(reg) < 3:
            reg = cnts[i].squeeze()
            if reg.ndim != 2 or len(reg) < 3:
                continue
        cv2.fillPoly(out, [np.round(reg).astype(np.int32)], 0 if depth(i) % 2 else 255)
    return out
