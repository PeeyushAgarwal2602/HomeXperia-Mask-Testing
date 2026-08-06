import os
import re
import cv2
import time
import uuid
import torch
import numpy as np
from PIL import Image

# Global variables to hold the models in memory
_models_loaded = False
processor = None
segmenter = None
sam_predictor = None
_depth_processor = None
_depth_model = None
_birefnet = None
device = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_OBJECTS = {
    "wall": ["wall"],
    "floor": ["floor", "flooring"],
    "curtain": ["curtain", "blind", "drape"],
    "rug": ["rug", "carpet"],
    "window" : ["window"],
    "door" : ["door"]
}

TYPE_MAPPING = {
    "curtain": "window",
    "floor": "floor",
    "rug": "floor",
    "wall": "wall",
    "window": "window",
    "door": "door"
}

SUB_CATEGORY = {
    "curtain": "73189a1e-da26-447a-a8ff-43d8ae388bbf",
    # "floor": "1f20b5a7-1144-47ad-b57c-5f25609eb763",
    "floor": "b41c7377-c4a5-461a-b214-12cbc52eb17e",
    "rug": "b41c7377-c4a5-461a-b214-12cbc52eb17e",
    "wall": "68381f08-54d1-4836-bf06-1af233ecac81",
    "window": "d9c8e1b7-5c3a-4c8c-9b0a-2f1e5b6f8a2d",
    "door": "e1f2c3d4-5678-4abc-9def-1234567890ab"
}

# Objects that commonly OCCLUDE surfaces. They are segmented precisely and
# SUBTRACTED from surface masks, so an object yields a tight silhouette cut
# instead of one large rectangular bite.
OCCLUDER_OBJECTS = {
    "plant", "flora", "tree", "flower", "palm", "pot", "flowerpot", "vase",
    "lamp", "light", "floor lamp", "table lamp", "chandelier", "pendant", "pendent", "sconce",
    "fan", "sculpture", "ceiling fan",
    "wardrobe", "cabinet", "closet", "cupboard", "chest", "chest of drawers",
    "bookcase", "bookshelf", "shelf", "shelving",
    "refrigerator", "fridge", "washing machine",
    "sofa", "couch", "armchair", "chair", "bench",
    "table", "desk", "counter", "countertop",
    "bed", "headboard",
    "television", "tv", "monitor", "screen",
    "door", "sliding door",
    "radiator", "air conditioner", "ac unit",
}

# STRICT MODEL OWNERSHIP. Every occluder is handled by exactly one model, never both.
#
# FINE — anything containing genuinely fine structure — is owned by BiRefNet, whole
# object. Assignment is per OBJECT, not per part: a floor lamp is a thin pole plus a
# solid shade and base, a potted plant is fine foliage plus a solid pot. If any part is
# fine, BiRefNet takes all of it, because BiRefNet handles solid shapes perfectly well
# while SAM genuinely cannot resolve thin ones — the split is not symmetric.
#
# Everything else in OCCLUDER_OBJECTS is SOLID and owned by OneFormer + a constrained
# SAM refinement. SAM earns its place here for one specific reason: BiRefNet is a
# SALIENT-object model, and its failure mode is "nothing distinctly salient, return
# everything" — exactly what a sofa filling most of a crop triggers.
#
# label_matches compares SINGLE tokens, so multi-word entries can never match; every
# token below hits a real ADE20K-150 label OneFormer emits.
FINE_OCCLUDER_LABELS = {
    "plant", "flora", "tree", "flower", "palm", "branch", "vine", "leaf", "leaves",
    "pot", "flowerpot", "vase", "sculpture", "statue", "figurine",
    "lamp", "light", "chandelier", "pendant", "pendent", "sconce", "fan",
    "candle", "candlestick", "tripod", "stand",
}

ONEFORMER_EXTENT_CLASSES = {"wall", "floor", "curtain"}

# Surfaces from which precise occluder silhouettes should be subtracted.
OCCLUDER_SUBTRACT_CLASSES = {"curtain", "wall"}

# Occluder segments are admitted on an ABSOLUTE pixel floor, not a fraction of the
# image. A fraction scales the wrong way: 0.05% of a 3840x2063 upload is 3960 px, so a
# whole dried-branch segment was dropped HERE — before build_occluder_union_birefnet
# ever got the chance to split it into components — while the same branch in an
# 894x894 upload (400 px threshold) sailed through. That is the "twigs getting
# swallowed" complaint on 2bcd442b and 41384699: the branch was never an occluder
# candidate at all, so no amount of BiRefNet quality downstream could recover it.
# 100 px matches both the per-component floor in build_occluder_union_birefnet and the
# production pipeline's own find_occluder_candidates threshold.
OCCLUDER_MIN_PX = 100
SMALL_OBJECT_MIN_AREA = 0.005  # hotspot small-object filter (0.5% of the image)

# Half-width, in MASK pixels, of the anti-aliased alpha ramp written into the saved
# mask PNG — see antialias_mask_edge for why this exists and why it is a constant.
# --- fill-once / cut-once tuning ---
# BiRefNet's matte is thresholded to binary HERE and nowhere else: soft alpha is
# produced exactly once, at the very end, at render resolution. Carrying softness
# through intermediate stages is what made v2.6 look blurred — a geometric feather and
# a BiRefNet matte compounded into two overlapping soft bands.
# Slightly below mid-grey so the cut is marginally generous: if a thin object shows a
# 1-2px fringe of surface texture on its edge, LOWER this rather than dilating.
BIREFNET_CUT_LEVEL = 115

# A VOID hole (OneFormer labelled nothing) larger than this share of the image is not
# model uncertainty, it is an unlabelled real object. Halo clusters measured on room
# 6a717a84 (3840x2063) topped out at ~7k px = 0.09% of frame, so 0.5% clears every
# genuine halo with a wide margin while still refusing to swallow furniture.
VOID_FILL_MAX_FRAC = 0.005
# ...but a void hole that TOUCHES an occluder is that object's own unlabelled fringe,
# and a large object legitimately has a large fringe, so it gets a far more generous
# ceiling. Contact with an occluder is strong evidence about what the void IS, which a
# frame-relative area threshold alone can never provide. (Prod's equivalent branch had
# no ceiling at all here; this keeps one so a dresser fused to a plant still cannot get
# through — fused, their areas sum and blow past even this cap, and the conservative
# failure is to under-fill rather than to paint wallpaper over furniture.)
VOID_HALO_MAX_FRAC = 0.04

# Validation gate: reject the heal if the final mask outgrew OneFormer's own surface by
# more than this. A complete fill is a powerful operation and needs a guard — prod's
# bbox fill had no ceiling and that is why v2.1 abandoned it. Deliberately loose: this
# is a backstop against catastrophic overreach (wallpaper across a ceiling), not a
# policy on normal growth, and the precise protection is the structural-claim check
# beside it. Every rejection is logged, so if it fires on healthy masks it will show.
MAX_HEAL_GROWTH_FRAC = 0.5

# Render canvas long side (mirrors app.py upscale_image / MAX_DIM) and the width of the
# anti-aliased band, in RENDER pixels. Stating it in render px is the whole point: the
# v2.6 feather was 2.5 MASK px, which became 25 render px at a 5x upscale.
RENDER_MAX_DIM = 4500
AA_RENDER_PX = 1.5

# Classes that still get the guided-filter edge snap. Kept deliberately until the new
# boundary generation is verified to produce clean edges unaided; the wall has been
# excluded since v2.4 because it measurably looked better without it.
EDGE_REFINE_CLASSES = {"curtain", "floor", "rug", "window", "door"}

# Max enclosed-hole size to fill, as a fraction of the image area. Holes larger
# than this are real objects sitting on/in the surface (a table on the floor, a
# window in a wall) and MUST stay cut out. Floor/rug are kept tight so furniture
# is never swallowed; vertical surfaces allow slightly larger fold/gap fills.
DEFAULT_HOLE_FILL_FRAC = 0.003
HOLE_FILL_FRAC = {
    "floor": 0.0015,
    "rug": 0.0015,
    "wall": 0.002,
    "curtain": 0.004,
    "window": 0.004,
}

DEBUG_SEG = True
_DEBUG_MASK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Debugs", "Masks")
_DEBUG_MASK_DIR = os.path.normpath(_DEBUG_MASK_DIR)
try:
    os.makedirs(_DEBUG_MASK_DIR, exist_ok=True)
except Exception:
    pass


def load_models_if_needed():
    global _models_loaded, processor, segmenter, sam_predictor
    if _models_loaded: return

    print(f"➡ [INFO] Loading OneFormer & SAM-HQ models to {device.upper()}...")
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    from segment_anything_hq import sam_model_registry, SamPredictor # type:ignore

    # Load OneFormer.
    processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    segmenter = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large").to(device)

    # Load SAM-HQ (ViT-B)
    sam_checkpoint = "sam_hq_vit_b.pth"
    model_type = "vit_b"
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "map_location": device})
    try:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    finally:
        torch.load = _orig_load
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    print("✅ [SUCCESS] All Models Loaded Successfully!")
    _models_loaded = True

# Depth Anything V2 (metric, indoor) — reconstructs the floor PLANE so the rug
# visualizer gets a perspective-correct floor quad. Loaded lazily on first use.
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

def load_depth_model_if_needed():
    global _depth_processor, _depth_model
    if _depth_model is not None:
        return
    print(f"➡ [INFO] Loading Depth-Anything-V2 (metric) to {device.upper()}...")
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    _depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    _depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(device).eval()
    print("✅ [SUCCESS] Depth model loaded.")

def get_metric_depth(image_cv2):
    """Return a per-pixel METRIC depth map (HxW float, metres) for a BGR image."""
    import torch.nn.functional as F
    load_depth_model_if_needed()

    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    inputs = _depth_processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        predicted = _depth_model(**inputs).predicted_depth

    h, w = image_cv2.shape[:2]
    depth = F.interpolate(predicted.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False)[0, 0]
    return depth.cpu().numpy()

BIREFNET_MODEL_ID = "ZhengPeng7/BiRefNet_lite"
BIREFNET_INPUT_SIZE = 512  # do NOT raise: 1024 OOMs under multi-worker gunicorn

# Contrast applied to BiRefNet's sigmoid before it is used as an alpha matte.
# The raw sigmoid is a saliency score, not a calibrated alpha: left alone, a whole
# ambiguous region can sit at 0.3-0.7 and turn a large patch of surface
# semi-transparent. This gain keeps the soft band narrow (only p in
# [0.5 - 0.5/gain, 0.5 + 0.5/gain] stays intermediate — at gain 4 that is
# 0.375..0.625) so the silhouette is crisp with a genuinely anti-aliased edge,
# instead of either a hard staircase or a mushy blob.
BIREFNET_MATTE_GAIN = 4.0

def load_birefnet_if_needed():
    global _birefnet
    if _birefnet is not None:
        return
    print(f"➡ [INFO] Loading BiRefNet-lite to {device.upper()}...")
    from transformers import AutoModelForImageSegmentation
    _birefnet = AutoModelForImageSegmentation.from_pretrained(
        BIREFNET_MODEL_ID, trust_remote_code=True
    )
    _birefnet.eval().to(device)
    print("✅ [INFO] BiRefNet-lite loaded.")

def birefnet_fg_mask(image_pil, out_hw):
    """Run BiRefNet on a PIL crop; return a SOFT uint8 alpha matte resized to out_hw.

    The sigmoid is kept as a matte rather than thresholded at 0.5. BiRefNet predicts a
    genuinely sub-pixel silhouette — that is the whole reason it is in this pipeline —
    and `(pred > 0.5) * 255` threw it away, so a leaf edge arrived as a hard staircase
    on the 512-px inference grid. Worse, the old code thresholded and THEN resized with
    INTER_LINEAR, so the intermediate values that did survive were interpolation of an
    already-binarised mask: soft-looking, but carrying no more information than the
    binary. Scaling the probability first keeps the real thing.
    """
    import torchvision.transforms.functional as TF
    s = BIREFNET_INPUT_SIZE
    img = image_pil.convert("RGB").resize((s, s))
    t = torch.tensor(np.array(img)).float().permute(2, 0, 1) / 255.0
    t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    t = t.unsqueeze(0).to(device)
    with torch.no_grad():
        preds = _birefnet(t)
    pred = preds[-1].sigmoid().cpu().squeeze().numpy()
    alpha = np.clip((pred - 0.5) * BIREFNET_MATTE_GAIN + 0.5, 0.0, 1.0)
    soft = (alpha * 255.0 + 0.5).astype(np.uint8)
    return cv2.resize(soft, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)

def birefnet_detect_object(image, raw_component_mask, obj_bbox, known_area, width, height):
    """BiRefNet foreground detection for one object, with expand-and-retry.

    A labeled object can be much smaller than the real visual occluder it belongs
    to (foliage growing from a labeled "vase", which the panoptic model drops to
    void), so a too-tight first crop truncates it. But a tiny confident nub can be
    non-empty and not touch the crop border, satisfying a naive "done" check before
    the crop was ever widened — so "done" requires covering a meaningful share of
    what is already known to be labeled there, not just ">0 px".

    Returns (cx1, cy1, cx2, cy2, fg_crop) in image coordinates.
    """
    ox1, oy1, ox2, oy2 = obj_bbox
    best_fg_crop, best_box, best_area = None, None, -1
    min_good_area = max(50, int(0.4 * known_area))
    pad_frac = 0.15
    first_box = None
    for _ in range(4):
        pad = max(10, int(pad_frac * max(ox2 - ox1, oy2 - oy1)))
        cx1 = max(0, ox1 - pad); cy1 = max(0, oy1 - pad)
        cx2 = min(width, ox2 + pad); cy2 = min(height, oy2 + pad)
        if first_box is None:
            first_box = (cx1, cy1, cx2, cy2)
        fg_crop = birefnet_fg_mask(image.crop((cx1, cy1, cx2, cy2)), (cy2 - cy1, cx2 - cx1))
        # fg_crop is a SOFT matte. Every gate below is a decision about "is the object
        # here", so each one reads the >127 core, never the raw non-zero count — the
        # anti-aliased skirt around a thin twig is many pixels wide and would inflate
        # `area` enough to trip the degenerate test on a perfectly good detection.
        fg_bin = fg_crop > 127
        area = int(fg_bin.sum())
        # A detection covering nearly the whole crop is a degenerate result — the
        # model found nothing distinctly salient and defaulted to "everything".
        # Never trust it, not even as a fallback: blanking a whole rectangle of
        # wall around a thin object is worse than detecting nothing there.
        crop_pixels = (cy2 - cy1) * (cx2 - cx1)
        degenerate = area > 0.85 * crop_pixels
        if area > best_area and not degenerate:
            best_fg_crop, best_box, best_area = fg_crop, (cx1, cy1, cx2, cy2), area
        touches_edge = area > 0 and (
            fg_bin[0, :].any() or fg_bin[-1, :].any() or
            fg_bin[:, 0].any() or fg_bin[:, -1].any()
        )
        full_frame = (cx1 == 0 and cy1 == 0 and cx2 == width and cy2 == height)
        if not degenerate and ((area >= min_good_area and not touches_edge) or full_frame):
            break
        pad_frac *= 2.2

    if best_box is None:
        # Every attempt was degenerate — detect nothing rather than guess.
        cx1, cy1, cx2, cy2 = first_box
        return cx1, cy1, cx2, cy2, np.zeros((cy2 - cy1, cx2 - cx1), np.uint8)

    # Sanity-check the WINNING attempt only. Two independent signals, either one
    # enough to reject and fall back to the panoptic model's own pixels:
    #   1. total area vs known_area — catches a near-total miss.
    #   2. spatial extent vs the object's own bbox — catches a PARTIAL miss that
    #      area alone cannot see (a lamp's shade found but its pole never detected
    #      is still ~full width, so area lands mid-range and slips past an
    #      area-only threshold). A genuine miss never reaches across the full
    #      extent already known to be there; a legitimately thinner-or-holed
    #      silhouette always does.
    cx1, cy1, cx2, cy2 = best_box
    area_ratio = best_area / max(1, known_area)
    fys, fxs = np.where(best_fg_crop > 127)
    known_h, known_w = max(1, oy2 - oy1), max(1, ox2 - ox1)
    if len(fys) == 0:
        extent_frac = 0.0
    else:
        extent_frac = min((fys.max() - fys.min()) / known_h, (fxs.max() - fxs.min()) / known_w)
    if area_ratio < 0.2 or extent_frac < 0.5:
        best_fg_crop = raw_component_mask[cy1:cy2, cx1:cx2].copy()
    return cx1, cy1, cx2, cy2, best_fg_crop

def find_ade20k_id(label_name, id2label):
    possible_names = TARGET_OBJECTS.get(label_name, [label_name])
    found_ids = []
    for id_key, model_label in id2label.items():
        try:
            curr_id = int(id_key)
        except ValueError:
            continue
        for key in possible_names:
            if key in model_label.lower():
                found_ids.append(curr_id)
    return found_ids

def get_label_from_id(id2label, valid_id):
    if valid_id in id2label: return id2label[valid_id]
    if str(valid_id) in id2label: return id2label[str(valid_id)]
    return f"Unknown ({valid_id})"

def generate_color_map(segmentation_map, found_objects):
    h, w = segmentation_map.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(300, 3))
    for obj in found_objects:
        obj_id = obj['id']
        mask = (segmentation_map == obj_id)
        color_idx = obj_id % 300
        color_map[mask] = colors[color_idx]
    return color_map

def save_panoptic_label_debug(image_cv, segmentation_map, segments_info, id2label, out_path):
    """Every panoptic segment, coloured, with its raw label text drawn at its
    centroid, over the original photo at reduced opacity. For diagnosing what the
    model actually called a region rather than inferring it from mask shape."""
    h, w = segmentation_map.shape
    np.random.seed(7)
    colors = np.random.randint(60, 255, size=(300, 3))
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for seg in segments_info:
        mask = (segmentation_map == seg["id"])
        if not mask.any():
            continue
        overlay[mask] = colors[seg["id"] % 300]
    blended = cv2.addWeighted(image_cv, 0.45, overlay, 0.55, 0)
    for seg in segments_info:
        mask = (segmentation_map == seg["id"])
        if not mask.any():
            continue
        label = get_label_from_id(id2label, seg["label_id"])
        ys, xs = np.where(mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        text = f"{label}#{seg['id']}"
        cv2.putText(blended, text, (cx - 4 * len(text), cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(blended, text, (cx - 4 * len(text), cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, blended)



# Mask post-processing helpers

def _label_tokens(label):
    """Tokenize an ADE20k label ('palm, palm tree') into a set of words."""
    return set(t for t in re.split(r'[^a-z]+', str(label).lower()) if t)

def label_matches(model_label, keywords):
    """Exact-token match so 'tree' does NOT match 'street'/'streetlight'."""
    toks = _label_tokens(model_label)
    return any(k in toks for k in keywords)


def fill_small_holes(mask_img, image_area, max_hole_frac=DEFAULT_HOLE_FILL_FRAC):
    """Fill enclosed holes ONLY if they are smaller than max_hole_frac of the
    image. Large enclosed holes are real objects sitting on/in the surface (a
    table on the floor, a window in a wall) and must stay cut out."""
    _, binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    h, w = padded.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    flood = padded.copy()
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood = flood[1:h - 1, 1:w - 1]
    holes = cv2.bitwise_not(flood)  # only fully-enclosed holes are 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats((holes > 0).astype(np.uint8), connectivity=8)
    max_hole_area = max(1.0, max_hole_frac * float(image_area))
    fill = np.zeros_like(binary)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] <= max_hole_area:
            fill[labels == i] = 255
    return binary | fill

def keep_significant_components(mask_img, image_area, frac_image=0.0005, min_abs=200):
    """Keep the largest connected component PLUS any other component whose area
    is >= max(min_abs, frac_image * image_area)."""
    binary = (mask_img > 127).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return mask_img
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return np.zeros_like(mask_img)
    thresh = max(min_abs, frac_image * float(image_area))
    largest_label = int(np.argmax(areas)) + 1
    keep = (labels == largest_label)
    for i, a in enumerate(areas, start=1):
        if a >= thresh:
            keep |= (labels == i)
    # Copy the ORIGINAL values through rather than stamping 255. Identical for a
    # binary input, but this now also runs on masks carrying BiRefNet's soft alpha,
    # and stamping would flatten that matte back to a hard edge.
    return np.where(keep, mask_img, 0).astype(mask_img.dtype)

def _guided_filter(I, p, radius, eps):
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
    return mean_a * I + mean_b

def refine_mask_edges(mask_img, image_bgr, radius_frac=0.004, eps=1e-3):
    """Snap mask boundaries onto true image edges (fixes blobby/distorted
    boundaries) via guided-filter matting, then re-threshold to binary."""
    h, w = mask_img.shape[:2]
    radius = max(3, int(radius_frac * max(h, w)))
    guide = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    src = mask_img.astype(np.float32) / 255.0
    q = _guided_filter(guide, src, radius, eps)
    return (q >= 0.5).astype(np.uint8) * 255

def sample_positive_points(of_bool, bbox, max_points=8):
    """Distance-transform peak + a spatial grid of interior points, so SAM is
    guided to cover the WHOLE region instead of one local blob."""
    of_uint8 = of_bool.astype(np.uint8) * 255
    pts = []
    dist = cv2.distanceTransform(of_uint8, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    pts.append([int(max_loc[0]), int(max_loc[1])])

    x0, y0, x1, y1 = bbox
    gx = gy = 3
    for iy in range(gy):
        for ix in range(gx):
            cx0 = x0 + (x1 - x0) * ix // gx
            cx1 = x0 + (x1 - x0) * (ix + 1) // gx
            cy0 = y0 + (y1 - y0) * iy // gy
            cy1 = y0 + (y1 - y0) * (iy + 1) // gy
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            cell = of_bool[cy0:cy1, cx0:cx1]
            if int(cell.sum()) < 50:
                continue
            ys, xs = np.where(cell)
            mid = len(xs) // 2
            pts.append([int(cx0 + xs[mid]), int(cy0 + ys[mid])])

    uniq, seen = [], set()
    for px, py in pts:
        if (px, py) not in seen:
            seen.add((px, py))
            uniq.append([px, py])
    return uniq[:max_points]

def mask_to_sam_logits(of_bool, size=256, val=8.0):
    """Encode a binary mask as SAM low-res mask_input logits (fg=+val, bg=-val)."""
    small = cv2.resize(of_bool.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    logits = (small * 2.0 - 1.0) * val
    return logits[None, :, :].astype(np.float32)

def _sam_predict_safe(predictor, points, labels, box, mask_input):
    """Call SAM with graceful degradation if a kwarg combination is rejected."""
    pc = np.array(points) if points else None
    pl = np.array(labels) if labels else None
    try:
        return predictor.predict(point_coords=pc, point_labels=pl, box=box, mask_input=mask_input, multimask_output=True, hq_token_only=True)
    except Exception:
        try:
            return predictor.predict(point_coords=pc, point_labels=pl, box=box, multimask_output=True, hq_token_only=True)
        except Exception:
            return predictor.predict(box=box, multimask_output=True, hq_token_only=True)

def _best_iou_index(masks, of_bool):
    best_idx, best_iou = 0, -1.0
    for i, m in enumerate(masks):
        inter = np.logical_and(m, of_bool).sum()
        union = np.logical_or(m, of_bool).sum()
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou, best_idx = iou, i
    return best_idx

def refine_with_sam(predictor, of_bool, bbox, neg_points, image_shape, shrink_guard=0.7, constrain_frac=0.02):
    """SAM boundary refinement seeded by OneFormer (mask_input + multi-point +
    box prompt). The SAM result is constrained to a dilated OneFormer region so
    it cannot bleed into neighbours, and falls back to OneFormer if SAM collapses
    (shrink guard) — so the output is never drastically smaller than OneFormer."""
    H, W = image_shape
    of_area = int(of_bool.sum())
    if of_area == 0:
        return of_bool.astype(np.uint8) * 255

    pos_points = sample_positive_points(of_bool, bbox, max_points=8)
    if not pos_points:
        return of_bool.astype(np.uint8) * 255

    points = pos_points + list(neg_points)
    labels = [1] * len(pos_points) + [0] * len(neg_points)

    pad = int(0.02 * max(H, W))
    box = np.array([
        max(0, bbox[0] - pad), max(0, bbox[1] - pad),
        min(W - 1, bbox[2] + pad), min(H - 1, bbox[3] + pad)
    ])

    try:
        masks, _, _ = _sam_predict_safe(predictor, points, labels, box, mask_to_sam_logits(of_bool))
    except Exception as e:
        print(f"⚠ [WARN] SAM refine failed ({e}); using OneFormer mask.")
        return of_bool.astype(np.uint8) * 255

    sam_bool = masks[_best_iou_index(masks, of_bool)].astype(bool)

    k = max(3, int(constrain_frac * max(H, W)))
    of_dilated = cv2.dilate(of_bool.astype(np.uint8) * 255, np.ones((k, k), np.uint8)) > 127
    constrained = np.logical_and(sam_bool, of_dilated)

    if int(constrained.sum()) < shrink_guard * of_area:
        refined = of_bool  # SAM collapsed -> trust OneFormer's full extent
    else:
        refined = constrained
    return refined.astype(np.uint8) * 255

def build_unified_occluder_mask(predictor, occluder_segments, segmentation_map,
                               image_pil, id2label, room_id=None):
    """ONE binary occluder mask for the whole image, with strict model ownership.

    Replaces the three overlapping unions this pipeline used to carry (raw OneFormer,
    dilated SAM blob, crisp BiRefNet) and the per-surface choice between them. There is
    now a single mask, subtracted once, so the final silhouette of every occluder is by
    construction the silhouette this function produced — nothing downstream can fatten
    it. That is the whole point of fill-once/cut-once: measured on room 6a717a84, the old
    ordering left our hole matching the dilated SAM blob (IoU 0.641) better than it
    matched BiRefNet's matte (0.611), because the blob's damage was applied before the
    refill and the refill could not reach past it.

    NO DILATION anywhere. The dilated blob existed to close inter-leaf gaps so surface
    texture would not bleed through onto whatever sat behind an object. Under a complete
    fill that reason is gone: the gaps between leaves are genuinely surface, the fill
    restores them, and only the leaf silhouettes are cut.

    Returns (union_uint8_or_None, stats).
    """
    H, W = segmentation_map.shape[:2]
    if not occluder_segments:
        return None, {}

    fine, solid = [], []
    for occ in occluder_segments:
        lbl = get_label_from_id(id2label, occ["label_id"])
        (fine if label_matches(lbl, FINE_OCCLUDER_LABELS) else solid).append(occ)

    union = np.zeros((H, W), dtype=np.uint8)
    st = {"fine_objects": 0, "fine_fallback": 0, "solid_objects": 0,
          "fine_segments": len(fine), "solid_segments": len(solid)}

    # ---- FINE objects: BiRefNet only. SAM never touches these. ----
    if fine:
        load_birefnet_if_needed()
    for occ in fine:
        seg_bool = (segmentation_map == occ["segment_id"])
        if int(seg_bool.sum()) == 0:
            continue
        # Split by connected component: the panoptic model merges instances of one class
        # under a single id (verified: two separate dried-branch arrangements on opposite
        # walls shared one "plant" id), and a crop spanning both frames neither.
        num_cc, labels_cc = cv2.connectedComponents(seg_bool.astype(np.uint8))
        for cc in range(1, num_cc):
            comp = (labels_cc == cc)
            area = int(comp.sum())
            if area < OCCLUDER_MIN_PX:
                continue
            ys, xs = np.where(comp)
            obj_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            raw = comp.astype(np.uint8) * 255
            try:
                cx1, cy1, cx2, cy2, fg = birefnet_detect_object(
                    image_pil, raw, obj_bbox, area, W, H)
            except Exception as e:
                print(f"⚠ [WARN] BiRefNet failed on a fine occluder ({e}); "
                      f"using OneFormer pixels for it.")
                union = np.maximum(union, raw)
                st["fine_fallback"] += 1
                continue
            # Threshold the matte HERE. birefnet_detect_object returns a soft alpha
            # because a sub-pixel-accurate matte places the boundary better than a
            # coarse model's binary does — but the geometry the pipeline carries stays
            # binary, and the only anti-aliasing happens at render resolution.
            cut = (fg >= BIREFNET_CUT_LEVEL).astype(np.uint8) * 255
            union[cy1:cy2, cx1:cx2] = np.maximum(union[cy1:cy2, cx1:cx2], cut)
            st["fine_objects"] += 1
            if DEBUG_SEG and room_id:
                try:
                    cv2.imwrite(os.path.join(_DEBUG_MASK_DIR,
                        f"birefnet_input_occluder_{room_id}_{occ['segment_id']}_{cc}.png"),
                        cv2.cvtColor(np.array(image_pil.crop((cx1, cy1, cx2, cy2))),
                                    cv2.COLOR_RGB2BGR))
                except Exception:
                    pass

    # ---- SOLID objects: OneFormer's own label, sharpened by SAM. BiRefNet never
    # touches these. SAM REFINES rather than segments — prompted freely it merges
    # neighbours or picks the wrong object, but constrained to an existing label
    # (mask_input + dilated-label clamp + shrink guard) it is reliable.
    for occ in solid:
        of_bool = (segmentation_map == occ["segment_id"])
        if int(of_bool.sum()) == 0:
            continue
        bbox = occ["bbox"]
        pad = int(0.01 * max(H, W))
        box = np.array([
            max(0, bbox[0] - pad), max(0, bbox[1] - pad),
            min(W - 1, bbox[2] + pad), min(H - 1, bbox[3] + pad)
        ])
        pos_points = sample_positive_points(of_bool, bbox, max_points=4)
        try:
            masks, _, _ = _sam_predict_safe(predictor, pos_points, [1] * len(pos_points),
                                            box, mask_to_sam_logits(of_bool))
            sam_bool = masks[_best_iou_index(masks, of_bool)].astype(bool)
            k = max(3, int(0.015 * max(H, W)))
            of_dilated = cv2.dilate(of_bool.astype(np.uint8) * 255,
                                    np.ones((k, k), np.uint8)) > 127
            constrained = np.logical_and(sam_bool, of_dilated)
            # shrink guard: SAM collapsing must never shrink a solid object's cutout
            refined = (np.logical_or(constrained, of_bool)
                       if int(constrained.sum()) < 0.5 * int(of_bool.sum()) else constrained)
        except Exception as e:
            print(f"⚠ [WARN] SAM refine failed on a solid occluder ({e}); "
                  f"using OneFormer mask.")
            refined = of_bool
        union = np.maximum(union, refined.astype(np.uint8) * 255)
        st["solid_objects"] += 1

    print(f"➡ [INFO] Occluders: {st['fine_objects']} fine object(s) via BiRefNet "
          f"({st['fine_fallback']} fell back), {st['solid_objects']} solid via SAM; "
          f"from {len(fine)} fine + {len(solid)} solid segment(s)")
    return (union if union.any() else None), st

def find_enclosed_holes(mask_img):
    """Just the fully-enclosed interior holes of mask_img — pixels that are 0 but
    unreachable from the image border without crossing a 255 pixel."""
    _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    padded = cv2.copyMakeBorder(binary_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    h, w = padded.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    flood = padded.copy()
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood = flood[1:h - 1, 1:w - 1]
    return cv2.bitwise_not(flood)

def occluders_in_front_of(occluder_union, surface_uint8, labelled_bool=None,
                          ring_frac=0.012, share=0.25):
    """The occluder components that actually sit IN FRONT OF this surface.

    Only these count towards the surface's extent during the heal. Plain overlap cannot
    be the test: an occluder in front of a wall is labelled plant/lamp, so OneFormer
    already excluded it from the wall mask and the overlap is zero by construction. Nor
    can plain proximity be the test — a wall spans most of a photo, so nearly every
    object in the room is "near" it. What discriminates is what SURROUNDS the object: if
    a quarter of the confidently-labelled things ringing it are this surface, it is in
    front of this surface. A plant standing in a window is ringed by window, so the
    wall's fill never reaches through it.

    Two details that are easy to get wrong and both matter:
      * VOID ABSTAINS. The ring is measured only against pixels OneFormer labelled.
        Counting void in the denominator makes this test fail on exactly the objects it
        exists to serve — a plant is separated from the wall by the very unlabelled halo
        the heal is meant to recover, so a void-inclusive ring reads as 0% surface and
        the object is rejected.
      * The ring must be WIDE enough to cross that halo. Measured on room 6a717a84 the
        halo ran to 28px at p90 and 54px at worst on a 3840px frame, so the reach is
        1.2% of the long side (~46px there) rather than a few pixels.
    """
    if occluder_union is None:
        return np.zeros_like(surface_uint8)
    h, w = surface_uint8.shape[:2]
    # ring_frac is the REACH in pixels, so the kernel is 2r+1 — an (r,r) kernel only
    # reaches r/2 and would leave the ring stranded inside the halo it must cross.
    r = max(3, int(ring_frac * max(h, w)))
    ker = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    surf = surface_uint8 > 127
    known = labelled_bool if labelled_bool is not None else np.ones((h, w), bool)
    out = np.zeros_like(surface_uint8)
    num, labels = cv2.connectedComponents((occluder_union > 127).astype(np.uint8))
    for i in range(1, num):
        comp = (labels == i)
        ring = cv2.dilate(comp.astype(np.uint8), ker).astype(bool) & ~comp
        n_conf = int((ring & known).sum())
        if n_conf and int((ring & surf).sum()) / float(n_conf) >= share:
            out[comp] = 255
    return out

def heal_surface(surface_uint8, occ_front_uint8, structural_uint8, segmentation_map,
                 occluder_seg_ids, image_area, void_fill_max_frac=VOID_FILL_MAX_FRAC,
                 void_halo_max_frac=VOID_HALO_MAX_FRAC):
    """FILL ONCE: complete interior fill of the surface, protecting real openings.

    The occluders in front of the surface are part of the flood BARRIER while the extent
    is established, then cut away afterwards. That is what recovers the void halo:
    OneFormer will not label the fuzzy region around a vase or a dried branch — it
    labels a thin confident core and drops the rest to void — and measured on room
    6a717a84 against prod's fd3cf46c (same photo) that unlabelled envelope was 52% of
    the surface prod keeps and we deleted. Trapped between the occluder and the surface,
    it is an enclosed hole of the barrier, so a single fill recovers all of it. No
    footprint refill, no halo growth, no distance caps.

    A hole is only filled when we know what it is:
      * VOID touching an occluder -> that object's unlabelled fringe. Fill, generously.
      * VOID touching nothing      -> an unlabelled real object. Fill only if small.
      * an OCCLUDER segment    -> we cut it precisely one step later. Fill.
      * anything else          -> LEFT EXCLUDED.
    That last branch is the safe default and it is what protects windows, doors and
    arches without needing a label whitelist: an arch shows the next room's floor and
    wall, so the hole's dominant label is a different surface class and it survives
    untouched. It also protects objects this pipeline does not model at all — a mirror
    or a painting is not in OCCLUDER_OBJECTS, so nothing would ever cut it back out, and
    a size-based rule would happily paint wallpaper straight over it.
    """
    # The flood BARRIER is everything OneFormer was confident about: this surface, the
    # occluders in front of it, and all other labelled territory. Only VOID is floodable,
    # so an enclosed hole is void fully ringed by things we understand.
    #
    # Including `structural` here is not cosmetic — it is what makes the common case
    # work. A plant standing on the floor has its unlabelled halo open to the floor, and
    # a barrier of surface+occluder alone leaves that halo reachable from the frame edge,
    # so it is not "enclosed" and never gets recovered. The floor is a definite boundary
    # of the wall; treating it as one closes the halo. Same for a vase whose halo runs
    # down onto the table it stands on.
    barrier = np.maximum(np.maximum(surface_uint8, occ_front_uint8), structural_uint8)
    holes = find_enclosed_holes(barrier)
    # The EXTENT, though, is only ever surface + occluder. Structural territory bounds
    # the fill; it is never claimed by it.
    out = np.maximum(surface_uint8, occ_front_uint8)
    st = {"filled": 0, "kept_structural": 0, "kept_big_void": 0}
    if not holes.any():
        return out, st
    num, labels, cc_stats, _ = cv2.connectedComponentsWithStats(
        (holes > 0).astype(np.uint8), connectivity=8)
    max_void = max(200.0, void_fill_max_frac * float(image_area))
    max_halo = max(max_void, void_halo_max_frac * float(image_area))
    # dilate by one so a hole sitting flush against the occluder registers as touching
    occ_near = cv2.dilate(occ_front_uint8, np.ones((3, 3), np.uint8)) > 0
    for i in range(1, num):
        comp = (labels == i)
        area = int(cc_stats[i, cv2.CC_STAT_AREA])
        vals, counts = np.unique(segmentation_map[comp], return_counts=True)
        dominant = int(vals[int(np.argmax(counts))])
        if dominant == 0:                       # VOID — OneFormer labelled nothing
            cap = max_halo if bool((comp & occ_near).any()) else max_void
            if area <= cap:
                out[comp] = 255
                st["filled"] += 1
            else:
                st["kept_big_void"] += 1
        elif dominant in occluder_seg_ids:      # will be cut precisely in one step
            out[comp] = 255
            st["filled"] += 1
        else:                                   # opening / other surface / unknown object
            st["kept_structural"] += 1
    return out, st

def refine_boundary_scoped(mask_uint8, image_bgr, occluder_union, guard_frac=0.004):
    """Guided-filter edge snap, restricted to ARCHITECTURAL boundary.

    Kept on purpose: the new boundary generation has not yet been verified to produce
    clean edges unaided, and until it has, a lightweight edge-alignment step still earns
    its place. It is now scoped, though. The single subtraction owns every occluder
    silhouette, and a guided filter re-thickens those: it follows colour contrast only,
    so a brown branch against tan fabric or a pale lamp pole against a light wall bleeds
    the boundary well past the object. Excluding the occluder neighbourhood leaves the
    filter doing only what it is good at — snapping the wall/ceiling/floor/corner
    boundary onto a real image edge.
    """
    refined = refine_mask_edges(mask_uint8, image_bgr)
    if occluder_union is None:
        return refined
    k = max(3, int(guard_frac * max(mask_uint8.shape[:2])))
    near = cv2.dilate(occluder_union, np.ones((k, k), np.uint8)) > 0
    return np.where(near, mask_uint8, refined).astype(np.uint8)

def validate_heal(final_uint8, source_uint8, forbidden_bool, image_area,
                  max_growth_frac=MAX_HEAL_GROWTH_FRAC, forbid_tol_frac=0.002):
    """Sanity gate on the healed result. Returns (ok, reason).

    A complete fill is the most powerful operation in this pipeline and the one with the
    worst failure mode — wallpaper painted across a ceiling, with nothing downstream to
    catch it. Two independent checks, either one enough to reject and fall back to the
    unhealed mask: how much the mask grew beyond OneFormer's own surface, and whether it
    claimed pixels OneFormer confidently assigned to a different surface.
    """
    src = int((source_uint8 > 127).sum())
    fin = int((final_uint8 > 127).sum())
    if src == 0:
        return True, "empty source"
    growth = (fin - src) / float(src)
    if growth > max_growth_frac:
        return False, f"grew {growth * 100:.0f}% over OneFormer (cap {max_growth_frac * 100:.0f}%)"
    bad = int(((final_uint8 > 127) & forbidden_bool).sum())
    if bad > forbid_tol_frac * float(image_area):
        return False, f"claimed {bad}px belonging to another surface"
    return True, f"grew {growth * 100:.0f}%"

def postprocess_mask(mask_uint8, image_bgr, image_area, do_edge_refine=True,
                    max_hole_frac=DEFAULT_HOLE_FILL_FRAC, do_prune=True):
    """Shared cleanup: bridge small gaps, keep ALL significant components,
    fill only SMALL enclosed holes (so objects on the surface stay cut out),
    and snap edges to the image.

    do_prune=False defers the component pruning to AFTER the heal. Pruning first is a
    quiet way to lose wall: a sliver of surface visible between two objects can fall
    under the significance floor (3961 px on a 3840x2063 frame) and be dropped before
    the heal ever gets the chance to reconnect it to the main region. The final prune
    after the cut applies the same thresholds, so nothing noisy survives either way."""
    h, w = mask_uint8.shape[:2]
    k = max(3, int(0.004 * max(h, w)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    if do_prune:
        mask_uint8 = keep_significant_components(mask_uint8, image_area)
    mask_uint8 = fill_small_holes(mask_uint8, image_area, max_hole_frac)
    if do_edge_refine:
        mask_uint8 = refine_mask_edges(mask_uint8, image_bgr)
    return mask_uint8

def render_canvas_size(w, h, target=RENDER_MAX_DIM):
    """The canvas the renderers will actually composite on, so our resize is theirs.

    Mirrors app.py: upscale_image() leaves an image alone once its long side reaches
    4000 and otherwise scales the long side to 4500, and the MAX_DIM cap trims anything
    above 4500. Matching it means utils/wall.py's own
    `cv2.resize(mask, (W, H), INTER_NEAREST)` becomes a no-op instead of the 1.2x-5.03x
    magnification that created the staircase in the first place. An exact match is not
    critical — nearest-resampling an already-smooth mask by a percent or two is
    harmless — but being at 894 px when the canvas is 4500 px is not.
    """
    m = max(w, h)
    if 4000 <= m <= target:
        return w, h
    sc = float(target) / float(m)
    return max(1, int(w * sc)), max(1, int(h * sc))

def rasterise_at_render_res(mask_uint8, target=RENDER_MAX_DIM, aa_px=AA_RENDER_PX):
    """Resample the boundary as a distance FIELD, then rasterise at render resolution.

    This is the fix for both complaints at once — the staircase AND the blur — and it
    replaces the mask-space feather entirely.

    Upscaling a BINARY mask magnifies each boundary pixel into a k*k block: measured
    across the test rooms, k ran 1.17x to 5.03x and the median block reached 4 px.
    Feathering it in mask space does not help; it only spreads that block into a banded
    gradient, which is exactly the "blurred, just dulled" result of v2.6 (a 2.5 mask-px
    ramp becomes 25 render px at 5.03x, quantised into five visible steps).

    A signed distance field has neither problem, because it is a smooth continuous
    function rather than a step: interpolating it places the zero-crossing sub-pixel
    accurately between the low-resolution samples, so re-thresholding at render
    resolution rasterises the contour the mask implied. Measured on the same masks, the
    median staircase block drops from 4.0 px to 1.0 px, with a hard edge and no blur.

    Anti-aliasing is applied ONCE, here, and its width is stated in RENDER pixels —
    ~1.5 px, i.e. genuinely sub-pixel rather than a visible soft band. This is the only
    place in the pipeline that produces intermediate alpha; everything upstream is
    binary, which is what stops soft bands compounding.

    Note this adds no information. It removes a rasterisation artifact; a boundary the
    models placed 3 px wrong is still 3 px wrong, it merely stops looking blocky.
    """
    binary = (mask_uint8 > 127).astype(np.uint8)
    h, w = binary.shape[:2]
    rw, rh = render_canvas_size(w, h, target)
    if not binary.any() or binary.all():
        return cv2.resize(mask_uint8, (rw, rh), interpolation=cv2.INTER_NEAREST)
    # distanceTransform measures to the nearest zero WITHIN the image, so a mask running
    # off the frame keeps full alpha at the border instead of fading out.
    d_in = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    sdf = d_in - d_out                      # >0 inside, <0 outside, 0 on the boundary
    sc = float(rw) / float(w)
    big = cv2.resize(sdf, (rw, rh), interpolation=cv2.INTER_CUBIC) * sc   # native px -> render px
    alpha = np.clip(0.5 + big / max(1e-6, float(aa_px)), 0.0, 1.0)
    return (alpha * 255.0 + 0.5).astype(np.uint8)

def process_scene_pipeline(image: Image.Image, room_id: str, filename: str, masks_folder: str, generated_folder: str, server_base_url: str):
    
    load_models_if_needed() # Ensure models are loaded

    width, height = image.size
    image_area = width * height

    # Run OneFormer
    inputs = processor(images=image, task_inputs=["panoptic"], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = segmenter(**inputs)

    # Panoptic post-processing
    panoptic_result = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]

    segmentation_map = panoptic_result["segmentation"].cpu().numpy()
    segments_info = panoptic_result["segments_info"]

    id2label = segmenter.config.id2label

    # Pre-compute the ADE20k id set for each target class once (was recomputed per-segment in the original loop).
    target_ids = {ul: set(find_ade20k_id(ul, id2label)) for ul in TARGET_OBJECTS.keys()}

    found_objects = []
    hotspots = []
    occluder_segments = []
    instance_counts = {}

    # Iterate through segments — collect target surfaces AND occluders.
    for segment in segments_info:
        segment_id = segment["id"]
        label_id = segment["label_id"]
        model_label = get_label_from_id(id2label, label_id)

        seg_bool = (segmentation_map == segment_id)
        seg_count = int(seg_bool.sum())
        if seg_count == 0:
            continue
        seg_area_ratio = seg_count / float(image_area)

        # Collect occluders for precise subtraction later.
        if label_matches(model_label, OCCLUDER_OBJECTS) and seg_count >= OCCLUDER_MIN_PX:
            rows_o, cols_o = np.where(seg_bool)
            occluder_segments.append({
                "segment_id": segment_id,
                "label_id": label_id,
                "bbox": [int(np.min(cols_o)), int(np.min(rows_o)), int(np.max(cols_o)), int(np.max(rows_o))],
                "label": model_label,
            })

        # Match target surfaces (wall/floor/curtain/rug/window/door).
        matched_user_label = None
        for user_label in TARGET_OBJECTS.keys():
            if label_id in target_ids[user_label]:
                matched_user_label = user_label
                break

        if not matched_user_label:
            continue

        # FILTER SMALL OBJECTS (< 0.5% of the image area)
        if seg_area_ratio < SMALL_OBJECT_MIN_AREA:
            continue

        rows, cols = np.where(seg_bool)

        # Calculate Bounding Box
        y_min, y_max = int(np.min(rows)), int(np.max(rows))
        x_min, x_max = int(np.min(cols)), int(np.max(cols))
        bbox = [x_min, y_min, x_max, y_max]

        # Use Distance Transform to find the thickest part of the mask for better tooltip placement
        object_mask_uint8 = seg_bool.astype(np.uint8) * 255
        dist_transform = cv2.distanceTransform(object_mask_uint8, cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
        cx, cy = max_loc # max_loc is (x, y)

        # Calculate Relative % Coordinates using the accurate cx, cy
        perc_x = round(cx / width, 4)
        perc_y = round(cy / height, 4)

        # Keep tooltips slightly inside the absolute edges (between 3% and 97%)
        perc_x = max(0.03, min(0.97, perc_x))
        perc_y = max(0.03, min(0.97, perc_y))

        instance_counts[matched_user_label] = instance_counts.get(matched_user_label, 0) + 1
        found_objects.append({"id": segment_id})

        hotspot_type = TYPE_MAPPING.get(matched_user_label, "unknown")
        sub_category_id = SUB_CATEGORY.get(matched_user_label, "unknown")
        display_label = "Rugs" if matched_user_label == "rug" else matched_user_label.capitalize()

        hotspots.append({
            "image_hotspots_id": str(uuid.uuid4()),
            "type": hotspot_type,
            "label": display_label,
            "x": perc_x,
            "y": perc_y,
            "sub_category_id": sub_category_id,
            "bbox": bbox,
            "segment_id": segment_id, # For cross-reference with SAM later
            "mask_image": "",
            "_seg_class": matched_user_label,  # internal only; removed before returning
        })

    # Generate Color Map
    color_map_np = generate_color_map(segmentation_map, found_objects)
    color_map_img = Image.fromarray(color_map_np)
    map_filename = f"map_{filename}"
    color_map_img.save(os.path.join(generated_folder, map_filename))

    # SAM SETUP
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_rgb_sam = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb_sam)

    if DEBUG_SEG:
        try:
            save_panoptic_label_debug(image_cv, segmentation_map, segments_info, id2label,
                os.path.join(_DEBUG_MASK_DIR, f"oneformer_labels_{room_id}.png"))
        except Exception as _dbg_e:
            print(f"⚠ [WARN] Panoptic label debug image failed ({_dbg_e}); skipping.")

    # -> ONE unified occluder mask for the whole image (steps 4-5). Strict model
    #    ownership inside: BiRefNet owns fine objects, SAM owns solid ones, never both.
    occluder_union, _occ_stats = None, {}
    try:
        occluder_union, _occ_stats = build_unified_occluder_mask(
            sam_predictor, occluder_segments, segmentation_map, image,
            id2label, room_id=room_id)
    except Exception as _be:
        print(f"⚠ [WARN] Occluder pass failed ({_be}); surfaces keep raw OneFormer pixels.")

    # Occluder territory as OneFormer labelled it — needed so the heal can tell an
    # occluder hole (fill, it gets cut precisely) from a structural one (leave alone).
    _occ_ids = {o["segment_id"] for o in occluder_segments}
    occ_labelled_bool = (np.isin(segmentation_map, list(_occ_ids)) if _occ_ids
                         else np.zeros(segmentation_map.shape, bool))

    if DEBUG_SEG and occluder_union is not None:
        try:
            cv2.imwrite(os.path.join(_DEBUG_MASK_DIR, f"occluders_{room_id}.png"), occluder_union)
        except Exception:
            pass

    # -> Generate the final mask for each hotspot.
    for hotspot in hotspots:
        segment_id = hotspot['segment_id']
        seg_class = hotspot['_seg_class']
        bbox = hotspot['bbox']
        of_bool = (segmentation_map == segment_id)
        of_uint8 = of_bool.astype(np.uint8) * 255

        # Negative points: smaller foreground objects whose centre lies inside
        # this object's bbox (same intent as the original negative-point logic).
        current_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        neg_points = []
        for other in hotspots:
            if other['image_hotspots_id'] == hotspot['image_hotspots_id']:
                continue
            ob = other['bbox']
            ocx = int(other['x'] * width)
            ocy = int(other['y'] * height)
            oarea = (ob[2] - ob[0]) * (ob[3] - ob[1])
            if oarea < current_area and bbox[0] <= ocx <= bbox[2] and bbox[1] <= ocy <= bbox[3]:
                neg_points.append([ocx, ocy])

        # STRUCTURAL territory: everything OneFormer confidently labelled as something
        # else that is NOT an occluder — other surfaces, openings, and objects this
        # pipeline does not model. The mask may never claim these. NOTE the sentinel:
        # OneFormer builds its map with torch.zeros and increments BEFORE assigning, so
        # VOID IS 0 and ids start at 1 — the opposite of EoMT (void -1, ids from 0).
        # Using ">= 0" would treat void as another object and eat every unlabelled fringe.
        # Occluders are deliberately excluded here: their pixels belong to the surface's
        # extent during the fill and are removed by the single cut instead.
        structural_bool = ((segmentation_map != 0) & (segmentation_map != segment_id)
                           & ~occ_labelled_bool)
        structural_uint8 = structural_bool.astype(np.uint8) * 255

        try:
            # ---- 1. semantic surface ----
            if seg_class == "wall":
                # Wall keeps raw OneFormer pixels: its panoptic assignment is already
                # pixel-accurate about what is not wall, and both SAM refinement and the
                # guided filter measurably degraded it (v2.1/v2.4).
                surface = of_uint8
            else:
                refined = refine_with_sam(sam_predictor, of_bool, bbox, neg_points, (height, width))
                # For large "stuff" surfaces OneFormer's extent is the floor: SAM may
                # crisp or extend edges but must never carve away coverage.
                surface = cv2.bitwise_or(refined, of_uint8) if seg_class in ONEFORMER_EXTENT_CLASSES else refined

            do_cut = seg_class in OCCLUDER_SUBTRACT_CLASSES and occluder_union is not None
            # Hole filling is the heal's job for the classes that get one, and the heal
            # classifies every hole by label first. An unclassified small-hole fill here
            # would be the one unsafe filler left in the pipeline.
            max_hole_frac = 0.0 if do_cut else HOLE_FILL_FRAC.get(seg_class, DEFAULT_HOLE_FILL_FRAC)
            surface = postprocess_mask(surface, image_cv, image_area,
                                       max_hole_frac=max_hole_frac, do_edge_refine=False,
                                       do_prune=not do_cut)
            surface = cv2.bitwise_and(surface, cv2.bitwise_not(structural_uint8))

            # ---- 2. HEAL: fill once ----
            if do_cut:
                occ_front = occluders_in_front_of(occluder_union, surface,
                                                  labelled_bool=(segmentation_map != 0))
                healed, _hst = heal_surface(surface, occ_front, structural_uint8,
                                            segmentation_map, _occ_ids, image_area)
                healed = cv2.bitwise_and(healed, cv2.bitwise_not(structural_uint8))
                if _hst["filled"] or _hst["kept_big_void"] or _hst["kept_structural"]:
                    print(f"   [HEAL] {seg_class}: filled {_hst['filled']} void pocket(s); "
                          f"refused {_hst['kept_big_void']} too-large-to-trust"
                          + (f" + {_hst['kept_structural']} unrecognised" if _hst['kept_structural'] else ""))
            else:
                healed = surface

            # ---- 3. architectural edge alignment (scoped; see refine_boundary_scoped) ----
            if seg_class in EDGE_REFINE_CLASSES:
                healed = refine_boundary_scoped(healed, image_cv, occluder_union)
                healed = cv2.bitwise_and(healed, cv2.bitwise_not(structural_uint8))

            # ---- 4. CUT ONCE ----
            candidate = (cv2.bitwise_and(healed, cv2.bitwise_not(occluder_union))
                         if do_cut else healed)
            candidate = keep_significant_components(candidate, image_area)

            # ---- 5. VALIDATE, fall back if the heal overreached ----
            ok, why = validate_heal(candidate, of_uint8, structural_bool, image_area)
            if ok:
                mask_uint8 = candidate
            else:
                print(f"⚠ [HEAL-REJECT] {seg_class}: {why}; falling back to unhealed mask.")
                fb = cv2.bitwise_and(of_uint8, cv2.bitwise_not(structural_uint8))
                if do_cut:
                    fb = cv2.bitwise_and(fb, cv2.bitwise_not(occluder_union))
                mask_uint8 = keep_significant_components(fb, image_area)
        except Exception as e:
            print(f"⚠ [WARN] Mask generation failed for {seg_class} ({e}); using OneFormer mask.")
            mask_uint8 = of_uint8

        # ---- 6. rasterise at render resolution, anti-aliased once, at the very end ----
        mask_uint8 = rasterise_at_render_res(mask_uint8)

        mask_img = Image.fromarray(mask_uint8)
        mask_filename = f"mask_{room_id}_{hotspot['image_hotspots_id']}.png"
        mask_img.save(os.path.join(masks_folder, mask_filename))

        if DEBUG_SEG:
            try:
                cv2.imwrite(os.path.join(_DEBUG_MASK_DIR, f"{seg_class}_{room_id}_{hotspot['image_hotspots_id']}.png"), mask_uint8)
            except Exception:
                pass

        hotspot['mask_image'] = f"{server_base_url}/masks/{mask_filename}"
        del hotspot['_seg_class']  # strip internal key before returning

    return {
        "status": "success",
        "room_category_image_id": room_id,
        "image_url": f"{server_base_url}/uploads/{filename}",
        "hotspots": hotspots,
        "found_objects": found_objects,
        "image_dims": {"width": width, "height": height},
        "map_image_url": f"{server_base_url}/outputs/{map_filename}"
    }