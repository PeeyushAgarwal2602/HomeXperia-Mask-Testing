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

# The subset of OCCLUDER_OBJECTS worth spending a BiRefNet pass on: objects with
# genuinely fine structure, where a blob silhouette is visibly wrong. Solid
# furniture is deliberately excluded — a blob IS the correct silhouette for a
# sofa, so a BiRefNet call there costs a crop of inference for no visible gain.
# label_matches compares SINGLE tokens of the model label, so a multi-word entry
# such as "floor lamp" can never match anything — those were dead weight and are
# gone. Checked against the ADE20K-150 vocabulary OneFormer actually emits:
#   "plant, flora, plant life" / "tree" / "flower" / "palm, palm tree" /
#   "pot, flowerpot" / "vase" / "lamp" / "light, light source" /
#   "chandelier, pendant, pendent" / "sconce" / "fan" / "sculpture"
# every token below hits one of those, so the set does not need widening for
# OneFormer — the earlier suspicion about vocabulary drift was wrong.
BIREFNET_OCCLUDER_LABELS = {
    "plant", "flora", "tree", "flower", "palm", "branch",
    "pot", "flowerpot", "vase", "sculpture",
    "lamp", "light", "floor lamp", "table lamp", "chandelier", "pendant", "pendent", "sconce", "fan", "ceiling fan",
}

# Surfaces whose extent OneFormer is trusted to define (these are large "stuff"
# regions that get fragmented by occluders; SAM is only allowed to refine edges
# and ADD detail, never to shrink them below OneFormer's coverage).
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
MASK_FEATHER_PX = 2.5

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

def isolate_largest_blob(mask_img):
    _, binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find all external contours

    if not contours: return mask_img # Return original if somehow empty

    largest_contour = max(contours, key=cv2.contourArea) # Identify the contour with the maximum area
    clean_mask = np.zeros_like(mask_img) # Create a fresh black mask of the same dimensions

    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED) # Draw only the largest contour filled with white
    return clean_mask

def fill_internal_holes(mask_img):
    _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY) # Ensure the mask is strictly binary (0 or 255)
    padded = cv2.copyMakeBorder(binary_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0) # Pad the image with a 1-pixel black border.

    h, w = padded.shape[:2]
    flood_mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(padded, flood_mask, (0,0), 255)

    im_floodfill = padded[1:h-1, 1:w-1] # Remove the padding to restore original dimensions
    im_floodfill_inv = cv2.bitwise_not(im_floodfill) # Invert the flood-filled image. Now, only the enclosed holes are white.

    filled_mask = binary_mask | im_floodfill_inv # Bitwise OR combines the original mask with the isolated holes
    return isolate_largest_blob(filled_mask)

# Mask post-processing helpers

def _label_tokens(label):
    """Tokenize an ADE20k label ('palm, palm tree') into a set of words."""
    return set(t for t in re.split(r'[^a-z]+', str(label).lower()) if t)

def label_matches(model_label, keywords):
    """Exact-token match so 'tree' does NOT match 'street'/'streetlight'."""
    toks = _label_tokens(model_label)
    return any(k in toks for k in keywords)

def fill_enclosed_holes(mask_img):
    """Fill only fully-enclosed interior holes."""
    _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    padded = cv2.copyMakeBorder(binary_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    h, w = padded.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    flood = padded.copy()
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood = flood[1:h - 1, 1:w - 1]
    holes = cv2.bitwise_not(flood)
    return binary_mask | holes

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

def build_occluder_union_of(occluder_segments, segmentation_map):
    """Raw OneFormer-pixel union (no SAM expansion).
    OneFormer assigns inter-leaf gap pixels to wall, so this mask correctly
    excludes them — the SAM blob often fills those gaps, which would subtract
    real wall pixels and leave visible beige around leaves."""
    H, W = segmentation_map.shape[:2]
    if not occluder_segments:
        return None
    union = np.zeros((H, W), dtype=np.uint8)
    for occ in occluder_segments:
        of_bool = (segmentation_map == occ["segment_id"])
        if int(of_bool.sum()) == 0:
            continue
        union = cv2.bitwise_or(union, of_bool.astype(np.uint8) * 255)
    return union

def build_occluder_union(predictor, occluder_segments, segmentation_map, image_bgr):
    """Segment each occluder precisely (per-instance, box prompted SAM) and return
    the union of their refined masks. Dilated by the caller and subtracted from
    CURTAIN, where filling the inter-leaf gaps is wanted so curtain texture does
    not bleed through onto whatever sits behind the leaves."""
    H, W = segmentation_map.shape[:2]
    if not occluder_segments:
        return None

    union = np.zeros((H, W), dtype=np.uint8)
    for occ in occluder_segments:
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
        labels = [1] * len(pos_points)
        try:
            masks, _, _ = _sam_predict_safe(predictor, pos_points, labels, box, mask_to_sam_logits(of_bool))
            sam_bool = masks[_best_iou_index(masks, of_bool)].astype(bool)
            k = max(3, int(0.015 * max(H, W)))
            of_dilated = cv2.dilate(of_bool.astype(np.uint8) * 255, np.ones((k, k), np.uint8)) > 127
            constrained = np.logical_and(sam_bool, of_dilated)
            if int(constrained.sum()) < 0.5 * int(of_bool.sum()):
                refined = np.logical_or(constrained, of_bool)
            else:
                refined = constrained
        except Exception as e:
            print(f"⚠ [WARN] Occluder SAM refine failed ({e}); using OneFormer mask.")
            refined = of_bool

        ref_uint8 = refined.astype(np.uint8) * 255
        ref_uint8 = refine_mask_edges(ref_uint8, image_bgr, radius_frac=0.002)  # crisp leaf edges
        union = cv2.bitwise_or(union, ref_uint8)

    return union

def build_occluder_union_birefnet(occluder_segments, segmentation_map, image_pil, id2label, room_id=None):
    """Crisp union of the THIN/SPARSE occluders only, via BiRefNet — no dilation.

    Returns (crisp_union, footprints): `footprints` is the list of RAW OneFormer
    footprints of the components BiRefNet actually resolved.

    The caller needs them because SUBTRACTING THE CRISP UNION ALONE IS A NO-OP.
    OneFormer has already excluded the whole plant from the wall mask as one
    coarse blob, so removing a thin twig silhouette from an already-missing region
    changes nothing. Measured on rooms 4448f78f, c77a842c and f4eec840: 100% of
    BiRefNet's cut pixels were ALREADY absent from the wall mask — the subtraction
    altered 0% of its own area, so BiRefNet's precision never reached the render.

    To make it visible the caller must FILL the coarse footprint back to wall and
    THEN carve the crisp silhouette out of it. That fill is the entire mechanism by
    which BiRefNet's quality becomes visible, and it is what the older EoMT
    production pipeline was doing (through a bbox fill) when its leaf cutouts
    looked right. Filling each occluder's OWN footprint instead of its bounding box
    keeps the benefit without the bbox fill's habit of swallowing unrelated objects.

    Two deliberate restrictions:

      * Only BIREFNET_OCCLUDER_LABELS classes. Solid furniture is already excluded
        from a wall mask by OneFormer's own labelling, so subtracting a SAM blob
        for it again buys nothing and risks a silhouette fatter than the object —
        which is exactly the beige halo the SAM path used to leave around leaves.

      * No dilation, ever. Dilating a thin-object silhouette closes the gaps
        between twigs, and those gaps are where the wallpaper must stay visible.

    Split by connected component, because the panoptic model merges instances of
    one class under a single segment id (verified: two separate dried-branch
    arrangements on opposite sides of a room shared one "plant" id), and a single
    crop spanning both would frame neither properly.
    """
    H, W = segmentation_map.shape[:2]
    if not occluder_segments:
        return None, []

    thin = [o for o in occluder_segments
            if label_matches(get_label_from_id(id2label, o["label_id"]), BIREFNET_OCCLUDER_LABELS)]
    if not thin:
        print("➡ [INFO] No thin/sparse occluders in this scene; wall keeps raw OneFormer pixels.")
        return None, []

    load_birefnet_if_needed()
    union = np.zeros((H, W), dtype=np.uint8)
    footprints = []
    n_done = 0
    for occ in thin:
        seg_bool = (segmentation_map == occ["segment_id"])
        if int(seg_bool.sum()) == 0:
            continue
        num_cc, labels_cc = cv2.connectedComponents(seg_bool.astype(np.uint8))
        for cc in range(1, num_cc):
            comp = (labels_cc == cc)
            area = int(comp.sum())
            if area < 100:  # stray/noise pixels
                continue
            ys, xs = np.where(comp)
            obj_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            raw = comp.astype(np.uint8) * 255
            try:
                cx1, cy1, cx2, cy2, fg = birefnet_detect_object(
                    image_pil, raw, obj_bbox, area, W, H)
            except Exception as e:
                print(f"⚠ [WARN] BiRefNet occluder detect failed ({e}); using OneFormer pixels.")
                union = np.maximum(union, raw)
                continue
            union[cy1:cy2, cx1:cx2] = np.maximum(union[cy1:cy2, cx1:cx2], fg)
            # Offer the footprint for refilling unless the detection is so weak
            # that refilling would paint surface over the object. The ONLY
            # dangerous case is a near-empty silhouette: if BiRefNet came back a
            # SUPERSET of the footprint (it found foliage OneFormer dropped to
            # void — the common case) then footprint is a subset of fg and the
            # refill is arithmetically a no-op, so it needs no gate at all.
            # An earlier "< 0.92 * area" test had this backwards and rejected 8 of
            # 12 perfectly good refills across the v2.1 rooms, which is why the
            # lamp in 16239afa kept its ~6px border.
            # >127 core on both sides — fg is a soft matte now, so bitwise_and would
            # count anti-aliased skirt pixels as detection.
            fg_in_fp = int(np.count_nonzero((fg > 127) & (raw[cy1:cy2, cx1:cx2] > 127)))
            if fg_in_fp >= max(50, int(0.25 * area)):
                footprints.append(raw)
            n_done += 1
            if DEBUG_SEG and room_id:
                try:
                    cv2.imwrite(os.path.join(_DEBUG_MASK_DIR,
                        f"birefnet_input_occluder_{room_id}_{occ['segment_id']}_{cc}.png"),
                        cv2.cvtColor(np.array(image_pil.crop((cx1, cy1, cx2, cy2))), cv2.COLOR_RGB2BGR))
                except Exception:
                    pass

    print(f"➡ [INFO] BiRefNet refined {n_done} thin occluder component(s), "
          f"{len(footprints)} tight enough to refill; "
          f"from {len(thin)} of {len(occluder_segments)} occluder segment(s)")
    return (union if union.any() else None), footprints

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

def reveal_occluder_holes(occluder_mask, surface_extent_mask, min_hole_px=20):
    """Recover the SEE-THROUGH gaps inside an occluder's silhouette.

    A rope loop, or the gap between two twigs, is an enclosed hole in the
    occluder's shape through which the surface behind is genuinely visible. These
    are most of what makes the production masks look crisp: measured on the same
    room, prod carries 30 and 54 enclosed holes where ours carried 4 and 3.

    Only revealed when the hole's horizontal span falls almost entirely within
    columns the surface actually reaches — otherwise a sparse object straddling
    the surface's edge would have gaps opening onto a window revealed as surface.
    surface_extent_mask must be the surface's own mask from BEFORE any occluder
    subtraction.
    """
    holes = find_enclosed_holes(occluder_mask)
    if not holes.any():
        return holes
    col_reach = surface_extent_mask.any(axis=0)
    num, labels = cv2.connectedComponents(holes)
    revealed = np.zeros_like(holes)
    for i in range(1, num):
        blob = labels == i
        # SIZE GATE. BiRefNet's silhouettes are fine enough that a hardcoded 20px
        # floor let through a swarm of noise specks: measured over this run's wall
        # and curtain masks, 179 + 101 tiny islands appeared where the previous
        # build had ZERO, and the median small component was 12px with p75 at
        # 135px. After the renderer's ~3.3x nearest upscale each of those paints a
        # visible block, which reads as pixelated edges. Genuine see-through gaps
        # (a rope loop, a gap between twigs) are far larger and survive this.
        if int(blob.sum()) < min_hole_px:
            continue
        cols = np.unique(np.where(blob)[1])
        if col_reach[cols].mean() >= 0.9:
            revealed[blob] = 255
    return revealed

VOID_HALO_DIST_FRAC = 0.015  # how far a void halo may reach from its occluder
VOID_HALO_MAX_GROWTH = 2.0   # halo may not exceed this multiple of the footprint's area

def grow_footprint_into_void(fp, mask_uint8, void_bool,
                             dist_frac=VOID_HALO_DIST_FRAC,
                             max_growth=VOID_HALO_MAX_GROWTH):
    """Grow an occluder's footprint outward into the adjacent VOID halo.

    This is the single biggest reason the production pipeline's cutouts look crisper
    than ours. OneFormer will not label the fuzzy region around a vase or a dried
    branch — it labels a thin confident core and drops the rest to void. Measured on
    room 6a717a84 against prod's fd3cf46c (same photo, 3840x2063): 52% of the curtain
    prod keeps and we delete is covered by NO occluder mask at all. It is that
    unlabelled envelope — 8px thick at the median, 28px at p90, 54px at worst.

    Prod recovers it by filling the surface's whole bounding box and then carving the
    BiRefNet silhouette out of it. That works, but a bbox fill also swallows anything
    else inside the box, which is exactly why v2.1 replaced it with a footprint fill.
    The footprint, though, is the LABELLED core — smaller than the halo that needs
    recovering — so a footprint fill can never undo damage extending beyond it.

    The test for "is this halo mine" is ENCLOSURE by (footprint + this surface): flood
    from outside and keep the void that cannot be reached. That single test does all the
    work, because every other labelled segment is left out of the flood barrier and so
    counts as outside — a void shell ringed by curtain is unreachable and qualifies,
    while a shell opening onto a window or a bed is reachable through it and does not.
    It is prod's flood fill, scoped to the occluder's neighbourhood instead of the whole
    bounding box, which is precisely the bbox fill's swallowing risk removed.

    Two bounds on top:
      * distance — the halo may only reach dist_frac of the long side from the
        footprint. A halo is a thin shell; an unlabelled dresser abutting the plant is
        not, and this stops the fill running down its whole face.
      * growth — the halo may not exceed max_growth times the footprint's own area. A
        shell around an object is comparable to the object; anything far larger is not
        a shell, so the whole growth is abandoned rather than half-applied.
    """
    if void_bool is None or not void_bool.any():
        return fp
    h, w = fp.shape[:2]
    d = max(3, int(dist_frac * max(h, w)))
    ys, xs = np.where(fp > 127)
    if len(ys) == 0:
        return fp
    # Work inside the footprint's neighbourhood only — nothing beyond bbox+d is
    # reachable anyway, and a full-frame dilate per footprint costs ~0.5s at
    # 3840x2063. The +2 margin keeps the flood's outside ring inside the crop.
    m = d + 2
    y0, y1 = max(0, int(ys.min()) - m), min(h, int(ys.max()) + m + 1)
    x0, x1 = max(0, int(xs.min()) - m), min(w, int(xs.max()) + m + 1)
    fp_c = fp[y0:y1, x0:x1]
    near = cv2.dilate(fp_c, np.ones((2 * d + 1, 2 * d + 1), np.uint8)) > 0

    # Barrier = the occluder plus this surface. Everything else — other segments and
    # any void that escapes — is "outside".
    barrier = ((fp_c > 127) | (mask_uint8[y0:y1, x0:x1] > 127)).astype(np.uint8) * 255
    trapped = find_enclosed_holes(barrier) > 0
    fill = trapped & void_bool[y0:y1, x0:x1] & near & ~(fp_c > 127)
    n_fill = int(fill.sum())
    if n_fill == 0 or n_fill > max_growth * max(1, int((fp_c > 127).sum())):
        return fp
    out = fp.copy()
    out[y0:y1, x0:x1][fill] = 255
    return out

def apply_occluder_cutout(mask_uint8, footprints, crisp_union, image_area, ring_frac=0.006,
                          void_bool=None):
    """FILL each resolved occluder's coarse footprint, THEN carve its crisp
    silhouette. Used by both wall and curtain.

    Subtracting the crisp union on its own is a measured no-op: OneFormer has
    already removed the whole object from the surface as one coarse blob, so on
    rooms 4448f78f / c77a842c / f4eec840 100% of BiRefNet's cut pixels were
    already absent and the subtraction changed 0% of its own area. Restoring the
    footprint first is what lets the precise silhouette define the boundary, and
    therefore the only way BiRefNet's quality reaches the render.

    A footprint is only refilled when it is genuinely embedded in THIS surface —
    at least half of the ring around it must already belong to the mask. A plant
    standing in front of a window shares no border with the wall, and refilling it
    there would paint wallpaper across the window between its leaves.
    """
    if crisp_union is None:
        return mask_uint8, 0
    h, w = mask_uint8.shape[:2]
    rk = max(3, int(ring_frac * max(h, w)))
    ker = np.ones((rk, rk), np.uint8)
    n_filled = 0
    n_halo = 0
    halo_union = np.zeros_like(mask_uint8)
    for fp in footprints or []:
        ring = cv2.bitwise_and(cv2.dilate(fp, ker), cv2.bitwise_not(fp))
        ring_px = int(np.count_nonzero(ring))
        if ring_px == 0:
            continue
        # The decision to refill still rests on the ORIGINAL footprint's ring — that
        # gate is what keeps a plant standing in front of a window from painting
        # wallpaper between its leaves. Only WHAT gets refilled grows.
        if int(np.count_nonzero(cv2.bitwise_and(ring, mask_uint8))) / float(ring_px) >= 0.5:
            fill = grow_footprint_into_void(fp, mask_uint8, void_bool)
            if int(np.count_nonzero(fill)) > int(np.count_nonzero(fp)):
                n_halo += 1
                halo_union = np.maximum(halo_union, cv2.bitwise_and(fill, cv2.bitwise_not(fp)))
            mask_uint8 = np.maximum(mask_uint8, fill)
            n_filled += 1
    # Soft carve: crisp_union is BiRefNet's alpha matte, so the surface keeps
    # (1 - occluder_alpha). bitwise_and/bitwise_not would quantise that back to a
    # hard edge and re-introduce the staircase this matte exists to avoid.
    mask_uint8 = np.minimum(mask_uint8, 255 - crisp_union)
    return keep_significant_components(mask_uint8, image_area), (n_filled, n_halo), halo_union

def postprocess_mask(mask_uint8, image_bgr, image_area, do_edge_refine=True, max_hole_frac=DEFAULT_HOLE_FILL_FRAC):
    """Shared cleanup: bridge small gaps, keep ALL significant components,
    fill only SMALL enclosed holes (so objects on the surface stay cut out),
    and snap edges to the image."""
    h, w = mask_uint8.shape[:2]
    k = max(3, int(0.004 * max(h, w)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_uint8 = keep_significant_components(mask_uint8, image_area)
    mask_uint8 = fill_small_holes(mask_uint8, image_area, max_hole_frac)
    if do_edge_refine:
        mask_uint8 = refine_mask_edges(mask_uint8, image_bgr)
    return mask_uint8

def antialias_mask_edge(mask_uint8, width=MASK_FEATHER_PX):
    """Write the mask with an ANTI-ALIASED boundary instead of a hard 0/255 step.

    This is the fix for the staircase ("pixelated") mask edges, and the artifact is
    created at RENDER time, not here. Every renderer magnifies the mask with nearest
    neighbour — utils/wall.py, utils/curtain.py, utils/floor.py, utils/rugs.py and
    utils/wall_depth.py all do
        cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    and then feed it to the blend as an alpha with only a 3x3 Gaussian on top. Masks
    are saved at the UPLOAD resolution while the render canvas is always ~4500 px on
    its long side (app.py upscale_image), so a mask boundary pixel is blown up into a
    k*k block with k = 4500 / max(mask_w, mask_h). Measured over the v2.3 test rooms
    that k ran from 1.17x (3840 px upload) to 5.03x (894 px upload), and the feedback
    tracked it exactly: no pixelation complaint at 1.17x, "low strength, acceptable" at
    1.59x, and a complaint on every room from 2.93x up. A hard binary edge simply
    cannot survive a 5x nearest upscale, no matter how good the mask geometry is —
    the worst-offending room's masks were the CLEANEST of the set.
    Nothing downstream re-thresholds the alpha before blending, so a grayscale ramp in
    the PNG reaches the blend intact and the magnified edge reads as a smooth gradient
    rather than a row of blocks.

    A SIGNED DISTANCE ramp, not a Gaussian blur. A blur sums contributions from both
    walls of a narrow gap and fills it in: at sigma 1.5 a 3 px see-through gap between
    twigs came back 69% closed, which would attack the exact thing the crisp cutouts
    are for. The distance field has no such cross-talk — measured, a 3 px gap stays
    fully open, a 3 px sliver of surface keeps full alpha, and because alpha crosses
    0.5 exactly on the original boundary the >127 set is bit-identical to the binary
    mask (IoU 1.0000), so every threshold-127 consumer downstream is untouched.

    `width` is a constant in MASK pixels on purpose — it needs no knowledge of the
    render size. The ramp is magnified by the same k as the staircase it hides, so the
    ratio between them is scale-invariant and one value covers every upload size.
    """
    binary = (mask_uint8 > 127).astype(np.uint8)
    if not binary.any() or binary.all():
        return mask_uint8  # nothing to feather; also guards distanceTransform
    # distanceTransform measures to the nearest zero pixel WITHIN the image, so a mask
    # running off the frame keeps full alpha at the border instead of fading out.
    d_in = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    sdf = d_in - d_out  # > 0 inside, < 0 outside, 0 on the boundary
    alpha = np.clip(0.5 + sdf / (2.0 * float(width)), 0.0, 1.0)
    ramp = (alpha * 255.0 + 0.5).astype(np.uint8)
    # Pass an already-soft matte straight through. Where the incoming mask carries
    # genuinely intermediate values it is BiRefNet's alpha on an occluder silhouette —
    # real, image-derived sub-pixel detail — and this purely geometric ramp knows
    # nothing that could improve it. It must be `where`, not a min/max blend: a matte
    # value below 127 reads as background to the binary above, so the ramp drives it to
    # 0 and any blend would erase the very detail this is meant to protect.
    is_soft = (mask_uint8 > 4) & (mask_uint8 < 251)
    return np.where(is_soft, mask_uint8, ramp).astype(np.uint8)

def _dump_halo_debug(halo, room_id, seg_class, hotspot_id):
    """Save the void halo this surface recovered, so a run can be measured after the
    fact — the halo fill is the change most likely to need its bounds retuned."""
    if not (DEBUG_SEG and room_id) or halo is None or not halo.any():
        return
    try:
        cv2.imwrite(os.path.join(_DEBUG_MASK_DIR,
                    f"halo_{seg_class}_{room_id}_{hotspot_id}.png"), halo)
    except Exception:
        pass

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

    # -> Build occluder masks ONCE. Three variants, each for a different consumer:
    #    occluder_union_of : raw OneFormer pixels (reference/debug only).
    #    occluder_dilated  : SAM-refined blob + dilation -> CURTAIN, where filling
    #                        inter-leaf gaps stops curtain texture bleeding through.
    #    occluder_birefnet : crisp, undilated, thin/sparse classes only -> WALL,
    #                        where the gaps between twigs must keep their wallpaper.
    occluder_union_of = build_occluder_union_of(occluder_segments, segmentation_map)
    occluder_union = build_occluder_union(sam_predictor, occluder_segments, segmentation_map, image_cv)
    occluder_dilated = None
    if occluder_union is not None:
        d = max(2, int(0.002 * max(width, height)))
        occluder_dilated = cv2.dilate(occluder_union, np.ones((d, d), np.uint8))

    occluder_birefnet, occluder_footprints = None, []
    try:
        occluder_birefnet, occluder_footprints = build_occluder_union_birefnet(
            occluder_segments, segmentation_map, image, id2label, room_id=room_id)
    except Exception as _be:
        print(f"⚠ [WARN] BiRefNet occluder pass failed ({_be}); wall keeps raw OneFormer pixels.")

    # Wherever BiRefNet resolved an object, BiRefNet OWNS that object's boundary:
    # drop the SAM blob for it entirely, by whole connected component.
    #
    # This used to subtract only a dilate(footprint, 0.006*long_side) neighbourhood
    # from the blob, and that is nowhere near enough. Measured on room 6a717a84
    # (3840x2063, so the old kernel was 23px): the dilated SAM union is 4.44x the area
    # of the BiRefNet union, and in the occluder neighbourhood our resulting hole
    # matched the SAM blob (IoU 0.641) BETTER than it matched BiRefNet (0.611) — i.e.
    # the blob, not the matte, was still defining the silhouette. Prod's equivalent
    # hole matches its own BiRefNet cutout at IoU 0.713. Our hole came out 1.30x the
    # size of prod's. That gap IS the blobbiness.
    #
    # Dropping the whole component is the right unit: a SAM blob component and the
    # footprint inside it are the same physical object, so no part of that blob has
    # any business trimming the surface once a precise matte exists for it. Blobs with
    # no resolved footprint (solid furniture) are untouched and still close inter-leaf
    # gaps for the curtain, which is what they were added for.
    if occluder_dilated is not None and occluder_footprints:
        _fp_all = np.zeros_like(occluder_dilated)
        for _fp in occluder_footprints:
            _fp_all = np.maximum(_fp_all, _fp)
        _num_b, _lab_b = cv2.connectedComponents((occluder_dilated > 127).astype(np.uint8))
        _kept = np.zeros_like(occluder_dilated)
        _n_owned = 0
        for _bi in range(1, _num_b):
            _comp = (_lab_b == _bi)
            if (_comp & (_fp_all > 127)).any():
                _n_owned += 1
                continue
            _kept[_comp] = 255
        occluder_dilated = _kept
        if _n_owned:
            print(f"➡ [INFO] BiRefNet owns {_n_owned} of {_num_b - 1} SAM occluder blob(s); "
                  f"those blobs no longer trim any surface.")

    if DEBUG_SEG:
        for _nm, _m in (("occluders", occluder_union),
                        ("occluders_of", occluder_union_of),
                        ("occluders_birefnet", occluder_birefnet)):
            if _m is None:
                continue
            try:
                cv2.imwrite(os.path.join(_DEBUG_MASK_DIR, f"{_nm}_{room_id}.png"), _m)
            except Exception:
                pass

    # VOID is what OneFormer declined to label at all — sentinel 0, ids from 1 (the
    # opposite of EoMT's -1/from-0). It is the raw material for the halo fill in
    # grow_footprint_into_void: the unlabelled envelope around a vase or a branch.
    void_bool = (segmentation_map == 0)

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

        # Every other labelled segment. VOID IS 0 for OneFormer (see below).
        other_of = ((segmentation_map != 0) & (segmentation_map != segment_id)).astype(np.uint8) * 255

        try:
            # Wall: OneFormer panoptic already gives pixel-perfect wall/plant
            # separation — no gap at all. SAM refinement and refine_mask_edges both
            # introduce a gap by avoiding plants. So for wall, use OF pixels
            # directly, with only morphological closing to fill tiny holes: no SAM,
            # no edge refinement. The ONE subtraction it gets is the crisp,
            # undilated BiRefNet silhouette of thin occluders, which resolves the
            # twigs OneFormer's own grid cannot and leaves their gaps intact.
            if seg_class == "wall":
                surface = of_uint8
                max_hole_frac = HOLE_FILL_FRAC.get(seg_class, DEFAULT_HOLE_FILL_FRAC)
                mask_uint8 = postprocess_mask(surface, image_cv, image_area,
                                              max_hole_frac=max_hole_frac, do_edge_refine=False)
                # Hard-subtract every OTHER labelled segment. OneFormer's own
                # panoptic assignment is authoritative about what is not wall, and
                # without this nothing removes coverage the closing/hole-fill above
                # pushed past the boundary. NOTE the sentinel: OneFormer builds its
                # map with torch.zeros and increments BEFORE assigning, so VOID IS 0
                # and ids start at 1 — the opposite of EoMT (void -1, ids from 0).
                # Using ">= 0" here would treat void as another object and eat every
                # mask's unlabelled fringe.
                # NO other_of here, deliberately. It was added to fix CURTAIN bleed,
                # which was measured; the wall never had a bleed complaint. What it
                # DID cost the wall is edge quality: other_of is OneFormer's raw
                # upsampled label map, whose boundary is a coarse axis-aligned
                # staircase, and stamping it onto the wall took the mean boundary
                # run from ~2px to 8.47px. segmentation_v2.1.py had no other_of on
                # the wall and its edges were acceptable, so the wall goes back to
                # raw OneFormer pixels exactly as it was there.
                pre_occluder_mask = mask_uint8.copy()
                mask_uint8, (_nf, _nh), _halo = apply_occluder_cutout(
                    mask_uint8, occluder_footprints, occluder_birefnet, image_area,
                    void_bool=void_bool)
                if _nf:
                    print(f"   [CUTOUT] wall: refilled+carved {_nf} occluder footprint(s), "
                          f"{_nh} grown into their void halo ({int(np.count_nonzero(_halo))} px)")
                _dump_halo_debug(_halo, room_id, seg_class, hotspot['image_hotspots_id'])
            else:
                refined = refine_with_sam(sam_predictor, of_bool, bbox, neg_points, (height, width))

                # For large "stuff" surfaces, OneFormer's extent is the floor: SAM may
                # crisp/extend edges but must never carve away coverage (this removes
                # the big plant "bite" from curtains and recovers split wall/floor parts).
                if seg_class in ONEFORMER_EXTENT_CLASSES:
                    surface = cv2.bitwise_or(refined, of_uint8)
                else:
                    surface = refined

                max_hole_frac = HOLE_FILL_FRAC.get(seg_class, DEFAULT_HOLE_FILL_FRAC)
                # do_edge_refine=False: the guided filter runs ONCE per mask, in the
                # shared tail below. It used to run here too, so every non-wall class
                # got two passes — and each pass roughens the boundary it is meant to
                # clean, because it follows local colour contrast and a textured
                # wall/fabric makes the 0.5 crossing wander per-pixel. Measured on a
                # geometrically perfect ellipse over a real textured guide: boundary
                # wiggle 0.30px clean -> 0.52px after one pass -> 0.64px after two,
                # with specks appearing at two. That doubling is what v2.3 introduced
                # and why the pixelation read as a NEW global issue even though the
                # renderer's nearest upscale had always been there. The late pass is
                # the one worth keeping: it runs after other_of, so it is the pass
                # that actually has the raw label staircase to undo.
                mask_uint8 = postprocess_mask(surface, image_cv, image_area,
                                              max_hole_frac=max_hole_frac, do_edge_refine=False)

                # Hard-subtract every OTHER labelled segment — see the wall branch
                # for why the sentinel is "!= 0". This is the fix for the global
                # curtain bleed: refine_with_sam may expand 0.02*long_side (~30px
                # at 1536) past OneFormer's curtain and `bitwise_or(refined,
                # of_uint8)` makes that permanent, so without this nothing removes
                # it. Measured over the v2.1 rooms: 87k px of curtain mask was
                # sitting on wall / window / floor / rug masks in 9 of 10 rooms.
                mask_uint8 = cv2.bitwise_and(mask_uint8, cv2.bitwise_not(other_of))
                pre_occluder_mask = mask_uint8.copy()

                # Occluders WITHOUT a precise silhouette still get the dilated SAM
                # blob, which closes inter-leaf gaps so curtain texture cannot bleed
                # through onto whatever is behind them.
                if seg_class in OCCLUDER_SUBTRACT_CLASSES and occluder_dilated is not None:
                    mask_uint8 = cv2.bitwise_and(mask_uint8, cv2.bitwise_not(occluder_dilated))
                    mask_uint8 = keep_significant_components(mask_uint8, image_area)

                # Occluders WITH one get fill-and-carve, so the boundary is
                # BiRefNet's silhouette rather than a fattened blob.
                mask_uint8, (_nf, _nh), _halo = apply_occluder_cutout(
                    mask_uint8, occluder_footprints, occluder_birefnet, image_area,
                    void_bool=void_bool)
                if _nf:
                    print(f"   [CUTOUT] {seg_class}: refilled+carved {_nf} occluder footprint(s), "
                          f"{_nh} grown into their void halo ({int(np.count_nonzero(_halo))} px)")
                _dump_halo_debug(_halo, room_id, seg_class, hotspot['image_hotspots_id'])
            # ---- Shared tail, matching the production pipeline's ordering ----
            # 1. other_of above imposes OneFormer's RAW upsampled label grid on the
            #    boundary, which is a pure horizontal/vertical staircase. Measured
            #    against prod on the same rooms, skipping this step left our masks
            #    blockier on every single one (axis-aligned boundary-step fraction
            #    0.890/0.814/0.898 vs prod's 0.876/0.783/0.835). Snap back onto real
            #    image edges to undo it — this is THE fix for the pixelated edges,
            #    and the wall path previously had no edge refinement at all.
            # Only the non-wall classes need this. They DO get other_of (that is
            # the curtain-bleed fix) so their boundary carries the raw label
            # staircase and has to be snapped back onto real image edges — that
            # pass measurably improved the wall from 8.47 to 6.23 mean run when the
            # wall still had other_of. With the wall now back on raw OneFormer
            # pixels it has no staircase to undo, and v2.1 proved it looks right
            # without any edge pass, so it is left alone.
            if seg_class != "wall":
                mask_uint8 = refine_mask_edges(mask_uint8, image_cv)

            if occluder_birefnet is not None:
                # 2. The guided filter above follows colour contrast only, so where
                #    an occluder's colour is close to the surface's it re-thickens
                #    the boundary the carve just made precise. Re-apply the crisp
                #    cut so low-contrast objects are no blurrier than high-contrast
                #    ones. This is what makes a final edge pass SAFE on the wall.
                #    Soft (min against 255-alpha), not bitwise: refine_mask_edges above
                #    re-binarised the mask, so this is also the step that puts
                #    BiRefNet's matte back on the occluder boundary for the non-wall
                #    classes. Doing it with bitwise_and would throw the matte away
                #    again and leave the silhouette hard-edged.
                mask_uint8 = np.minimum(mask_uint8, 255 - occluder_birefnet)
                # 3. Recover the see-through gaps — rope loops, gaps between twigs.
                #    Recomputed here, AFTER the edge pass, because that pass can
                #    blur or erase a small revealed island.
                try:
                    _min_hole = max(150, int(0.0002 * image_area))
                    mask_uint8 = np.maximum(
                        mask_uint8, reveal_occluder_holes(
                            occluder_birefnet, pre_occluder_mask, min_hole_px=_min_hole))
                except Exception as _rh:
                    print(f"⚠ [WARN] Hole reveal failed ({_rh}); skipping.")
        except Exception as e:
            print(f"⚠ [WARN] Mask generation failed for {seg_class} ({e}); using OneFormer mask.")
            mask_uint8 = of_uint8

        # Anti-alias the boundary LAST, after the except-fallback above, so the
        # OneFormer-only fallback mask is written the same way as a fully refined one.
        mask_uint8 = antialias_mask_edge(mask_uint8)

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