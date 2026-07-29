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

# Objects that commonly OCCLUDE surfaces (plants, trees, vases, pots). They are
# segmented precisely and SUBTRACTED from surface masks, so a plant/vase yields
# a tight silhouette cut instead of one large rectangular bite.
OCCLUDER_OBJECTS = {"plant", "tree", "flower", "palm", "pot", "flowerpot", "vase"}

# Decorative/furniture objects that commonly occlude WALL or CURTAIN surfaces
# and may have attached fine detail with no EoMT label of its own (e.g. a
# plant growing from a labeled "vase"). Re-checked via BiRefNet on whichever
# surface(s) they overlap, using one cached detection per real object shared
# across surfaces — see get_cached_occluder_fg.
BIREFNET_RECHECK_LABELS = {
    "lamp", "floor lamp", "table lamp", "light", "chandelier",
    "sconce", "wall lamp", "wall light", "pendant", "pendant light",
    "vase", "plant", "potted plant", "tree", "flower", "branch",
    "sculpture", "statue", "figurine", "candle", "candlestick",
    "stand", "tripod", "pot", "flowerpot",
}

# Surfaces whose extent OneFormer is trusted to define (these are large "stuff"
# regions that get fragmented by occluders; SAM is only allowed to refine edges
# and ADD detail, never to shrink them below OneFormer's coverage).
ONEFORMER_EXTENT_CLASSES = {"wall", "floor", "curtain"}

# Surfaces from which precise occluder silhouettes should be subtracted.
OCCLUDER_SUBTRACT_CLASSES = {"curtain", "wall"}

OCCLUDER_MIN_AREA = 0.0005     # ignore occluder segments below 0.05% of the image
SMALL_OBJECT_MIN_AREA = 0.005  # hotspot small-object filter (0.5% of the image)

# Max enclosed-hole size to fill, as a fraction of the image area. Holes larger
# than this are real objects sitting on/in the surface (a table on the floor, a
# window in a wall) and MUST stay cut out. Floor/rug are kept tight so furniture
# is never swallowed; vertical surfaces allow slightly larger fold/gap fills.
DEFAULT_HOLE_FILL_FRAC = 0.003

# How far past OneFormer's boundary refine_with_sam may grow a mask, as a
# fraction of the long side (30px at 1536). It bounds how much unlabeled fringe a
# mask can have GAINED, so strip_unlabeled_halo reuses it to bound how much it may
# reclaim — the two must not drift apart.
SAM_EXPANSION_FRAC = 0.02

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

    print(f"➡ [INFO] Loading EoMT & SAM-HQ models to {device.upper()}...")
    from transformers import EomtForUniversalSegmentation, AutoImageProcessor
    from segment_anything_hq import sam_model_registry, SamPredictor # type:ignore

    # Load EoMT (panoptic, ADE20K) — drop-in replacement for OneFormer
    processor = AutoImageProcessor.from_pretrained("tue-mps/ade20k_panoptic_eomt_large_640")
    segmenter = EomtForUniversalSegmentation.from_pretrained("tue-mps/ade20k_panoptic_eomt_large_640").to(device)

    # Load SAM-HQ — patch torch.load so CPU-only machines work
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
BIREFNET_INPUT_SIZE = 512  # reduce for CPU; use 1024 on GPU for best quality

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
    """Run BiRefNet on a PIL image; return uint8 fg mask resized to out_hw (H, W)."""
    import torchvision.transforms.functional as TF
    s = BIREFNET_INPUT_SIZE
    img = image_pil.convert("RGB").resize((s, s))
    t = torch.tensor(np.array(img)).float().permute(2, 0, 1) / 255.0
    t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    t = t.unsqueeze(0).to(device)
    with torch.no_grad():
        preds = _birefnet(t)
    pred = preds[-1].sigmoid().cpu().squeeze().numpy()
    mask = (pred > 0.5).astype(np.uint8) * 255
    return cv2.resize(mask, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)

def find_occluder_candidates(segmentation_map, segments_info, id2label, exclude_seg_id, label_set, bbox, region_mask=None):
    """Find every real occluder object — by connected component, not just by
    segment id — whose pixels meaningfully overlap the given surface bbox. A
    single EoMT instance can merge multiple physically separate objects under
    one id (two potted plants on opposite sides of a room both labeled
    "plant"); splitting by connected component keeps them distinct so a
    smaller-but-real object isn't discarded just because a bigger one shares
    its label. Returns a list of (segment_id, component_label, bbox, pixel_area)
    tuples, each using that object's own FULL, unclipped extent (not cropped
    to the surface's bbox — clipping can truncate an object right at the
    boundary, e.g. a vase whose base sits below a curtain's hem).

    region_mask (optional): the surface's own true, possibly non-rectangular
    extent (with tolerance), used INSTEAD of the bbox rectangle for the
    overlap test. bbox alone can't tell a real occluder from an object
    sitting in a totally unrelated part of the same oversized bounding box —
    verified: a vase on a table in front of a curtain still got treated as a
    wall candidate (and sent through a full BiRefNet pass) purely because
    EoMT merged the whole room's wall into one segment spanning past the
    curtain gap, and the bbox of that segment covers the vase's location
    too. Omit for surfaces whose bbox is already a reasonable proxy for
    their extent (curtain)."""
    x1, y1, x2, y2 = bbox
    region = (region_mask > 0) if region_mask is not None else None
    candidates = []
    for seg in segments_info:
        if seg["id"] == exclude_seg_id:
            continue
        lbl = get_label_from_id(id2label, seg["label_id"])
        if not label_matches(lbl, label_set):
            continue
        seg_bool_full = (segmentation_map == seg["id"])
        if region is not None:
            if not np.logical_and(seg_bool_full, region).any():
                continue
        elif not seg_bool_full[y1:y2 + 1, x1:x2 + 1].any():
            continue
        seg_mask_full = seg_bool_full.astype(np.uint8)
        num_cc, labels_cc = cv2.connectedComponents(seg_mask_full)
        if num_cc <= 1:
            continue
        crop_labels = labels_cc[region] if region is not None else labels_cc[y1:y2 + 1, x1:x2 + 1]
        crop_labels = crop_labels[crop_labels > 0]
        if crop_labels.size == 0:
            continue
        overlap_counts = np.bincount(crop_labels)
        for label_id in np.nonzero(overlap_counts)[0]:
            if overlap_counts[label_id] < 100:  # filters stray/noise pixels
                continue
            component_bool = (labels_cc == label_id)
            ys, xs = np.where(component_bool)
            candidates.append((
                seg["id"],
                int(label_id),
                (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                int(component_bool.sum()),
                component_bool.astype(np.uint8) * 255,
            ))
    return candidates

def birefnet_detect_object(
    image,
    raw_component_mask,
    obj_bbox,
    known_area,
    width,
    height,
):
    """BiRefNet foreground detection for a single object, with expand-and-
    retry. A labeled object (e.g. "vase") can be much smaller than the real
    visual occluder it's part of (a plant growing from it with no label of
    its own — EoMT drops thin/sparse foliage to void); a too-tight first crop
    truncates it. But a tiny confident nub (a few px in the darkest part of a
    leaf) can be non-empty and not touch the crop border, satisfying a naive
    "done" check before the crop was ever widened enough to see the rest —
    so "done" requires the detection to cover a meaningful share of what we
    already know is labeled there (known_area), not just ">0 px". Tracks the
    largest result seen as a fallback in case no attempt clears that bar
    (e.g. a too-large later retry dilutes the object back into nothing).
    Returns (cx1, cy1, cx2, cy2, fg_crop) in image coordinates."""
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
        crop_pil = image.crop((cx1, cy1, cx2, cy2))
        fg_crop = birefnet_fg_mask(crop_pil, (cy2 - cy1, cx2 - cx1))
        area = int(np.count_nonzero(fg_crop))
        # A detection covering nearly the whole crop is a degenerate/failed
        # result — BiRefNet found nothing distinctly salient and defaulted
        # to "everything" — not a real thin-object silhouette (a branch, a
        # thin stand). Never trust it, not even as a last-resort fallback:
        # blanking out a whole rectangle of curtain/wall around a thin
        # object is worse than detecting nothing there.
        crop_pixels = (cy2 - cy1) * (cx2 - cx1)
        degenerate = area > 0.85 * crop_pixels
        if area > best_area and not degenerate:
            best_fg_crop, best_box, best_area = fg_crop, (cx1, cy1, cx2, cy2), area
        touches_edge = area > 0 and (
            fg_crop[0, :].any() or fg_crop[-1, :].any() or
            fg_crop[:, 0].any() or fg_crop[:, -1].any()
        )
        full_frame = (cx1 == 0 and cy1 == 0 and cx2 == width and cy2 == height)
        if not degenerate and ((area >= min_good_area and not touches_edge) or full_frame):
            break
        pad_frac *= 2.2
    if best_box is None:
        # Every attempt was degenerate — detect nothing rather than guess.
        cx1, cy1, cx2, cy2 = first_box
        best_fg_crop = np.zeros((cy2 - cy1, cx2 - cx1), np.uint8)
        best_box = first_box
    else:
        # Sanity-check the WINNING attempt only, once — never mid-loop (a
        # per-attempt version tried earlier let a later, genuinely worse
        # attempt's raw-EoMT substitution outbid an already-good earlier
        # attempt purely by pixel count).
        #
        # Two independent signals, either one enough to reject:
        # 1. Total area vs known_area — catches a near-total miss (the
        #    object barely detected at all: <1% kept).
        # 2. Spatial extent vs the object's own known bbox — catches a
        #    PARTIAL miss that (1) alone can't see: a lamp's shade found
        #    but its base/pole never detected AT ALL is still ~wide (the
        #    shade spans the object's full width) so area alone can land
        #    anywhere from 20-50% and slip past a pure area threshold: this
        #    exact case measured 22% height coverage against >=90% for
        #    every confirmed-legitimate trim (a vase's decorative ring cut
        #    out kept 100%/173%; a lamp's shade+arm kept 100%/100%; a
        #    lamp's pole+base — trimmed down from a fatter raw disc, not a
        #    hole and not confined to a slim boundary ring, yet still a
        #    correct trim — kept 90%/107%). A genuine miss never reaches
        #    across the full extent of what EoMT already knows is there;
        #    a legitimate trim (thinner, or with a hole) always does.
        cx1, cy1, cx2, cy2 = best_box
        area_ratio = best_area / known_area
        fys, fxs = np.where(best_fg_crop > 0)
        known_h, known_w = max(1, oy2 - oy1), max(1, ox2 - ox1)
        if len(fys) == 0:
            extent_frac = 0.0
        else:
            height_frac = (fys.max() - fys.min()) / known_h
            width_frac = (fxs.max() - fxs.min()) / known_w
            extent_frac = min(height_frac, width_frac)
        if area_ratio < 0.2 or extent_frac < 0.5:
            best_fg_crop = raw_component_mask[cy1:cy2, cx1:cx2].copy()
    cx1, cy1, cx2, cy2 = best_box
    return cx1, cy1, cx2, cy2, best_fg_crop

def get_cached_occluder_fg(
    cache,
    segment_id,
    component_label,
    raw_component_mask,
    obj_bbox,
    known_area,
    image,
    width,
    height,
):
    """The same physical object often overlaps more than one surface's bbox
    (a sconce straddling the seam between two adjacent curtain panels, or a
    plant behind both a curtain and the wall next to it). Cache BiRefNet's
    detection per (segment_id, component_label) so a given object only ever
    runs through the model once per image, and every surface touching it
    gets the identical, best-quality result instead of a second independent
    — and possibly different-quality — detection."""
    key = (segment_id, component_label)
    if key in cache:
        return cache[key]
    result = birefnet_detect_object(
        image,
        raw_component_mask,
        obj_bbox,
        known_area,
        width,
        height,
    )
    cache[key] = result
    return result

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

def save_eomt_label_debug(image_cv, segmentation_map, segments_info, id2label, out_path):
    """Every EoMT segment, colored, with its raw label text drawn at its
    centroid — over the original photo at reduced opacity. For diagnosing
    what EoMT actually called a given region (e.g. is a stray blob labeled
    "plant", "curtain", or "wall") rather than inferring it from mask shape."""
    h, w = segmentation_map.shape
    np.random.seed(7)
    colors = np.random.randint(60, 255, size=(300, 3))
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for seg in segments_info:
        seg_id = seg["id"]
        mask = (segmentation_map == seg_id)
        if not mask.any():
            continue
        overlay[mask] = colors[seg_id % 300]
    blended = cv2.addWeighted(image_cv, 0.45, overlay, 0.55, 0)
    for seg in segments_info:
        seg_id = seg["id"]
        mask = (segmentation_map == seg_id)
        if not mask.any():
            continue
        label = get_label_from_id(id2label, seg["label_id"])
        ys, xs = np.where(mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        text = f"{label}#{seg_id}"
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

def find_enclosed_holes(mask_img):
    """Return just the fully-enclosed interior holes of mask_img (not the
    filled result) — pixels that are 0 but unreachable from the image
    border without crossing a 255 pixel. Used to recover a genuine physical
    opening in a detected occluder (e.g. a vase's decorative ring cutout)
    that lets the surface behind it show through: BiRefNet's own per-object
    prediction can already mark that gap correctly as non-occluder, but as
    an isolated island fully surrounded by occluder pixels it's exactly the
    shape an earlier keep_significant_components pass (run right after the
    SAM-based occluder subtraction, well before BiRefNet) drops as an
    insignificant disconnected component — silently losing it regardless of
    what BiRefNet finds later. Recomputing the hole directly from the final
    occluder shape and re-adding it to the surface mask recovers it
    regardless of when upstream it was dropped."""
    _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    padded = cv2.copyMakeBorder(binary_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    h, w = padded.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    flood = padded.copy()
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood = flood[1:h - 1, 1:w - 1]
    return cv2.bitwise_not(flood)

def reveal_occluder_holes(occluder_mask, surface_extent_mask):
    """Enclosed holes in an occluder's shape are only sometimes a
    legitimate see-through gap onto the surface — a vase's decorative
    cutout sits entirely over the curtain (revealing it is correct), but a
    thin sparse object (a branch) can straddle the surface's own true edge,
    with some of its inter-twig gaps opening onto the surface and others
    opening onto whatever is past the surface's edge (a window). Revealing
    those would incorrectly paint surface texture past where the surface
    actually is. surface_extent_mask must be the surface's OWN mask from
    BEFORE any occluder subtraction (its natural, un-occluded span) — a
    hole is only revealed if its horizontal span falls (almost) entirely
    within columns that span actually reaches at some row."""
    holes = find_enclosed_holes(occluder_mask)
    if not holes.any():
        return holes
    col_reach = surface_extent_mask.any(axis=0)
    num, labels = cv2.connectedComponents(holes)
    revealed = np.zeros_like(holes)
    for i in range(1, num):
        blob = labels == i
        if int(blob.sum()) < 20:
            continue
        cols = np.unique(np.where(blob)[1])
        if col_reach[cols].mean() >= 0.9:
            revealed[blob] = 255
    return revealed

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
    out = np.zeros_like(mask_img)
    out[labels == largest_label] = 255
    for i, a in enumerate(areas, start=1):
        if a >= thresh:
            out[labels == i] = 255
    return out

def strip_unlabeled_halo(mask_uint8, segmentation_map, segment_id, other_of, protect=None,
                         max_reach_frac=SAM_EXPANSION_FRAC):
    """Give back the UNLABELED fringe that belongs to a neighbouring object.

    EoMT rarely labels an object right up to its true boundary — it leaves a band
    of void (-1) around every lamp, vase and pillow. Those pixels are invisible
    to the `other_of` hard-subtract, which can only remove pixels that actually
    carry another segment's id, while refine_with_sam may expand constrain_frac
    (2% of the long side = 30px at 1536) past OneFormer's boundary and
    `bitwise_or(refined, of_uint8)` then makes that expansion permanent. Verified
    on room da5c6d5d: curtain#5's mask reached x=717 where its fabric actually
    starts at x=748 — the void band around the table lamp standing in front of
    it, which is how curtain texture ended up painted across that lamp and the
    flowers beside it.

    Ambiguous void is resolved by NEAREST LABEL rather than by eroding a fixed
    width: a void pixel is dropped only when it sits closer to some other labeled
    segment than to this surface's own labeled pixels. That keeps genuine wall
    right next to a pillow (its nearest label is still the wall, so it stays and
    gets papered) while releasing fringe that hugs an object, and it splits a
    genuinely ambiguous gap down the middle instead of at an invented offset. A
    fixed erosion width can't do either: the width that reclaims a 30px gap also
    carves an un-textured trench around every object in the room.

    max_reach_frac caps how far out the rule may act, so a large void region far
    from this surface's labeled core is never surrendered wholesale to a distant
    object. Only void is ever touched — pixels EoMT labeled as this surface are
    always kept, and labeled occluder pixels (lamp/vase/flower) are left to
    BiRefNet.
    """
    void = (segmentation_map < 0)
    if not void.any():
        return mask_uint8
    candidate = (mask_uint8 > 127) & void
    if protect is not None:
        candidate &= (protect == 0)
    if not candidate.any():
        return mask_uint8
    own_labeled = (segmentation_map == segment_id)
    if not own_labeled.any():
        return mask_uint8

    h, w = mask_uint8.shape[:2]
    # distanceTransform measures to the nearest ZERO, so each source is inverted.
    d_self = cv2.distanceTransform((~own_labeled).astype(np.uint8) * 255, cv2.DIST_L2, 3)
    d_other = cv2.distanceTransform(cv2.bitwise_not(other_of), cv2.DIST_L2, 3)
    max_reach = max(3.0, max_reach_frac * max(h, w))

    strip = candidate & (d_other < d_self) & (d_other <= max_reach)
    if not strip.any():
        return mask_uint8
    out = mask_uint8.copy()
    out[strip] = 0
    return out

def prune_speckles(mask_uint8, image_area, frac_image=0.0001, min_abs=64, protect=None):
    """Final speckle filter, run after every edge pass and subtraction.

    postprocess_mask already calls keep_significant_components, but that happens
    BEFORE the bbox hole fills, before the _dropped_wall restore (which
    deliberately re-adds sub-threshold components) and before two more
    guided-filter passes — each of which can (re)introduce islands. So
    sub-threshold blobs routinely survive into the saved mask: one measured wall
    mask kept components of 395/87/46/13px against its own 786px threshold, and
    at the ~3x render upscale even the 46px one paints a ~20x20px blob, which is
    the speckle visible beside the door.

    Threshold is deliberately 5x looser than keep_significant_components' own
    500ppm so this cannot silently undo the _dropped_wall restore. Measured over
    400 wall masks the distribution is cleanly bimodal — median non-largest
    component ~2ppm of the image, legitimate second regions (a soffit above a
    window, a wall face past a curtain) above ~1000ppm — so 100ppm removes 91%
    of islands while touching nothing real.
    """
    pruned = keep_significant_components(mask_uint8, image_area, frac_image=frac_image, min_abs=min_abs)
    if protect is not None and protect.any():
        pruned = cv2.bitwise_or(pruned, cv2.bitwise_and(mask_uint8, protect))
    return pruned

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
        return predictor.predict(point_coords=pc, point_labels=pl, box=box, mask_input=mask_input, multimask_output=True)
    except Exception:
        try:
            return predictor.predict(point_coords=pc, point_labels=pl, box=box, multimask_output=True)
        except Exception:
            return predictor.predict(box=box, multimask_output=True)

def _best_iou_index(masks, of_bool):
    best_idx, best_iou = 0, -1.0
    for i, m in enumerate(masks):
        inter = np.logical_and(m, of_bool).sum()
        union = np.logical_or(m, of_bool).sum()
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou, best_idx = iou, i
    return best_idx

def refine_with_sam(predictor, of_bool, bbox, neg_points, image_shape, shrink_guard=0.7,
                    constrain_frac=SAM_EXPANSION_FRAC):
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

def build_occluder_union(predictor, occluder_segments, segmentation_map, image_bgr):
    """Segment each plant/tree/vase/pot occluder precisely (per-instance, box
    prompted SAM) and return the union of their refined masks. This is what gets
    subtracted from curtain/wall masks to produce a tight silhouette cut."""
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

def process_scene_pipeline(image: Image.Image, room_id: str, filename: str, masks_folder: str, generated_folder: str, server_base_url: str):
    
    load_models_if_needed() # Ensure models are loaded
    
    width, height = image.size
    image_area = width * height

    # Run EoMT panoptic segmentation
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = segmenter(**inputs)

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

        # Collect occluders (plant/tree/vase/pot) for precise subtraction later.
        if label_matches(model_label, OCCLUDER_OBJECTS) and seg_area_ratio >= OCCLUDER_MIN_AREA:
            rows_o, cols_o = np.where(seg_bool)
            occluder_segments.append({
                "segment_id": segment_id,
                "bbox": [int(np.min(cols_o)), int(np.min(rows_o)), int(np.max(cols_o)), int(np.max(rows_o))],
                "label": model_label,
            })

        # Match target surfaces (wall/floor/curtain/rug/window).
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
            save_eomt_label_debug(image_cv, segmentation_map, segments_info, id2label,
                os.path.join(_DEBUG_MASK_DIR, f"eomt_labels_{room_id}.png"))
        except Exception as _eomt_dbg_e:
            print(f"⚠ [WARN] EoMT label debug image failed ({_eomt_dbg_e}); skipping.")

    # -> Build occluder masks (plants/vases) ONCE — two variants:
    #    occluder_dilated : SAM-refined blob + dilation → used for CURTAIN so the
    #                       curtain fill recovers fabric behind branches/leaves.
    #    occluder_of      : raw EoMT pixels, no dilation → used for WALL so the
    #                       inter-branch gap pixels (EoMT labels them as "wall") are
    #                       NOT removed and still receive the wallpaper texture.
    occluder_union = build_occluder_union(sam_predictor, occluder_segments, segmentation_map, image_cv)
    occluder_dilated = None
    occluder_of = None
    if occluder_union is not None:
        d = max(2, int(0.002 * max(width, height)))
        occluder_dilated = cv2.dilate(occluder_union, np.ones((d, d), np.uint8))
    if occluder_segments:
        _occ_of = np.zeros((height, width), np.uint8)
        for _oseg in occluder_segments:
            _occ_of |= (segmentation_map == _oseg["segment_id"]).astype(np.uint8) * 255
        occluder_of = _occ_of
        if DEBUG_SEG:
            try:
                cv2.imwrite(os.path.join(_DEBUG_MASK_DIR, f"occluders_{room_id}.png"),
                            occluder_union if occluder_union is not None else occluder_of)
            except Exception:
                pass

    # Shared across every hotspot below — see get_cached_occluder_fg.
    _occluder_fg_cache = {}

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

        try:
            refined = refine_with_sam(sam_predictor, of_bool, bbox, neg_points, (height, width))

            # For large "stuff" surfaces, OneFormer's extent is the floor: SAM may
            # crisp/extend edges but must never carve away coverage (this removes
            # the big plant "bite" from curtains and recovers split wall/floor parts).
            if seg_class in ONEFORMER_EXTENT_CLASSES:
                surface = cv2.bitwise_or(refined, of_uint8)
            else:
                surface = refined

            max_hole_frac = HOLE_FILL_FRAC.get(seg_class, DEFAULT_HOLE_FILL_FRAC)
            mask_uint8 = postprocess_mask(surface, image_cv, image_area, max_hole_frac=max_hole_frac)
            # Snapshot BEFORE any occluder subtraction — the surface's own
            # column-reach, used later to tell a legitimate see-through gap
            # in an occluder (fully inside the surface's own span) apart
            # from background peeking past where the surface actually ends.
            pre_occluder_mask = mask_uint8.copy()

            # Subtract precise occluder silhouettes (plants/vases) from surfaces.
            if seg_class in OCCLUDER_SUBTRACT_CLASSES:
                if seg_class == "wall" and occluder_of is not None:
                    # Raw EoMT pixels — inter-branch/leaf gaps stay as wall so
                    # wallpaper shows through the vase/plant silhouette.
                    mask_uint8 = cv2.bitwise_and(mask_uint8, cv2.bitwise_not(occluder_of))
                elif occluder_dilated is not None:
                    # Dilated SAM blob — fills gaps so curtain texture doesn't
                    # bleed through to objects behind the branches.
                    mask_uint8 = cv2.bitwise_and(mask_uint8, cv2.bitwise_not(occluder_dilated))
                mask_uint8 = keep_significant_components(mask_uint8, image_area)

            # Hard-subtract pixels OF definitively assigned to other objects.
            # SAM refine can expand ~38px beyond the OF boundary; this prevents it
            # bleeding into lamps, mirrors, bed, paintings that OF correctly identified.
            #
            # VOID is -1, not 0. EoMT ships its own compute_segments (distinct from
            # Mask2Former's, which does start ids at 1): transformers 4.56 fills the
            # map via `torch.zeros(...) - 1` and assigns `current_segment_id` BEFORE
            # incrementing it, so ids start at 0 and 0 is an ordinary segment — in
            # this room set it is usually whichever surface scored first, typically
            # the floor (verified: `floor#0` in the eomt_labels debug overlays).
            # `!= 0` therefore treated that one real segment as void and never
            # subtracted it from any other surface's mask, and when segment 0 was
            # itself the hotspot it wrongly counted void as "another object".
            other_of = ((segmentation_map >= 0) & (segmentation_map != segment_id)).astype(np.uint8) * 255
            mask_uint8 = cv2.bitwise_and(mask_uint8, cv2.bitwise_not(other_of))

            # Wall: subtract foreground objects (plants, vases, lamps…) that EoMT
            # mislabelled as wall or that SAM expanded into from unlabeled pixels.
            # Uses the same per-object candidate discovery + cached BiRefNet
            # detection as the curtain path below, so an object handled once
            # (for a curtain, say) gets reused here at identical quality instead
            # of a second, independently-run — and possibly worse — detection.
            fg_combined = None
            if seg_class == "wall":
                try:
                    fg_combined = np.zeros((height, width), np.uint8)
                    # Fill-first step (ported from the curtain path's Step 1):
                    # without it, any gap between plant leaves that EoMT left
                    # void — or a fragment of the wall's own segment dropped by
                    # keep_significant_components for being too small in
                    # isolation — never gets a chance to be recovered before
                    # Step 2's BiRefNet cut runs; it just stays whatever
                    # `other_of`'s hard-subtract above already zeroed out,
                    # leaving Step 2's precise silhouette subtracted from a
                    # ragged, incomplete base instead of a clean filled one.
                    _wall_region_tol = cv2.dilate(pre_occluder_mask, np.ones((25, 25), np.uint8))
                    labeled_as_other_wall = np.zeros((height, width), np.uint8)
                    occluder_labeled_wall = np.zeros((height, width), np.uint8)
                    for _seg in segments_info:
                        if _seg["id"] == segment_id:
                            continue
                        _lbl = get_label_from_id(id2label, _seg["label_id"])
                        _seg_bool = (segmentation_map == _seg["id"])
                        _is_recheck_label = label_matches(_lbl, BIREFNET_RECHECK_LABELS)
                        # Split by connected component — same granularity
                        # find_occluder_candidates uses — before testing
                        # overlap. EoMT can merge multiple physically separate
                        # objects sharing one label under a single segment id
                        # (verified: two potted plants AND two free-standing
                        # decorative vase/branch objects all labeled "plant"
                        # under one id; also verified: a floor lamp's shade and
                        # its pole/base labeled as two disconnected components
                        # sharing one "lamp" id). Testing overlap on the WHOLE
                        # segment let one component's overlap (the pole,
                        # genuinely near the wall) vouch for a totally
                        # unrelated component of the same label (the shade,
                        # sitting mostly over the window) — promising Step 2
                        # would cut it, when Step 2's own per-component test
                        # (rightly) never finds it a candidate at all. Painted
                        # solid white forever.
                        _num_cc, _labels_cc = cv2.connectedComponents(_seg_bool.astype(np.uint8))
                        for _cc_id in range(1, _num_cc):
                            _comp_bool = (_labels_cc == _cc_id)
                            _overlap_px = int(np.logical_and(_comp_bool, _wall_region_tol > 0).sum())
                            if _is_recheck_label and _overlap_px >= 100:
                                occluder_labeled_wall[_comp_bool] = 255
                            else:
                                labeled_as_other_wall[_comp_bool] = 255
                    wx1, wy1, wx2, wy2 = bbox
                    bbox_fill_wall = np.zeros((height, width), np.uint8)
                    bbox_fill_wall[wy1:wy2 + 1, wx1:wx2 + 1] = 255
                    hole_wall = cv2.bitwise_and(bbox_fill_wall, cv2.bitwise_not(mask_uint8))
                    hole_wall = cv2.bitwise_and(hole_wall, cv2.bitwise_not(labeled_as_other_wall))
                    if hole_wall.any():
                        max_fill_area = max(200, DEFAULT_HOLE_FILL_FRAC * image_area)
                        _dk = max(15, int(0.008 * max(width, height)))
                        near_object_wall = cv2.dilate(labeled_as_other_wall, np.ones((_dk, _dk), np.uint8))
                        _any_occluder_raw = np.zeros_like(hole_wall)
                        for _oseg in occluder_segments:
                            _any_occluder_raw |= (segmentation_map == _oseg["segment_id"]).astype(np.uint8) * 255
                        # A void gap immediately touching/inside an occluder's
                        # silhouette is that object's own fuzzy, unlabeled edge
                        # — EoMT rarely labels 100% of an object's pixels right
                        # to its true boundary — not a genuine small gap in the
                        # wall itself. `_any_occluder_raw` alone (exact label
                        # match) never overlaps this halo by construction, so
                        # it fell through to the small-hole-fill branch below
                        # and got painted solid wall, with its removal resting
                        # entirely on BiRefNet's later per-object cut. Verified:
                        # when that BiRefNet detection comes back degenerate/
                        # empty for a small or ambiguous crop, nothing ever
                        # undoes the fill, leaving a solid wall island sitting
                        # inside the object. Dilating catches the halo so it's
                        # deferred to fg_combined (BiRefNet's precise cut, or —
                        # if that also finds nothing — simply left unfilled)
                        # instead of blindly trusted to wall.
                        _any_occluder_dilated = cv2.dilate(_any_occluder_raw, np.ones((_dk, _dk), np.uint8))
                        num_hw, labels_hw, stats_hw, _ = cv2.connectedComponentsWithStats(hole_wall, connectivity=8)
                        hole_wall_capped = np.zeros_like(hole_wall)
                        for _hwi in range(1, num_hw):
                            comp_mask = (labels_hw == _hwi)
                            # A component pixel that's neither EoMT's own wall
                            # label, nor any other labeled object, nor a known
                            # occluder is truly unexplained — void, not a
                            # member of segments_info at all. A LARGE
                            # truly-void chunk means EoMT never recognized
                            # whatever is actually there — a real, unlabeled
                            # piece of furniture (verified: a dark wood
                            # dresser EoMT never labeled as anything).
                            # Checked FIRST, before any branch below —
                            # verified this exact case reaches connected-
                            # components as ONE blob merging the dresser's
                            # void together with a vase's own (already
                            # correctly-excluded) pixels sitting on top of
                            # it, since they're physically touching in the
                            # image; that merge made the component visible to
                            # the occluder_labeled_wall branch below, which
                            # has no size limit at all and blindly filled the
                            # WHOLE merged blob — including the unrelated
                            # void — as wall. A blob this size is never a
                            # genuine small hole/halo regardless of which
                            # object it happens to touch, so it's excluded up
                            # front rather than patched per-branch. Leaving it
                            # unfilled (not even deferred to fg_combined,
                            # which nothing will ever resolve for an object
                            # with no label to find a candidate under) is the
                            # safe side: at worst a real small wall gap goes
                            # unpainted, never furniture painted as wall.
                            truly_void = np.logical_and(
                                np.logical_and(comp_mask, ~of_bool),
                                ~np.logical_or(labeled_as_other_wall > 0, occluder_labeled_wall > 0)
                            )
                            if int(np.count_nonzero(truly_void)) > max_fill_area:
                                continue
                            if np.logical_and(comp_mask, occluder_labeled_wall > 0).any():
                                hole_wall_capped[comp_mask] = 255
                            elif np.logical_and(comp_mask, _any_occluder_dilated > 0).any():
                                fg_combined[comp_mask] = 255
                            elif (stats_hw[_hwi, cv2.CC_STAT_AREA] <= max_fill_area
                                  and not np.logical_and(comp_mask, near_object_wall > 0).any()):
                                hole_wall_capped[comp_mask] = 255
                            else:
                                # Same fix as the curtain path: reached when
                                # the component is large, or small but
                                # touching another labeled object's (pillow,
                                # bed, furniture — anything in
                                # labeled_as_other_wall, not just
                                # occluder-type) dilated halo — that halo is
                                # the object's own fuzzy edge, not a genuine
                                # wall gap. near_object is now ALWAYS
                                # deferred, no size-cap escape, so it's only
                                # painted if BiRefNet elsewhere confirms it as
                                # real wall, never blindly trusted by area.
                                own_wall_part = np.logical_and(comp_mask, of_bool)
                                near_object = np.logical_and(own_wall_part, near_object_wall > 0)
                                far_from_object = np.logical_and(own_wall_part, near_object_wall == 0)
                                hole_wall_capped[far_from_object] = 255
                                fg_combined[near_object] = 255
                                fg_combined[np.logical_and(comp_mask, ~of_bool)] = 255
                        hole_wall = hole_wall_capped
                    if hole_wall.any():
                        mask_uint8 = cv2.bitwise_or(mask_uint8, hole_wall)

                    wall_candidates = find_occluder_candidates(
                        segmentation_map, segments_info, id2label,
                        segment_id, BIREFNET_RECHECK_LABELS, bbox,
                        region_mask=_wall_region_tol,
                    )
                    print("wall candidates:")
                    for c in wall_candidates:
                        print(c[0], c[2], c[3])
                    if wall_candidates:
                        load_birefnet_if_needed()
                    for _oi, (
                        _seg_id,
                        _comp_label,
                        _obj_bbox,
                        _known_area,
                        _raw_component_mask,
                    ) in enumerate(wall_candidates):
                            _cx1, _cy1, _cx2, _cy2, _fg_crop = get_cached_occluder_fg(
                            _occluder_fg_cache,
                            _seg_id,
                            _comp_label,
                            _raw_component_mask,
                            _obj_bbox,
                            _known_area,
                            image,
                            width,
                            height
                            )
                            fg_combined[_cy1:_cy2, _cx1:_cx2] = np.maximum(
                                fg_combined[_cy1:_cy2, _cx1:_cx2], _fg_crop
                            )
                            if DEBUG_SEG:
                                try:
                                    _hid = hotspot['image_hotspots_id']
                                    cv2.imwrite(
                                        os.path.join(_DEBUG_MASK_DIR,
                                            f"birefnet_input_wall_{room_id}_{_hid}_{_oi}.png"),
                                        cv2.cvtColor(np.array(image.crop((_cx1, _cy1, _cx2, _cy2))), cv2.COLOR_RGB2BGR)
                                    )
                                except Exception:
                                    pass
                    if fg_combined is not None and fg_combined.any():
                        if DEBUG_SEG:
                            try:
                                _hid = hotspot['image_hotspots_id']
                                cv2.imwrite(os.path.join(_DEBUG_MASK_DIR,
                                    f"birefnet_cutout_wall_{room_id}_{_hid}.png"), fg_combined)
                            except Exception:
                                pass
                        mask_uint8 = cv2.bitwise_and(mask_uint8, cv2.bitwise_not(fg_combined))
                        _pre_kc_wall = mask_uint8.copy()
                        mask_uint8 = keep_significant_components(mask_uint8, image_area)
                        # Restore whatever keep_significant_components just
                        # pruned as a "noise island", except pixels BiRefNet
                        # itself claims as the object, and only within
                        # _wall_region_tol (the wall's own pre-occlusion
                        # reach) so this can't resurrect an object sitting
                        # outside the wall's real territory.
                        _dropped_wall = cv2.bitwise_and(_pre_kc_wall, cv2.bitwise_not(mask_uint8))
                        _dropped_wall = cv2.bitwise_and(_dropped_wall, cv2.bitwise_not(fg_combined))
                        _dropped_wall = cv2.bitwise_and(_dropped_wall, _wall_region_tol)
                        mask_uint8 = cv2.bitwise_or(mask_uint8, _dropped_wall)
                        # Hole-reveal and the re-cut against the later guided-
                        # filter edge pass both happen once, shared with the
                        # curtain path, further down — see _birefnet_occluder_mask.
                except Exception as _we:
                    print(f"⚠ [WARN] BiRefNet wall step failed ({_we}); skipping.")

            # Curtain hole completion via BiRefNet:
            # 1. Fill the full curtain bbox (excluding structural gaps like window/wall).
            # 2. Crop the image to the curtain panel and run BiRefNet → it treats the
            #    curtain fabric as background and vase/branches as salient foreground.
            # 3. Subtract the BiRefNet foreground from the filled curtain mask.
            fg_full = None
            if seg_class == "curtain":
                try:
                    x1, y1, x2, y2 = bbox
                    # Step 1: fill curtain gaps inside the bbox.
                    # Exclude any EoMT-labeled non-curtain segment EXCEPT known
                    # foreground occluders (lamp, vase, plant…) — those sit IN FRONT
                    # of the curtain and EoMT often bleeds their label onto adjacent
                    # curtain fabric. We include those pixels so the fill recovers the
                    # curtain fabric behind them; BiRefNet subtracts the actual object.
                    # Everything else (wall, floor, pillow, bed, sofa, headboard…) is
                    # excluded so the fill never bleeds below the curtain hem.
                    # Within a curtain bbox, EoMT often labels curtain fabric as
                    # "window/windowpane" because it can see the window behind.
                    # Include these too so the fill recovers that curtain coverage
                    # (window/windowpane need no BiRefNet re-detection, so they're
                    # not part of BIREFNET_RECHECK_LABELS itself).
                    _OCCLUDER_LABELS = BIREFNET_RECHECK_LABELS | {"window", "windowpane"}
                    # A flat, width-independent cutoff line here (an earlier
                    # attempt: curtain's own max-y + a fixed buffer, applied
                    # uniformly across the whole bbox) doesn't follow any real
                    # object's contour — it cuts some pillows too high and
                    # others not enough depending on where they sit relative
                    # to that one global row. The connected-component size
                    # cap below is what actually guards against void bleed
                    # (a pillow/headboard being filled in as curtain); no
                    # separate flat cap is needed on top of it.
                    bbox_fill = np.zeros((height, width), np.uint8)
                    bbox_fill[y1:y2 + 1, x1:x2 + 1] = 255
                    labeled_as_other = np.zeros((height, width), np.uint8)
                    occluder_labeled = np.zeros((height, width), np.uint8)
                    for _seg in segments_info:
                        if _seg["id"] == segment_id:
                            continue
                        _lbl = get_label_from_id(id2label, _seg["label_id"])
                        if not label_matches(_lbl, _OCCLUDER_LABELS):
                            labeled_as_other[segmentation_map == _seg["id"]] = 255
                        elif label_matches(_lbl, BIREFNET_RECHECK_LABELS):
                            occluder_labeled[segmentation_map == _seg["id"]] = 255
                    fg_full = np.zeros((height, width), np.uint8)
                    hole = cv2.bitwise_and(bbox_fill, cv2.bitwise_not(mask_uint8))
                    hole = cv2.bitwise_and(hole, cv2.bitwise_not(labeled_as_other))
                    if hole.any():
                        # A hole here is either a genuine small labeling gap in
                        # the curtain fabric (recoverable — e.g. behind sparse
                        # branches), a large VOID region where EoMT failed to
                        # confidently classify a real object (a pillow, a
                        # headboard) as anything at all, OR a labeled occluder
                        # (lamp, vase, plant…) that Step 1 deliberately treats
                        # as fillable so Step 2's BiRefNet can cut its precise
                        # shape back out. Void pixels are NOT tracked in
                        # labeled_as_other — void isn't a member of
                        # segments_info, it's the absence of one — so without
                        # a cap, a whole unlabeled pillow silently gets filled
                        # in as "curtain" and painted with texture. But that
                        # same size cap must NOT apply to a component that
                        # touches an occluder-labeled segment: a real lamp or
                        # vase is routinely bigger than the cap, and capping
                        # it here would exclude it from the fill entirely —
                        # leaving it (and any void pixels merged into the same
                        # connected blob) permanently excluded, well beyond
                        # BiRefNet's own precise shape, since it never gets a
                        # chance to be filled-then-precisely-cut in Step 2.
                        max_fill_area = max(200, DEFAULT_HOLE_FILL_FRAC * image_area)
                        # Size alone can't tell "EoMT mislabeled a real object
                        # as curtain" apart from "this is just a big stretch
                        # of genuine curtain visible behind sparse leaves" —
                        # both look like a large of_bool sub-region. Only
                        # proximity to a real, known object (labeled_as_other)
                        # tells them apart: a mislabel happens right at that
                        # object's boundary, genuine fabric doesn't. Only
                        # of_bool pixels near such an object go through the
                        # size cap; of_bool pixels far from any go through
                        # regardless of size (same fix as the wall path).
                        _dk = max(15, int(0.008 * max(width, height)))
                        near_object_curtain = cv2.dilate(labeled_as_other, np.ones((_dk, _dk), np.uint8))
                        # Same fix as the wall path: a void gap immediately
                        # touching/inside an occluder's silhouette is that
                        # object's own fuzzy, unlabeled edge, not a genuine
                        # small gap in the curtain fabric. `occluder_labeled`
                        # alone (exact label match) never overlaps this halo
                        # by construction, so it fell through to the
                        # small-hole-fill branch below and got painted solid
                        # curtain, with its removal resting entirely on
                        # BiRefNet's later per-object cut (fg_full). Deferring
                        # the halo to fg_full instead means it's either
                        # precisely cut by BiRefNet or — if that also finds
                        # nothing — simply left unfilled, instead of blindly
                        # trusted to curtain.
                        _occluder_labeled_dilated = cv2.dilate(occluder_labeled, np.ones((_dk, _dk), np.uint8))
                        num_hole, labels_hole, stats_hole, _ = cv2.connectedComponentsWithStats(hole, connectivity=8)
                        hole_capped = np.zeros_like(hole)
                        for _hi in range(1, num_hole):
                            comp_mask = (labels_hole == _hi)
                            if np.logical_and(comp_mask, occluder_labeled > 0).any():
                                hole_capped[comp_mask] = 255
                            elif np.logical_and(comp_mask, _occluder_labeled_dilated > 0).any():
                                fg_full[comp_mask] = 255
                            elif (stats_hole[_hi, cv2.CC_STAT_AREA] <= max_fill_area
                                  and not np.logical_and(comp_mask, near_object_curtain > 0).any()):
                                hole_capped[comp_mask] = 255
                            else:
                                # Reached only when the component is either
                                # large, or small but touching another
                                # labeled object's (e.g. pillow, bed,
                                # furniture — anything in labeled_as_other,
                                # not just occluder-type) dilated halo. That
                                # halo is the object's own fuzzy, unlabeled
                                # edge, not a genuine curtain gap — verified:
                                # a pillow's boundary halo was small enough to
                                # pass the old area-only cap and got painted
                                # solid curtain right up to the pillow's edge,
                                # with no BiRefNet mechanism (pillows aren't
                                # occluder-type) to ever cut it back out.
                                # near_object is now ALWAYS deferred — no
                                # size-cap escape — so it only gets painted
                                # if BiRefNet elsewhere calls it real curtain,
                                # never blindly trusted by area alone.
                                own_curtain_part = np.logical_and(comp_mask, of_bool)
                                near_object = np.logical_and(own_curtain_part, near_object_curtain > 0)
                                far_from_object = np.logical_and(own_curtain_part, near_object_curtain == 0)
                                hole_capped[far_from_object] = 255
                                fg_full[near_object] = 255
                                fg_full[np.logical_and(comp_mask, ~of_bool)] = 255
                        hole = hole_capped
                    if hole.any():
                        mask_uint8 = cv2.bitwise_or(mask_uint8, hole)

                    # Step 2: BiRefNet PER individual foreground occluder — crop
                    # tightly around each object's OWN full shape, not the whole
                    # curtain panel. A thin twig/vase is a tiny fraction of the full
                    # curtain bbox, which reads as "no salient object" to BiRefNet
                    # (verified: raw logits all background over the full-panel crop);
                    # cropped tightly around just that object, the same model
                    # detects it reliably (mirrors the per-occluder wall path above).
                    # Same shared candidate discovery + cached detection as the wall
                    # path, so an object already handled for the wall (or another
                    # curtain whose bbox happens to overlap this one) is reused here
                    # instead of re-run.
                    #
                    # Candidates are restricted to BIREFNET_RECHECK_LABELS — i.e.
                    # exactly the set Step 1 filled back in (window/windowpane
                    # excluded, since those need no re-detection). Furniture
                    # (bed/table/pillow/…) was never excluded from labeled_as_other,
                    # so it's already handled by the plain hard-subtract without
                    # costing a BiRefNet call.
                    fg_object_bboxes = find_occluder_candidates(
                        segmentation_map, segments_info, id2label,
                        segment_id, BIREFNET_RECHECK_LABELS, bbox,
                    )

                    # fg_full already carries any void blobs confirmed as real
                    # unlabeled objects above; merge the labeled-occluder
                    # detections into it rather than overwriting.
                    if fg_object_bboxes:
                        load_birefnet_if_needed()
                        for _oi, (_seg_id, _comp_label, _obj_bbox, _known_area, _raw_component_mask) in enumerate(fg_object_bboxes):
                            cx1, cy1, cx2, cy2, fg_crop = get_cached_occluder_fg(
                                _occluder_fg_cache, _seg_id, _comp_label, _raw_component_mask,  _obj_bbox,
                                _known_area, image, width, height,
                            )
                            fg_full[cy1:cy2, cx1:cx2] = np.maximum(fg_full[cy1:cy2, cx1:cx2], fg_crop)
                            if DEBUG_SEG:
                                try:
                                    hotspot_id = hotspot['image_hotspots_id']
                                    cv2.imwrite(
                                        os.path.join(_DEBUG_MASK_DIR,
                                            f"birefnet_input_{seg_class}_{room_id}_{hotspot_id}_{_oi}.png"),
                                        cv2.cvtColor(np.array(image.crop((cx1, cy1, cx2, cy2))), cv2.COLOR_RGB2BGR)
                                    )
                                except Exception:
                                    pass
                        fg_full = cv2.bitwise_and(fg_full, cv2.bitwise_not(labeled_as_other))
                        if DEBUG_SEG:
                            try:
                                hotspot_id = hotspot['image_hotspots_id']
                                cv2.imwrite(
                                    os.path.join(_DEBUG_MASK_DIR,
                                        f"birefnet_cutout_{seg_class}_{room_id}_{hotspot_id}.png"),
                                    fg_full
                                )
                            except Exception:
                                pass
                        mask_uint8 = cv2.bitwise_and(
                            mask_uint8, cv2.bitwise_not(fg_full)
                        )
                except Exception as _be:
                    print(f"⚠ [WARN] BiRefNet curtain refinement failed ({_be}); skipping.")

            if DEBUG_SEG and seg_class in ("wall", "curtain"):
                try:
                    cv2.imwrite(os.path.join(_DEBUG_MASK_DIR,
                        f"after_birefnet_{seg_class}_{room_id}_{hotspot['image_hotspots_id']}.png"), mask_uint8)
                except Exception:
                    pass

            # Final edge pass: snap boundary to actual image edges after all
            # subtractions — OF segment boundaries are pixel-grid rough, this
            # smooths them onto real colour transitions in the photo.
            mask_uint8 = refine_mask_edges(mask_uint8, image_cv)
            # A revealed occluder hole is by construction a small island fully
            # enclosed by the occluder — exactly the shape the hygiene passes at
            # the end of this block would otherwise throw away (the halo strip
            # because the hole sits inside another object's dilated ring, the
            # speckle prune because it is disconnected and tiny). Track every
            # reveal so both passes can exempt it.
            revealed_holes = np.zeros((height, width), np.uint8)
            _birefnet_occluder_mask = fg_full if seg_class == "curtain" else fg_combined
            if _birefnet_occluder_mask is not None:
                # The guided-filter snap above follows color contrast; where
                # an occluder's color is close to the surface's own (a brown
                # branch against tan/beige curtain fabric, a pale lamp pole
                # against a light wall — verified on both: BiRefNet's own
                # cutout was precise, but the edge pass alone widened it by
                # up to 7x beyond the true shape there), it bleeds the
                # boundary wider than the object actually is. Re-apply the
                # already-precise BiRefNet cut — on WHICHEVER surface this
                # is — so low-contrast objects don't end up with a blurrier
                # boundary than high-contrast ones.
                mask_uint8 = cv2.bitwise_and(mask_uint8, cv2.bitwise_not(_birefnet_occluder_mask))
                # A detected occluder can have a genuine physical opening cut
                # through it (verified: this vase's decorative ring) that lets
                # the surface behind show through. Re-check for it here, AFTER
                # the guided-filter pass above — that pass can itself blur or
                # shrink a small revealed-hole island, so recomputing last
                # (rather than earlier, before a step that can undo it) is what
                # makes it actually survive to the saved mask.
                revealed_holes = reveal_occluder_holes(_birefnet_occluder_mask, pre_occluder_mask)
                mask_uint8 = cv2.bitwise_or(mask_uint8, revealed_holes)
            # Post-refinement cleanup for wall: strip any OTHER labeled
            # segment (window, windowpane, pillow, bed, headboard…) the
            # guided-filter edge pass just above may have bled into. This
            # mirrors the curtain-only cleanup below, closing the same gap
            # for wall — verified: a decorative branch crossing in front of
            # a window, with BiRefNet's own cutout not 100% covering its
            # thinnest twigs, left small gaps where the guided filter (which
            # only follows color contrast, oblivious to WHY a pixel was
            # excluded) smeared wallpaper texture back across the window's
            # own already-correctly-excluded segment. Unlike curtain, window/
            # windowpane is NOT kept here — a window is a genuinely separate
            # surface from wall, never an intentional fill-then-cut occluder
            # the way a lamp/vase/plant is.
            if seg_class == "wall":
                for _seg in segments_info:
                    if _seg["id"] == segment_id:
                        continue
                    _lbl = get_label_from_id(id2label, _seg["label_id"])
                    if not label_matches(_lbl, BIREFNET_RECHECK_LABELS):
                        mask_uint8[segmentation_map == _seg["id"]] = 0

            # Post-refinement cleanup for curtains only: strip non-occluder
            # segments the guided filter may have bled into (pillow, wall, etc.).
            # Occluder labels (lamp, vase, plant) are kept — their pixels were
            # intentionally included in the fill so BiRefNet can handle them.
            if seg_class == "curtain":
                _OCC_POST = {
                    "lamp", "floor lamp", "table lamp", "light", "chandelier",
                    "vase", "plant", "potted plant", "tree", "flower", "branch",
                    "sculpture", "statue", "figurine", "candle", "candlestick",
                    "stand", "tripod",
                    "window", "windowpane",
                }
                for _seg in segments_info:
                    if _seg["id"] == segment_id:
                        continue
                    _lbl = get_label_from_id(id2label, _seg["label_id"])
                    if not label_matches(_lbl, _OCC_POST):
                        mask_uint8[segmentation_map == _seg["id"]] = 0

            if seg_class == "curtain" and fg_full is not None:
                # Second safety net, curtain-only: the _OCC_POST cleanup just
                # above zeroes pixels by raw EoMT segment id and doesn't know
                # about the hole reveal that already ran earlier — if a hole
                # happens to overlap one of those "other" segments, this
                # would silently undo it. Re-apply once more, truly last.
                _rev_again = reveal_occluder_holes(fg_full, pre_occluder_mask)
                mask_uint8 = cv2.bitwise_or(mask_uint8, _rev_again)
                revealed_holes = cv2.bitwise_or(revealed_holes, _rev_again)

            # --- Hygiene: runs after EVERY edge pass and subtraction above, so
            # nothing downstream can reintroduce what these two remove. ---
            mask_uint8 = strip_unlabeled_halo(
                mask_uint8, segmentation_map, segment_id, other_of, protect=revealed_holes
            )
            mask_uint8 = prune_speckles(mask_uint8, image_area, protect=revealed_holes)
        except Exception as e:
            print(f"⚠ [WARN] Mask generation failed for {seg_class} ({e}); using OneFormer mask.")
            mask_uint8 = of_uint8

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