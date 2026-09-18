import numpy as np
from PIL import Image, ImageFilter


def round_to_multiple(x, base=16):
    return max(base, base * round(x / base))


def resize_to_max_dim(img, max_dim=1024, resample=Image.LANCZOS):
    """Resize so the longer side == max_dim, preserving aspect ratio,
    and round both dims to a multiple of 16."""
    w, h = img.size
    scale = max_dim / max(w, h)
    new_w = round_to_multiple(int(w * scale))
    new_h = round_to_multiple(int(h * scale))
    return img.resize((new_w, new_h), resample)


def standardize_pair(room_img, mask_img, max_dim=1024):
    """Resize a room image + its mask to the same standardized size.
    Room image uses smooth resampling; mask uses NEAREST to keep edges sharp.
    Returns (room_img_resized, mask_img_resized).
    """
    room_resized = resize_to_max_dim(room_img, max_dim, resample=Image.LANCZOS)
    target_size = room_resized.size  # (w, h)
    mask_resized = mask_img.resize(target_size, Image.NEAREST)
    return room_resized, mask_resized


def union_bbox(boxes):
    """Union multiple [x_min, y_min, x_max, y_max] boxes into one box
    that covers all of them."""
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def expand_bbox(bbox, margin_frac, img_w, img_h):
    """Expand a bbox by margin_frac on each side (relative to the box's own
    width/height), clamped to the image bounds, so the crop keeps some
    surrounding context (wall, rod, floor) instead of just the window itself."""
    x_min, y_min, x_max, y_max = bbox
    mx = (x_max - x_min) * margin_frac
    my = (y_max - y_min) * margin_frac
    return [
        max(0, int(x_min - mx)),
        max(0, int(y_min - my)),
        min(img_w, int(x_max + mx)),
        min(img_h, int(y_max + my)),
    ]


def prepare_room_and_mask(room_img, mask_img, bbox=None, margin_frac=0.2, max_dim=1024):
    """Prepare a room image + mask pair for Flux generation.

    If bbox is given (a single [x_min,y_min,x_max,y_max], or a list of boxes
    to union across multiple hotspots), crop both images to that region
    (expanded by margin_frac for context) before standardizing. If no bbox
    is given, standardize the raw full room image + full raw mask untouched.

    Returns (prepared_room, prepared_mask, crop_box) where crop_box is the
    region (in original image coordinates) that was cropped, or None if no
    crop was applied — the caller needs crop_box to paste the result back.
    """
    if bbox:
        box = union_bbox(bbox) if isinstance(bbox[0], (list, tuple)) else list(bbox)

        w, h = room_img.size
        box = expand_bbox(box, margin_frac, w, h)

        room_crop = room_img.crop(tuple(box))
        mask_crop = mask_img.crop(tuple(box))
        room_std, mask_std = standardize_pair(room_crop, mask_crop, max_dim=max_dim)
        return room_std, mask_std, box

    room_std, mask_std = standardize_pair(room_img, mask_img, max_dim=max_dim)
    return room_std, mask_std, None


def compute_sheer_center_mask(mask_img, left_frac=0.25, right_frac=0.75):
    """Derive the CENTER-gap-only mask a two-stage sheer style's stage 2
    needs. Stage 1 (opaque side panels) uses the full window mask unchanged
    - the prompt alone is what makes it leave the center open - but stage 2
    (the sheer panel layered on top of stage 1's output) must be restricted
    to only the center gap, otherwise it would repaint stage 1's
    already-generated side panels too.

    left_frac/right_frac mark the center gap's boundaries as fractions of
    the window's own bbox width (0.25/0.75 -> center gap is the middle
    50%). Cropped to the window's actual bbox first, so this doesn't touch
    anything outside it.

    Returns a PIL "L" image the same size as mask_img.
    """
    mask_np = np.array(mask_img.convert("L"))
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return Image.fromarray(np.zeros_like(mask_np))

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    window_width = x2 - x1

    left_boundary = int(x1 + window_width * left_frac)
    right_boundary = int(x1 + window_width * right_frac)

    center_mask = np.zeros_like(mask_np)
    center_mask[y1:y2 + 1, left_boundary:right_boundary] = \
        mask_np[y1:y2 + 1, left_boundary:right_boundary]

    return Image.fromarray(center_mask)


def paste_crop_back(full_room_img, crop_result_img, crop_box, mask_full_img=None, feather_px=15):
    """Paste a generated crop back into the full-resolution room image at
    crop_box. If mask_full_img (the full-resolution curtain mask) is given,
    the paste is feathered along the mask edges so the crop boundary and the
    curtain edge don't show a visible seam; otherwise it's a hard paste over
    the whole crop_box rectangle.
    """
    x0, y0, x1, y1 = crop_box
    target_size = (x1 - x0, y1 - y0)
    crop_resized = crop_result_img.resize(target_size, Image.LANCZOS)

    result = full_room_img.copy()
    if mask_full_img is not None:
        mask_crop = mask_full_img.crop((x0, y0, x1, y1)).convert("L")
        mask_crop = mask_crop.filter(ImageFilter.GaussianBlur(feather_px))
        result.paste(crop_resized, (x0, y0), mask_crop)
    else:
        result.paste(crop_resized, (x0, y0))
    return result
