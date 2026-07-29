import cv2
import numpy as np

"""Shared mask -> alpha resampling for every texture renderer.

Masks are generated ONCE at upload resolution (utils/segmentation.py) but
composited onto a canvas that app.upscale_image has pushed to ~4500px, so every
renderer has to bridge a ~3x resolution gap. Resampling the binary mask with
INTER_NEAREST — what all four renderers used to do — turns each mask pixel into
a hard 3x3 block, and the fixed (3, 3) GaussianBlur in the blend is a no-op at
that size, so every boundary was composited as a razor-sharp staircase. Real
photographic edges carry a few pixels of softness; a hard-cut texture boundary
reads as "pasted on" even where the mask geometry is correct.

resize_mask_alpha resamples through a signed distance field instead. The SDF is
smooth and continuous, so interpolating it places the 0.5 level set back on the
original contour with sub-pixel accuracy instead of quantising it to blocks, and
the feather ramp is then specified in CANVAS pixels rather than being inherited
from whatever resolution the mask happened to be generated at.

Geometry consumers are unaffected: thresholding the returned alpha at 127 still
yields a binary mask, now with a sub-pixel-accurate boundary.
"""

# Feather ramp width as a fraction of the canvas's longest side. ~5px at 4500px:
# wide enough to hide a +/-1 mask-pixel staircase (~3 canvas px after upscale)
# and to match the softness the Lanczos upscale already gave the photo, short of
# looking blurry.
FEATHER_FRAC = 0.0012
MIN_FEATHER_PX = 1.5


def _signed_distance(binary):
    """Distance to the mask boundary in mask pixels, positive inside.

    distanceTransform measures to the nearest pixel of the opposite class, so the
    innermost/outermost ring both come back at 1.0 and the raw difference jumps
    -1 -> +1 with no zero crossing on the contour itself. Pulling 0.5 off the
    magnitude puts those rings at +/-0.5 and the contour back at 0, which is both
    the geometrically correct sub-pixel position and what lets the alpha ramp
    produce intermediate values instead of snapping straight from 0 to 255.
    """
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(cv2.bitwise_not(binary), cv2.DIST_L2, 3)
    sdf = inside - outside
    return sdf - np.sign(sdf) * 0.5


def resize_mask_alpha(mask_img, width, height, feather_frac=FEATHER_FRAC):
    """Resample a mask to (width, height) as a soft uint8 alpha.

    Interiors saturate at 255 and only the boundary is feathered, so masked
    coverage is unchanged — this softens HOW the edge is composited, not WHERE
    it is. Also normalises the two mask sources: segmentation.py writes a strict
    binary PNG, while app.get_or_create_mask's SAM-API path writes an
    INTER_LINEAR-resized mask that was never re-thresholded.
    """
    if mask_img is None:
        return None
    if mask_img.ndim == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    if not binary.any():
        return np.zeros((height, width), np.uint8)

    mh, mw = binary.shape[:2]
    # SDF distances are in MASK pixels; convert to canvas pixels so the ramp
    # width stays constant regardless of the upload's resolution.
    scale = 0.5 * (width / float(mw) + height / float(mh))
    # The ramp also has to span at least ~1.5 mask pixels, or the SDF's own 1px
    # quantisation leaves nothing between 0 and 255 to interpolate (only bites
    # when the canvas is barely larger than the mask; app.upscale_image normally
    # gives a ~3x gap).
    feather = max(MIN_FEATHER_PX, feather_frac * max(width, height), 1.5 * scale)

    sdf = _signed_distance(binary)
    # Only the boundary band matters, and distanceTransform returns very large
    # values deep inside a large region (unbounded when a mask has no background
    # at all, which overflowed the scaling below). Clamping to twice the ramp
    # keeps the field small, keeps INTER_LINEAR away from steep far-field
    # gradients, and costs nothing since everything beyond saturates anyway.
    band = 2.0 * feather / scale
    np.clip(sdf, -band, band, out=sdf)
    if (mw, mh) != (width, height):
        sdf = cv2.resize(sdf, (width, height), interpolation=cv2.INTER_LINEAR)

    # alpha = 0.5 on the contour, saturating +/- feather/2 either side of it.
    np.multiply(sdf, scale / feather, out=sdf)
    np.add(sdf, 0.5, out=sdf)
    np.clip(sdf, 0.0, 1.0, out=sdf)
    return (sdf * 255.0).astype(np.uint8)


def hard_mask(alpha, thresh=127):
    """Binary view of a soft alpha, for geometry/coverage consumers."""
    return (alpha > thresh).astype(np.uint8) * 255
