import os
import cv2
import numpy as np

DEBUG_REFERENCES = os.getenv("HX_DEBUG_REFERENCES", "1") not in ("0", "false", "False")
DEBUG_REF_DIR = os.path.join("Debugs", "References")
DEBUG_MAX_WIDTH = 1600

# BGR, by sample status.
_STATUS_COLOUR = {
    "selected":     (60, 200, 60),     # green  - this one set the scale
    "rejected":     (60, 60, 235),     # red    - measured badly or gated out
    "not-selected": (40, 190, 235),    # amber  - usable, but out-scored
    "skipped":      (150, 150, 150),   # grey   - a higher-priority class won
}
_DEFAULT_COLOUR = (200, 120, 200)
_KNOWN_COLOUR = (245, 245, 245)    # the "this is what known_ft looks like" bar
_RULER_COLOUR = (235, 225, 90)     # 1 ft ticks in corrected feet


def _draw_label(canvas, x, y, lines, colour, scale):
    """Text with a filled backing box, clamped inside the canvas."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = max(1, int(round(scale * 1.6)))
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
    box_w = max(w for w, _ in sizes) + 12
    line_h = max(h for _, h in sizes) + 8
    box_h = line_h * len(lines) + 6

    H, W = canvas.shape[:2]
    x = int(np.clip(x, 0, max(0, W - box_w)))
    y = int(np.clip(y, 0, max(0, H - box_h)))

    cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), (24, 24, 24), -1)
    cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), colour, max(1, thick))

    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (x + 6, y + line_h * (i + 1) - 4), font, scale, colour, thick, cv2.LINE_AA)

def _summary_lines(sample):
    """Compact lines describing one reference."""
    head = "{0} #{1}  {2}".format(sample.get("label", "?"),
                                  sample.get("index", "?"),
                                  str(sample.get("status", "?")).upper())

    if sample.get("measured_ft") is not None:
        detail = "{0} ft measured vs {1} ft known  ->  x{2}".format(
            sample.get("measured_ft"), sample.get("known_ft"), sample.get("scale"))
        extra = "footprint {0} x {1} ft ({2} axis)  {3} m away  score {4}".format(
            sample.get("footprint_short_ft"), sample.get("footprint_long_ft"),
            sample.get("measured_axis", "?"), sample.get("median_depth_m"),
            sample.get("score"))
        lines = [head, detail, extra]

    else:
        lines = [head, "not measured ({0} ft expected)".format(sample.get("known_ft"))]

    if sample.get("reason"):
        lines.append(str(sample["reason"]))
    return lines


def _scaled(points, view_scale):
    """Geometry arrives in room-image pixels; the overlay may be downscaled."""
    return [(int(round(px * view_scale)), int(round(py * view_scale))) for px, py in points]


def _draw_measurement(canvas, geom, sample, colour, view_scale, font_scale, is_selected):
    """Draw the EXACT geometry the scale correction was computed from"""
    if not geom: return

    foot = geom.get("footprint_px")
    if foot:
        pts = np.array(_scaled(foot, view_scale), np.int32)
        cv2.polylines(canvas, [pts], True, colour, max(1, int(canvas.shape[1] / 1100)))

    measured = geom.get("measured_px")

    if measured:
        (ax, ay), (bx, by) = _scaled(measured, view_scale)
        thick = max(3, int(canvas.shape[1] / 420))
        cv2.line(canvas, (ax, ay), (bx, by), colour, thick, cv2.LINE_AA)

        for cx_, cy_ in ((ax, ay), (bx, by)):
            cv2.circle(canvas, (cx_, cy_), thick + 2, colour, -1, cv2.LINE_AA)

        _draw_label(canvas, (ax + bx) // 2, (ay + by) // 2,
                    ["MEASURED {0} ft".format(sample.get("measured_ft"))],
                    colour, font_scale)

    known = geom.get("known_px")
    if known:
        (ax, ay), (bx, by) = _scaled(known, view_scale)
        thick = max(2, int(canvas.shape[1] / 600))
        cv2.line(canvas, (ax, ay), (bx, by), _KNOWN_COLOUR, thick, cv2.LINE_AA)

        for cx_, cy_ in ((ax, ay), (bx, by)):
            cv2.circle(canvas, (cx_, cy_), thick + 2, _KNOWN_COLOUR, -1, cv2.LINE_AA)

        _draw_label(canvas, (ax + bx) // 2, (ay + by) // 2,
                    ["KNOWN {0} ft".format(sample.get("known_ft"))],
                    _KNOWN_COLOUR, font_scale)

    # The ruler is the payoff — only worth drawing for the object that won.
    if is_selected:
        for tick in geom.get("ruler_px") or []:
            (ax, ay), (bx, by) = _scaled([tick["a"], tick["b"]], view_scale)

            whole = tick["ft"] % 2 == 0
            cv2.line(canvas, (ax, ay), (bx, by), _RULER_COLOUR,
                    max(1, int(canvas.shape[1] / (900 if whole else 1400))), cv2.LINE_AA)

            if whole:
                cv2.putText(canvas, "{0}ft".format(tick["ft"]), (bx + 4, by + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8,
                            _RULER_COLOUR, max(1, int(font_scale * 1.6)), cv2.LINE_AA)


def save_reference_debug(room_img, detections, samples, scale_factor, key, out_dir=DEBUG_REF_DIR):
    """Write the overlay + one raw mask PNG per detected reference."""
    if room_img is None or not detections:
        return None
    os.makedirs(out_dir, exist_ok=True)

    H0, W0 = room_img.shape[:2]
    view_scale = min(1.0, DEBUG_MAX_WIDTH / float(W0))

    if view_scale < 1.0:
        canvas = cv2.resize(room_img, (int(W0 * view_scale), int(H0 * view_scale)), interpolation=cv2.INTER_AREA)
    else:
        canvas = room_img.copy()

    H, W = canvas.shape[:2]
    font_scale = max(0.42, W / 1900.0)

    by_index = {s.get("index"): s for s in (samples or [])}
    tint = canvas.copy()
    outlines = []

    for index, det in enumerate(detections):
        mask = det.get("mask")
        if mask is None:
            continue
        sample = by_index.get(index, {"index": index, "label": det.get("label"), "status": "not-evaluated"})
        colour = _STATUS_COLOUR.get(sample.get("status"), _DEFAULT_COLOUR)

        # ---- raw mask, at its own resolution ----
        cv2.imwrite(os.path.join(out_dir, "{0}_{1}_{2}_{3}.png".format(
            key, index, det.get("label", "obj"), sample.get("status", "na"))), mask)

        # ---- overlay ----
        small = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        tint[small > 127] = colour
        contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(biggest)
            outlines.append((contours, colour, sample, x, y, w, h))

    canvas = cv2.addWeighted(canvas, 0.72, tint, 0.28, 0)

    for contours, colour, sample, x, y, w, h in outlines:
        cv2.drawContours(canvas, contours, -1, colour, max(2, int(W / 700)))

        # The measurement geometry sits on top of the silhouette.
        _draw_measurement(canvas, sample.get("_geom"), sample, colour, view_scale,font_scale, sample.get("status") == "selected")

        # Label below the object when there is no room above it.
        lines = _summary_lines(sample)
        above = y - int(28 * font_scale * len(lines)) - 10
        _draw_label(canvas, x, above if above > 0 else y + h + 6, lines, colour, font_scale)

    selected = next((s for s in (samples or []) if s.get("status") == "selected"), None)
    header = [
        "SCALE x{0:.3f}".format(float(scale_factor)),
        "from: {0}".format("{0} #{1} ({2} ft measured -> {3} ft known)".format(
            selected.get("label"), selected.get("index"),
            selected.get("measured_ft"), selected.get("known_ft"))
            if selected else "no usable reference - depth left uncorrected"),
        "priority: bed > door > chair",
        "thick bar = measured span   white bar = known size   yellow = 1ft ticks",
    ]
    _draw_label(canvas, 10, 10, header, (60, 200, 60) if selected else (40, 190, 235), font_scale * 1.15)

    overlay_path = os.path.join(out_dir, "{0}_overlay.jpg".format(key))
    cv2.imwrite(overlay_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("   [REF-DEBUG] wrote {0} (+{1} mask png)".format(overlay_path, len(outlines)))
    return overlay_path