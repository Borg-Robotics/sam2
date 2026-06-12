"""Barcode detection (pyzbar) and overlay drawing."""

import cv2
from pyzbar.pyzbar import decode


def detect_barcode(cfg, frame_bgr):
    results = decode(frame_bgr)
    scale_used = 1.0

    if (
        len(results) == 0
        and cfg.barcode_decode_upscale is not None
        and cfg.barcode_decode_upscale > 1.0
    ):
        upscaled = cv2.resize(
            frame_bgr,
            None,
            fx=cfg.barcode_decode_upscale,
            fy=cfg.barcode_decode_upscale,
            interpolation=cv2.INTER_CUBIC,
        )
        results = decode(upscaled)
        scale_used = cfg.barcode_decode_upscale

    if len(results) == 0:
        return None

    r = results[0]

    try:
        barcode_data = r.data.decode("utf-8")
    except Exception:
        barcode_data = str(r.data)

    x, y, w, h = r.rect

    if scale_used != 1.0:
        x = int(x / scale_used)
        y = int(y / scale_used)
        w = int(w / scale_used)
        h = int(h / scale_used)

    return {
        "type": str(r.type),
        "data": barcode_data,
        "rect": (int(x), int(y), int(w), int(h)),
        "decoded_count": int(len(results)),
    }


def draw_barcode_overlay(frame_bgr, barcode):
    if barcode is None:
        return frame_bgr

    x, y, w, h = barcode["rect"]

    cv2.rectangle(
        frame_bgr,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3,
    )

    cv2.putText(
        frame_bgr,
        f"{barcode['type']}: {barcode['data']}",
        (x, max(y - 12, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return frame_bgr
