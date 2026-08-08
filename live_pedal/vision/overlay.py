"""The camera preview window.

Its job is calibration, not decoration: you need to be able to see at a glance
whether the tracker has your hand, which gesture values it is reading, and what
those values are doing to the sound. Everything drawn here answers one of those
three questions.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..ipc import FEATURE_INDEX, FEATURE_NAMES
from ..layout import ParamLayout

# Landmark index pairs forming the hand skeleton.
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

BG = (24, 22, 20)
FG = (235, 235, 235)
DIM = (130, 128, 125)
ACCENT = (90, 200, 255)       # BGR: amber
GOOD = (120, 220, 130)
WARN = (80, 160, 250)
BAD = (80, 80, 240)

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_W = 250


def _bar(img, x, y, w, h, value, colour, bg=(60, 58, 55)):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    filled = int(w * min(max(value, 0.0), 1.0))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x + filled, y + h), colour, -1)


def draw_skeleton(frame, lm_xy: np.ndarray, colour=ACCENT) -> None:
    h, w = frame.shape[:2]
    pts = [(int(p[0] * w), int(p[1] * h)) for p in lm_xy]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], colour, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 5 if i in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, p, r, FG, -1, cv2.LINE_AA)


def compose(
    frame: np.ndarray,
    features: np.ndarray,
    param_values: np.ndarray,
    layout: ParamLayout,
    shown_params: list[int],
    status: dict,
) -> np.ndarray:
    """Return the full window image: camera in the middle, panels either side."""
    h, w = frame.shape[:2]
    canvas = np.full((h, w + PANEL_W * 2, 3), BG, dtype=np.uint8)
    canvas[:, PANEL_W : PANEL_W + w] = frame

    _draw_features(canvas, features, h)
    _draw_params(canvas, param_values, layout, shown_params, PANEL_W + w, h)
    _draw_status(canvas, status, PANEL_W, w)
    return canvas


def _draw_features(canvas, features, h) -> None:
    cv2.putText(canvas, "GESTURE", (14, 26), FONT, 0.5, ACCENT, 1, cv2.LINE_AA)
    y = 48
    present = features[FEATURE_INDEX["present"]] > 0.5

    for name in FEATURE_NAMES:
        if name in ("fps", "present"):
            continue
        i = FEATURE_INDEX[name]
        val = float(features[i])
        is_pose = name.startswith("pose_")

        if is_pose:
            colour = GOOD if val > 0.5 else (70, 70, 70)
            cv2.circle(canvas, (20, y - 4), 5, colour, -1, cv2.LINE_AA)
            cv2.putText(canvas, name[5:], (34, y), FONT, 0.4,
                        FG if val > 0.5 else DIM, 1, cv2.LINE_AA)
            y += 19
        else:
            cv2.putText(canvas, name, (14, y - 6), FONT, 0.38,
                        FG if present else DIM, 1, cv2.LINE_AA)
            _bar(canvas, 14, y, PANEL_W - 34, 7, val,
                 ACCENT if present else (70, 70, 70))
            cv2.putText(canvas, f"{val:.2f}", (PANEL_W - 44, y + 7),
                        FONT, 0.34, DIM, 1, cv2.LINE_AA)
            y += 26
        if y > h - 14:
            break

    if not present:
        cv2.putText(canvas, "no hand -- values held", (14, h - 12),
                    FONT, 0.38, WARN, 1, cv2.LINE_AA)


def _draw_params(canvas, values, layout: ParamLayout, shown, x0, h) -> None:
    cv2.putText(canvas, "EFFECTS", (x0 + 14, 26), FONT, 0.5, ACCENT, 1, cv2.LINE_AA)
    y = 48
    last_effect = None

    for i in shown:
        name = layout.names[i]
        effect = name.split(".")[0]
        if effect != last_effect:
            if y > 48:
                y += 6
            cv2.putText(canvas, effect.upper(), (x0 + 14, y), FONT, 0.4,
                        (150, 190, 220), 1, cv2.LINE_AA)
            y += 16
            last_effect = effect
        if y > h - 20:
            break

        val = float(values[i])
        u = layout.normalise(i, val)
        short = name.split(".", 1)[1]
        unit = layout.units[i]

        is_enable = short == "enable"
        colour = (GOOD if val > 0.5 else (70, 70, 70)) if is_enable else ACCENT

        cv2.putText(canvas, short, (x0 + 20, y), FONT, 0.36, DIM, 1, cv2.LINE_AA)
        txt = f"{val:.0f}{unit}" if abs(val) >= 100 else f"{val:.2f}{unit}"
        cv2.putText(canvas, txt, (x0 + PANEL_W - 78, y), FONT, 0.36, FG, 1, cv2.LINE_AA)
        y += 5
        _bar(canvas, x0 + 20, y, PANEL_W - 40, 6, u, colour)
        y += 20


def _draw_status(canvas, status: dict, x0: int, w: int) -> None:
    parts = [
        f"cam {status.get('fps', 0):.0f} fps",
        f"dsp {status.get('cpu', 0.0):.0f}%",
        f"xruns {status.get('xruns', 0)}",
    ]
    colour = GOOD
    if status.get("xruns", 0) > 0:
        colour = WARN
    if status.get("cpu", 0.0) > 70:
        colour = BAD

    cv2.rectangle(canvas, (x0, 0), (x0 + w, 26), (16, 15, 14), -1)
    cv2.putText(canvas, "   ".join(parts), (x0 + 10, 18), FONT, 0.45,
                colour, 1, cv2.LINE_AA)

    flags = []
    if status.get("bypass"):
        flags.append("BYPASS")
    if status.get("hold"):
        flags.append("HOLD")
    if flags:
        text = "  ".join(flags)
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.45, 1)
        cv2.putText(canvas, text, (x0 + w - tw - 10, 18), FONT, 0.45,
                    WARN, 1, cv2.LINE_AA)

    hint = "q quit   b bypass   h hold   space panic   r reset view"
    hh = canvas.shape[0]
    cv2.putText(canvas, hint, (x0 + 10, hh - 10), FONT, 0.38, DIM, 1, cv2.LINE_AA)
