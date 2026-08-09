"""The vision process: camera in, parameter targets out.

This runs as a *separate process*, not a thread, and that is the single most
important structural decision in the project. MediaPipe inference and OpenCV
rendering both hold Python's global interpreter lock in bursts. In one process
those bursts would contend with the audio callback -- which also needs the GIL,
on a hard deadline -- and you would hear it as intermittent clicks. Separate
processes have separate interpreters, so the audio thread never waits for the
camera.

The two talk only through shared memory: this side writes parameter targets,
reads back smoothed values for the display, and sets flag bits for the keyboard
shortcuts.
"""

from __future__ import annotations

import time
import traceback

import cv2
import numpy as np

from ..config import VisionConfig
from ..ipc import (
    FLAG_BYPASS,
    FLAG_HOLD,
    FLAG_PANIC,
    FLAG_QUIT,
    STAT_CPU_PERMILLE,
    STAT_XRUNS,
    VectorChannel,
)
from ..layout import ParamLayout
from ..mapping import Mapper
from . import overlay
from .camera import open_camera
from .features import FeatureExtractor
from .tracker import HandTracker, select_hand

WINDOW = "live_pedal -- gesture control"


def run_vision(
    vcfg: VisionConfig,
    layout: ParamLayout,
    mappings: list[dict],
    target_name: str,
    display_name: str,
    model_path: str,
    chord_labels: dict[str, list[str]] | None = None,
) -> None:
    """Child process entry point. Never raises into the parent; reports and exits."""
    target = None
    try:
        n = len(layout)
        target = VectorChannel(n, np.float64, name=target_name)
        display = VectorChannel(n, np.float64, name=display_name)

        mapper = Mapper(mappings, layout)
        extractor = FeatureExtractor(smoothing=vcfg.smoothing)
        tracker = HandTracker(
            model_path,
            num_hands=1 if vcfg.hand == "any" else 2,
            min_detection_confidence=vcfg.min_detection_confidence,
            min_presence_confidence=vcfg.min_presence_confidence,
            min_tracking_confidence=vcfg.min_tracking_confidence,
        )
        cam = open_camera(
            vcfg.camera_index, vcfg.width, vcfg.height, vcfg.fps, vcfg.exposure
        )
        measured = cam.measure_fps(0.7)
        print(f"  camera   : [{vcfg.camera_index}] via {cam.backend}, "
              f"{cam.size[0]}x{cam.size[1]}, {measured:.0f} fps measured "
              f"(exposure {vcfg.exposure})")
        if measured < 20.0:
            print("             low frame rate limits how fast gestures track. "
                  "More light, or a more negative 'exposure', will help.")

        _loop(vcfg, cam, tracker, extractor, mapper, target, display, layout,
              chord_labels or {})

    except Exception:
        print("\n--- vision process failed ---")
        traceback.print_exc()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if target is not None:
            target.set_flag(FLAG_QUIT)


def _loop(vcfg, cam, tracker, extractor, mapper, target, display, layout,
          chord_labels) -> None:
    n = len(layout)
    display_values = np.array(layout.defaults, dtype=np.float64)

    # Chord selectors, so the window and the console can name what a gesture
    # just chose. Without this, "did the camera see me?" and "is the chord
    # wrong?" look identical from where you are standing, holding a guitar.
    chord_slots = [
        (layout.index_of(name), labels)
        for name, labels in chord_labels.items()
        if name in layout.names
    ]
    last_chord: list[int] = [-1] * len(chord_slots)
    warned_handedness = False

    # Only show parameters something can actually change, plus every enable
    # switch. A full dump of 80 numbers is unreadable while you are playing.
    shown = sorted(
        mapper.targeted_params
        | {i for i, nm in enumerate(layout.names) if nm.endswith(".enable")}
    )

    t_start = time.perf_counter()
    fps = 0.0
    frame_times: list[float] = []
    last_frame = time.perf_counter()
    show = vcfg.show_window
    seq = 0

    while True:
        if target.flags & FLAG_QUIT:
            break

        frame_bgr, seq = cam.read_latest(seq, timeout=1.0)
        if frame_bgr is None:
            continue

        now = time.perf_counter()
        dt = now - last_frame
        last_frame = now
        frame_times.append(dt)
        if len(frame_times) > 30:
            frame_times.pop(0)
        if frame_times:
            mean_dt = sum(frame_times) / len(frame_times)
            fps = 1.0 / mean_dt if mean_dt > 1e-6 else 0.0

        h, w = frame_bgr.shape[:2]
        aspect = w / h if h else 1.0

        # Detect on the unmirrored frame so handedness labels stay correct.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hands = tracker.detect(rgb, int((now - t_start) * 1000.0))
        hand = select_hand(hands, vcfg.hand)

        # A hand in frame that the config refuses to use looks exactly like no
        # hand at all, and it is a very easy way to spend an evening waving at
        # a camera that is deliberately ignoring you.
        if hands and hand is None and not warned_handedness:
            warned_handedness = True
            seen = ", ".join(sorted({h.label for h in hands}))
            print(f"\n  note: tracking a {seen} hand but the preset asks for "
                  f"'{vcfg.hand}', so gestures are being ignored.\n"
                  f"        Use the other hand, or set vision.hand: any.")

        if hand is not None:
            features = extractor.extract(
                hand.image_lm, hand.world_lm, aspect, now, fps
            )
        else:
            features = extractor.no_hand(fps)

        hold = bool(target.flags & FLAG_HOLD)
        targets = mapper.update(features, hold=hold)
        target.write(targets)

        # Report chord changes on the console too: it is the only feedback
        # there is with --no-window, and it confirms the gesture registered
        # even when the pad happens to be silent.
        current_chord = ""
        for slot, (idx, labels) in enumerate(chord_slots):
            value = int(min(max(round(targets[idx]), 0), len(labels) - 1))
            current_chord = labels[value]
            if value != last_chord[slot]:
                last_chord[slot] = value
                print(f"\n  chord -> {labels[value]}")

        if not show:
            continue

        # --- display --------------------------------------------------------
        display.read_into(display_values)
        view = cv2.flip(frame_bgr, 1) if vcfg.mirror else frame_bgr
        view = np.ascontiguousarray(view)

        if hand is not None:
            lm = hand.image_lm.copy()
            if vcfg.mirror:
                lm[:, 0] = 1.0 - lm[:, 0]
            overlay.draw_skeleton(view, lm)
            label = hand.label
            cv2.putText(view, f"{label} hand  {hand.score:.2f}", (10, h - 12),
                        overlay.FONT, 0.45, overlay.GOOD, 1, cv2.LINE_AA)

        if current_chord:
            overlay.draw_chord(view, current_chord)

        flags = target.flags
        status = {
            "fps": fps,
            "cpu": display.get_stat(STAT_CPU_PERMILLE) / 10.0,
            "xruns": display.get_stat(STAT_XRUNS),
            "bypass": bool(flags & FLAG_BYPASS),
            "hold": bool(flags & FLAG_HOLD),
        }
        canvas = overlay.compose(view, features, display_values, layout, shown, status)
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            target.set_flag(FLAG_QUIT)
            break
        elif key == ord("b"):
            target.toggle_flag(FLAG_BYPASS)
        elif key == ord("h"):
            target.toggle_flag(FLAG_HOLD)
        elif key == ord(" "):
            # Momentary: clear it next frame so the engine resets once.
            target.set_flag(FLAG_PANIC)
        elif key == ord("r"):
            extractor.reset()
        elif target.flags & FLAG_PANIC:
            target.clear_flag(FLAG_PANIC)

        # If the user closed the window with the X button, stop.
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                target.set_flag(FLAG_QUIT)
                break
        except cv2.error:
            target.set_flag(FLAG_QUIT)
            break

    cam.release()
    tracker.close()
