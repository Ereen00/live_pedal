"""Thin wrapper over MediaPipe's HandLandmarker task.

MediaPipe 1.0 removed the old ``mp.solutions.hands`` API, so this uses the Tasks
API, which needs a ``.task`` model file on disk. ``tools/fetch_model.py`` gets it.

Running mode is VIDEO rather than LIVE_STREAM: VIDEO is synchronous, so a frame
goes in and landmarks come out on the same call. LIVE_STREAM would hand results
back on a callback thread later, which buys nothing here -- we are already off
the audio thread -- and makes it harder to reason about how stale the gesture
data is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass
class Hand:
    label: str                 # "Left" or "Right", as the person's own hand
    score: float
    image_lm: np.ndarray       # (21, 3) normalised to the frame
    world_lm: np.ndarray       # (21, 3) metres, centred on the hand


class HandTracker:
    def __init__(
        self,
        model_path: str | Path,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"hand landmark model not found at {model_path}\n"
                f"Fetch it with:  python tools/fetch_model.py\n"
                f"(or download {MODEL_URL} to that path)"
            )

        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[Hand]:
        """``rgb`` must be the *unmirrored* frame.

        MediaPipe's handedness labels assume a normal camera image. If you flip
        the frame before detection, every "Left" becomes a "Right" and picking a
        hand by name stops working -- so mirror only for display.
        """
        # Timestamps must strictly increase or the task raises.
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        hands: list[Hand] = []
        for i, lms in enumerate(result.hand_landmarks):
            image_lm = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
            if i < len(result.hand_world_landmarks):
                w = result.hand_world_landmarks[i]
                world_lm = np.array([[p.x, p.y, p.z] for p in w], dtype=np.float64)
            else:
                world_lm = None  # type: ignore[assignment]

            label, score = "Unknown", 0.0
            if i < len(result.handedness) and result.handedness[i]:
                cat = result.handedness[i][0]
                label = cat.category_name
                score = float(cat.score)

            hands.append(Hand(label=label, score=score,
                              image_lm=image_lm, world_lm=world_lm))
        return hands

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


def select_hand(hands: list[Hand], want: str) -> Hand | None:
    """Pick the controlling hand.

    ``want`` is "left", "right" or "any". With "any", the highest-confidence
    hand wins. This is what keeps your fretting hand from grabbing the controls
    when it wanders into frame.
    """
    if not hands:
        return None
    want = (want or "any").lower()
    if want in ("left", "right"):
        matches = [h for h in hands if h.label.lower() == want]
        if matches:
            return max(matches, key=lambda h: h.score)
        return None
    return max(hands, key=lambda h: h.score)
