"""Turn 21 hand landmarks into the handful of numbers you can map to knobs.

Two coordinate systems are in play and each is used for what it is good at:

*Image landmarks* are normalised to the frame, so they tell you **where** the
hand is -- position, apparent size, rotation on screen.

*World landmarks* are metric and centred on the hand itself, so they tell you
what **shape** the hand is in regardless of where it is or how it is turned.
Finger curl measured from image coordinates collapses the moment you point a
finger at the camera; measured in world coordinates it stays honest.

The raw ratios here are deliberately left in their natural units and only
roughly normalised. Hands differ, cameras differ, and the mapping layer already
has ``in_lo``/``in_hi`` for calibration -- so the honest thing is to produce a
consistent number and let you set its useful range per mapping, rather than to
bake in constants that suit one person's hand.
"""

from __future__ import annotations

import math

import numpy as np

from ..ipc import FEATURE_INDEX, N_FEATURES

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

# A straight finger has tip-to-knuckle distance equal to the sum of its bone
# lengths (ratio 1.0); a fully closed one folds back to roughly 0.45.
CURL_STRAIGHT = 0.97
CURL_CLOSED = 0.45

PINCH_MIN = 0.20        # thumb and index touching, in hand-size units
PINCH_MAX = 1.30
SPREAD_MIN = 0.70       # index to pinky tip, fingers together
SPREAD_MAX = 2.10
SPEED_FULL = 2.5        # frame-widths per second that reads as "fast"

# A finger is either up or down; nothing in between is useful for choosing a
# chord. These two thresholds decide which, with a wide gap between them so a
# finger hovering near the boundary cannot flicker. Crossing EXTEND_ON marks it
# up and it stays up until it crosses all the way back past EXTEND_OFF.
#
# Deciding per finger and then reading the *pattern* is much more forgiving
# than testing every curl against a tight window at once: a pinky that only
# ever half-straightens still reads as up, where a scheme demanding all four
# fingers below one strict number would simply never see an open hand.
EXTEND_ON = 0.42
EXTEND_OFF = 0.62
FOUR = ("index", "middle", "ring", "pinky")


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _norm(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clamp01((v - lo) / (hi - lo))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class FeatureExtractor:
    """Stateful because smoothing, velocity and pose hysteresis all need history."""

    def __init__(self, smoothing: float = 0.35, pose_frames: int = 3):
        self.smoothing = float(min(max(smoothing, 0.0), 0.95))
        self.pose_frames = max(int(pose_frames), 1)

        self.values = np.zeros(N_FEATURES, dtype=np.float32)
        self._smoothed = np.zeros(N_FEATURES, dtype=np.float32)
        self._have_history = False
        self._prev_xy = np.zeros(2, dtype=np.float64)
        self._prev_t: float | None = None
        self._pose_counts: dict[str, int] = {}
        self._extended: dict[str, bool] = {name: False for name in FINGERS}

        # Features that must never be smoothed: booleans and diagnostics.
        self._discrete = {
            FEATURE_INDEX[n]
            for n in FEATURE_INDEX
            if n.startswith("pose_") or n in ("present", "fps")
        }

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._have_history = False
        self._prev_t = None
        self._pose_counts.clear()
        for name in self._extended:
            self._extended[name] = False

    def no_hand(self, fps: float) -> np.ndarray:
        """Emit a vector for "nothing detected", holding shape values steady.

        Only ``present`` and the pose flags drop to zero. Everything else keeps
        its last value, which is what makes the mapper's ``when: present`` gate
        able to freeze a parameter cleanly instead of lurching.
        """
        self.values[:] = self._smoothed
        self.values[FEATURE_INDEX["present"]] = 0.0
        for name, i in FEATURE_INDEX.items():
            if name.startswith("pose_"):
                self.values[i] = 0.0
        self.values[FEATURE_INDEX["fps"]] = fps
        self._pose_counts.clear()
        self._prev_t = None
        return self.values

    # ------------------------------------------------------------------

    def extract(
        self,
        image_lm: np.ndarray,
        world_lm: np.ndarray | None,
        aspect: float,
        now: float,
        fps: float,
    ) -> np.ndarray:
        """``image_lm`` is (21,3) normalised to the frame; ``world_lm`` is (21,3)
        in metres, centred on the hand."""
        raw = np.zeros(N_FEATURES, dtype=np.float32)
        f = FEATURE_INDEX

        # --- shape, from world landmarks where available -------------------
        shape_lm = world_lm if world_lm is not None else self._isotropic(image_lm, aspect)
        scale = _dist(shape_lm[WRIST], shape_lm[MIDDLE_MCP])
        if scale < 1e-6:
            scale = 1e-6

        curls = {}
        for name, (mcp, pip, dip, tip) in FINGERS.items():
            span = _dist(shape_lm[mcp], shape_lm[tip])
            bones = (
                _dist(shape_lm[mcp], shape_lm[pip])
                + _dist(shape_lm[pip], shape_lm[dip])
                + _dist(shape_lm[dip], shape_lm[tip])
            )
            ratio = span / bones if bones > 1e-9 else 1.0
            # 1 = fully curled, 0 = straight
            curls[name] = 1.0 - _norm(ratio, CURL_CLOSED, CURL_STRAIGHT)
            raw[f[f"curl_{name}"]] = curls[name]

        # Openness ignores the thumb: it folds differently and dominates the
        # average in a way that does not match what the hand looks like.
        four = (curls["index"] + curls["middle"] + curls["ring"] + curls["pinky"]) / 4.0
        raw[f["openness"]] = 1.0 - four

        raw[f["pinch"]] = _norm(
            _dist(shape_lm[THUMB_TIP], shape_lm[INDEX_TIP]) / scale, PINCH_MIN, PINCH_MAX
        )
        raw[f["spread"]] = _norm(
            _dist(shape_lm[INDEX_TIP], shape_lm[PINKY_TIP]) / scale, SPREAD_MIN, SPREAD_MAX
        )

        # --- position and orientation, from image landmarks ----------------
        cx = float(image_lm[MIDDLE_MCP, 0])
        cy = float(image_lm[MIDDLE_MCP, 1])
        raw[f["x"]] = _clamp01(cx)
        raw[f["y"]] = _clamp01(cy)

        img_scale = _dist(
            self._iso_point(image_lm[WRIST], aspect),
            self._iso_point(image_lm[MIDDLE_MCP], aspect),
        )
        # Apparent size: bigger = closer. 0.35 of the frame height is about as
        # close as you get before the hand leaves the shot.
        raw[f["size"]] = _norm(img_scale, 0.05, 0.35)

        # Roll: the knuckle line's angle on screen.
        dx = (image_lm[PINKY_MCP, 0] - image_lm[INDEX_MCP, 0]) * aspect
        dy = image_lm[PINKY_MCP, 1] - image_lm[INDEX_MCP, 1]
        raw[f["roll"]] = _clamp01((math.atan2(dy, dx) + math.pi) / (2.0 * math.pi))

        # Tilt: wrist-to-knuckles axis, 1 = fingers up.
        tx = (image_lm[MIDDLE_MCP, 0] - image_lm[WRIST, 0]) * aspect
        ty = image_lm[MIDDLE_MCP, 1] - image_lm[WRIST, 1]
        raw[f["tilt"]] = _clamp01(0.5 - math.atan2(ty, abs(tx) + 1e-6) / math.pi)

        # --- velocity -------------------------------------------------------
        if self._prev_t is not None:
            dt = max(now - self._prev_t, 1e-3)
            move = math.hypot((cx - self._prev_xy[0]) * aspect, cy - self._prev_xy[1])
            raw[f["speed"]] = _clamp01(move / dt / SPEED_FULL)
        self._prev_xy[0] = cx
        self._prev_xy[1] = cy
        self._prev_t = now

        raw[f["present"]] = 1.0
        raw[f["fps"]] = fps

        # --- smoothing -------------------------------------------------------
        if self._have_history and self.smoothing > 0.0:
            a = self.smoothing
            for i in range(N_FEATURES):
                if i in self._discrete:
                    self._smoothed[i] = raw[i]
                else:
                    self._smoothed[i] = self._smoothed[i] * a + raw[i] * (1.0 - a)
        else:
            self._smoothed[:] = raw
            self._have_history = True

        # --- poses, from the smoothed curls, with hysteresis ---------------
        self._classify(self._smoothed)

        self.values[:] = self._smoothed
        return self.values

    # ------------------------------------------------------------------

    def _classify(self, v: np.ndarray) -> None:
        f = FEATURE_INDEX
        pinch = v[f["pinch"]]

        # First decide, with hysteresis, which fingers are up.
        for name in FINGERS:
            curl = float(v[f[f"curl_{name}"]])
            if self._extended[name]:
                if curl > EXTEND_OFF:
                    self._extended[name] = False
            elif curl < EXTEND_ON:
                self._extended[name] = True

        e = self._extended
        up = sum(1 for name in FOUR if e[name])

        # Then read the pattern. These six are mutually exclusive by
        # construction -- no two can be true at once -- which is what lets
        # several of them latch the same parameter without fighting.
        candidates = {
            "pose_fist": up == 0 and not e["thumb"],
            "pose_thumbup": up == 0 and e["thumb"],
            "pose_point": up == 1 and e["index"],
            "pose_peace": up == 2 and e["index"] and e["middle"],
            # Three or four counts as an open hand: the pinky is the finger
            # people straighten least, and losing it should not cost the pose.
            "pose_open": up >= 3 and e["index"],
            # The OK ring: index folded down to meet the thumb, rest up.
            "pose_ok": pinch < 0.18 and not e["index"] and up >= 2,
        }

        # Require the same answer for a few consecutive frames before acting.
        # Without this, a hand passing through a shape on its way somewhere else
        # fires a toggle you did not ask for.
        for name, hit in candidates.items():
            count = self._pose_counts.get(name, 0)
            count = min(count + 1, self.pose_frames) if hit else 0
            self._pose_counts[name] = count
            v[FEATURE_INDEX[name]] = 1.0 if count >= self.pose_frames else 0.0

    @staticmethod
    def _iso_point(p: np.ndarray, aspect: float) -> np.ndarray:
        return np.array([p[0] * aspect, p[1]], dtype=np.float64)

    @staticmethod
    def _isotropic(lm: np.ndarray, aspect: float) -> np.ndarray:
        out = lm.copy()
        out[:, 0] *= aspect
        return out
