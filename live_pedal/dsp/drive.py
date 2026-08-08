"""Overdrive / distortion / fuzz.

Signal path: input gain into the nonlinearity (optionally 4x oversampled), then
a DC blocker, then a one-knob tone lowpass, then output level.

The DC blocker is not optional. The asymmetric curves deliberately produce a DC
offset -- that asymmetry is where the even-harmonic warmth comes from -- and DC
riding into a delay or reverb downstream eats headroom and makes everything
sound choked.
"""

from __future__ import annotations

import numpy as np

from . import kernels as K
from .base import Effect, ParamSpec, db_to_lin

# Waveshaper curves, matching the ``kind`` selector in kernels._shape.
SHAPES = ("soft", "hard", "fuzz", "tube", "fold")


class Drive(Effect):
    kind = "drive"
    PARAMS = (
        ParamSpec("gain_db", 12.0, 0.0, 42.0, smooth_ms=25.0, unit="dB"),
        ParamSpec("tone", 0.6, 0.0, 1.0, smooth_ms=30.0),
        ParamSpec("level", 0.5, 0.0, 1.5, smooth_ms=25.0),
        ParamSpec("asym", 0.0, -0.4, 0.4, smooth_ms=30.0),
        # Discrete: 0=soft 1=hard 2=fuzz 3=tube 4=fold
        ParamSpec("shape", 0.0, 0.0, 4.0, smooth_ms=0.0),
        # Discrete: 1 = 4x oversampled (clean), 0 = raw (cheaper, grittier)
        ParamSpec("oversample", 1.0, 0.0, 1.0, smooth_ms=0.0),
    )

    TONE_LO = 700.0
    TONE_HI = 9000.0

    def prepare(self, sr: int, block: int) -> None:
        super().prepare(sr, block)
        self._w = np.zeros(block, dtype=np.float64)
        self._fir = K.OS_FIR
        self._up_hist = np.zeros(K.OS_PHASE, dtype=np.float64)
        self._down_hist = np.zeros(K.OS_TAPS, dtype=np.float64)
        self._os_state = np.zeros(2, dtype=np.float64)
        self._dc_state = np.zeros(1, dtype=np.float64)
        self._tone_state = np.zeros(1, dtype=np.float64)

    def reset(self) -> None:
        self._up_hist[:] = 0.0
        self._down_hist[:] = 0.0
        self._os_state[:] = 0.0
        self._dc_state[:] = 0.0
        self._tone_state[:] = 0.0

    def _tone_hz(self, t: float) -> float:
        t = min(max(t, 0.0), 1.0)
        return float(self.TONE_LO * (self.TONE_HI / self.TONE_LO) ** t)

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        ga, gb = self._ab("gain_db")
        da, db_ = db_to_lin(ga), db_to_lin(gb)
        shape = min(max(self._i("shape"), 0), 4)
        asym = self._v("asym")

        if self._v("oversample") >= 0.5:
            K.drive_os4(
                x, self._w, shape, da, db_, asym,
                self._fir, self._up_hist, self._down_hist, self._os_state,
            )
        else:
            K.drive_simple(x, self._w, shape, da, db_, asym)

        # DC blocker at ~18 Hz.
        g_dc = self._tan_g(18.0)
        K.svf_1pole(self._w, self._w, g_dc, g_dc, 1, self._dc_state)

        # Tone: single lowpass, the way a real one-knob overdrive works.
        ta, tb = self._ab("tone")
        K.svf_1pole(
            self._w, self._w,
            self._tan_g(self._tone_hz(ta)), self._tan_g(self._tone_hz(tb)),
            0, self._tone_state,
        )

        la, lb = self._ab("level")
        K.gain_ramp(self._w, y, la, lb)
