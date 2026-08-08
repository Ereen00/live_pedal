"""Utility processors."""

from __future__ import annotations

import numpy as np

from . import kernels as K
from .base import Effect, ParamSpec


class Volume(Effect):
    """Gesture-controlled expression/volume pedal.

    Put one of these at the end of the chain and map it to palm height for
    swells, or at the front to push the drive harder without changing overall
    level.
    """

    kind = "volume"
    PARAMS = (
        ParamSpec("gain", 1.0, 0.0, 2.0, smooth_ms=20.0),
    )

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        ga, gb = self._ab("gain")
        K.gain_ramp(x, y, ga, gb)


class Tone(Effect):
    """Simple tilt tone control: one knob, dark to bright."""

    kind = "tone"
    PARAMS = (
        ParamSpec("tone", 0.5, 0.0, 1.0, smooth_ms=30.0),
    )

    LO = 300.0
    HI = 12000.0

    def prepare(self, sr: int, block: int) -> None:
        super().prepare(sr, block)
        self._state = np.zeros(1, dtype=np.float64)

    def reset(self) -> None:
        self._state[:] = 0.0

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        ta, tb = self._ab("tone")

        def hz(t: float) -> float:
            t = min(max(t, 0.0), 1.0)
            return self.LO * (self.HI / self.LO) ** t

        K.svf_1pole(x, y, self._tan_g(hz(ta)), self._tan_g(hz(tb)), 0, self._state)
