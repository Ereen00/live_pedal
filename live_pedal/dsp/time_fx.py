"""Time-domain effects: feedback delay and reverb."""

from __future__ import annotations

import numpy as np

from . import kernels as K
from .base import Effect, ParamSpec


class Delay(Effect):
    """Feedback delay with damping in the repeat path.

    Sweeping ``time_ms`` with a gesture pitch-bends the repeats, because the
    read pointer is moving relative to the write pointer -- the same tape-echo
    behaviour you get from turning the delay-time knob on a real unit. That is a
    feature, and it is why the delay time is interpolated across the block
    rather than snapped.
    """

    kind = "delay"
    MAX_MS = 2000.0
    PARAMS = (
        ParamSpec("time_ms", 350.0, 10.0, MAX_MS, smooth_ms=80.0, unit="ms", curve="log"),
        ParamSpec("feedback", 0.35, 0.0, 0.95, smooth_ms=60.0),
        ParamSpec("mix", 0.3, 0.0, 1.0, smooth_ms=60.0),
        ParamSpec("damping", 0.4, 0.0, 1.0, smooth_ms=60.0),
    )

    def prepare(self, sr: int, block: int) -> None:
        super().prepare(sr, block)
        size = int(sr * self.MAX_MS * 0.001) + block + 4
        self._buf = np.zeros(size, dtype=np.float64)
        self._state = np.zeros(1, dtype=np.float64)
        self._damp_state = np.zeros(1, dtype=np.float64)

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._state[:] = 0.0
        self._damp_state[:] = 0.0

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        ms = self.sr * 0.001
        ta, tb = self._ab("time_ms")
        # Damping 0 = bright repeats, 1 = dark. Maps to a one-pole coefficient.
        damp = self._v("damping")
        g = float(np.clip(1.0 - damp * 0.97, 0.03, 1.0))
        K.delay_line(
            x, y, self._buf, self._state,
            ta * ms, tb * ms,
            self._v("feedback"), self._v("mix"),
            g, self._damp_state,
        )


class Reverb(Effect):
    """Freeverb: eight damped parallel combs into four series allpasses.

    Cheap, stable, and it adds no latency of its own -- the first sample out is
    the first sample in. Buffer lengths are the original Freeverb tuning scaled
    from its 44.1 kHz design rate to whatever the interface is running at.
    """

    kind = "reverb"
    PARAMS = (
        ParamSpec("size", 0.7, 0.0, 1.0, smooth_ms=100.0),
        ParamSpec("damping", 0.5, 0.0, 1.0, smooth_ms=100.0),
        ParamSpec("mix", 0.25, 0.0, 1.0, smooth_ms=80.0),
        ParamSpec("predelay_ms", 0.0, 0.0, 200.0, smooth_ms=0.0, unit="ms"),
    )

    def prepare(self, sr: int, block: int) -> None:
        super().prepare(sr, block)
        scale = sr / 44100.0

        self._comb_len = np.maximum((K.COMB_TUNING * scale).astype(np.int64), 8)
        self._ap_len = np.maximum((K.ALLPASS_TUNING * scale).astype(np.int64), 8)
        self._comb_buf = np.zeros(int(self._comb_len.sum()), dtype=np.float64)
        self._ap_buf = np.zeros(int(self._ap_len.sum()), dtype=np.float64)
        self._comb_idx = np.zeros(len(self._comb_len), dtype=np.int64)
        self._ap_idx = np.zeros(len(self._ap_len), dtype=np.int64)
        self._comb_store = np.zeros(len(self._comb_len), dtype=np.float64)

        pre_size = int(sr * 0.001 * 200.0) + block + 4
        self._pre_buf = np.zeros(pre_size, dtype=np.float64)
        self._pre_state = np.zeros(1, dtype=np.float64)
        self._pre_damp = np.zeros(1, dtype=np.float64)

        self._wet = np.zeros(block, dtype=np.float64)
        self._src = np.zeros(block, dtype=np.float64)

    def reset(self) -> None:
        self._comb_buf[:] = 0.0
        self._ap_buf[:] = 0.0
        self._comb_idx[:] = 0
        self._ap_idx[:] = 0
        self._comb_store[:] = 0.0
        self._pre_buf[:] = 0.0
        self._pre_state[:] = 0.0
        self._pre_damp[:] = 0.0

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        pre = self._v("predelay_ms")
        if pre > 0.5:
            d = pre * self.sr * 0.001
            K.delay_line(
                x, self._src, self._pre_buf, self._pre_state,
                d, d, 0.0, 1.0, 1.0, self._pre_damp,
            )
            src = self._src
        else:
            src = x

        # Freeverb's usable feedback range; below ~0.7 the tail is too short to
        # read as a room, above ~0.98 it never decays.
        feedback = 0.70 + 0.28 * float(np.clip(self._v("size"), 0.0, 1.0))
        damp = float(np.clip(self._v("damping"), 0.0, 1.0)) * 0.4

        K.reverb(
            src, self._wet,
            self._comb_buf, self._comb_len, self._comb_idx, self._comb_store,
            self._ap_buf, self._ap_len, self._ap_idx,
            feedback, damp, 0.0,
        )

        ma, mb = self._ab("mix")
        K.crossfade_ramp(y, x, self._wet, ma, mb)
