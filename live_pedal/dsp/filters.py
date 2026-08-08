"""Filter-based effects: wah, sweepable filter, and a 3-band EQ.

These are the effects that respond best to continuous hand control, because a
filter sweep is something the ear reads as a single expressive gesture rather
than as a parameter change.
"""

from __future__ import annotations

import numpy as np

from . import kernels as K
from .base import Effect, ParamSpec, db_to_lin


class Wah(Effect):
    """Resonant bandpass sweep -- the classic wah voicing.

    A real wah is a bandpass with a pronounced resonant peak that rides up and
    down in frequency. Here the pedal position is whatever gesture you map to
    ``freq``. Peak gain is only partially compensated (``sqrt(k)``) so the
    resonance still sings the way the hardware does instead of being flattened
    to unity.
    """

    kind = "wah"
    PARAMS = (
        ParamSpec("freq", 800.0, 250.0, 3200.0, smooth_ms=12.0, unit="Hz", curve="log"),
        ParamSpec("q", 4.0, 1.0, 12.0, smooth_ms=40.0),
        ParamSpec("mix", 1.0, 0.0, 1.0, smooth_ms=40.0),
        ParamSpec("level", 1.0, 0.0, 2.0, smooth_ms=40.0),
    )

    def prepare(self, sr: int, block: int) -> None:
        super().prepare(sr, block)
        self._state = np.zeros(2, dtype=np.float64)
        self._wet = np.zeros(block, dtype=np.float64)

    def reset(self) -> None:
        self._state[:] = 0.0

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        fa, fb = self._ab("freq")
        qa, qb = self._ab("q")
        ka = 1.0 / max(qa, 0.5)
        kb = 1.0 / max(qb, 0.5)
        K.svf(x, self._wet, self._tan_g(fa), self._tan_g(fb), ka, kb, 1, self._state)

        # The SVF bandpass peaks at 1/k, i.e. Q. Compensating by sqrt(k) removes
        # half of that in dB terms, leaving the resonance audible as a voice
        # rather than as a volume jump.
        la, lb = self._ab("level")
        K.gain_ramp(self._wet, self._wet, np.sqrt(ka) * la, np.sqrt(kb) * lb)

        ma, mb = self._ab("mix")
        K.crossfade_ramp(y, x, self._wet, ma, mb)


class Filter(Effect):
    """General sweepable state-variable filter: LP / BP / HP / notch / peak."""

    kind = "filter"
    PARAMS = (
        ParamSpec("cutoff", 1200.0, 60.0, 14000.0, smooth_ms=12.0, unit="Hz", curve="log"),
        ParamSpec("resonance", 1.0, 0.5, 15.0, smooth_ms=40.0),
        # Discrete selector: never interpolate it.
        ParamSpec("mode", 0.0, 0.0, 4.0, smooth_ms=0.0),
        ParamSpec("mix", 1.0, 0.0, 1.0, smooth_ms=40.0),
    )

    def prepare(self, sr: int, block: int) -> None:
        super().prepare(sr, block)
        self._state = np.zeros(2, dtype=np.float64)
        self._wet = np.zeros(block, dtype=np.float64)

    def reset(self) -> None:
        self._state[:] = 0.0

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        ca, cb = self._ab("cutoff")
        ra, rb = self._ab("resonance")
        mode = min(max(self._i("mode"), 0), 4)
        K.svf(
            x, self._wet,
            self._tan_g(ca), self._tan_g(cb),
            1.0 / max(ra, 0.5), 1.0 / max(rb, 0.5),
            mode, self._state,
        )
        ma, mb = self._ab("mix")
        K.crossfade_ramp(y, x, self._wet, ma, mb)


class EQ3(Effect):
    """Three-band EQ: low shelf, peaking mid, high shelf.

    Coefficients are recomputed once per block in Python. That is ~40 flops per
    band per block -- negligible next to the sample loop -- and it means the
    bands can be swept by a gesture without needing a per-sample coefficient
    update.
    """

    kind = "eq3"
    PARAMS = (
        ParamSpec("low_db", 0.0, -18.0, 18.0, smooth_ms=50.0, unit="dB"),
        ParamSpec("mid_db", 0.0, -18.0, 18.0, smooth_ms=50.0, unit="dB"),
        ParamSpec("high_db", 0.0, -18.0, 18.0, smooth_ms=50.0, unit="dB"),
        ParamSpec("mid_freq", 800.0, 200.0, 5000.0, smooth_ms=50.0, unit="Hz", curve="log"),
        ParamSpec("mid_q", 0.9, 0.2, 6.0, smooth_ms=50.0),
    )

    LOW_FREQ = 180.0
    HIGH_FREQ = 3200.0

    def prepare(self, sr: int, block: int) -> None:
        super().prepare(sr, block)
        self._s_low = np.zeros(2, dtype=np.float64)
        self._s_mid = np.zeros(2, dtype=np.float64)
        self._s_high = np.zeros(2, dtype=np.float64)
        self._tmp = np.zeros(block, dtype=np.float64)

    def reset(self) -> None:
        self._s_low[:] = 0.0
        self._s_mid[:] = 0.0
        self._s_high[:] = 0.0

    # RBJ audio EQ cookbook coefficients.
    def _shelf(self, freq: float, gain_db: float, high: bool):
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * freq / self.sr
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / 0.9 - 1.0) + 2.0)
        tsa = 2.0 * np.sqrt(A) * alpha
        if high:
            b0 = A * ((A + 1) + (A - 1) * cw + tsa)
            b1 = -2 * A * ((A - 1) + (A + 1) * cw)
            b2 = A * ((A + 1) + (A - 1) * cw - tsa)
            a0 = (A + 1) - (A - 1) * cw + tsa
            a1 = 2 * ((A - 1) - (A + 1) * cw)
            a2 = (A + 1) - (A - 1) * cw - tsa
        else:
            b0 = A * ((A + 1) - (A - 1) * cw + tsa)
            b1 = 2 * A * ((A - 1) - (A + 1) * cw)
            b2 = A * ((A + 1) - (A - 1) * cw - tsa)
            a0 = (A + 1) + (A - 1) * cw + tsa
            a1 = -2 * ((A - 1) + (A + 1) * cw)
            a2 = (A + 1) + (A - 1) * cw - tsa
        return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0

    def _peaking(self, freq: float, gain_db: float, q: float):
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * freq / self.sr
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / (2.0 * max(q, 0.1))
        b0 = 1 + alpha * A
        b1 = -2 * cw
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cw
        a2 = 1 - alpha / A
        return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        low_db = self._v("low_db")
        mid_db = self._v("mid_db")
        high_db = self._v("high_db")

        src = x
        if abs(low_db) > 0.01:
            b0, b1, b2, a1, a2 = self._shelf(self.LOW_FREQ, low_db, high=False)
            K.biquad(src, self._tmp, b0, b1, b2, a1, a2, self._s_low)
            src = self._tmp
        if abs(mid_db) > 0.01:
            b0, b1, b2, a1, a2 = self._peaking(
                self._v("mid_freq"), mid_db, self._v("mid_q")
            )
            K.biquad(src, y, b0, b1, b2, a1, a2, self._s_mid)
            src = y
        if abs(high_db) > 0.01:
            b0, b1, b2, a1, a2 = self._shelf(self.HIGH_FREQ, high_db, high=True)
            K.biquad(src, y, b0, b1, b2, a1, a2, self._s_high)
            src = y
        if src is not y:
            y[:] = src


class Gate(Effect):
    """Noise gate with hold.

    High-gain settings amplify everything, including the hum a single-coil
    picks up between phrases. The hold stage keeps the gate open briefly after
    the signal drops below threshold so decaying notes are not truncated.
    """

    kind = "gate"
    PARAMS = (
        ParamSpec("threshold_db", -55.0, -80.0, -10.0, smooth_ms=0.0, unit="dB"),
        ParamSpec("attack_ms", 2.0, 0.1, 50.0, smooth_ms=0.0, unit="ms"),
        ParamSpec("release_ms", 120.0, 5.0, 1000.0, smooth_ms=0.0, unit="ms"),
        ParamSpec("hold_ms", 60.0, 0.0, 500.0, smooth_ms=0.0, unit="ms"),
    )

    def prepare(self, sr: int, block: int) -> None:
        super().prepare(sr, block)
        self._state = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self._state[:] = 0.0

    def _coef(self, ms: float) -> float:
        ms = max(ms, 0.01)
        return float(np.exp(-1.0 / (self.sr * ms * 0.001)))

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        K.noise_gate(
            x, y,
            db_to_lin(self._v("threshold_db")),
            self._coef(self._v("attack_ms")),
            self._coef(self._v("release_ms")),
            self._v("hold_ms") * 0.001 * self.sr,
            self._state,
        )
