"""Modulation effects: tremolo, vibrato, chorus, flanger, phaser.

Chorus, flanger and vibrato are the same machine -- a delay line whose length is
wobbled by an LFO -- differing only in delay range, feedback and mix. They share
``_ModDelay`` rather than being three copies of the same code.

Every one of these has an internal LFO, but note that you can also drive the
``rate`` and ``depth`` from a gesture, or set ``rate`` to zero and sweep the
delay directly with your hand for manual flange.
"""

from __future__ import annotations

import numpy as np

from . import kernels as K
from .base import Effect, ParamSpec

# LFO waveform selector, matching kernels._lfo.
SHAPES = ("sine", "triangle", "square", "ramp")


class Tremolo(Effect):
    """Amplitude modulation."""

    kind = "tremolo"
    PARAMS = (
        ParamSpec("rate", 5.0, 0.05, 20.0, smooth_ms=40.0, unit="Hz", curve="log"),
        ParamSpec("depth", 0.6, 0.0, 1.0, smooth_ms=40.0),
        ParamSpec("shape", 0.0, 0.0, 3.0, smooth_ms=0.0),
    )

    def prepare(self, sr: int, block: int) -> None:
        super().prepare(sr, block)
        self._state = np.zeros(1, dtype=np.float64)

    def reset(self) -> None:
        self._state[:] = 0.0

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        K.tremolo(
            x, y,
            self._v("depth"),
            self._phase_inc(self._v("rate")),
            min(max(self._i("shape"), 0), 3),
            self._state,
        )


class _ModDelay(Effect):
    """Shared implementation for vibrato / chorus / flanger."""

    MAX_DELAY_MS = 120.0

    def prepare(self, sr: int, block: int) -> None:
        super().prepare(sr, block)
        size = int(sr * self.MAX_DELAY_MS * 0.001) + block + 4
        self._buf = np.zeros(size, dtype=np.float64)
        self._state = np.zeros(2, dtype=np.float64)   # [write index, lfo phase]

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._state[:] = 0.0

    def _run(self, x, y, delay_ms_a, delay_ms_b, depth_ms, rate, shape, feedback, mix):
        ms = self.sr * 0.001
        K.mod_delay(
            x, y, self._buf, self._state,
            delay_ms_a * ms, delay_ms_b * ms,
            depth_ms * ms,
            self._phase_inc(rate),
            shape,
            feedback, mix,
            0.0,
        )


class Vibrato(_ModDelay):
    """Pitch modulation: a fully wet modulated delay, no dry signal at all."""

    kind = "vibrato"
    MAX_DELAY_MS = 60.0
    PARAMS = (
        ParamSpec("rate", 5.5, 0.1, 14.0, smooth_ms=40.0, unit="Hz", curve="log"),
        ParamSpec("depth_ms", 2.0, 0.0, 12.0, smooth_ms=40.0, unit="ms"),
    )

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        depth = self._v("depth_ms")
        # Centre the sweep so the delay never goes negative.
        self._run(x, y, depth + 1.0, depth + 1.0, depth, self._v("rate"), 0, 0.0, 1.0)


class Chorus(_ModDelay):
    """Longer delay, gentle modulation, blended with dry: thickening."""

    kind = "chorus"
    MAX_DELAY_MS = 120.0
    PARAMS = (
        ParamSpec("rate", 0.8, 0.02, 8.0, smooth_ms=50.0, unit="Hz", curve="log"),
        ParamSpec("depth_ms", 3.0, 0.0, 15.0, smooth_ms=50.0, unit="ms"),
        ParamSpec("delay_ms", 18.0, 5.0, 45.0, smooth_ms=50.0, unit="ms"),
        ParamSpec("mix", 0.45, 0.0, 1.0, smooth_ms=50.0),
        ParamSpec("feedback", 0.0, -0.7, 0.7, smooth_ms=50.0),
        ParamSpec("shape", 0.0, 0.0, 3.0, smooth_ms=0.0),
    )

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        da, db_ = self._ab("delay_ms")
        self._run(
            x, y, da, db_, self._v("depth_ms"), self._v("rate"),
            min(max(self._i("shape"), 0), 3),
            self._v("feedback"), self._v("mix"),
        )


class Flanger(_ModDelay):
    """Short delay plus heavy feedback: the jet-plane sweep."""

    kind = "flanger"
    MAX_DELAY_MS = 40.0
    PARAMS = (
        ParamSpec("rate", 0.35, 0.01, 6.0, smooth_ms=50.0, unit="Hz", curve="log"),
        ParamSpec("depth_ms", 3.5, 0.0, 9.0, smooth_ms=50.0, unit="ms"),
        ParamSpec("delay_ms", 1.2, 0.2, 12.0, smooth_ms=50.0, unit="ms"),
        ParamSpec("feedback", 0.6, -0.95, 0.95, smooth_ms=50.0),
        ParamSpec("mix", 0.5, 0.0, 1.0, smooth_ms=50.0),
        ParamSpec("shape", 1.0, 0.0, 3.0, smooth_ms=0.0),
    )

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        da, db_ = self._ab("delay_ms")
        self._run(
            x, y, da, db_, self._v("depth_ms"), self._v("rate"),
            min(max(self._i("shape"), 0), 3),
            self._v("feedback"), self._v("mix"),
        )


class Phaser(Effect):
    """Cascaded allpass notches swept by an LFO.

    Unlike the delay-based effects a phaser has no delay line, so it adds zero
    latency -- worth knowing if you are chasing the lowest possible round trip
    and still want movement in the sound.
    """

    kind = "phaser"
    MAX_STAGES = 12
    PARAMS = (
        ParamSpec("rate", 0.5, 0.01, 8.0, smooth_ms=50.0, unit="Hz", curve="log"),
        ParamSpec("depth", 0.8, 0.0, 1.0, smooth_ms=50.0),
        ParamSpec("centre", 700.0, 150.0, 4000.0, smooth_ms=50.0, unit="Hz", curve="log"),
        ParamSpec("feedback", 0.4, -0.9, 0.9, smooth_ms=50.0),
        ParamSpec("mix", 0.5, 0.0, 1.0, smooth_ms=50.0),
        ParamSpec("stages", 4.0, 2.0, float(MAX_STAGES), smooth_ms=0.0),
    )

    def prepare(self, sr: int, block: int) -> None:
        super().prepare(sr, block)
        self._ap = np.zeros(self.MAX_STAGES, dtype=np.float64)
        self._fb = np.zeros(1, dtype=np.float64)
        self._phase = 0.0

    def reset(self) -> None:
        self._ap[:] = 0.0
        self._fb[:] = 0.0
        self._phase = 0.0

    def _g_for(self, hz: float) -> float:
        """Allpass coefficient placing the pole at ``hz``.

        The kernel implements H(z) = (z^-1 - g)/(1 - g*z^-1), whose 90-degree
        phase point sits at the frequency satisfying this bilinear mapping.
        """
        t = np.tan(np.pi * min(max(hz, 20.0), self.sr * 0.45) / self.sr)
        return float((1.0 - t) / (1.0 + t))

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        rate = self._v("rate")
        depth = self._v("depth")
        centre = self._v("centre")
        n = x.shape[0]

        # Advance the LFO once per block; the kernel interpolates the resulting
        # coefficient across the block so the sweep stays smooth.
        p0 = self._phase
        p1 = (self._phase + rate * n / self.sr) % 1.0
        self._phase = p1

        span = 1.0 + 2.5 * depth          # octaves of sweep either side
        f0 = centre * span ** np.sin(2.0 * np.pi * p0)
        f1 = centre * span ** np.sin(2.0 * np.pi * p1)

        K.phaser(
            x, y,
            min(max(self._i("stages"), 2), self.MAX_STAGES),
            self._g_for(f0), self._g_for(f1),
            self._v("feedback"), self._v("mix"),
            self._ap, self._fb,
        )
