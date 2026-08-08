"""Pitch effects: analog-style octaver and a delay-line pitch shifter.

A warning that belongs in the code and not just the README: these two work in
completely different ways and have completely different costs.

``Octaver`` is the analog trick -- rectification for the upper octave, a
zero-crossing flip-flop for the lower one. It adds **zero latency**, costs
almost nothing, tracks single notes well and turns to mush on chords. That is
not a bug; the hardware it imitates behaves identically.

``PitchShift`` is a real (if simple) shifter and can transpose anything, but it
works by crossfading two read pointers through a delay line, so it inherently
adds about half its window length in latency and warbles on sustained material.
Leave it out of the chain if you are chasing the lowest possible round trip.
"""

from __future__ import annotations

import numpy as np

from . import kernels as K
from .base import Effect, ParamSpec


class Octaver(Effect):
    kind = "octaver"
    PARAMS = (
        ParamSpec("dry", 1.0, 0.0, 1.5, smooth_ms=40.0),
        ParamSpec("down", 0.6, 0.0, 1.5, smooth_ms=40.0),
        ParamSpec("up", 0.0, 0.0, 1.5, smooth_ms=40.0),
        # How aggressively to lowpass before the divider. Lower tracks better on
        # the low strings; higher keeps more bite on the high ones.
        ParamSpec("track_hz", 900.0, 200.0, 3000.0, smooth_ms=60.0, unit="Hz", curve="log"),
    )

    def prepare(self, sr: int, block: int) -> None:
        super().prepare(sr, block)
        self._track = np.zeros(block, dtype=np.float64)
        self._down = np.zeros(block, dtype=np.float64)
        self._up = np.zeros(block, dtype=np.float64)
        self._s_track = np.zeros(1, dtype=np.float64)
        self._s_down = np.zeros(1, dtype=np.float64)
        self._s_up = np.zeros(1, dtype=np.float64)
        self._flip = np.array([1.0, 0.0], dtype=np.float64)

    def reset(self) -> None:
        self._s_track[:] = 0.0
        self._s_down[:] = 0.0
        self._s_up[:] = 0.0
        self._flip[0] = 1.0
        self._flip[1] = 0.0

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        dry = self._v("dry")
        down_lvl = self._v("down")
        up_lvl = self._v("up")

        K.gain_ramp(x, y, dry, dry)

        if down_lvl > 1e-4:
            # Lowpass first: the flip-flop triggers on zero crossings, and
            # harmonics crossing zero would make it octave-divide erratically.
            ga, gb = (self._tan_g(v) for v in self._ab("track_hz"))
            K.svf_1pole(x, self._track, ga, gb, 0, self._s_track)
            K.octave_down(self._track, self._down, self._flip)
            # Smooth the square edges the divider produces.
            g = self._tan_g(1400.0)
            K.svf_1pole(self._down, self._down, g, g, 0, self._s_down)
            K.mix_into(y, self._down, down_lvl)

        if up_lvl > 1e-4:
            K.octave_up(x, self._up)
            # Rectification leaves a big DC term; strip it before mixing.
            g = self._tan_g(60.0)
            K.svf_1pole(self._up, self._up, g, g, 1, self._s_up)
            K.mix_into(y, self._up, up_lvl)


class PitchShift(Effect):
    kind = "pitchshift"
    MAX_WINDOW_MS = 120.0
    PARAMS = (
        ParamSpec("semitones", 7.0, -24.0, 24.0, smooth_ms=60.0, unit="st"),
        ParamSpec("window_ms", 55.0, 15.0, MAX_WINDOW_MS, smooth_ms=0.0, unit="ms"),
        ParamSpec("mix", 0.5, 0.0, 1.0, smooth_ms=60.0),
        # Quantise to semitones so a sweeping hand lands on real intervals
        # instead of gliding through microtones. 0 = continuous glide.
        ParamSpec("quantise", 1.0, 0.0, 1.0, smooth_ms=0.0),
    )

    def prepare(self, sr: int, block: int) -> None:
        super().prepare(sr, block)
        size = int(sr * self.MAX_WINDOW_MS * 0.001) * 2 + block + 8
        self._buf = np.zeros(size, dtype=np.float64)
        self._state = np.zeros(2, dtype=np.float64)
        self._wet = np.zeros(block, dtype=np.float64)

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._state[:] = 0.0

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        st = self._v("semitones")
        if self._v("quantise") >= 0.5:
            st = round(st)
        ratio = float(2.0 ** (st / 12.0))
        window = max(self._v("window_ms"), 5.0) * self.sr * 0.001

        K.pitch_shift(x, self._wet, self._buf, self._state, ratio, window)
        ma, mb = self._ab("mix")
        K.crossfade_ramp(y, x, self._wet, ma, mb)
