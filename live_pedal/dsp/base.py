"""Effect base class and parameter description.

The design constraint that shapes everything here: the audio callback must not
allocate. Python allocation can trigger a garbage collection pause, and a pause
longer than the block period is a dropout. So every effect allocates all of its
buffers in ``prepare()`` and touches nothing but preallocated memory in
``process()``.

Parameters live in a single flat float64 table owned by the chain, not on the
effect objects. Each effect holds two numpy *views* into that table:

    self.pa -- parameter values at the start of the block
    self.pb -- parameter values at the end of the block

Kernels interpolate between them, which is what makes a parameter swept by a
moving hand sound smooth instead of stepped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ParamSpec:
    """Description of one automatable parameter.

    ``smooth_ms`` of 0 means "apply instantly" and is what discrete selector
    parameters (filter mode, clipping type) use -- interpolating those would
    produce meaningless intermediate values.
    """

    name: str
    default: float
    lo: float
    hi: float
    smooth_ms: float = 25.0
    unit: str = ""
    curve: str = "lin"          # "lin" or "log", used by the mapper and display

    def clamp(self, v: float) -> float:
        return min(max(v, self.lo), self.hi)

    def normalise(self, v: float) -> float:
        """Map a real value into 0..1 for the overlay display."""
        if self.hi <= self.lo:
            return 0.0
        if self.curve == "log" and self.lo > 0.0:
            return float(np.log(max(v, self.lo) / self.lo) / np.log(self.hi / self.lo))
        return float((v - self.lo) / (self.hi - self.lo))

    def denormalise(self, u: float) -> float:
        """Map 0..1 back to a real value, honouring the curve."""
        u = min(max(u, 0.0), 1.0)
        if self.curve == "log" and self.lo > 0.0:
            return float(self.lo * (self.hi / self.lo) ** u)
        return float(self.lo + u * (self.hi - self.lo))


class Effect:
    """Base class for every processor in the chain.

    Subclasses declare ``PARAMS``, allocate in ``prepare()`` and implement
    ``process(x, y)``. ``process`` must treat ``x`` as read-only and write its
    result to ``y``; the chain relies on ``x`` surviving so it can crossfade the
    effect in and out.
    """

    kind: str = "effect"
    PARAMS: tuple[ParamSpec, ...] = ()

    def __init__(self, name: str | None = None, **overrides: float):
        self.name = name or self.kind
        self.sr = 48000
        self.block = 256
        self.overrides = dict(overrides)
        # Filled in by the chain when the parameter table is built.
        self.pa: np.ndarray = np.zeros(len(self.PARAMS))
        self.pb: np.ndarray = np.zeros(len(self.PARAMS))
        self._idx = {s.name: i for i, s in enumerate(self.PARAMS)}

    # -- lifecycle ----------------------------------------------------------

    def prepare(self, sr: int, block: int) -> None:
        """Allocate every buffer this effect will ever need. Called once."""
        self.sr = sr
        self.block = block

    def reset(self) -> None:
        """Clear filter/delay state. Called on panic and on sample rate change."""

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    # -- parameter access ---------------------------------------------------

    def _ab(self, name: str) -> tuple[float, float]:
        """Start-of-block and end-of-block value of a parameter."""
        i = self._idx[name]
        return float(self.pa[i]), float(self.pb[i])

    def _v(self, name: str) -> float:
        """End-of-block value, for parameters that are not swept per sample."""
        return float(self.pb[self._idx[name]])

    def _i(self, name: str) -> int:
        return int(round(self.pb[self._idx[name]]))

    def defaults(self) -> np.ndarray:
        vals = np.array([s.default for s in self.PARAMS], dtype=np.float64)
        for k, v in self.overrides.items():
            if k in self._idx:
                vals[self._idx[k]] = self.PARAMS[self._idx[k]].clamp(float(v))
        return vals

    # -- helpers used by subclasses ----------------------------------------

    def _tan_g(self, hz: float) -> float:
        """Prewarped SVF coefficient for a cutoff in Hz."""
        hz = min(max(hz, 10.0), self.sr * 0.45)
        return float(np.tan(np.pi * hz / self.sr))

    def _phase_inc(self, hz: float) -> float:
        """LFO phase increment per sample for a rate in Hz."""
        return float(hz / self.sr)


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))
