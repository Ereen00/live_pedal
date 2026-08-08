"""A picklable description of the chain's parameter table.

The vision process needs to know what parameters exist and what their legal
ranges are so it can turn gestures into values, but it has no reason to build
the DSP objects themselves -- that would allocate every delay line twice and
trigger a second round of JIT compilation in the child.

So the parent builds the real ``Chain``, exports this plain-data layout, and
hands it to the child through the process arguments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParamLayout:
    names: list[str]
    lo: list[float]
    hi: list[float]
    defaults: list[float]
    curves: list[str]          # "lin" or "log", the natural scaling of the value
    units: list[str]

    def __len__(self) -> int:
        return len(self.names)

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc

    def denormalise(self, i: int, u: float) -> float:
        u = min(max(u, 0.0), 1.0)
        lo, hi = self.lo[i], self.hi[i]
        if self.curves[i] == "log" and lo > 0.0:
            return float(lo * (hi / lo) ** u)
        return float(lo + u * (hi - lo))

    def normalise(self, i: int, v: float) -> float:
        lo, hi = self.lo[i], self.hi[i]
        if hi <= lo:
            return 0.0
        if self.curves[i] == "log" and lo > 0.0:
            import math

            return float(math.log(max(v, lo) / lo) / math.log(hi / lo))
        return float((v - lo) / (hi - lo))

    def clamp(self, i: int, v: float) -> float:
        return min(max(v, self.lo[i]), self.hi[i])
