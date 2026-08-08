"""Routes gesture features to effect parameters.

A mapping is one line of YAML saying "this hand measurement drives this knob".
It runs in the vision process at camera rate, so clarity matters far more than
speed here -- the audio thread never sees this code.

    mappings:
      - source: openness          # a name from ipc.FEATURE_NAMES
        target: wah.freq          # <effect name>.<parameter>
        lo: 320                   # value when the source reads 0
        hi: 2800                  # value when the source reads 1
        curve: scurve             # how the input is shaped on the way
        scale: log                # how the output range is traversed

Three modes:

    continuous  (default) interpolate the output across the range
    gate        output ``hi`` while the source is above ``threshold``, else ``lo``
    toggle      flip between ``lo`` and ``hi`` on each rising edge

By default every mapping is gated on ``present``: when the tracked hand leaves
the frame the parameter freezes at its last value instead of snapping. That is
almost always what you want while playing -- you drop your hand to the strings
and the tone stays where you left it. Set ``when: always`` to opt out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..ipc import FEATURE_INDEX, FEATURE_NAMES
from ..layout import ParamLayout

CURVES = ("lin", "square", "sqrt", "scurve")
SCALES = ("lin", "log")
MODES = ("continuous", "gate", "toggle", "latch")

# Modes that only write on an edge. Several of these can share one target
# without fighting -- that is exactly how a set of poses selects one chord.
EDGE_MODES = ("toggle", "latch")


class MappingError(ValueError):
    """Raised for a malformed mapping, with a message aimed at the person
    editing the YAML rather than at a developer."""


@dataclass
class _Rule:
    source: int
    target: int
    target_name: str
    source_name: str
    out_lo: float
    out_hi: float
    in_lo: float
    in_hi: float
    curve: str
    scale: str
    invert: bool
    mode: str
    threshold: float
    gate: int                 # feature index to gate on, -1 for none
    smooth: float
    # toggle state
    latched: bool = False
    armed: bool = True
    value: float = field(default=0.0)
    initialised: bool = False


def _shape(u: float, curve: str) -> float:
    if curve == "square":
        return u * u
    if curve == "sqrt":
        return float(np.sqrt(u))
    if curve == "scurve":
        return u * u * (3.0 - 2.0 * u)
    return u


class Mapper:
    def __init__(self, mappings: list[dict], layout: ParamLayout):
        self.layout = layout
        self.rules: list[_Rule] = [self._compile(m, i) for i, m in enumerate(mappings)]
        self._targets = np.array(layout.defaults, dtype=np.float64)
        self.warnings: list[str] = self._check_conflicts()

    def _check_conflicts(self) -> list[str]:
        """Flag parameters driven by more than one rule.

        Rules are applied in order, so two rules on one parameter means the
        later one wins every frame and the earlier one silently does nothing.
        That is usually a mistake -- but not always: two rules with different
        ``when`` gates are a legitimate way to change what a gesture does
        depending on hand pose. So this warns rather than refuses.
        """
        seen: dict[int, list[_Rule]] = {}
        for r in self.rules:
            seen.setdefault(r.target, []).append(r)

        out = []
        for target, rules in seen.items():
            if len(rules) < 2:
                continue
            if all(r.mode in EDGE_MODES for r in rules):
                continue          # edge-driven: they take turns, not fight
            if len({r.gate for r in rules}) == len(rules):
                continue          # each has a distinct gate: probably deliberate
            names = ", ".join(r.source_name for r in rules)
            out.append(
                f"{self.layout.names[target]} is driven by {len(rules)} mappings "
                f"({names}); '{rules[-1].source_name}' will win and the others "
                f"will have no effect."
            )
        return out

    # ------------------------------------------------------------------
    # Compilation and validation
    # ------------------------------------------------------------------

    def _compile(self, m: dict, n: int) -> _Rule:
        where = f"mapping #{n + 1}"
        if not isinstance(m, dict):
            raise MappingError(f"{where}: expected a mapping, got {type(m).__name__}")

        src = m.get("source")
        dst = m.get("target")
        if not src:
            raise MappingError(f"{where}: missing 'source'")
        if not dst:
            raise MappingError(f"{where}: missing 'target'")

        if src not in FEATURE_INDEX:
            raise MappingError(
                f"{where}: unknown source {src!r}.\n"
                f"  Available gestures: {', '.join(FEATURE_NAMES)}"
            )
        try:
            ti = self.layout.index_of(str(dst))
        except KeyError:
            raise MappingError(
                f"{where}: unknown target {dst!r}.\n"
                f"  Available parameters: {', '.join(self.layout.names)}"
            ) from None

        mode = str(m.get("mode", "continuous")).lower()
        if mode not in MODES:
            raise MappingError(f"{where}: mode must be one of {MODES}, got {mode!r}")
        curve = str(m.get("curve", "lin")).lower()
        if curve not in CURVES:
            raise MappingError(f"{where}: curve must be one of {CURVES}, got {curve!r}")

        default_scale = self.layout.curves[ti]
        scale = str(m.get("scale", default_scale)).lower()
        if scale not in SCALES:
            raise MappingError(f"{where}: scale must be one of {SCALES}, got {scale!r}")

        out_lo = float(m.get("lo", self.layout.lo[ti]))
        out_hi = float(m.get("hi", self.layout.hi[ti]))
        if scale == "log" and (out_lo <= 0.0 or out_hi <= 0.0):
            raise MappingError(
                f"{where}: scale 'log' needs lo and hi above zero "
                f"(got lo={out_lo}, hi={out_hi}). Use scale: lin instead."
            )

        in_lo = float(m.get("in_lo", 0.0))
        in_hi = float(m.get("in_hi", 1.0))
        if abs(in_hi - in_lo) < 1e-9:
            raise MappingError(f"{where}: in_lo and in_hi must differ")

        when = m.get("when", "present")
        if when in (None, False, "always", "none"):
            gate = -1
        elif when in FEATURE_INDEX:
            gate = FEATURE_INDEX[when]
        else:
            raise MappingError(
                f"{where}: 'when' must be a gesture name or 'always', got {when!r}"
            )
        # A mapping driven by 'present' itself must not be gated on it, or it
        # could never turn back on once the hand left.
        if src == "present":
            gate = -1

        smooth = float(m.get("smooth", 0.0))
        if not 0.0 <= smooth < 1.0:
            raise MappingError(f"{where}: 'smooth' must be in [0, 1), got {smooth}")

        return _Rule(
            source=FEATURE_INDEX[src],
            target=ti,
            target_name=str(dst),
            source_name=str(src),
            out_lo=out_lo,
            out_hi=out_hi,
            in_lo=in_lo,
            in_hi=in_hi,
            curve=curve,
            scale=scale,
            invert=bool(m.get("invert", False)),
            mode=mode,
            threshold=float(m.get("threshold", 0.5)),
            gate=gate,
            smooth=smooth,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _output(self, r: _Rule, u: float) -> float:
        u = min(max(u, 0.0), 1.0)
        u = _shape(u, r.curve)
        if r.invert:
            u = 1.0 - u
        if r.scale == "log":
            return float(r.out_lo * (r.out_hi / r.out_lo) ** u)
        return float(r.out_lo + u * (r.out_hi - r.out_lo))

    def update(self, features: np.ndarray, hold: bool = False) -> np.ndarray:
        """Evaluate every rule and return the full target vector.

        The returned array is owned by the Mapper and reused, so the caller
        should write it to shared memory rather than keep a reference.
        """
        if hold:
            return self._targets

        for r in self.rules:
            if r.gate >= 0 and features[r.gate] < 0.5:
                continue                       # hand gone: hold last value

            f = float(features[r.source])

            if r.mode == "continuous":
                u = (f - r.in_lo) / (r.in_hi - r.in_lo)
                val = self._output(r, u)
            elif r.mode == "gate":
                val = r.out_hi if f >= r.threshold else r.out_lo
            elif r.mode == "latch":
                # Write once, on the rising edge, then leave the parameter
                # alone. This is what lets four different poses each select a
                # chord: whichever fired last owns the value, and releasing the
                # gesture does not revert it.
                high = f >= r.threshold
                if high and r.armed:
                    r.armed = False
                    self._targets[r.target] = self.layout.clamp(r.target, r.out_hi)
                    r.value = r.out_hi
                    r.initialised = True
                elif not high:
                    r.armed = True
                continue
            else:                              # toggle
                high = f >= r.threshold
                if high and r.armed:
                    r.latched = not r.latched
                    r.armed = False
                elif not high:
                    r.armed = True             # rearm once the gesture is released
                val = r.out_hi if r.latched else r.out_lo

            if r.smooth > 0.0 and r.initialised:
                val = r.value + (val - r.value) * (1.0 - r.smooth)
            r.value = val
            r.initialised = True

            self._targets[r.target] = self.layout.clamp(r.target, val)

        return self._targets

    # ------------------------------------------------------------------

    def describe(self) -> str:
        if not self.rules:
            return "  (no mappings -- the chain will run at its configured defaults)"
        lines = []
        for r in self.rules:
            arrow = "->"
            if r.mode == "latch":
                lines.append(
                    f"  {r.source_name:<12} {arrow} {r.target_name:<26} "
                    f"= {r.out_hi:<8g} [latch]"
                )
                continue
            extra = ""
            if r.mode != "continuous":
                extra += f"  [{r.mode} @ {r.threshold:g}]"
            if r.invert:
                extra += "  [inverted]"
            if r.gate < 0:
                extra += "  [always]"
            lines.append(
                f"  {r.source_name:<12} {arrow} {r.target_name:<26} "
                f"{r.out_lo:>8.3g} .. {r.out_hi:<8.3g} "
                f"({r.curve}/{r.scale}){extra}"
            )
        return "\n".join(lines)

    @property
    def targeted_params(self) -> set[int]:
        return {r.target for r in self.rules}
