"""The effect chain and its parameter table.

All automatable values for the whole chain live in four parallel float64 arrays:

    target   what the mapper wants (written by the gesture layer)
    current  the smoothed value at the end of this block
    prev     the smoothed value at the end of the previous block
    coeff    per-parameter one-pole smoothing coefficient

Effects read ``prev``/``current`` through numpy views (``pa``/``pb``) handed to
them at build time, and kernels interpolate between the two across the block.
Because the views are created once and the arrays are never reallocated, the
whole thing runs without a single allocation per block.

Each effect additionally gets one implicit ``<name>.enable`` parameter. It is a
smoothed 0..1 value, not a boolean, so an effect can be crossfaded in by a
gesture -- half-engaged wah is a real and useful sound -- and so toggling one
never clicks.
"""

from __future__ import annotations

import time

import numpy as np

from ..dsp import EFFECTS, Effect, ParamSpec
from ..dsp.kernels import crossfade_ramp, smooth_params
from ..layout import ParamLayout

ENABLE_SPEC = ParamSpec("enable", 1.0, 0.0, 1.0, smooth_ms=15.0)

# Discrete selector parameters are far more readable as words than as magic
# numbers, so YAML may spell them either way.
ENUMS: dict[str, tuple[str, ...]] = {
    "shape": ("soft", "hard", "fuzz", "tube", "fold"),      # drive only
    "mode": ("lp", "bp", "hp", "notch", "peak"),
    "lfo": ("sine", "triangle", "square", "ramp"),
    "wave": ("saw", "square", "sine", "triangle"),          # chord pad
}
LFO_KINDS = ("tremolo", "chorus", "flanger", "vibrato")


def resolve_value(kind: str, key: str, value) -> float:
    """Turn a YAML scalar into the float the parameter table expects."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    table: tuple[str, ...] | None = None
    if key == "shape":
        table = ENUMS["lfo"] if kind in LFO_KINDS else ENUMS["shape"]
    elif key == "mode":
        table = ENUMS["mode"]
    elif key == "wave":
        table = ENUMS["wave"]
    if table and text in table:
        return float(table.index(text))
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"cannot interpret {kind}.{key} = {value!r}"
            + (f"; expected one of {table}" if table else "")
        ) from exc


class Chain:
    def __init__(self, effects: list[Effect], sr: int, block: int):
        self.effects = effects
        self.sr = sr
        self.block = block

        # --- lay out the parameter table -----------------------------------
        self.param_names: list[str] = []
        self.param_specs: list[ParamSpec] = []
        self._enable_idx: list[int] = []
        self._slices: list[tuple[int, int]] = []

        cursor = 0
        for fx in effects:
            start = cursor
            for spec in fx.PARAMS:
                self.param_names.append(f"{fx.name}.{spec.name}")
                self.param_specs.append(spec)
                cursor += 1
            self._slices.append((start, cursor))
            self.param_names.append(f"{fx.name}.enable")
            self.param_specs.append(ENABLE_SPEC)
            self._enable_idx.append(cursor)
            cursor += 1

        n = cursor
        self.n_params = n
        self.index = {name: i for i, name in enumerate(self.param_names)}

        self.target = np.zeros(n, dtype=np.float64)
        self.current = np.zeros(n, dtype=np.float64)
        self.prev = np.zeros(n, dtype=np.float64)
        self.coeff = np.zeros(n, dtype=np.float64)
        self._lo = np.array([s.lo for s in self.param_specs], dtype=np.float64)
        self._hi = np.array([s.hi for s in self.param_specs], dtype=np.float64)

        block_period = block / sr
        for i, spec in enumerate(self.param_specs):
            if spec.smooth_ms <= 0.0:
                self.coeff[i] = 0.0          # snap: discrete selectors
            else:
                tau = spec.smooth_ms * 0.001
                self.coeff[i] = float(np.exp(-block_period / tau))

        # --- defaults ------------------------------------------------------
        for fx, (a, b) in zip(effects, self._slices):
            self.target[a:b] = fx.defaults()
            self.target[b] = float(fx.overrides.get("enable", 1.0))
        self.current[:] = self.target
        self.prev[:] = self.target

        # --- wire effects to their views, then let them allocate ------------
        # Views are attached first because prepare() may read its own defaults
        # -- the chord pad, for instance, builds its oscillator table from the
        # unison and detune settings.
        for fx, (a, b) in zip(effects, self._slices):
            fx.pa = self.prev[a : b + 1]
            fx.pb = self.current[a : b + 1]
            fx.prepare(sr, block)

        # --- ping-pong work buffers ----------------------------------------
        self._a = np.zeros(block, dtype=np.float64)
        self._b = np.zeros(block, dtype=np.float64)
        self._norm = np.zeros(n, dtype=np.float32)

    # ------------------------------------------------------------------
    # Construction from config
    # ------------------------------------------------------------------

    @classmethod
    def from_spec(cls, spec: list[dict], sr: int, block: int) -> "Chain":
        """Build from a list of ``{type, name, params...}`` dicts."""
        effects: list[Effect] = []
        used: set[str] = set()
        for entry in spec:
            kind = entry.get("type")
            if kind not in EFFECTS:
                raise ValueError(
                    f"unknown effect type {kind!r}; known types: {sorted(EFFECTS)}"
                )
            name = entry.get("name") or kind
            if name in used:
                raise ValueError(f"duplicate effect name {name!r} in chain")
            used.add(name)

            effect_cls = EFFECTS[kind]
            param_names = {s.name for s in effect_cls.PARAMS}
            setup_keys = set(effect_cls.SETUP_KEYS)

            overrides: dict[str, float] = {}
            setup: dict = {}
            for k, v in entry.items():
                if k in ("type", "name"):
                    continue
                if k in param_names:
                    overrides[k] = resolve_value(kind, k, v)
                elif k in setup_keys:
                    # Not a number: chord tables and the like.
                    setup[k] = v
                else:
                    known = sorted(param_names | setup_keys)
                    raise ValueError(
                        f"effect {name!r} ({kind}) has no option {k!r}.\n"
                        f"  Valid options: {', '.join(known)}"
                    )

            effects.append(effect_cls(name=name, setup=setup, **overrides))
        return cls(effects, sr, block)

    # ------------------------------------------------------------------
    # Real-time path
    # ------------------------------------------------------------------

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        """Run the chain. ``x`` and ``y`` are float64 blocks of length ``block``.

        Allocation-free: the only Python-level work per block is the loop over
        effects and a handful of scalar reads.
        """
        self.prev[:] = self.current
        smooth_params(self.current, self.target, self.coeff)

        a = self._a
        b = self._b
        a[:] = x

        for fx, ei in zip(self.effects, self._enable_idx):
            ea = self.prev[ei]
            eb = self.current[ei]
            if ea < 1e-4 and eb < 1e-4:
                continue                       # fully bypassed: skip the work
            fx.process(a, b)
            if ea > 0.9999 and eb > 0.9999:
                a, b = b, a                    # fully wet: swap, no crossfade
            else:
                crossfade_ramp(b, a, b, ea, eb)
                a, b = b, a

        y[:] = a
        # Keep the ping-pong buffers attached to the object; which physical
        # array ended up as `a` depends on how many swaps happened.
        self._a = a
        self._b = b

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def set(self, name: str, value: float) -> None:
        i = self.index[name]
        spec = self.param_specs[i]
        self.target[i] = spec.clamp(float(value))

    def get(self, name: str) -> float:
        return float(self.current[self.index[name]])

    def set_normalised(self, name: str, u: float) -> None:
        i = self.index[name]
        self.target[i] = self.param_specs[i].denormalise(u)

    def snapshot_normalised(self) -> np.ndarray:
        """0..1 view of every current value, for the overlay display."""
        for i, spec in enumerate(self.param_specs):
            self._norm[i] = spec.normalise(self.current[i])
        return self._norm

    def warmup(self, verbose: bool = False) -> float:
        """Force every JIT kernel to compile before the stream opens.

        Numba compiles on first call with a given argument type. The first call
        of a kernel costs anywhere from milliseconds to a second; inside the
        audio callback that is a guaranteed dropout, and on a strict driver it
        kills the stream outright.

        Warming up with only the current settings is not enough. An effect that
        is switched off is skipped entirely, and a branch that is not taken --
        the non-oversampled drive path, a filter mode you are not using, the
        octave-up mixer -- stays uncompiled until the moment a gesture selects
        it, mid-performance. So this walks every effect through every discrete
        setting it has, and pushes the continuous mix/level parameters off zero
        so the guarded branches actually execute.

        Returns how long it took, in seconds.
        """
        t0 = time.perf_counter()
        saved_current = self.current.copy()
        saved_target = self.target.copy()
        saved_prev = self.prev.copy()

        rng = np.random.default_rng(0)
        x = (rng.standard_normal(self.block) * 0.1).astype(np.float64)
        y = np.zeros(self.block, dtype=np.float64)

        for fx, (a, b) in zip(self.effects, self._slices):
            for variant in self._variants(fx, a, b):
                self.current[a:b] = variant
                self.prev[a:b] = variant
                for _ in range(2):
                    fx.process(x, y)
            # Event-gated effects need a push to reach their real work.
            fx.warmup_hook(x, y)
            if verbose:
                print(f"    warmed {fx.name}")

        self.current[:] = saved_current
        self.prev[:] = saved_prev
        self.target[:] = saved_target

        # Now the chain-level path: partial enables exercise crossfade_ramp,
        # full enables exercise the swap path.
        for enable in (0.5, 1.0):
            for ei in self._enable_idx:
                self.prev[ei] = enable
                self.current[ei] = enable
            for _ in range(2):
                self.process(x, y)

        self.current[:] = saved_current
        self.prev[:] = saved_prev
        self.target[:] = saved_target
        self.reset()
        return time.perf_counter() - t0

    def _variants(self, fx: Effect, a: int, b: int) -> list[np.ndarray]:
        """Parameter sets that between them take every branch in ``fx``."""
        base = self.current[a:b].copy()
        specs = fx.PARAMS

        # Push anything that gates a branch off zero, so guarded code runs.
        engaged = base.copy()
        for i, spec in enumerate(specs):
            if spec.name in ("mix", "level", "dry", "down", "up", "depth",
                             "depth_ms", "feedback", "predelay_ms",
                             "low_db", "mid_db", "high_db"):
                engaged[i] = spec.clamp(0.5 * (spec.lo + spec.hi))
                if spec.lo < 0.0 < spec.hi:
                    engaged[i] = spec.hi * 0.5

        variants = [base, engaged]

        # Every value of every discrete selector.
        for i, spec in enumerate(specs):
            if spec.smooth_ms == 0.0 and spec.hi > spec.lo and spec.hi - spec.lo <= 12:
                for v in range(int(spec.lo), int(spec.hi) + 1):
                    var = engaged.copy()
                    var[i] = float(v)
                    variants.append(var)
        return variants

    def layout(self) -> ParamLayout:
        """Plain-data description of the table, for the vision process."""
        return ParamLayout(
            names=list(self.param_names),
            lo=[s.lo for s in self.param_specs],
            hi=[s.hi for s in self.param_specs],
            defaults=[float(v) for v in self.target],
            curves=[s.curve for s in self.param_specs],
            units=[s.unit for s in self.param_specs],
        )

    def reset(self) -> None:
        for fx in self.effects:
            fx.reset()
        self._a[:] = 0.0
        self._b[:] = 0.0

    def describe(self) -> str:
        lines = []
        for fx, (a, b) in zip(self.effects, self._slices):
            lines.append(f"  {fx.name} ({fx.kind})")
            for i in range(a, b):
                spec = self.param_specs[i]
                unit = f" {spec.unit}" if spec.unit else ""
                lines.append(
                    f"      {self.param_names[i]:<28} = {self.current[i]:>9.3f}{unit}"
                    f"   [{spec.lo:g} .. {spec.hi:g}]"
                )
            lines.append(f"      {self.param_names[b]:<28} = {self.current[b]:>9.3f}")
        return "\n".join(lines)
