"""Benchmark and self-test the DSP chain offline -- no sound card required.

Runs every effect, checks the output is finite and sane, and reports how much of
the real-time budget each one costs. Run this after changing any kernel:

    python tools/bench_dsp.py
    python tools/bench_dsp.py --block 128 --sr 48000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_pedal.audio.chain import Chain          # noqa: E402
from live_pedal.dsp import ALL_EFFECTS            # noqa: E402


def make_signal(sr: int, block: int, n_blocks: int) -> np.ndarray:
    """A plucked-note-ish test signal: decaying harmonic stack plus noise."""
    n = block * n_blocks
    t = np.arange(n) / sr
    env = np.exp(-t * 2.0)
    sig = sum(np.sin(2 * np.pi * 110.0 * k * t) / k for k in (1, 2, 3, 5))
    return ((sig * env) * 0.3 + np.random.randn(n) * 1e-4).astype(np.float64)


def bench_one(kind_cls, sr: int, block: int, n_blocks: int, signal: np.ndarray):
    chain = Chain.from_spec([{"type": kind_cls.kind}], sr, block)
    x = np.zeros(block, dtype=np.float64)
    y = np.zeros(block, dtype=np.float64)

    # Warm up the JIT so compilation time is not counted as runtime.
    for _ in range(4):
        x[:] = signal[:block]
        chain.process(x, y)
    chain.reset()

    peak = 0.0
    bad = 0
    t0 = time.perf_counter()
    for b in range(n_blocks):
        x[:] = signal[b * block : (b + 1) * block]
        chain.process(x, y)
        peak = max(peak, float(np.max(np.abs(y))))
        if not np.all(np.isfinite(y)):
            bad += 1
    elapsed = time.perf_counter() - t0

    per_block_us = elapsed / n_blocks * 1e6
    budget_us = block / sr * 1e6
    return per_block_us, budget_us, peak, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=400)
    args = ap.parse_args()

    signal = make_signal(args.sr, args.block, args.blocks)
    budget = args.block / args.sr * 1e6
    print(f"sample rate {args.sr} Hz, block {args.block} "
          f"({budget:.0f} us of real time per block)\n")
    print(f"{'effect':<14} {'us/block':>10} {'% budget':>10} {'peak out':>10} {'nonfinite':>10}")
    print("-" * 58)

    failures = 0
    total = 0.0
    for cls in ALL_EFFECTS:
        us, budget_us, peak, bad = bench_one(cls, args.sr, args.block, args.blocks, signal)
        total += us
        flag = ""
        if bad:
            flag = "  <-- NONFINITE OUTPUT"
            failures += 1
        elif peak > 20.0:
            flag = "  <-- suspicious level"
            failures += 1
        print(f"{cls.kind:<14} {us:>10.1f} {100*us/budget_us:>9.1f}% "
              f"{peak:>10.3f} {bad:>10d}{flag}")

    print("-" * 58)
    print(f"{'ALL SERIAL':<14} {total:>10.1f} {100*total/budget:>9.1f}%")
    print("\n(Every effect enabled at once is a worst case you would not "
          "normally run;\n a realistic chain uses 4-6 of these.)")

    # Full-chain test: everything at once, to prove they compose.
    print("\n=== full chain (every effect in series) ===")
    spec = [{"type": c.kind} for c in ALL_EFFECTS]
    chain = Chain.from_spec(spec, args.sr, args.block)
    x = np.zeros(args.block, dtype=np.float64)
    y = np.zeros(args.block, dtype=np.float64)
    for _ in range(4):
        chain.process(x, y)
    chain.reset()

    worst = 0.0
    times = []
    for b in range(args.blocks):
        x[:] = signal[b * args.block : (b + 1) * args.block]
        t0 = time.perf_counter()
        chain.process(x, y)
        times.append((time.perf_counter() - t0) * 1e6)
        worst = max(worst, float(np.max(np.abs(y))))
        if not np.all(np.isfinite(y)):
            print("  NONFINITE OUTPUT in full chain")
            failures += 1
            break
    times_arr = np.array(times)
    print(f"  params: {chain.n_params}")
    print(f"  median {np.median(times_arr):.0f} us | p95 {np.percentile(times_arr, 95):.0f} us "
          f"| max {times_arr.max():.0f} us | budget {budget:.0f} us")
    print(f"  peak output {worst:.3f}")

    if failures:
        print(f"\nFAILED: {failures} problem(s)")
        return 1
    print("\nAll effects produced finite, sane output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
