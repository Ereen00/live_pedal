"""Measure the real round-trip latency of your rig.

The driver's own figure is a promise, not a measurement. This plays a short
chirp, records what comes back, and cross-correlates the two to find the true
delay from "sample leaves the computer" to "sample returns".

    You need a physical loopback: a cable from an output back into an input,
    or just hold the guitar cable's plug against the output jack. The signal
    only has to be audible to the input, not clean.

    python tools/measure_latency.py
    python tools/measure_latency.py --blocksize 128 --hostapi wasapi

The number it prints is what you actually feel when you play. Anything under
about 10 ms is transparent; 10-20 ms is playable but noticeable on percussive
attacks; above 25 ms fights you.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_pedal.audio import devices                           # noqa: E402


def make_chirp(sr: int, ms: float = 6.0) -> np.ndarray:
    n = int(sr * ms / 1000.0)
    t = np.arange(n) / sr
    f0, f1 = 800.0, 6000.0
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * ms / 1000.0) * t**2)
    window = np.hanning(n)
    return (np.sin(phase) * window * 0.6).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samplerate", type=int, default=48000)
    ap.add_argument("--blocksize", type=int, default=256)
    ap.add_argument("--hostapi", default="auto")
    ap.add_argument("--input-device", default="USB AUDIO")
    ap.add_argument("--output-device", default="USB AUDIO")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=0.6)
    args = ap.parse_args()

    choice = devices.resolve(
        input_spec=args.input_device,
        output_spec=args.output_device,
        hostapi=args.hostapi,
        samplerate=args.samplerate,
    )
    print(choice.summary())
    print()

    din = sd.query_devices(choice.input_index)
    dout = sd.query_devices(choice.output_index)
    ch_in = max(1, min(int(din["max_input_channels"]), 2))
    ch_out = max(1, min(int(dout["max_output_channels"]), 2))

    sr = args.samplerate
    if not devices.check_samplerate(choice, sr, ch_in, ch_out):
        sr = int(din["default_samplerate"])
        print(f"note: falling back to {sr} Hz")

    chirp = make_chirp(sr)
    total = int(sr * args.seconds)

    results = []
    for run in range(args.repeats):
        played = {"i": 0}
        recorded = np.zeros(total, dtype=np.float32)
        rec_pos = {"i": 0}
        # Let the stream settle before firing, so we do not measure start-up.
        start_at = int(sr * 0.15)

        def cb(indata, outdata, frames, t, status):
            outdata.fill(0)
            p = played["i"]
            if start_at <= p:
                off = p - start_at
                if off < len(chirp):
                    take = min(frames, len(chirp) - off)
                    outdata[:take, 0] = chirp[off : off + take]
                    for c in range(1, outdata.shape[1]):
                        outdata[:take, c] = outdata[:take, 0]
            played["i"] = p + frames

            r = rec_pos["i"]
            take = min(frames, total - r)
            if take > 0:
                recorded[r : r + take] = indata[:take, 0]
                rec_pos["i"] = r + take

        with sd.Stream(
            device=(choice.input_index, choice.output_index),
            samplerate=sr, blocksize=args.blocksize, dtype="float32",
            channels=(ch_in, ch_out), latency="low",
            extra_settings=choice.extra_settings, callback=cb,
        ):
            deadline = time.time() + args.seconds + 1.0
            while rec_pos["i"] < total and time.time() < deadline:
                time.sleep(0.01)

        peak = float(np.max(np.abs(recorded)))
        if peak < 0.002:
            print(f"  run {run+1}: nothing came back (peak {peak:.5f}) "
                  f"-- is the loopback connected and the output turned up?")
            continue

        corr = np.correlate(recorded.astype(np.float64),
                            chirp.astype(np.float64), mode="valid")
        idx = int(np.argmax(np.abs(corr)))
        delay_samples = idx - start_at
        if delay_samples <= 0:
            print(f"  run {run+1}: correlation peak at {idx}, cannot resolve")
            continue
        ms = delay_samples / sr * 1000.0
        results.append(ms)
        print(f"  run {run+1}: {ms:6.2f} ms  ({delay_samples} samples, "
              f"peak {peak:.3f})")

    if not results:
        print("\nNo usable measurement. Connect an output back to an input and "
              "make sure the level is audible.")
        return 1

    arr = np.array(results)
    block_ms = args.blocksize / sr * 1000.0
    print(f"\nround trip: median {np.median(arr):.2f} ms "
          f"(min {arr.min():.2f}, max {arr.max():.2f})")
    print(f"one block at {args.blocksize} samples / {sr} Hz = {block_ms:.2f} ms")
    print(f"that is about {np.median(arr)/block_ms:.1f} block periods of "
          f"buffering end to end")

    med = float(np.median(arr))
    if med < 10:
        print("\nTransparent. You will not feel this.")
    elif med < 20:
        print("\nPlayable, but you may notice it on hard attacks. "
              "Try a smaller --blocksize, or ASIO if you are not on it.")
    else:
        print("\nHigh enough to fight you. Install an ASIO driver "
              "(or ASIO4ALL) and re-run with --hostapi asio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
