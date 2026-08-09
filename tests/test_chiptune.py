"""The 8-bit guitar voice, verified offline against a synthetic guitar.

Every check here corresponds to something you can actually hear:

  - the waveform really is a pulse, not a distorted sine
  - the bit crusher really quantises, and to the number of steps asked for
  - the downsampler really holds samples, and aliases because of it
  - the pluck envelope really shortens a note that would otherwise ring
  - nothing spatial is added -- a dry signal in stays dry
  - the output stays finite and bounded at silly settings

    python tests/test_chiptune.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_pedal.audio.chain import Chain                       # noqa: E402
from live_pedal.dsp import kernels as K                        # noqa: E402

SR = 48000
BLOCK = 256

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def db(x: float) -> float:
    return -120.0 if x < 1e-9 else 20.0 * np.log10(x)


def note(n: int, freq: float = 196.0, decay: float = 1.2,
         amp: float = 0.3) -> np.ndarray:
    """A plucked note: sharp attack, a few harmonics, exponential decay."""
    t = np.arange(n) / SR
    env = np.exp(-t * decay)
    sig = sum(np.sin(2 * np.pi * freq * k * t) / k for k in (1, 2, 3))
    return (sig * env * amp).astype(np.float64)


def make_chain(**overrides) -> Chain:
    entry = {
        "type": "chiptune",
        "square": 1.0,
        "pulse_width": 0.5,
        "bits": 4,
        "crush_hz": 6000,
        "tone": 12000,
        "pluck": 0.0,
        "decay_ms": 400,
        "level": 1.0,
        "mix": 1.0,
    }
    entry.update(overrides)
    return Chain.from_spec([entry], SR, BLOCK)


def run(chain: Chain, signal: np.ndarray) -> np.ndarray:
    n_blocks = len(signal) // BLOCK
    out = np.zeros(n_blocks * BLOCK)
    x = np.zeros(BLOCK)
    y = np.zeros(BLOCK)
    for b in range(n_blocks):
        x[:] = signal[b * BLOCK : (b + 1) * BLOCK]
        chain.process(x, y)
        out[b * BLOCK : (b + 1) * BLOCK] = y
    return out


def envelope_of(x: np.ndarray, win: int = 2048) -> np.ndarray:
    n = len(x) // win
    return np.array([np.max(np.abs(x[i * win : (i + 1) * win])) for i in range(n)])


# ---------------------------------------------------------------------------


def test_squares_the_waveform() -> None:
    """A pulse wave spends its time at two levels, not spread across many."""
    print("\npulse shaping: the waveform becomes two-valued")
    sig = note(BLOCK * 200, decay=0.0)         # steady, so the envelope is flat
    seg = slice(BLOCK * 100, BLOCK * 160)

    plain = run(make_chain(square=0.0, bits=16, crush_hz=24000), sig)[seg]
    pulsed = run(make_chain(square=1.0, bits=16, crush_hz=24000), sig)[seg]

    # RMS over peak is the cleanest single number: a square wave spends all
    # its time at full amplitude and scores 1.0, a sine scores 0.707, and a
    # guitar with its high crest factor scores lower still. Comparing against
    # a fixed level would not work here because the pulse tracks the note's
    # envelope, so "full amplitude" is a moving target.
    def flatness(x: np.ndarray) -> float:
        peak = float(np.max(np.abs(x)))
        return 0.0 if peak < 1e-9 else float(np.sqrt(np.mean(x * x)) / peak)

    a, b = flatness(plain), flatness(pulsed)
    check("untouched guitar has a peaky waveform", a < 0.6, f"rms/peak {a:.2f}")
    check("pulsed guitar is close to a square", b > 0.85, f"rms/peak {b:.2f}")

    # And it should be far richer in harmonics than the input.
    def brightness(x: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1 / SR)
        return float(spec[freqs > 2000].sum() / max(spec.sum(), 1e-12))

    check("pulse wave is much brighter than the guitar",
          brightness(pulsed) > brightness(sig[seg]) * 3.0,
          f"{brightness(pulsed):.3f} vs {brightness(sig[seg]):.3f}")


def test_pulse_width_changes_the_tone() -> None:
    """Off-centre duty makes the wave asymmetric -- the thin, nasal sound."""
    print("\npulse width: duty cycle moves off 50%")
    sig = note(BLOCK * 200, decay=0.0)
    seg = slice(BLOCK * 100, BLOCK * 160)

    duties = {}
    for pw in (0.5, 0.25):
        out = run(make_chain(pulse_width=pw, bits=16, crush_hz=24000), sig)[seg]
        duties[pw] = float(np.mean(out > 0.0))

    check("0.5 is a symmetric square", abs(duties[0.5] - 0.5) < 0.08,
          f"{duties[0.5] * 100:.0f}% high")
    check("0.25 is visibly narrower", duties[0.25] < duties[0.5] - 0.12,
          f"{duties[0.25] * 100:.0f}% high vs {duties[0.5] * 100:.0f}%")


def crush_only(sig: np.ndarray, bits: int, crush_hz: float,
               square: float = 0.0, pw: float = 0.5) -> np.ndarray:
    """Run the crusher kernel alone.

    The quantisation grid and the held plateaus are both destroyed by the DC
    blocker and the tone filter that follow in the real chain, so measuring
    them at the effect's output would only ever measure those filters. This
    checks the stage that is actually supposed to do the work.
    """
    out = np.zeros(len(sig) // BLOCK * BLOCK)
    state = np.zeros(3)
    levels = 2.0 ** (bits - 1)
    step = max(SR / crush_hz, 1.0)
    y = np.zeros(BLOCK)
    for b in range(len(out) // BLOCK):
        x = np.ascontiguousarray(sig[b * BLOCK : (b + 1) * BLOCK])
        K.chip_crush(x, y, square, square, pw, levels, step,
                     1 - np.exp(-1 / (SR * 0.001)),
                     1 - np.exp(-1 / (SR * 0.030)), state)
        out[b * BLOCK : (b + 1) * BLOCK] = y
    return out


def test_bit_depth() -> None:
    """The crusher must land on 2**(bits-1) steps, not merely sound rough."""
    print("\nbit crusher: the signal lands on a coarse grid")
    sig = note(BLOCK * 120, decay=0.3)

    for bits in (3, 5, 8):
        out = crush_only(sig, bits=bits, crush_hz=24000)
        step = 1.0 / 2.0 ** (bits - 1)
        err = np.abs(out / step - np.round(out / step))
        check(f"{bits} bits: samples sit on the {step:.4f} grid",
              float(np.max(err)) < 1e-9, f"worst error {np.max(err):.2e}")

    n_coarse = len(np.unique(crush_only(sig, bits=2, crush_hz=24000)))
    n_fine = len(np.unique(crush_only(sig, bits=12, crush_hz=24000)))
    check("fewer bits means fewer distinct levels", n_coarse < n_fine / 10,
          f"{n_coarse} levels vs {n_fine}")

    # And the crushing must survive into the effect's real output: quieter
    # signals have fewer steps left to work with, so the noise floor rises.
    def noise_floor(bits: int) -> float:
        clean = run(make_chain(bits=16, crush_hz=24000, square=0.0), sig)
        rough = run(make_chain(bits=bits, crush_hz=24000, square=0.0), sig)
        return float(np.sqrt(np.mean((rough - clean) ** 2)))

    check("3 bits is far noisier than 10", noise_floor(3) > noise_floor(10) * 20,
          f"{db(noise_floor(3)):.0f} dB vs {db(noise_floor(10)):.0f} dB of error")


def test_downsampler_holds_samples() -> None:
    print("\ndownsampler: samples are held, and that aliases")
    sig = note(BLOCK * 120, decay=0.3)
    ratio = SR / 4000.0                       # hold each sample ~12 times

    seg = crush_only(sig, bits=16, crush_hz=4000)[BLOCK * 20 : BLOCK * 100]
    repeats = float(np.mean(np.diff(seg) == 0.0))
    check("most samples repeat the previous one",
          repeats > 1.0 - 2.0 / ratio, f"{repeats * 100:.0f}% held")

    unchanged = crush_only(sig, bits=16, crush_hz=48000)[BLOCK * 20 : BLOCK * 100]
    check("at full rate nothing is held",
          float(np.mean(np.diff(unchanged) == 0.0)) < 0.05)

    # Aliasing: a 5 kHz tone sampled at 4 kHz folds back to about 1 kHz, and
    # that folded partial is the sound of the effect. Measured at the real
    # output, because it has to survive the tone filter to be audible.
    t = np.arange(BLOCK * 200) / SR
    tone_in = (0.4 * np.sin(2 * np.pi * 5000 * t)).astype(np.float64)
    folded = run(make_chain(crush_hz=4000, bits=16, square=0.0, tone=12000),
                 tone_in)[BLOCK * 40 :]
    spec = np.abs(np.fft.rfft(folded * np.hanning(len(folded))))
    freqs = np.fft.rfftfreq(len(folded), 1 / SR)
    band = (freqs > 800) & (freqs < 1200)
    quiet = (freqs > 1500) & (freqs < 3000)
    check("a folded partial appears below the input frequency",
          spec[band].max() > np.median(spec[quiet]) * 20,
          f"{db(spec[band].max() / spec.max()):.0f} dB relative to the loudest")


def test_pluck_shortens_the_note() -> None:
    print("\npluck: a long note is cut short")
    sig = note(BLOCK * 400, decay=0.7)        # rings for seconds on its own

    def tail_at(seconds: float, **kw) -> float:
        out = run(make_chain(**kw), sig)
        i = int(seconds * SR)
        return float(np.max(np.abs(out[i : i + BLOCK * 8])))

    ringing = tail_at(1.0, pluck=0.0)
    plucked = tail_at(1.0, pluck=1.0, decay_ms=250)
    check("without pluck the note is still ringing", ringing > 0.05,
          f"{db(ringing):.0f} dB")
    check("with pluck it is gone", plucked < ringing * 0.1,
          f"{db(plucked):.0f} dB vs {db(ringing):.0f} dB")

    # It must retrigger, or only the first note of a phrase would be heard.
    phrase = np.concatenate([note(BLOCK * 60, decay=3.0) for _ in range(4)])
    out = run(make_chain(pluck=1.0, decay_ms=200), phrase)
    peaks = [float(np.max(np.abs(out[i * BLOCK * 60 : (i + 1) * BLOCK * 60])))
             for i in range(4)]
    check("every note in a phrase retriggers the envelope",
          min(peaks) > max(peaks) * 0.4,
          "  ".join(f"{db(p):.0f}dB" for p in peaks))


def test_stays_dry() -> None:
    """No reverb, no delay, no tail of any kind -- this was the whole request."""
    print("\ndryness: nothing is added after the signal stops")
    sig = np.concatenate([note(BLOCK * 60, decay=6.0), np.zeros(BLOCK * 200)])
    out = run(make_chain(pluck=0.0), sig)

    # A gate would be needed to make this exactly zero; what matters is that
    # nothing keeps sounding once the input has gone.
    quiet_in = float(np.max(np.abs(sig[BLOCK * 120 :])))
    quiet_out = float(np.max(np.abs(out[BLOCK * 120 :])))
    check("input is silent by then", quiet_in < 1e-9)
    check("output is silent too -- no tail", quiet_out < 1e-3,
          f"{db(quiet_out):.0f} dB")

    # And no DC offset, which an off-centre pulse width would otherwise leave.
    for pw in (0.5, 0.2, 0.8):
        out = run(make_chain(pulse_width=pw), note(BLOCK * 200, decay=0.0))
        offset = float(np.mean(out[BLOCK * 100 :]))
        check(f"no DC offset at pulse_width {pw}", abs(offset) < 0.02,
              f"mean {offset:+.4f}")


def test_mix_and_bypass() -> None:
    print("\nmix: 0 returns the guitar untouched")
    sig = note(BLOCK * 80)
    out = run(make_chain(mix=0.0), sig)
    err = float(np.max(np.abs(out - sig[: len(out)])))
    check("mix 0 is bit-identical to the input", err < 1e-12, f"max error {err:.2e}")


def test_output_is_sane() -> None:
    print("\noutput stays finite and bounded at extreme settings")
    sig = note(BLOCK * 120, amp=0.9, decay=0.2)
    out = run(make_chain(square=1.0, pulse_width=0.05, bits=1,
                         crush_hz=500, tone=12000, pluck=1.0,
                         decay_ms=20, level=4.0), sig)
    check("all output finite", bool(np.all(np.isfinite(out))))
    check("output bounded", float(np.max(np.abs(out))) < 8.0,
          f"peak {np.max(np.abs(out)):.2f}")


def main() -> int:
    print("=" * 70)
    print("chiptune self-test")
    print("=" * 70)

    test_squares_the_waveform()
    test_pulse_width_changes_the_tone()
    test_bit_depth()
    test_downsampler_holds_samples()
    test_pluck_shortens_the_note()
    test_stays_dry()
    test_mix_and_bypass()
    test_output_is_sane()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
