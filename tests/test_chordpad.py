"""Chord pad behaviour, verified offline against a synthetic guitar signal.

Checks the things that were actually asked for:

  - nothing sounds until a note is played
  - a chord appears when a note is struck
  - it does not stop dead when you stop playing, it fades
  - the armed chord persists until a different gesture arrives
  - the notes that come out are the notes that were asked for

    python tests/test_chordpad.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_pedal.audio.chain import Chain                       # noqa: E402
from live_pedal.dsp.chords import midi_to_hz, note_to_midi     # noqa: E402

SR = 48000
BLOCK = 256

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def db(x: float) -> float:
    return -120.0 if x < 1e-6 else 20.0 * np.log10(x)


def pluck(n: int, freq: float = 82.41, decay: float = 2.5,
          amp: float = 0.35) -> np.ndarray:
    """A plucked note: sharp attack, harmonics, exponential decay."""
    t = np.arange(n) / SR
    env = np.exp(-t * decay)
    sig = sum(np.sin(2 * np.pi * freq * k * t) / k for k in (1, 2, 3, 4))
    return (sig * env * amp).astype(np.float64)


def make_chain(**overrides) -> Chain:
    entry = {
        "type": "chordpad",
        "level": 0.5,
        "attack_ms": 50,
        "decay_ms": 300,
        "sustain": 0.8,
        "release_ms": 1200,
        "threshold_db": -34,
        "retrigger_ms": 150,
        "hold_db": -46,
        # Off unless a test is specifically about following, so the envelope
        # tests measure the envelope rather than the guitar's decay.
        "follow": 0,
        "change_on_trigger": 1,
        "unison": 1,
        "detune": 0,
        "cutoff": 12000,
        "resonance": 0.7,
    }
    entry.update(overrides)
    return Chain.from_spec([entry], SR, BLOCK)


def run(chain: Chain, signal: np.ndarray) -> np.ndarray:
    """Push a signal through and return the pad only (output minus input)."""
    n_blocks = len(signal) // BLOCK
    out = np.zeros(n_blocks * BLOCK)
    x = np.zeros(BLOCK)
    y = np.zeros(BLOCK)
    for b in range(n_blocks):
        x[:] = signal[b * BLOCK : (b + 1) * BLOCK]
        chain.process(x, y)
        out[b * BLOCK : (b + 1) * BLOCK] = y
    return out - signal[: len(out)]


def envelope_of(x: np.ndarray, win: int = 2048) -> np.ndarray:
    n = len(x) // win
    return np.array([np.max(np.abs(x[i * win : (i + 1) * win])) for i in range(n)])


# ---------------------------------------------------------------------------


def test_note_names() -> None:
    print("\nnote name parsing")
    cases = [("C4", 60), ("A4", 69), ("E2", 40), ("G#3", 56), ("Ab3", 56),
             ("B2", 47), ("D#3", 51), ("F2", 41), ("C3", 48)]
    ok = all(note_to_midi(n) == m for n, m in cases)
    check("scientific pitch notation", ok)
    check("A4 is 440 Hz", abs(midi_to_hz(69) - 440.0) < 1e-9)
    check("E2 is 82.41 Hz", abs(midi_to_hz(40) - 82.4069) < 0.01,
          f"{midi_to_hz(40):.3f} Hz")
    try:
        note_to_midi("H9")
        check("rejects a bad note name", False)
    except ValueError:
        check("rejects a bad note name", True)


def test_silence_stays_silent() -> None:
    print("\nsilence in, silence out (no background drone)")
    chain = make_chain()
    pad = run(chain, np.zeros(BLOCK * 60))
    peak = float(np.max(np.abs(pad)))
    check("nothing sounds without a note", peak < 1e-9, f"peak {peak:.2e}")


def test_note_triggers_pad() -> None:
    print("\nplaying a note brings the chord in")
    chain = make_chain()
    sig = np.concatenate([np.zeros(BLOCK * 10), pluck(BLOCK * 60)])
    pad = run(chain, sig)

    before = float(np.max(np.abs(pad[: BLOCK * 10])))
    after = float(np.max(np.abs(pad[BLOCK * 12 : BLOCK * 40])))
    check("silent before the note", before < 1e-9, f"peak {before:.2e}")
    check("audible after the note", after > 0.02, f"peak {after:.3f} ({db(after):.0f} dB)")


def test_release_is_gradual() -> None:
    print("\nthe chord fades out, it does not cut off")
    release_ms = 1200
    chain = make_chain(release_ms=release_ms)
    # A short note, then a long silence to watch the tail.
    sig = np.concatenate([pluck(BLOCK * 40, decay=6.0), np.zeros(BLOCK * 400)])
    pad = run(chain, sig)

    env = envelope_of(pad)
    peak_i = int(np.argmax(env))
    peak = env[peak_i]
    check("pad reached a usable level", peak > 0.02, f"{db(peak):.0f} dB")

    # Find where it falls 40 dB below its peak.
    target = peak * 10 ** (-40 / 20)
    below = np.nonzero(env[peak_i:] < target)[0]
    if len(below) == 0:
        check("tail decays to -40 dB", False, "never decayed")
        return
    fall_ms = below[0] * 2048 / SR * 1000.0
    check("tail decays to -40 dB", True, f"{fall_ms:.0f} ms")

    # Gradual means it takes a substantial fraction of release_ms, and is
    # certainly not one block.
    check("fade is gradual, not a cut", fall_ms > 200,
          f"{fall_ms:.0f} ms (a hard cut would be ~5 ms)")

    # And it must be monotonic-ish over the tail: no sudden drop to zero.
    tail = env[peak_i : peak_i + below[0]]
    jumps = np.diff(tail)
    worst = float(np.min(jumps)) if len(jumps) else 0.0
    check("no abrupt drop inside the tail", worst > -peak * 0.35,
          f"largest single-step drop {worst / max(peak,1e-9) * 100:.0f}% of peak")


def test_correct_pitches() -> None:
    print("\nthe chord that comes out is the chord that was asked for")
    chain = make_chain(
        chords=[{"name": "Em", "notes": ["E2", "B2", "E3", "G3", "B3"]}],
        attack_ms=5, decay_ms=50, sustain=1.0, release_ms=6000, cutoff=14000,
    )
    sig = np.concatenate([pluck(BLOCK * 200, decay=0.4)])
    pad = run(chain, sig)

    seg = pad[BLOCK * 60 : BLOCK * 60 + 32768]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1 / SR)

    expected = [midi_to_hz(note_to_midi(n)) for n in ["E2", "B2", "E3", "G3", "B3"]]
    found = []
    for f in expected:
        band = (freqs > f * 0.985) & (freqs < f * 1.015)
        # Compare against the local neighbourhood, not the global max, so a
        # loud low note does not mask the detection of a quiet high one.
        near = (freqs > f * 0.75) & (freqs < f * 1.33)
        peak_in_band = spec[band].max() if band.any() else 0.0
        median_near = np.median(spec[near]) if near.any() else 1.0
        found.append(peak_in_band > median_near * 8)

    for n, f, ok in zip(["E2", "B2", "E3", "G3", "B3"], expected, found):
        check(f"{n} ({f:.1f} Hz) present", ok)


def test_chord_change_waits_for_next_note() -> None:
    print("\na new gesture does not change the chord under a ringing pad")
    chain = make_chain(change_on_trigger=1, release_ms=4000)
    fx = chain.effects[0]
    i = chain.index["chordpad.chord"]

    x = np.zeros(BLOCK)
    y = np.zeros(BLOCK)
    sig = pluck(BLOCK * 30)

    for b in range(30):                       # play a note on chord 0
        x[:] = sig[b * BLOCK : (b + 1) * BLOCK]
        chain.process(x, y)
    check("chord 0 is active", fx._active_chord == 0, f"active={fx._active_chord}")

    chain.target[i] = 3.0                     # gesture arms chord 3
    x[:] = 0.0
    for _ in range(40):                       # pad still ringing, no new note
        chain.process(x, y)
    check("armed chord does not take over while ringing",
          fx._active_chord == 0, f"active={fx._active_chord}")

    for b in range(20):                       # now play a new note
        x[:] = sig[b * BLOCK : (b + 1) * BLOCK]
        chain.process(x, y)
    check("new note switches to the armed chord",
          fx._active_chord == 3, f"active={fx._active_chord}")


def test_chord_changes_immediately() -> None:
    """With change_on_trigger off, a gesture is audible without playing.

    This is the default now. Waiting for the next note is musically tidier but
    it makes the gestures impossible to learn: you make a shape, hear nothing
    change, and cannot tell whether the camera missed you or the setting is
    working as designed.
    """
    print("\nchange_on_trigger 0: the gesture takes effect straight away")
    chain = make_chain(change_on_trigger=0, release_ms=4000)
    fx = chain.effects[0]
    i = chain.index["chordpad.chord"]

    x = np.zeros(BLOCK)
    y = np.zeros(BLOCK)
    sig = pluck(BLOCK * 30)
    for b in range(30):
        x[:] = sig[b * BLOCK : (b + 1) * BLOCK]
        chain.process(x, y)

    chain.target[i] = 3.0
    x[:] = 0.0
    chain.process(x, y)
    check("chord switches on the very next block", fx._active_chord == 3,
          f"active={fx._active_chord}")


def test_follow_kills_the_pad_with_the_note() -> None:
    """follow high: the chord is a shadow of the note, not a separate tail."""
    print("\nfollow: the chord dies when the note dies")
    # hold_db is deliberately far below anything the note reaches, so the ADSR
    # gate never opens the release by itself. Whatever difference shows up is
    # follow's doing and nothing else's.
    common = dict(release_ms=2000, attack_ms=20, decay_ms=200, sustain=0.9,
                  hold_db=-75)
    # A short, fast-decaying note, then real silence.
    sig = np.concatenate([pluck(BLOCK * 30, decay=8.0), np.zeros(BLOCK * 300)])
    # Two sampling windows, well after the note has died. Following should be
    # quieter at both, and the gap should widen -- the point is that it keeps
    # going down rather than settling at some reduced level.
    windows = {"600 ms": (0.6, 0.8), "1.2 s": (1.2, 1.4)}

    levels: dict[str, dict[str, float]] = {}
    for name, follow in (("free", 0.0), ("following", 0.9)):
        pad = run(make_chain(follow=follow, follow_db=-14, **common), sig)
        peak = float(np.max(np.abs(pad)))
        check(f"{name}: pad was audible", peak > 0.01, f"{db(peak):.0f} dB")
        levels[name] = {
            when: float(np.max(np.abs(pad[int(a * SR) : int(b * SR)])))
            for when, (a, b) in windows.items()
        }
        if name == "following":
            env = envelope_of(pad)
            worst = float(np.min(np.diff(env[int(np.argmax(env)):])))
            check("following still fades rather than cutting",
                  worst > -peak * 0.35,
                  f"largest step {worst / peak * 100:.0f}% of peak")

    # The bar is not tighter than this because the level that follow tracks is
    # smoothed over 180 ms; that lag is exactly what keeps the fade smooth
    # instead of gating the pad off the moment the string stops.
    for when, bar in (("600 ms", 0.45), ("1.2 s", 0.20)):
        ratio = levels["following"][when] / max(levels["free"][when], 1e-12)
        check(f"following is much quieter at {when}", ratio < bar,
              f"{db(levels['following'][when]):.0f} dB vs "
              f"{db(levels['free'][when]):.0f} dB  (ratio {ratio:.2f})")


def test_follow_tracks_dynamics() -> None:
    """A quiet note gets a quiet chord."""
    print("\nfollow: the chord tracks how hard you played")
    loud = run(make_chain(follow=1.0, follow_db=-14),
               pluck(BLOCK * 120, decay=1.0, amp=0.35))
    quiet = run(make_chain(follow=1.0, follow_db=-14),
                pluck(BLOCK * 120, decay=1.0, amp=0.03))
    pl = float(np.max(np.abs(loud)))
    pq = float(np.max(np.abs(quiet)))
    check("loud note gives a louder chord", pl > pq * 2.0,
          f"{db(pl):.0f} dB vs {db(pq):.0f} dB")


def test_chord_persists() -> None:
    print("\nthe selected chord stays selected across many notes")
    chain = make_chain()
    fx = chain.effects[0]
    chain.target[chain.index["chordpad.chord"]] = 2.0

    x = np.zeros(BLOCK)
    y = np.zeros(BLOCK)
    sig = pluck(BLOCK * 25)
    for _ in range(4):                        # four separate notes
        for b in range(25):
            x[:] = sig[b * BLOCK : (b + 1) * BLOCK]
            chain.process(x, y)
        x[:] = 0.0
        for _ in range(30):
            chain.process(x, y)
    check("still on chord 2 after four notes", fx._active_chord == 2,
          f"active={fx._active_chord}")


def test_output_is_sane() -> None:
    print("\noutput stays finite and bounded")
    chain = make_chain(level=1.5, resonance=12.0, unison=5, detune=40)
    sig = np.concatenate([pluck(BLOCK * 100, amp=0.9), np.zeros(BLOCK * 100)])
    n_blocks = len(sig) // BLOCK
    x = np.zeros(BLOCK)
    y = np.zeros(BLOCK)
    worst = 0.0
    finite = True
    for b in range(n_blocks):
        x[:] = sig[b * BLOCK : (b + 1) * BLOCK]
        chain.process(x, y)
        worst = max(worst, float(np.max(np.abs(y))))
        finite &= bool(np.all(np.isfinite(y)))
    check("all output finite", finite)
    check("output bounded at extreme settings", worst < 8.0, f"peak {worst:.2f}")


def main() -> int:
    print("=" * 70)
    print("chord pad self-test")
    print("=" * 70)

    test_note_names()
    test_silence_stays_silent()
    test_note_triggers_pad()
    test_release_is_gradual()
    test_correct_pitches()
    test_chord_change_waits_for_next_note()
    test_chord_changes_immediately()
    test_follow_kills_the_pad_with_the_note()
    test_follow_tracks_dynamics()
    test_chord_persists()
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
