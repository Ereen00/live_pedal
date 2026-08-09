"""Gesture maths, verified against synthetic hands.

You cannot regression-test hand tracking by waving at a webcam. So this builds
landmark sets by construction -- a hand whose joints are bent by a known angle
-- and checks that the features come out where they should and that the mapping
layer turns them into the right parameter values.

    python tests/test_gestures.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_pedal.ipc import FEATURE_INDEX                       # noqa: E402
from live_pedal.layout import ParamLayout                      # noqa: E402
from live_pedal.mapping import Mapper, MappingError            # noqa: E402
from live_pedal.vision.features import FeatureExtractor        # noqa: E402

BONE = 0.030          # metres per phalanx
PALM = 0.085          # wrist to middle knuckle

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def make_hand(bend_deg: float, thumb_bend_deg: float | None = None,
              spread: float = 1.0, per_finger: dict[int, float] | None = None
              ) -> np.ndarray:
    """Build 21 landmarks for a hand with the given joint bend.

    ``bend_deg`` is how far each joint rotates: 0 is a straight finger, 70+ is
    a closed fist. ``per_finger`` overrides the bend for individual fingers
    (keyed 0=thumb .. 4=pinky).
    """
    lm = np.zeros((21, 3), dtype=np.float64)
    lm[0] = (0.0, 0.0, 0.0)                        # wrist

    # Knuckles fan out across the palm; y is "up the hand".
    mcp_x = np.array([-0.035, -0.020, 0.0, 0.018, 0.034]) * spread
    mcp_y = np.array([0.030, 0.075, 0.080, 0.075, 0.068])

    starts = [1, 5, 9, 13, 17]
    per_finger = per_finger or {}

    for fi in range(5):
        base = starts[fi]
        bend = per_finger.get(fi, thumb_bend_deg if fi == 0 and
                              thumb_bend_deg is not None else bend_deg)
        pos = np.array([mcp_x[fi], mcp_y[fi], 0.0])
        lm[base] = pos
        # Direction starts along the finger and rotates by `bend` at each joint.
        angle = math.atan2(mcp_y[fi] - 0.0, mcp_x[fi] - 0.0) if fi == 0 else math.pi / 2
        for j in range(3):
            angle -= math.radians(bend)
            pos = pos + BONE * np.array([math.cos(angle), math.sin(angle), 0.0])
            lm[base + 1 + j] = pos

    # Make the wrist-to-middle-knuckle distance the expected palm size.
    lm[:, :2] *= PALM / np.linalg.norm(lm[9, :2])
    return lm


def to_image(world: np.ndarray, cx=0.5, cy=0.5, scale=1.6) -> np.ndarray:
    """Project a world hand into normalised image coordinates."""
    img = world.copy()
    img[:, 0] = cx + world[:, 0] * scale
    img[:, 1] = cy - world[:, 1] * scale      # image y grows downward
    return img


def extract(world: np.ndarray, extractor: FeatureExtractor | None = None,
            t: float = 0.0) -> np.ndarray:
    ex = extractor or FeatureExtractor(smoothing=0.0)
    return ex.extract(to_image(world), world, aspect=4 / 3, now=t, fps=30.0).copy()


# ---------------------------------------------------------------------------


def test_openness() -> None:
    print("\nopenness: flat hand vs closed fist")
    flat = extract(make_hand(0.0))
    fist = extract(make_hand(75.0))
    o_flat = flat[FEATURE_INDEX["openness"]]
    o_fist = fist[FEATURE_INDEX["openness"]]
    check("open hand reads high", o_flat > 0.85, f"openness={o_flat:.3f}")
    check("fist reads low", o_fist < 0.15, f"openness={o_fist:.3f}")
    check("they are well separated", o_flat - o_fist > 0.7,
          f"gap={o_flat - o_fist:.3f}")


def test_per_finger_curl() -> None:
    print("\nper-finger curl: index straight, others closed (a 'point')")
    hand = make_hand(80.0, per_finger={1: 0.0})
    f = extract(hand)
    ci = f[FEATURE_INDEX["curl_index"]]
    cm = f[FEATURE_INDEX["curl_middle"]]
    check("index reads straight", ci < 0.2, f"curl_index={ci:.3f}")
    check("middle reads curled", cm > 0.8, f"curl_middle={cm:.3f}")


def test_poses() -> None:
    print("\npose classification (needs 3 consistent frames)")
    ex = FeatureExtractor(smoothing=0.0, pose_frames=3)
    for _ in range(4):
        f = extract(make_hand(80.0), ex, t=_ * 0.03)
    check("fist detected", f[FEATURE_INDEX["pose_fist"]] > 0.5)
    check("open not detected", f[FEATURE_INDEX["pose_open"]] < 0.5)

    ex2 = FeatureExtractor(smoothing=0.0, pose_frames=3)
    for i in range(4):
        f = extract(make_hand(0.0), ex2, t=i * 0.03)
    check("open detected", f[FEATURE_INDEX["pose_open"]] > 0.5)
    check("fist not detected", f[FEATURE_INDEX["pose_fist"]] < 0.5)

    print("  hysteresis: a single stray frame must not fire a pose")
    ex3 = FeatureExtractor(smoothing=0.0, pose_frames=3)
    extract(make_hand(0.0), ex3, t=0.0)
    f = extract(make_hand(80.0), ex3, t=0.03)      # one fist frame only
    check("one frame is not enough", f[FEATURE_INDEX["pose_fist"]] < 0.5)


def settle(world: np.ndarray, frames: int = 5) -> np.ndarray:
    """Hold one hand shape long enough for the pose hysteresis to accept it."""
    ex = FeatureExtractor(smoothing=0.0, pose_frames=3)
    f = None
    for i in range(frames):
        f = extract(world, ex, t=i * 0.03)
    return f


def test_chord_poses() -> None:
    """The four poses that select chords must be reliable and exclusive.

    This is the test that matters most in practice: a pose that needs a
    textbook hand shape is a pose that never fires while you are holding a
    guitar, and two poses that overlap fight over the same chord parameter.
    """
    print("\nchord poses: each fires, and only one at a time")
    OPEN, FIST = 3.0, 80.0
    shapes = {
        # Deliberately sloppy: fingers not fully straight, fist not fully
        # closed, pinky lazy -- the way a hand actually looks mid-song.
        "pose_open": make_hand(18.0, per_finger={4: 30.0}),
        "pose_fist": make_hand(FIST, thumb_bend_deg=70.0),
        "pose_point": make_hand(FIST, thumb_bend_deg=70.0,
                                per_finger={1: OPEN}),
        "pose_peace": make_hand(FIST, thumb_bend_deg=70.0,
                                per_finger={1: OPEN, 2: OPEN}),
    }
    pose_names = [n for n in FEATURE_INDEX if n.startswith("pose_")]

    for want, world in shapes.items():
        f = settle(world)
        fired = [n for n in pose_names if f[FEATURE_INDEX[n]] > 0.5]
        check(f"{want[5:]} detected", f[FEATURE_INDEX[want]] > 0.5,
              f"fired={fired or 'nothing'}")
        check(f"{want[5:]} is the only pose", fired == [want],
              f"fired={fired}")


def test_pose_hysteresis_gap() -> None:
    """A finger hovering near the threshold must not chatter.

    Without hysteresis a half-bent finger crossing back and forth over one
    threshold retriggers the latch every few frames, and the chord flickers
    between two values while your hand is perfectly still.
    """
    print("\npose stability: a finger held near the boundary does not chatter")
    ex = FeatureExtractor(smoothing=0.0, pose_frames=3)
    flips = 0
    was = None
    for i in range(40):
        # Ring finger wobbles by a couple of degrees right at the boundary.
        wobble = 44.0 + (2.0 if i % 2 else -2.0)
        f = extract(make_hand(80.0, thumb_bend_deg=70.0,
                              per_finger={1: 3.0, 3: wobble}), ex, t=i * 0.03)
        now = f[FEATURE_INDEX["pose_point"]] > 0.5
        if was is not None and now != was:
            flips += 1
        was = now
    check("pose does not flicker", flips <= 1, f"{flips} change(s) in 40 frames")


def test_latch_selects_chords() -> None:
    """Four poses, one chord parameter, no interference."""
    print("\nmapping: four latched poses select four chords")
    layout = ParamLayout(
        names=["chordpad.chord", "chordpad.enable"],
        lo=[0.0, 0.0], hi=[15.0, 1.0], defaults=[0.0, 1.0],
        curves=["lin", "lin"], units=["", ""],
    )
    poses = ["pose_open", "pose_fist", "pose_point", "pose_peace"]
    m = Mapper(
        [{"source": p, "target": "chordpad.chord", "hi": i, "mode": "latch"}
         for i, p in enumerate(poses)],
        layout,
    )
    check("no conflict warning for latched rules", not m.warnings,
          "; ".join(m.warnings))

    f = np.zeros(len(FEATURE_INDEX), dtype=np.float32)
    f[FEATURE_INDEX["present"]] = 1.0

    # Out of order, to prove rule order does not decide the winner.
    for want, pose in [(2, "pose_point"), (0, "pose_open"),
                       (3, "pose_peace"), (1, "pose_fist")]:
        f[FEATURE_INDEX[pose]] = 1.0
        got = m.update(f)[0]
        check(f"{pose[5:]} selects chord {want}", got == want, f"got {got:g}")
        f[FEATURE_INDEX[pose]] = 0.0
        held = m.update(f)[0]
        check(f"{pose[5:]} released, chord holds", held == want, f"got {held:g}")

    f[FEATURE_INDEX["present"]] = 0.0
    check("chord survives the hand leaving frame", m.update(f)[0] == 1.0)


def test_position() -> None:
    print("\nposition tracking")
    left = extract(make_hand(0.0), FeatureExtractor(smoothing=0.0))
    world = make_hand(0.0)
    img_r = to_image(world, cx=0.8)
    ex = FeatureExtractor(smoothing=0.0)
    right = ex.extract(img_r, world, 4 / 3, 0.0, 30.0).copy()
    check("x follows the hand", right[FEATURE_INDEX["x"]] >
          left[FEATURE_INDEX["x"]] + 0.2,
          f"{left[FEATURE_INDEX['x']]:.2f} -> {right[FEATURE_INDEX['x']]:.2f}")


def test_hand_loss_holds_values() -> None:
    print("\nlosing the hand freezes the shape features")
    ex = FeatureExtractor(smoothing=0.0)
    f = extract(make_hand(20.0), ex)
    held_openness = f[FEATURE_INDEX["openness"]]
    gone = ex.no_hand(30.0)
    check("present drops to 0", gone[FEATURE_INDEX["present"]] == 0.0)
    check("openness is held, not zeroed",
          close(gone[FEATURE_INDEX["openness"]], held_openness, 1e-6),
          f"held={gone[FEATURE_INDEX['openness']]:.3f}")


# ---------------------------------------------------------------------------


def make_layout() -> ParamLayout:
    return ParamLayout(
        names=["wah.freq", "wah.enable", "volume.gain", "drive.enable"],
        lo=[250.0, 0.0, 0.0, 0.0],
        hi=[3200.0, 1.0, 2.0, 1.0],
        defaults=[700.0, 1.0, 1.0, 0.0],
        curves=["log", "lin", "lin", "lin"],
        units=["Hz", "", "", ""],
    )


def test_mapping_range() -> None:
    print("\nmapping: openness sweeps the wah")
    layout = make_layout()
    m = Mapper([{"source": "openness", "target": "wah.freq",
                 "lo": 320, "hi": 2600, "scale": "log"}], layout)

    f = np.zeros(len(FEATURE_INDEX), dtype=np.float32)
    f[FEATURE_INDEX["present"]] = 1.0

    f[FEATURE_INDEX["openness"]] = 0.0
    check("closed hand -> low end", close(m.update(f)[0], 320.0, 1.0),
          f"{m.update(f)[0]:.1f} Hz")

    f[FEATURE_INDEX["openness"]] = 1.0
    check("open hand -> high end", close(m.update(f)[0], 2600.0, 1.0),
          f"{m.update(f)[0]:.1f} Hz")

    f[FEATURE_INDEX["openness"]] = 0.5
    mid = m.update(f)[0]
    geometric = math.sqrt(320 * 2600)
    check("halfway is geometric, not arithmetic (log scale)",
          close(mid, geometric, 5.0),
          f"{mid:.0f} Hz vs geometric {geometric:.0f}, arithmetic {(320+2600)/2:.0f}")


def test_mapping_hold_on_hand_loss() -> None:
    print("\nmapping: parameters freeze when the hand leaves")
    layout = make_layout()
    m = Mapper([{"source": "openness", "target": "wah.freq", "lo": 320, "hi": 2600}],
               layout)
    f = np.zeros(len(FEATURE_INDEX), dtype=np.float32)
    f[FEATURE_INDEX["present"]] = 1.0
    f[FEATURE_INDEX["openness"]] = 1.0
    m.update(f)
    held = m.update(f)[0]

    f[FEATURE_INDEX["present"]] = 0.0
    f[FEATURE_INDEX["openness"]] = 0.0        # value changes, but hand is gone
    after = m.update(f)[0]
    check("value held after hand loss", close(after, held, 1e-6),
          f"{held:.0f} -> {after:.0f} Hz")


def test_toggle() -> None:
    print("\nmapping: toggle latches on rising edge only")
    layout = make_layout()
    m = Mapper([{"source": "pose_fist", "target": "drive.enable",
                 "lo": 0, "hi": 1, "mode": "toggle"}], layout)
    f = np.zeros(len(FEATURE_INDEX), dtype=np.float32)
    f[FEATURE_INDEX["present"]] = 1.0
    i = layout.index_of("drive.enable")

    f[FEATURE_INDEX["pose_fist"]] = 1.0
    check("first fist turns it on", m.update(f)[i] == 1.0)
    check("holding the fist does not turn it off", m.update(f)[i] == 1.0)

    f[FEATURE_INDEX["pose_fist"]] = 0.0
    m.update(f)
    check("still on after releasing", m.update(f)[i] == 1.0)

    f[FEATURE_INDEX["pose_fist"]] = 1.0
    check("second fist turns it off", m.update(f)[i] == 0.0)


def test_invert_and_curves() -> None:
    print("\nmapping: invert and response curves")
    layout = make_layout()
    f = np.zeros(len(FEATURE_INDEX), dtype=np.float32)
    f[FEATURE_INDEX["present"]] = 1.0
    i = layout.index_of("volume.gain")

    m = Mapper([{"source": "y", "target": "volume.gain",
                 "lo": 0.0, "hi": 2.0, "invert": True}], layout)
    f[FEATURE_INDEX["y"]] = 0.0               # top of frame
    check("inverted: top of frame is loud", close(m.update(f)[i], 2.0, 1e-6))
    f[FEATURE_INDEX["y"]] = 1.0
    check("inverted: bottom of frame is silent", close(m.update(f)[i], 0.0, 1e-6))

    m2 = Mapper([{"source": "y", "target": "volume.gain",
                  "lo": 0.0, "hi": 2.0, "curve": "square"}], layout)
    f[FEATURE_INDEX["y"]] = 0.5
    check("square curve bends the middle down",
          close(m2.update(f)[i], 0.5, 1e-6), f"{m2.update(f)[i]:.3f} (linear would be 1.0)")


def test_mapping_errors() -> None:
    print("\nmapping: bad config is rejected with a useful message")
    layout = make_layout()
    for bad, why in [
        ({"source": "nonsense", "target": "wah.freq"}, "unknown gesture"),
        ({"source": "openness", "target": "nope.nope"}, "unknown parameter"),
        ({"source": "openness"}, "missing target"),
        ({"source": "openness", "target": "wah.freq", "scale": "log",
          "lo": 0, "hi": 100}, "log scale through zero"),
        ({"source": "openness", "target": "wah.freq", "mode": "wobble"},
         "unknown mode"),
    ]:
        try:
            Mapper([bad], layout)
            check(f"rejects {why}", False, "no error raised")
        except MappingError as exc:
            first = str(exc).splitlines()[0]
            check(f"rejects {why}", True, f'"{first[:60]}"')


def main() -> int:
    print("=" * 70)
    print("gesture and mapping self-test")
    print("=" * 70)

    test_openness()
    test_per_finger_curl()
    test_poses()
    test_chord_poses()
    test_pose_hysteresis_gap()
    test_latch_selects_chords()
    test_position()
    test_hand_loss_holds_values()
    test_mapping_range()
    test_mapping_hold_on_hand_loss()
    test_toggle()
    test_invert_and_curves()
    test_mapping_errors()

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
