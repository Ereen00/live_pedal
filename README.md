# live_pedal

A guitar effects processor you play with your hands.

Guitar goes into your audio interface. The signal is processed live and comes
straight back out. While you play with one hand, the other hand moves in front
of the webcam and shapes the sound.

The default rig is an 8-bit one: the guitar is squared into a pulse wave and
crushed down to four bits, bone dry — no reverb, no delay, nothing spatial. Your
free hand picks a chord, and that chord appears underneath the note you play and
dies with it.

No DAW, no plugin host, no MIDI controller. One command.

```
python run.py
```

---

## What it actually does

```
  guitar ──► audio interface ──► [ gate ─► chiptune ─► chordpad ─► volume ] ──► speakers
                                                ▲
                                                │  parameter targets (shared memory)
                                                │
  webcam ──► hand landmarks ──► gesture features ──► mapping rules
```

Two processes. One owns the sound card, one owns the camera. They share a small
block of memory and nothing else.

---

## Why it is built this way

**Python, with the sample loops compiled by Numba.** The obvious objection to
Python for real-time audio is the one that matters: an interpreted per-sample
loop cannot run at 48 kHz. Numba compiles those loops to machine code ahead of
time and releases the GIL while they run, which removes the objection entirely.
Measured on the development machine:

| | |
|---|---|
| All 16 effects in series, per 256-sample block | **157 µs** |
| Real time available per block | 5333 µs |
| **Fraction of budget used** | **2.9 %** |

The DSP is not the bottleneck and never comes close to being it. What limits
latency is the audio driver, so that is where the effort went.

**Vision runs in a separate process, not a thread.** MediaPipe inference and
OpenCV rendering both hold the GIL in bursts. In one process those bursts
contend with the audio callback, which needs the GIL on a hard deadline, and you
hear it as intermittent clicks. Separate processes have separate interpreters.

**The mapping from gesture to knob happens on the vision side.** The audio
callback does one memcpy of a parameter vector and nothing else. Mapping at
camera rate loses nothing, because the camera is the real rate limit and the
audio thread smooths between updates anyway.

**Nothing allocates in the audio callback.** Every buffer is allocated at
startup; the callback only writes into preallocated arrays. The garbage
collector is frozen and disabled while the stream is live.

**Every kernel is compiled before the stream opens.** This one was found the
hard way: the first callback took **930 ms** compiling JIT code, and the driver
killed the stream. Warmup now walks every effect through every discrete setting
it has, so a branch cannot compile itself mid-performance the moment a gesture
selects it. First callback is now ~150 µs.

---

## Setup

Requires Python 3.11+ and Windows (it will run elsewhere, but the low-latency
driver notes below are Windows-specific).

```bash
git clone https://github.com/Ereen00/live_pedal.git
cd live_pedal

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
python tools/fetch_model.py          # ~8 MB hand tracking model
```

Check what it can see, without touching anything:

```bash
python tools/list_devices.py         # audio devices and expected latency
python tools/test_output.py --sweep  # beep each output: which can you hear?
python run.py --dry-run              # validate config, print the rig
python tests/test_gestures.py        # gesture maths self-test
python tests/test_chordpad.py        # chord pad self-test
python tests/test_chiptune.py        # 8-bit voice self-test
python tools/bench_dsp.py            # DSP benchmark and sanity check
```

Then plug in and go:

```bash
python run.py
```

Press `q` in the camera window to stop, or `Ctrl+C` in the terminal.

> **Use the virtualenv's Python.** If you open a fresh terminal and forget to
> `activate`, `python run.py` will be the *system* Python, which does not have
> the dependencies. `run.bat` sidesteps this entirely — it always uses
> `.venv\Scripts\python.exe` and passes your arguments through:
>
> ```bash
> run.bat                 # same as python run.py
> run.bat -c lead
> run.bat --list-devices
> ```

---

## Latency

This is the number that decides whether the thing is playable. Under ~10 ms is
transparent; 10–20 ms is playable but noticeable on hard attacks; above 25 ms
fights you.

**Measure it, don't guess.** Connect an output back into an input and run:

```bash
python tools/measure_latency.py
```

It plays a chirp, records what returns, and cross-correlates. That is the real
figure, including everything the driver does.

### The three things that control it

**1. The host API — by far the biggest factor.**

| API | Typical round trip |
|---|---|
| ASIO | 2–8 ms |
| WDM-KS | 5–15 ms |
| WASAPI | 8–20 ms |
| DirectSound | 30 ms+ |
| MME | 60 ms+ |

`hostapi: auto` picks the best available in that order. On the development
machine, which has **no ASIO driver installed**, WDM-KS reported 20 ms round
trip against WASAPI's 47 ms — so auto-selection matters even without ASIO.

**Installing an ASIO driver is the single biggest improvement available.** Use
your interface's own driver if it has one, otherwise
[ASIO4ALL](https://asio4all.org) wraps almost any Windows device. Then set
`hostapi: asio`.

**2. Block size.** `blocksize: 128` halves the buffering of `256`. The DSP has
enough headroom that 128 is worth trying first — measured at 10 % of budget with
zero dropouts.

**3. Camera frame rate — this is your *gesture* latency.** Every frame you don't
get is another 33 ms your hand can move before the sound knows. The default
setting that ruins this is auto-exposure: in room light the driver lengthens
the integration time to brighten the picture and the frame rate collapses.
Measured on the development webcam:

| | |
|---|---|
| Auto exposure | 10 fps |
| Manual exposure (`exposure: -6`) | **30 fps** |

So `exposure` defaults to manual. If the picture is too dark to track, **add
light before raising it** — at `-4` the same camera dropped back to 16 fps.

---

## Gestures

These are the names you use on the left of a mapping. The camera window shows
all of them live as bars, so you can see exactly what your hand is producing.

| Continuous | |
|---|---|
| `openness` | 0 = tight fist, 1 = flat open palm |
| `pinch` | 0 = thumb and index touching, 1 = far apart |
| `spread` | 0 = fingers together, 1 = splayed |
| `x`, `y` | palm position in frame (`y` is 0 at the **top**) |
| `size` | apparent hand size — a proxy for distance to the camera |
| `roll` | palm rotation |
| `tilt` | wrist-to-knuckle axis, 1 = fingers up |
| `curl_thumb` … `curl_pinky` | per-finger curl |
| `speed` | how fast the hand is moving |
| `present` | 1 while a hand is tracked |

| Discrete poses | Fingers |
|---|---|
| `pose_fist` | none up, thumb tucked |
| `pose_thumbup` | none up, thumb out |
| `pose_point` | index only |
| `pose_peace` | index and middle |
| `pose_open` | index plus at least two more |
| `pose_ok` | thumb and index touching, the rest up |

Poses are decided by asking which fingers are up and then reading the pattern,
rather than by testing every curl against one tight window. That matters in
practice: a pinky that only ever half-straightens still counts as up, where a
scheme demanding all four fingers below a single strict number would simply
never see an open hand. Each finger's up/down decision has a wide hysteresis
gap, so one hovering near the boundary cannot chatter.

The six are **mutually exclusive** — no two can be true at once — which is what
lets four of them latch the same parameter without fighting over it.

Poses also need three consistent frames before they fire, so a hand passing
through a shape on its way somewhere else does not trigger anything.

If the chord never changes, the most common cause is `vision.hand`. Set to
`right` or `left`, a hand the tracker labels the other way is ignored entirely
and looks exactly like no hand at all; the console says so when it happens.
`hand: any` uses whichever hand is in frame.

**When the tracked hand leaves the frame, every parameter freezes where it is.**
You can drop your hand to the strings and the tone stays put.

---

## Effects

| | |
|---|---|
| `chiptune` | 8-bit voice: pulse wave, bit crusher, downsampler, pluck — see below |
| `chordpad` | gesture-selected chord synth, triggered by playing — see below |
| `gate` | noise gate with hold |
| `drive` | overdrive/distortion/fuzz, 5 curves, 4× oversampled |
| `wah` | resonant bandpass sweep |
| `filter` | sweepable LP / BP / HP / notch / peak |
| `eq3` | low shelf, peaking mid, high shelf |
| `tone` | one-knob tilt |
| `octaver` | analog-style octave up and down — **zero added latency** |
| `pitchshift` | ±24 semitones — *adds latency, see below* |
| `phaser` | 2–12 swept allpass stages, zero added latency |
| `flanger`, `chorus`, `vibrato` | modulated delay |
| `tremolo` | amplitude modulation |
| `delay` | up to 2 s, damped feedback |
| `reverb` | Freeverb, zero added latency |
| `volume` | expression pedal |

Two notes worth knowing before you build a chain:

- **`octaver` and `pitchshift` are not the same kind of thing.** The octaver uses
  the analog tricks — rectification for the upper octave, a zero-crossing
  flip-flop for the lower — so it costs nothing and adds no latency, but it
  tracks single notes and turns to mush on chords, exactly like the hardware it
  imitates. `pitchshift` transposes anything, but it works by crossfading two
  read pointers through a delay line, so it adds roughly half its window length
  in latency and warbles on sustained chords. Leave it out if you are chasing
  the lowest round trip.
- **Sweeping `delay.time_ms` with a gesture pitch-bends the repeats.** That is
  deliberate — it is what a real tape echo does when you turn the knob.

---

## The 8-bit voice

`chiptune` is the only effect on the guitar in the default rig, and it is
deliberately dry — there is nothing spatial anywhere in it. Four stages:

| Stage | Parameters | What it does |
|---|---|---|
| Square | `square`, `pulse_width` | Replaces the waveform with a pulse of the same amplitude. `square: 0` leaves the guitar alone, `1` is a pure pulse. `pulse_width` off 0.5 gives the thin, nasal duty-cycle tone. |
| Crush | `bits`, `crush_hz` | Quantises to `2**(bits-1)` steps and holds each sample. The aliasing is the effect, so it is left unfiltered. |
| Tone | `tone` | One pole of lowpass, to stop the aliasing getting shrill. |
| Pluck | `pluck`, `decay_ms` | A short envelope restarted by every note. This is what makes it read as chiptune rather than as a broken amp: a console voice has no sustain. |

The pulse tracks the note's envelope rather than being a constant buzz, so your
picking dynamics survive.

| If | Change |
|---|---|
| Too clean, not retro enough | Lower `bits` to 3, lower `crush_hz` to 4000 |
| Too harsh | Raise `bits`, lower `tone` |
| Notes cut off too fast | Raise `decay_ms`, or lower `pluck` |
| Notes ring like a normal guitar | Raise `pluck` toward 1, lower `decay_ms` |
| You want the real guitar back underneath | Lower `mix` below 1 |

---

## The chord pad

The default rig uses gestures for one thing: choosing a chord. It is not a
background drone — **nothing sounds until you actually strike a string.** The
chord then comes in underneath what you played and fades away as the note does.

| Gesture | Chord |
|---|---|
| Open hand (three or more fingers up) | **Em** |
| Fist | **E** |
| Index finger up | **Fm** |
| Two fingers (peace) | **B7b9** |

The chord's name is drawn large on the camera window. If it is not changing,
look there first — it tells you immediately whether the camera saw the gesture
or the chord itself is the problem.

The choice **latches**: drop your hand and the chord stays selected until you
make a different gesture. By default a new gesture takes effect immediately, so
you can hear the camera register it without playing; set `change_on_trigger: 1`
and it waits for your next note instead, which never changes the chord
underneath a pad that is still ringing.

### How it decides you played a note

A fast envelope of the guitar is compared against a slower baseline, so it
detects the *rise* rather than the level — which means a note played while the
pad is still decaying retriggers it, not just the first note out of silence.

`sensitivity` is how much louder than the moment before counts as a new note,
and useful values are small. A note picked 130 ms after the last one has only
decayed to about 0.7 of its level, so the rise the detector sees is on the order
of 10 percent; anything much above `1.15` hears the first note of a fast run and
nothing after it.

### How it stops

`follow` scales the pad by how loud the guitar actually is. At `0.85` the chord
is a shadow of the note — it comes up with it and goes down with it, no tail of
its own. Turn it down and the pad gets a life of its own, outliving the note by
`release_ms`. Set `follow_db` near how hard you actually play; put it far below
and the pad sits at full volume for most of the note and only thins out at the
very end, which defeats the purpose.

| If | Change |
|---|---|
| It triggers on finger noise | Raise `threshold_db` toward `-28` |
| It misses quiet notes | Lower `threshold_db` toward `-46` |
| It misses notes in a fast run | Lower `sensitivity` toward `1.05` |
| One pick stroke fires it twice | Raise `retrigger_ms` |
| The chord hangs around too long | Raise `follow`, or lower `release_ms` |
| The chord dies before the note does | Lower `follow`, or lower `follow_db` |
| The chord drowns the guitar | Lower `level` |

### Changing the chords

Any chords, any voicing. Notes are scientific pitch notation, `C4` = middle C:

```yaml
  - type: chordpad
    chords:
      - name: Em
        notes: [E2, B2, E3, G3,  B3]
      - name: Am7
        notes: [A2, E3, G3, C4, E4]
```

To reassign a gesture, change one `source:` line — the `hi:` value is the index
into that chord list:

```yaml
  - source: pose_ok        # was pose_point
    target: chordpad.chord
    hi: 2                  # third chord in the list
    mode: latch
```

Other poses available: `pose_ok`, `pose_thumbup`.

---

## Assigning gestures

This is the part you will actually edit. A mapping is one line saying "this hand
measurement drives this knob":

```yaml
mappings:
  - source: openness          # a gesture name from the table above
    target: wah.freq          # <effect>.<parameter>
    lo: 320                   # value when the gesture reads 0
    hi: 2600                  # value when it reads 1
    scale: log                # traverse the range logarithmically
    curve: scurve             # ease in and out rather than tracking linearly
```

| Field | |
|---|---|
| `lo` / `hi` | output range. Put `lo` above `hi` to reverse it |
| `in_lo` / `in_hi` | input range — **use this to calibrate to your hand** |
| `curve` | `lin`, `square`, `sqrt`, `scurve` — shapes the response |
| `scale` | `lin` or `log` — frequencies want `log` |
| `invert` | swap the ends |
| `mode` | `continuous` (default), `gate`, `toggle`, `latch` |
| `smooth` | extra smoothing, 0 to <1 |
| `when` | only apply while this gesture is true (default: `present`) |

Every effect also has an implicit `<effect>.enable`, and it is a *smoothed 0–1
value*, not a switch. So a gesture can fade an effect in halfway — a
half-engaged wah is a real sound — and toggling one never clicks.

Hands differ. If a gesture doesn't reach the end of its range, don't change the
maths — narrow the input:

```yaml
  - source: openness
    target: wah.freq
    in_lo: 0.15      # treat 0.15 as fully closed
    in_hi: 0.85      # and 0.85 as fully open
    lo: 320
    hi: 2600
```

Check your edits without plugging anything in:

```bash
python run.py --dry-run
```

It validates every mapping, prints the whole rig, and tells you if two mappings
are fighting over the same parameter.

---

## Presets

```bash
python run.py -c lowlatency    # minimal chain, 128-sample blocks
python run.py -c lead          # drive, octave, wah on palm tilt, slapback
python run.py -c ambient       # long delay, big reverb, pinch for a fifth above
```

A preset can inherit from another with `extends:`, so you can keep device
settings in one place and swap only the chain and mappings.

---

## Controls

In the camera window:

| | |
|---|---|
| `q` / `Esc` | quit |
| `b` | bypass — pass the guitar through untouched |
| `h` | hold — freeze parameters, ignore the hand |
| `space` | panic — mute and clear all effect state |
| `r` | reset gesture smoothing |

---

## Command line

```
python run.py                          # default rig
python run.py -c ambient               # a preset
python run.py --list-devices           # what sound cards are visible
python run.py --dry-run                # validate config, open nothing
python run.py --no-window              # headless, lowest CPU
python run.py --no-vision              # audio only, chain at its defaults
python run.py --hostapi asio           # force a host API
python run.py --blocksize 128          # override block size
python run.py --input-device 12        # index or name substring
python run.py --seconds 30             # stop automatically (soak testing)
```

---

## Troubleshooting

**No sound — but the meters are moving.** Read the meters first, they tell you
which half is wrong:

```
in ###############-   -0dB    out ####------------  -23dB    dsp 4.0%  xruns 0
```

If `out` shows a level, the audio is fine and going somewhere — just not
somewhere you are listening. **Your audio interface is its own sound card.**
When `output` says `USB AUDIO CODEC`, sound goes to the interface's outputs, and
your laptop speakers and Windows volume slider have nothing to do with it. Plug
headphones into the *interface*.

To find out what you can actually hear:

```bash
python tools/test_output.py --sweep     # beeps every output in turn
```

Then set what you heard:

```bash
run.bat --output-device 25              # or put it in the config
```

Using the interface for input and the laptop for output works for testing, but
the two run on independent clocks and will slowly drift, which you may hear as
occasional clicks. Same device for both is the right answer.

**No sound and the `in` meter never moves.** The guitar is on a different input
— try `input_channel: 1`, or `python run.py --list-devices` and set the device
explicitly.

**`in` sits at `-0dB`.** That is full scale: the converter is clipping before
live_pedal ever sees the signal, and no software setting can undo it. Turn the
**gain knob on the interface** down until hard strums peak around −12 to −6 dB.
(`input_gain_db` is applied after the converter, so it cannot fix this.)

**Dropouts (`xruns` climbing).** Raise `blocksize` to 512. If that fixes it,
the driver could not keep up, not the DSP — the terminal shows the actual DSP
load, and if it reads a few percent the problem is upstream. Install an ASIO
driver.

**Hand not tracked.** Watch the camera window. If the frame rate is below 20,
add light. Set `hand: left` or `hand: right` so your fretting hand cannot grab
the controls.

**A gesture doesn't reach the end of its range.** Narrow `in_lo`/`in_hi` — see
above.

**`ModuleNotFoundError: No module named 'sounddevice'`.** You are running the
system Python instead of the virtualenv. Use `run.bat`, or
`.venv\Scripts\activate` first.

**`hand model not found`.** Run `python tools/fetch_model.py`.

**Sound is fine but gestures lag.** That is camera frame rate, not audio. See
the exposure section.

---

## Layout

```
run.py                      entry point, orchestrates both processes
run.bat                     launcher that always uses the virtualenv
live_pedal/
  ipc.py                    lock-free shared memory (seqlock)
  config.py                 YAML loading, presets, inheritance
  layout.py                 picklable parameter table description
  audio/
    devices.py              host API selection and latency expectations
    chain.py                effect chain, parameter table, JIT warmup
    engine.py               the real-time callback
  dsp/
    kernels.py              Numba-compiled primitives
    base.py                 effect base class, parameter specs
    drive.py filters.py modulation.py time_fx.py pitch.py utility.py
  vision/
    camera.py               threaded capture, exposure control
    tracker.py              MediaPipe hand landmarker
    features.py             landmarks to gesture features
    overlay.py              the preview window
    process.py              the vision process main loop
  mapping/mapper.py         gesture to parameter routing
configs/                    presets
tools/                      device list, latency measurement, benchmark
tests/                      gesture and mapping self-tests
```

---

## Status

Working and tested end to end on the development machine: 2253 blocks with zero
dropouts, worst callback 377 µs against a 5333 µs budget, camera at 30 fps.

Currently mono internally (guitar is a mono source) and duplicated to both
outputs. Stereo effects would be the natural next step.
