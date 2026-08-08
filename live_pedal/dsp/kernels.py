"""Numba-compiled DSP primitives.

Everything here is an ``njit`` function operating on preallocated arrays. That
matters for two reasons:

1. Recursive filters need a per-sample loop, which is hopeless in pure Python
   (48000 iterations/second/filter) but compiles to tight machine code here.
2. ``nogil=True`` means these kernels release the GIL while they run, so the
   audio callback is not fighting the rest of the interpreter for it.

State is always passed in as a small float64 array that the caller owns, so the
kernels themselves are pure and can be reused by several effect instances.

Convention: kernels write into ``y`` and never allocate. Where a parameter can
be swept by a gesture, the kernel takes ``*_a`` (value at the start of the block)
and ``*_b`` (value at the end) and interpolates across the block. That is what
keeps a fast hand movement from producing zipper noise.
"""

from __future__ import annotations

import numpy as np
from numba import njit

JIT = dict(cache=True, nogil=True, fastmath=True)

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


@njit(inline="always", **JIT)
def _clip(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@njit(inline="always", **JIT)
def _read_frac(buf, pos):
    """Linear-interpolated read from a circular buffer at fractional index."""
    n = buf.shape[0]
    i0 = int(pos)
    frac = pos - i0
    i0 = i0 % n
    i1 = i0 + 1
    if i1 >= n:
        i1 = 0
    return buf[i0] * (1.0 - frac) + buf[i1] * frac


@njit(inline="always", **JIT)
def _lfo(phase, shape):
    """Unipolar-to-bipolar LFO. 0=sine 1=triangle 2=square 3=ramp."""
    if shape == 0:
        return np.sin(TWO_PI * phase)
    elif shape == 1:
        t = 4.0 * phase
        if t < 1.0:
            return t
        elif t < 3.0:
            return 2.0 - t
        else:
            return t - 4.0
    elif shape == 2:
        return 1.0 if phase < 0.5 else -1.0
    else:
        return 2.0 * phase - 1.0


# ---------------------------------------------------------------------------
# State variable filter (Cytomic / Andy Simper topology-preserving transform)
# ---------------------------------------------------------------------------
# This is the workhorse for anything a gesture sweeps: wah, filter, phaser
# stages. Unlike a biquad it stays stable and well-behaved when the cutoff is
# modulated fast, which is exactly what a hand does.


@njit(**JIT)
def svf(x, y, g_a, g_b, k_a, k_b, mode, state):
    """Sweepable 2-pole SVF. mode: 0=LP 1=BP 2=HP 3=notch 4=peak.

    ``g = tan(pi * fc / sr)``, ``k = 1/Q``. Both are interpolated across the
    block so a fast sweep does not step.
    """
    n = x.shape[0]
    ic1 = state[0]
    ic2 = state[1]
    inv_n = 1.0 / n
    dg = (g_b - g_a) * inv_n
    dk = (k_b - k_a) * inv_n
    g = g_a
    k = k_a
    for i in range(n):
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v0 = x[i]
        v3 = v0 - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == 0:
            out = v2
        elif mode == 1:
            out = v1
        elif mode == 2:
            out = v0 - k * v1 - v2
        elif mode == 3:
            out = v0 - k * v1
        else:
            out = v0 - 2.0 * k * v1
        y[i] = out
        g += dg
        k += dk
    # Flush denormals: a decaying filter left alone can spend cycles in
    # denormal arithmetic, which on some CPUs is catastrophically slow.
    if abs(ic1) < 1e-20:
        ic1 = 0.0
    if abs(ic2) < 1e-20:
        ic2 = 0.0
    state[0] = ic1
    state[1] = ic2


@njit(**JIT)
def svf_1pole(x, y, g_a, g_b, highpass, state):
    """1-pole TPT filter, for tone controls and DC blocking."""
    n = x.shape[0]
    s = state[0]
    dg = (g_b - g_a) / n
    g = g_a
    for i in range(n):
        v = (x[i] - s) * g / (1.0 + g)
        lp = v + s
        s = lp + v
        y[i] = (x[i] - lp) if highpass else lp
        g += dg
    if abs(s) < 1e-20:
        s = 0.0
    state[0] = s


@njit(**JIT)
def biquad(x, y, b0, b1, b2, a1, a2, state):
    """Direct form II transposed biquad, fixed coefficients for the block."""
    n = x.shape[0]
    z1 = state[0]
    z2 = state[1]
    for i in range(n):
        xi = x[i]
        out = b0 * xi + z1
        z1 = b1 * xi - a1 * out + z2
        z2 = b2 * xi - a2 * out
        y[i] = out
    if abs(z1) < 1e-20:
        z1 = 0.0
    if abs(z2) < 1e-20:
        z2 = 0.0
    state[0] = z1
    state[1] = z2


# ---------------------------------------------------------------------------
# Envelope following and dynamics
# ---------------------------------------------------------------------------


@njit(**JIT)
def envelope(x, atk, rel, state):
    """Peak envelope follower. Returns the envelope at the end of the block."""
    e = state[0]
    for i in range(x.shape[0]):
        a = abs(x[i])
        if a > e:
            e = atk * e + (1.0 - atk) * a
        else:
            e = rel * e + (1.0 - rel) * a
    if e < 1e-20:
        e = 0.0
    state[0] = e
    return e


@njit(**JIT)
def noise_gate(x, y, thresh, atk, rel, hold_samples, state):
    """Downward-expanding gate with hold, so note tails are not chopped.

    state = [envelope, gain, hold_counter]
    """
    n = x.shape[0]
    e = state[0]
    gain = state[1]
    hold = state[2]
    for i in range(n):
        a = abs(x[i])
        if a > e:
            e = 0.7 * e + 0.3 * a          # fast attack on the detector
        else:
            e = 0.9995 * e                  # slow release on the detector
        if e > thresh:
            hold = hold_samples
            target = 1.0
        else:
            if hold > 0.0:
                hold -= 1.0
                target = 1.0
            else:
                target = 0.0
        if target > gain:
            gain = atk * gain + (1.0 - atk) * target
        else:
            gain = rel * gain + (1.0 - rel) * target
        y[i] = x[i] * gain
    if e < 1e-20:
        e = 0.0
    state[0] = e
    state[1] = gain
    state[2] = hold
    return e


# ---------------------------------------------------------------------------
# Waveshaping and oversampled distortion
# ---------------------------------------------------------------------------


@njit(inline="always", **JIT)
def _shape(v, kind, drive, asym):
    """Single-sample waveshaper. kind: 0=soft 1=hard 2=fuzz 3=tube 4=fold."""
    v = v * drive + asym
    if kind == 0:                                    # smooth overdrive
        return np.tanh(v)
    elif kind == 1:                                  # hard clip
        return _clip(v, -1.0, 1.0)
    elif kind == 2:                                  # fuzz: aggressive, gated edges
        if v > 0.0:
            return 1.0 - np.exp(-v)
        else:
            return -1.0 + np.exp(v)
    elif kind == 3:                                  # asymmetric tube-ish curve
        if v > 0.0:
            return v / (1.0 + v)
        else:
            return v / (1.0 - 0.6 * v)
    else:                                            # wave folder
        return np.sin(v * 1.5707963267948966)


# 4x oversampling FIR, 32 taps of windowed sinc at fs/4. Length is a multiple of
# 4 so it splits cleanly into 4 polyphase branches of 8 taps each: the upsampler
# then never multiplies by the stuffed zeros, and the decimator only evaluates
# the one output phase it keeps. That is ~70 MACs/sample instead of ~260.
# Group delay is 16 taps at 4x = 4 samples at base rate (0.08ms) -- inaudible.
OS_TAPS = 32
OS_PHASE = OS_TAPS // 4          # taps per polyphase branch


def _design_os_fir(taps: int = OS_TAPS, cutoff: float = 0.25) -> np.ndarray:
    """Unit-DC-gain lowpass. The x4 interpolation gain is applied in the kernel."""
    n = np.arange(taps) - (taps - 1) / 2.0
    h = np.sinc(2.0 * cutoff * n) * np.blackman(taps)
    return (h / h.sum()).astype(np.float64)


OS_FIR = _design_os_fir()


@njit(**JIT)
def drive_os4(x, y, kind, drive_a, drive_b, asym, fir, up_hist, down_hist, state):
    """4x oversampled waveshaping using polyphase interpolation/decimation.

    Waveshaping a guitar signal generates harmonics far above Nyquist; without
    oversampling they fold back as inharmonic aliasing, which is the fizzy,
    "digital" quality that makes cheap distortion plugins unpleasant. Running
    the nonlinearity at 4x and filtering before decimating pushes those
    products out of the audible band.

    ``up_hist`` holds OS_PHASE input samples, ``down_hist`` holds OS_TAPS
    samples at 4x rate; both are circular and carried across blocks so there is
    no seam. state = [up_write, down_write].
    """
    n = x.shape[0]
    ddrive = (drive_b - drive_a) / n
    drive = drive_a
    uw = int(state[0])
    dw = int(state[1])

    for i in range(n):
        # --- polyphase upsample: one input in, four 4x samples out --------
        up_hist[uw] = x[i]
        for s in range(4):
            acc = 0.0
            for t in range(OS_PHASE):
                idx = uw - t
                if idx < 0:
                    idx += OS_PHASE
                acc += fir[4 * t + s] * up_hist[idx]
            # --- nonlinearity at 4x, then straight into the decimator -----
            down_hist[dw] = _shape(acc * 4.0, kind, drive, asym)
            dw += 1
            if dw >= OS_TAPS:
                dw = 0
        uw += 1
        if uw >= OS_PHASE:
            uw = 0

        # --- decimate: evaluate the filter once, keep this one phase ------
        acc = 0.0
        for t in range(OS_TAPS):
            idx = dw - 1 - t
            if idx < 0:
                idx += OS_TAPS
            acc += fir[t] * down_hist[idx]
        y[i] = acc
        drive += ddrive

    state[0] = uw
    state[1] = dw


@njit(**JIT)
def drive_simple(x, y, kind, drive_a, drive_b, asym):
    """Non-oversampled waveshaper, for when CPU matters more than aliasing."""
    n = x.shape[0]
    ddrive = (drive_b - drive_a) / n
    drive = drive_a
    for i in range(n):
        y[i] = _shape(x[i], kind, drive, asym)
        drive += ddrive


# ---------------------------------------------------------------------------
# Delay lines
# ---------------------------------------------------------------------------


@njit(**JIT)
def mod_delay(x, y, buf, state, base_a, base_b, depth, rate_inc, shape,
              feedback, mix, spread):
    """Modulated delay: the engine behind chorus, flanger and vibrato.

    ``base_*`` is delay in samples (interpolated across the block), ``depth`` is
    the LFO excursion in samples. state = [write_index, lfo_phase].
    """
    n = x.shape[0]
    size = buf.shape[0]
    w = int(state[0])
    phase = state[1]
    dbase = (base_b - base_a) / n
    base = base_a
    for i in range(n):
        d = base + depth * (0.5 * (_lfo(phase, shape) + 1.0)) + spread
        if d < 1.0:
            d = 1.0
        if d > size - 2:
            d = size - 2
        rpos = w - d
        while rpos < 0.0:
            rpos += size
        wet = _read_frac(buf, rpos)
        buf[w] = x[i] + wet * feedback
        y[i] = x[i] * (1.0 - mix) + wet * mix
        w += 1
        if w >= size:
            w = 0
        phase += rate_inc
        if phase >= 1.0:
            phase -= 1.0
        base += dbase
    state[0] = w
    state[1] = phase


@njit(**JIT)
def delay_line(x, y, buf, state, delay_a, delay_b, feedback, mix, damp_g, damp_state):
    """Feedback delay with a one-pole lowpass in the feedback path.

    Damping in the loop is what makes repeats decay naturally instead of
    turning into harsh metallic noise. state = [write_index].
    """
    n = x.shape[0]
    size = buf.shape[0]
    w = int(state[0])
    d_inc = (delay_b - delay_a) / n
    d = delay_a
    lp = damp_state[0]
    for i in range(n):
        dd = d
        if dd < 1.0:
            dd = 1.0
        if dd > size - 2:
            dd = size - 2
        rpos = w - dd
        while rpos < 0.0:
            rpos += size
        wet = _read_frac(buf, rpos)
        lp = lp + damp_g * (wet - lp)
        buf[w] = x[i] + lp * feedback
        y[i] = x[i] * (1.0 - mix) + wet * mix
        w += 1
        if w >= size:
            w = 0
        d += d_inc
    if abs(lp) < 1e-20:
        lp = 0.0
    damp_state[0] = lp
    state[0] = w


# ---------------------------------------------------------------------------
# Reverb (Freeverb-style: parallel damped combs into series allpasses)
# ---------------------------------------------------------------------------

COMB_TUNING = np.array([1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617], dtype=np.int64)
ALLPASS_TUNING = np.array([556, 441, 341, 225], dtype=np.int64)


@njit(**JIT)
def reverb(x, y, comb_buf, comb_len, comb_idx, comb_store,
           ap_buf, ap_len, ap_idx, feedback, damp, width_unused):
    """Mono Freeverb. Buffers are flat arrays indexed by offset tables.

    ``comb_buf``/``ap_buf`` are single flat float64 arrays; ``*_len`` holds the
    length of each section and the offsets are the running sums. Keeping them
    flat avoids ragged-array support issues in nopython mode.
    """
    n = x.shape[0]
    n_comb = comb_len.shape[0]
    n_ap = ap_len.shape[0]
    damp1 = damp
    damp2 = 1.0 - damp

    for i in range(n):
        inp = x[i] * 0.015          # Freeverb's fixed input scaling
        acc = 0.0

        off = 0
        for c in range(n_comb):
            L = comb_len[c]
            idx = comb_idx[c]
            out = comb_buf[off + idx]
            store = comb_store[c] * damp1 + out * damp2
            comb_store[c] = store
            comb_buf[off + idx] = inp + store * feedback
            idx += 1
            if idx >= L:
                idx = 0
            comb_idx[c] = idx
            acc += out
            off += L

        off = 0
        for a in range(n_ap):
            L = ap_len[a]
            idx = ap_idx[a]
            bufout = ap_buf[off + idx]
            out = -acc + bufout
            ap_buf[off + idx] = acc + bufout * 0.5
            idx += 1
            if idx >= L:
                idx = 0
            ap_idx[a] = idx
            acc = out
            off += L

        y[i] = acc

    for c in range(n_comb):
        if abs(comb_store[c]) < 1e-20:
            comb_store[c] = 0.0


# ---------------------------------------------------------------------------
# Phaser: cascade of modulated 1-pole allpass stages
# ---------------------------------------------------------------------------


@njit(**JIT)
def phaser(x, y, n_stages, g_a, g_b, feedback, mix, ap_state, fb_state):
    n = x.shape[0]
    dg = (g_b - g_a) / n
    g = g_a
    fb = fb_state[0]
    for i in range(n):
        v = x[i] + fb * feedback
        for s in range(n_stages):
            z = ap_state[s]
            out = -g * v + z
            ap_state[s] = v + g * out
            v = out
        fb = v
        y[i] = x[i] * (1.0 - mix) + v * mix
        g += dg
    if abs(fb) < 1e-20:
        fb = 0.0
    fb_state[0] = fb


# ---------------------------------------------------------------------------
# Tremolo and volume
# ---------------------------------------------------------------------------


@njit(**JIT)
def tremolo(x, y, depth, rate_inc, shape, state):
    n = x.shape[0]
    phase = state[0]
    for i in range(n):
        m = 1.0 - depth * (0.5 * (1.0 - _lfo(phase, shape)))
        y[i] = x[i] * m
        phase += rate_inc
        if phase >= 1.0:
            phase -= 1.0
    state[0] = phase


@njit(**JIT)
def gain_ramp(x, y, g_a, g_b):
    n = x.shape[0]
    dg = (g_b - g_a) / n
    g = g_a
    for i in range(n):
        y[i] = x[i] * g
        g += dg


# ---------------------------------------------------------------------------
# Octave generation
# ---------------------------------------------------------------------------


@njit(**JIT)
def octave_down(x, y, state):
    """Zero-crossing flip-flop octave divider, the trick analog OC-2 style
    pedals use. Cheap, zero added latency, tracks single notes well and gets
    confused by chords -- exactly like the hardware."""
    n = x.shape[0]
    flip = state[0]
    prev = state[1]
    for i in range(n):
        v = x[i]
        if prev <= 0.0 and v > 0.0:
            flip = -flip
        prev = v
        y[i] = abs(v) * flip
    state[0] = flip
    state[1] = prev


@njit(**JIT)
def octave_up(x, y):
    """Full-wave rectification doubles the fundamental.

    Rectifying leaves a large DC offset, so the caller must high-pass the
    result before mixing it back in."""
    for i in range(x.shape[0]):
        y[i] = abs(x[i]) * 2.0


@njit(**JIT)
def pitch_shift(x, y, buf, state, ratio, window):
    """Two-tap crossfading delay-line pitch shifter.

    Two read pointers drift through the delay line at a rate set by ``ratio``,
    half a window apart, crossfaded with a raised-sine so the splice is masked.
    This is how classic hardware whammy/octave units work: cheap and low
    latency, at the cost of some warble on sustained chords.

    state = [write_index, phase]
    """
    n = x.shape[0]
    size = buf.shape[0]
    w = int(state[0])
    phase = state[1]
    rate = 1.0 - ratio
    half = window * 0.5
    for i in range(n):
        buf[w] = x[i]
        p1 = phase
        p2 = phase + half
        if p2 >= window:
            p2 -= window

        r1 = w - p1
        while r1 < 0.0:
            r1 += size
        r2 = w - p2
        while r2 < 0.0:
            r2 += size

        g1 = np.sin(np.pi * p1 / window)
        g2 = np.sin(np.pi * p2 / window)
        y[i] = _read_frac(buf, r1) * g1 + _read_frac(buf, r2) * g2

        w += 1
        if w >= size:
            w = 0
        phase += rate
        while phase >= window:
            phase -= window
        while phase < 0.0:
            phase += window
    state[0] = w
    state[1] = phase


# ---------------------------------------------------------------------------
# Mixing helpers
# ---------------------------------------------------------------------------


@njit(**JIT)
def mix_into(dst, src, amount):
    for i in range(dst.shape[0]):
        dst[i] += src[i] * amount


@njit(**JIT)
def crossfade(dst, dry, wet, mix):
    for i in range(dst.shape[0]):
        dst[i] = dry[i] * (1.0 - mix) + wet[i] * mix


@njit(**JIT)
def crossfade_ramp(dst, dry, wet, m_a, m_b):
    """Crossfade with the mix swept across the block.

    Used by the chain to engage and disengage effects. Ramping rather than
    switching is what stops a gesture toggling an effect from producing an
    audible click at the boundary."""
    n = dst.shape[0]
    dm = (m_b - m_a) / n
    m = m_a
    for i in range(n):
        dst[i] = dry[i] * (1.0 - m) + wet[i] * m
        m += dm


@njit(**JIT)
def smooth_params(current, target, coeff):
    """One-pole parameter smoothing, run once per block over the whole table.

    This is the single most important line in the project for making gestures
    feel good: raw landmark values jitter by a few percent frame to frame, and
    without smoothing that jitter is audible as grain on every parameter.
    """
    for i in range(current.shape[0]):
        c = coeff[i]
        current[i] = target[i] + (current[i] - target[i]) * c


@njit(**JIT)
def peak_level(x):
    p = 0.0
    for i in range(x.shape[0]):
        a = abs(x[i])
        if a > p:
            p = a
    return p


@njit(**JIT)
def sanitise(x, limit):
    """Last line of defence before the DAC: kill NaN/Inf and clamp.

    A runaway feedback path or a divide-by-zero in a swept filter can produce a
    NaN, and a NaN reaching the sound card is a very loud, very unpleasant
    event. Cheap insurance."""
    for i in range(x.shape[0]):
        v = x[i]
        if not (v == v) or v > 1e6 or v < -1e6:
            x[i] = 0.0
        elif v > limit:
            x[i] = limit
        elif v < -limit:
            x[i] = -limit
