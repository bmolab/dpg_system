"""The physical-modelling regression suite: every unit, without GUI or
audio device.

Run with the project environment:
    python dpg_system/tests/test_physical.py
94 checks of validated behavior -- pitch laws, decay laws, normalization
balances, the emergent physics (bow locking, brass staircases, strain's
regenerative squeal, bounce cadence) -- grown check by check alongside
the instruments themselves. Every 'ok' line was once a measurement that
steered a design decision; a FAIL means a law this rack relies on has
shifted."""
import os
import sys
import math
import numpy as np
from scipy.signal import hilbert as sig_hilbert
from scipy.signal import find_peaks as sig_find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
from dpg_system import synth_core as sc

SR = 44100
BLOCK = 512

# Compile kernels synchronously so the units render for real.
sc._warm_up_filter()
assert sc._svf_ready.is_set(), 'kernels failed to compile'


def run(unit, seconds, feed=None):
    blocks = int(seconds * SR / BLOCK)
    out = np.zeros(blocks * BLOCK, dtype=np.float64)
    for b in range(blocks):
        if feed is not None:
            feed(unit, b)
        unit.render(BLOCK)
        out[b * BLOCK:(b + 1) * BLOCK] = unit.out.array(BLOCK)
    return out


def peak_frequency(x):
    """Fundamental via autocorrelation: robust to a dominant harmonic."""
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode='full')[x.shape[0] - 1:]
    corr /= corr[0]
    min_lag = int(SR / 2000.0)
    # First lag where correlation rises again after the initial fall,
    # searched for the tallest peak past that point.
    trough = min_lag
    while trough < corr.shape[0] - 1 and corr[trough] > 0.0:
        trough += 1
    lag = trough + np.argmax(corr[trough:trough + SR // 20])
    return SR / lag


def spectral_peaks(x, count=8):
    window = np.hanning(x.shape[0])
    spectrum = np.abs(np.fft.rfft(x * window))
    spectrum[:4] = 0.0
    peaks = []
    for i in range(1, spectrum.shape[0] - 1):
        if spectrum[i] > spectrum[i - 1] and spectrum[i] > spectrum[i + 1]:
            peaks.append((spectrum[i], i * SR / x.shape[0]))
    peaks.sort(reverse=True)
    return sorted(f for _, f in peaks[:count])


failures = []


def check(name, condition, detail=''):
    status = 'ok' if condition else 'FAIL'
    print(f'{status:5s} {name} {detail}')
    if not condition:
        failures.append(name)


# --- string~: pluck, pitch, decay -------------------------------------------
s = sc.StringUnit(SR)
s.frequency_in.base = 220.0
s.decay_in.base = 2.0
s.fire()
audio = run(s, 1.0)
check('string sounds', np.max(np.abs(audio)) > 0.01,
      f'peak={np.max(np.abs(audio)):.3f}')
f = peak_frequency(audio[4096:4096 + 32768])
check('string pitch ~220', abs(f - 220.0) < 4.0, f'measured={f:.2f} Hz')

early = np.sqrt(np.mean(audio[2048:6144] ** 2))
late = np.sqrt(np.mean(audio[-8192:] ** 2))
check('string decays', late < early, f'early={early:.4f} late={late:.4f}')
check('string still ringing at 1s (t60=2s)', late > 1e-4, f'late={late:.5f}')

# quiet path: after long silence the outlet goes constant
s2 = sc.StringUnit(SR)
s2.decay_in.base = 0.3
s2.fire()
run(s2, 4.0)
s2.render(BLOCK)
check('string goes quiet-constant', s2.out.constant)

# pitch inlet: +1 octave
s3 = sc.StringUnit(SR)
s3.frequency_in.base = 220.0
s3.pitch_in.base = 1.0
s3.fire()
audio3 = run(s3, 0.8)
f3 = peak_frequency(audio3[4096:4096 + 32768])
check('string pitch inlet +1 oct ~440', abs(f3 - 440.0) < 6.0,
      f'measured={f3:.2f} Hz')

# stiffness on: still roughly in tune at the fundamental
s4 = sc.StringUnit(SR)
s4.frequency_in.base = 220.0
s4.stiffness_in.base = 0.3
s4.fire()
audio4 = run(s4, 0.8)
f4 = peak_frequency(audio4[4096:4096 + 32768])
check('string stiff fundamental near 220', abs(f4 - 220.0) < 8.0,
      f'measured={f4:.2f} Hz')

# tube mode: odd harmonics, pitch preserved
t = sc.StringUnit(SR)
t.frequency_in.base = 220.0
t.mode = 1
t.fire()
taudio = run(t, 0.8)
ft = peak_frequency(taudio[4096:4096 + 32768])
check('tube pitch ~220', abs(ft - 220.0) < 6.0, f'measured={ft:.2f} Hz')
window = np.hanning(16384)
spec = np.abs(np.fft.rfft(taudio[4096:4096 + 16384] * window))
bin_of = lambda hz: int(round(hz * 16384 / SR))
h2 = spec[bin_of(440) - 2:bin_of(440) + 3].max()
h3 = spec[bin_of(660) - 2:bin_of(660) + 3].max()
check('tube favours odd harmonics', h3 > 3.0 * h2,
      f'h2={h2:.2f} h3={h3:.2f}')

# trigger signal with velocity: two plucks, second half as tall
s5 = sc.StringUnit(SR)
s5.frequency_in.base = 220.0
s5.decay_in.base = 0.5
trig = sc.Signal()
s5.trigger_in.sources.append(trig)
levels = []
for velocity in (1.0, 0.5):
    trig.data[:BLOCK] = 0.0
    trig.data[10] = velocity
    trig.constant = False
    s5.render(BLOCK)
    trig.set_constant(0.0)
    burst = run(s5, 0.2)
    levels.append(np.max(np.abs(burst)))
check('trigger velocity scales pluck', 0.3 < levels[1] / levels[0] < 0.8,
      f'ratio={levels[1] / levels[0]:.2f}')

# continuous excitation through the audio inlet
s6 = sc.StringUnit(SR)
s6.frequency_in.base = 110.0
noise_source = sc.Signal()
s6.excite_in.sources.append(noise_source)
rng = np.random.default_rng(7)


def feed_noise(unit, block):
    noise_source.data[:BLOCK] = rng.uniform(-0.2, 0.2, BLOCK)
    noise_source.constant = False


bowed = run(s6, 0.5, feed_noise)
f6 = peak_frequency(bowed[8192:8192 + 16384])
check('bowed string resonates at pitch', abs(f6 - 110.0) < 4.0,
      f'measured={f6:.2f} Hz')

# DC excitation must not pin the line
s7 = sc.StringUnit(SR)
dc = sc.Signal()
dc.set_constant(0.5)
s7.excite_in.sources.append(dc)
dc_audio = run(s7, 0.5)
check('DC excitation stays bounded', np.max(np.abs(dc_audio)) < 2.0,
      f'peak={np.max(np.abs(dc_audio)):.3f}')

# --- modal~ ------------------------------------------------------------------
BELL = [
    (0.5, 0.8, 1.6), (1.0, 1.0, 1.0), (1.2, 0.8, 0.8), (1.5, 0.6, 0.7),
    (2.0, 0.7, 0.6), (2.5, 0.5, 0.5),
]
m = sc.ModalUnit(SR)
m.frequency_in.base = 440.0
m.decay_in.base = 3.0
m.set_modes(BELL)
m.fire()
maudio = run(m, 1.0)
check('modal sounds', np.max(np.abs(maudio)) > 0.01,
      f'peak={np.max(np.abs(maudio)):.3f}')
peaks = spectral_peaks(maudio[2048:2048 + 32768], count=6)
wanted = [440.0 * r for r, _, _ in BELL]
hits = sum(1 for w in wanted if any(abs(p - w) < 6.0 for p in peaks))
check('modal peaks land on mode table', hits >= 5,
      f'wanted={[round(w) for w in wanted]} got={[round(p) for p in peaks]}')

mearly = np.sqrt(np.mean(maudio[2048:6144] ** 2))
mlate = np.sqrt(np.mean(maudio[-8192:] ** 2))
check('modal decays', mlate < mearly, f'early={mearly:.4f} late={mlate:.4f}')

# hardness: soft strike has less high-mode energy than hard strike
def strike_energy(hardness):
    unit = sc.ModalUnit(SR)
    unit.frequency_in.base = 440.0
    unit.set_modes(BELL)
    unit.hardness_in.base = hardness
    unit.fire()
    audio = unit_audio = run(unit, 0.4)
    window = np.hanning(8192)
    spectrum = np.abs(np.fft.rfft(unit_audio[1024:1024 + 8192] * window))
    split = bin_high = int(800 * 8192 / SR)
    return spectrum[split:].sum() / max(1e-9, spectrum[:split].sum())


check('hardness brightens attack',
      strike_energy(1.0) > 1.5 * strike_energy(0.0),
      f'hard={strike_energy(1.0):.3f} soft={strike_energy(0.0):.3f}')

# transpose past Nyquist: no fold, no blowup
mh = sc.ModalUnit(SR)
mh.frequency_in.base = 440.0
mh.pitch_in.base = 6.0    # 28 kHz fundamental: most modes muted
mh.set_modes(BELL)
mh.fire()
haudio = run(mh, 0.3)
check('supersonic modes muted not folded', np.max(np.abs(haudio)) < 1.0,
      f'peak={np.max(np.abs(haudio)):.4f}')

# quiet path
mq = sc.ModalUnit(SR)
mq.set_modes([(1.0, 1.0, 0.1)])
mq.decay_in.base = 0.2
mq.fire()
run(mq, 3.0)
mq.render(BLOCK)
check('modal goes quiet-constant', mq.out.constant)

# audio-inlet drive: a sustained tone on a mode blooms, bounded -- the
# filter normalization at work (struck gains here would multiply by Q).
mx = sc.ModalUnit(SR)
mx.frequency_in.base = 300.0
mx.decay_in.base = 3.0
mx.set_modes(BELL)
mx_source = sc.Signal()
mx.excite_in.sources.append(mx_source)


def feed_on_mode(unit, block):
    n = np.arange(block * BLOCK, (block + 1) * BLOCK)
    mx_source.data[:BLOCK] = 0.5 * np.sin(2 * np.pi * 300.0 * n / SR)
    mx_source.constant = False


xaudio = run(mx, 3.0, feed_on_mode)
early_peak = np.max(np.abs(xaudio[:22050]))
late_peak = np.max(np.abs(xaudio[-8192:]))
check('modal drive audible within half a second', early_peak > 0.05,
      f'peak={early_peak:.3f}')
check('modal drive bounded when parked on-mode', late_peak < 6.0,
      f'peak={late_peak:.3f}')

# a bowed sawtooth (the patch that matters): harmonics land on 1x and 2x
# modes of the bell table, and the bank must answer promptly at dry 0
mb = sc.ModalUnit(SR)
mb.frequency_in.base = 220.0
mb.decay_in.base = 3.0
mb.set_modes(BELL)
mb_source = sc.Signal()
mb.excite_in.sources.append(mb_source)


def feed_saw(unit, block):
    n = np.arange(block * BLOCK, (block + 1) * BLOCK)
    phase = (n * 220.0 / SR) % 1.0
    mb_source.data[:BLOCK] = 0.4 * (2.0 * phase - 1.0)
    mb_source.constant = False


baudio = run(mb, 1.0, feed_saw)
bpeak = np.max(np.abs(baudio[11025:22050]))
check('bowing activates the bank promptly', bpeak > 0.05,
      f'peak at 0.25-0.5s={bpeak:.3f}')

print()
if failures:
    print('FAILURES:', failures)
    sys.exit(1)
print('all checks passed')

# --- wind~ ------------------------------------------------------------------
def run_wind(mode, freq, pressure, seconds=2.0, emb=0.5):
    unit = sc.WindUnit(SR)
    unit.mode = mode
    unit.frequency_in.base = freq
    unit.pressure_in.base = pressure
    unit.embouchure_in.base = emb
    return unit, run(unit, seconds)


_, quiet_air = run_wind(0, 220.0, 0.0, seconds=0.3)
check('reed silent with no breath', np.max(np.abs(quiet_air)) < 1e-6)

_, reed_low = run_wind(0, 220.0, 0.3)
check('reed below threshold: near silence',
      np.max(np.abs(reed_low[-22050:])) < 0.05,
      f'peak={np.max(np.abs(reed_low[-22050:])):.4f}')

for f in (110.0, 220.0, 440.0, 880.0):
    _, a = run_wind(0, f, 0.8)
    got = peak_frequency(a[-22050:])
    cents = 1200 * np.log2(got / f)
    check(f'reed in tune at {f:.0f}', abs(cents) < 12.0,
          f'got={got:.2f} ({cents:+.1f} cents)')

_, reed_tone = run_wind(0, 220.0, 0.8)
check('reed speaks', np.max(np.abs(reed_tone[-22050:])) > 0.3,
      f'peak={np.max(np.abs(reed_tone[-22050:])):.3f}')
spec = np.abs(np.fft.rfft(reed_tone[-16384:] * np.hanning(16384)))
b = lambda hz: int(round(hz * 16384 / SR))
even = spec[b(441) - 3:b(441) + 4].max()
odd = spec[b(661) - 3:b(661) + 4].max()
check('reed favours odd harmonics', odd > 2.0 * even,
      f'h2={even:.2f} h3={odd:.2f}')

_, flute_low = run_wind(1, 440.0, 0.5)
check('flute below threshold: breathy near-silence',
      np.max(np.abs(flute_low[-22050:])) < 0.1,
      f'peak={np.max(np.abs(flute_low[-22050:])):.4f}')

for f in (110.0, 220.0, 440.0, 880.0):
    _, a = run_wind(1, f, 0.95)
    got = peak_frequency(a[-22050:])
    cents = 1200 * np.log2(got / f) if got > 0 else 9999
    check(f'flute in tune at {f:.0f}', abs(cents) < 12.0,
          f'got={got:.2f} ({cents:+.1f} cents)')

_, flute_tone = run_wind(1, 440.0, 0.95)
check('flute speaks', np.max(np.abs(flute_tone[-22050:])) > 0.3,
      f'peak={np.max(np.abs(flute_tone[-22050:])):.3f}')

_, over = run_wind(1, 440.0, 0.95, emb=0.9)
got = peak_frequency(over[-22050:])
check('flute overblows to octave at long jet', got > 800.0, f'got={got:.1f}')

# pitch inlet works through _build_hertz
u, _ = run_wind(0, 220.0, 0.8, seconds=0.5)
u.pitch_in.base = 1.0
a = run(u, 1.5)
got = peak_frequency(a[-22050:])
check('wind pitch inlet +1 oct', abs(got - 440.0) < 8.0, f'got={got:.2f}')

# quiet path after breath released
u2, _ = run_wind(0, 220.0, 0.8, seconds=0.5)
u2.pressure_in.base = 0.0
run(u2, 2.0)
u2.render(BLOCK)
check('wind goes quiet-constant after release', u2.out.constant)

print()
if failures:
    print('FAILURES:', failures)
    sys.exit(1)
print('all wind checks passed')

# --- bow~ -------------------------------------------------------------------
def pitch_sub_aware(x):
    """Autocorrelation pitch that prefers the smallest comparable-peak lag,
    so a strongly periodic signal is not read at a multiple of its period."""
    x = x - np.mean(x)
    if np.max(np.abs(x)) < 1e-4:
        return 0.0
    corr = np.correlate(x, x, mode='full')[x.shape[0] - 1:]
    corr /= corr[0]
    trough = int(SR / 4000.0)
    while trough < corr.shape[0] - 1 and corr[trough] > 0.0:
        trough += 1
    window = corr[trough:trough + SR // 25]
    lag = trough + int(np.argmax(window))
    best = lag
    for k in (4, 3, 2):
        cand = int(round(lag / k))
        if cand < max(trough, 4):
            continue
        local = int(cand - 3 + np.argmax(corr[cand - 3:cand + 4]))
        if corr[local] >= 0.93 * corr[lag]:
            best = local
            break
    return SR / best


def run_bow_unit(freq, vel, force=0.5, seconds=2.0):
    unit = sc.BowUnit(SR)
    unit.frequency_in.base = freq
    unit.velocity_in.base = vel
    unit.force_in.base = force
    return unit, run(unit, seconds)


_, silent_bow = run_bow_unit(220.0, 0.0, seconds=0.3)
check('bow silent with bow lifted', np.max(np.abs(silent_bow)) < 1e-6)

for f in (110.0, 220.0, 440.0, 880.0):
    _, a = run_bow_unit(f, 0.8, 0.5)
    got = pitch_sub_aware(a[-22050:])
    cents = 1200 * np.log2(got / f) if got > 0 else 9999
    check(f'bow in tune at {f:.0f}', abs(cents) < 30.0,
          f'got={got:.2f} ({cents:+.1f} cents)')

_, tone = run_bow_unit(220.0, 0.8, 0.5)
check('bow speaks', np.max(np.abs(tone[-22050:])) > 0.2,
      f'peak={np.max(np.abs(tone[-22050:])):.3f}')

spec = np.abs(np.fft.rfft(tone[-16384:] * np.hanning(16384)))
b = lambda hz: int(round(hz * 16384 / SR))
h = [spec[b(220 * k) - 3:b(220 * k) + 4].max() for k in (1, 2, 3)]
check('bow sawtooth rolloff (h1>h2>h3)', h[0] > h[1] > h[2],
      f'h={[round(x, 1) for x in h]}')

_, soft = run_bow_unit(220.0, 0.4, 0.3)
_, hard = run_bow_unit(220.0, 1.0, 0.9)
check('bow dynamics: harder is louder',
      np.max(np.abs(hard[-22050:])) > 2.0 * np.max(np.abs(soft[-22050:])),
      f'soft={np.max(np.abs(soft[-22050:])):.3f} '
      f'hard={np.max(np.abs(hard[-22050:])):.3f}')

# whistle regime: fast light bow leaves the fundamental
_, whistle = run_bow_unit(440.0, 1.5, 0.0)
got = pitch_sub_aware(whistle[-22050:])
check('bow whistles when pushed light and fast', got > 700.0,
      f'got={got:.1f}')

# bow lifted mid-note: rings down and goes constant
ub, _ = run_bow_unit(220.0, 0.8, 0.5, seconds=0.5)
ub.velocity_in.base = 0.0
run(ub, 3.0)
ub.render(BLOCK)
check('bow goes quiet-constant after lift', ub.out.constant)

print()
if failures:
    print('FAILURES:', failures)
    sys.exit(1)
print('all bow checks passed')

# --- modal~ dry knob --------------------------------------------------------
def modal_with_input(dry_level, feed):
    unit = sc.ModalUnit(SR)
    unit.frequency_in.base = 300.0
    unit.decay_in.base = 3.0     # narrow modes: 777 Hz lands on none
    unit.set_modes(BELL)
    unit.dry_in.base = dry_level
    source = sc.Signal()
    unit.excite_in.sources.append(source)
    return run(unit, 0.6, lambda u, blk: feed(source, blk))


def feed_sine_777(source, blk):
    # 777 Hz: lands on none of the 300*ratio modes
    n = np.arange(blk * BLOCK, (blk + 1) * BLOCK)
    source.data[:BLOCK] = 0.5 * np.sin(2 * np.pi * 777.0 * n / SR)
    source.constant = False


wet_only = modal_with_input(0.0, feed_sine_777)
with_dry = modal_with_input(1.0, feed_sine_777)
r_wet = np.sqrt(np.mean(wet_only[-8192:] ** 2))
r_dry = np.sqrt(np.mean(with_dry[-8192:] ** 2))
check('dry passes what the modes reject', r_dry > 5.0 * r_wet,
      f'wet-only rms={r_wet:.4f} dry rms={r_dry:.4f}')
check('dry level is unity-ish', 0.25 < r_dry / (0.5 / np.sqrt(2)) < 1.3,
      f'ratio to source rms={r_dry / (0.5 / np.sqrt(2)):.2f}')

# strike pulse stays out of the dry path: a struck bell at dry 1 with no
# input should sound the same as at dry 0
md0 = sc.ModalUnit(SR)
md0.frequency_in.base = 440.0
md0.set_modes(BELL)
md0.fire()
a0 = run(md0, 0.5)
md1 = sc.ModalUnit(SR)
md1.frequency_in.base = 440.0
md1.set_modes(BELL)
md1.dry_in.base = 1.0
md1.fire()
a1 = run(md1, 0.5)
check('strike click stays out of dry tap',
      np.max(np.abs(a0 - a1)) < 1e-12,
      f'max diff={np.max(np.abs(a0 - a1)):.2e}')

print()
if failures:
    print('FAILURES:', failures)
    sys.exit(1)
print('all dry-knob checks passed')

# --- rub~ -------------------------------------------------------------------
GLASS = [(1.0, 1.0, 1.0), (2.32, 0.5, 0.75), (4.25, 0.3, 0.5),
         (6.63, 0.2, 0.32), (9.38, 0.1, 0.2)]


def run_rub(vel, force=0.4, freq=440.0, seconds=3.0, unit=None):
    if unit is None:
        unit = sc.RubUnit(SR)
        unit.frequency_in.base = freq
        unit.set_modes(GLASS)
    unit.velocity_in.base = vel
    unit.force_in.base = force
    return unit, run(unit, seconds)


def purity_and_dom(x):
    tail = x[-32768:]
    spec = np.abs(np.fft.rfft(tail * np.hanning(32768))) ** 2
    spec[:8] = 0
    dom = int(np.argmax(spec))
    return spec[dom - 3:dom + 4].sum() / max(1e-12, spec.sum()), dom * SR / 32768


_, silent = run_rub(0.0, seconds=0.3)
check('rub silent with bow lifted', np.max(np.abs(silent)) < 1e-6)

u, sung = run_rub(0.7)
peak = np.max(np.abs(sung[-16384:]))
pur, dom = purity_and_dom(sung)
check('rub sings', peak > 0.2, f'peak={peak:.3f}')
# A bowed thing is loud in proportion to how fast it is bowed, and the
# reason is kinematic: through the stuck part of each cycle the surface
# is carried at the speed of the hair, so the distance it travels goes
# with that speed. Six decibels a doubling. Without a real stick the
# amplitude is set instead by where friction balances damping, which
# hardly moves with bow speed -- so the old smooth friction curve was
# silent below the threshold where oscillation starts and very nearly
# full voice above it: twenty-five decibels of arrival for a one-and-a-
# half-fold change in speed, and not playable.
_rub_slope = []
_rub_prev = None
for _v in (0.05, 0.1, 0.2, 0.3, 0.45, 0.6):
    _, _ry = run_rub(_v, seconds=1.6)
    _rr = float(np.sqrt(np.mean(_ry[int(0.7 * SR):] ** 2)))
    if _rub_prev is not None and _rr > 1e-9 and _rub_prev[1] > 1e-9:
        _rub_slope.append(20 * np.log10(_rr / _rub_prev[1])
                          / np.log2(_v / _rub_prev[0]))
    _rub_prev = (_v, _rr)

check('rub is loud in proportion to bow speed, as Schelleng says',
      3.5 < float(np.median(_rub_slope)) < 8.5,
      f'{np.median(_rub_slope):.1f} dB per doubling '
      f'(6 is the law; a fixed-amplitude oscillator would give 0)')
check('rub speaks at a slow bow rather than waiting for a threshold',
      float(np.sqrt(np.mean(run_rub(0.05, seconds=1.6)[1][-8192:] ** 2)))
      > 1e-4,
      'a smooth friction curve was silent below about 0.6')

check('rub locks to fundamental, near-pure', pur > 0.8 and abs(dom - 440.0) < 10.0,
      f'purity={pur:.2f} dom={dom:.1f}')

_, squeal = run_rub(1.5)
pur2, dom2 = purity_and_dom(squeal)
check('rub squeals to a higher mode when pushed', dom2 > 700.0,
      f'dom={dom2:.1f}')

# release: bow to a stop and the glass keeps ringing
u3, _ = run_rub(0.7, seconds=2.0)
u3.velocity_in.base = 0.0
ring = run(u3, 1.0)
check('rub rings after the bow lifts',
      np.sqrt(np.mean(ring[-8192:] ** 2)) > 0.01,
      f'ring rms at +1s={np.sqrt(np.mean(ring[-8192:] ** 2)):.4f}')
run(u3, 30.0)
u3.render(BLOCK)
check('rub goes quiet-constant eventually', u3.out.constant)

# force presses the tone quieter without unlocking it
_, light = run_rub(0.7, force=0.1)
_, heavy = run_rub(0.7, force=0.9)
lp = np.max(np.abs(light[-16384:]))
hp_ = np.max(np.abs(heavy[-16384:]))
check('rub force presses quieter', hp_ < lp, f'light={lp:.3f} heavy={hp_:.3f}')

# ---------------------------------------------------------------- noise~
def run_noise(pressure=1.0, color=0.85, sputter=0.0, rate=10.0, seconds=2.0):
    u = sc.NoiseUnit(SR)
    u.pressure_in.base = pressure
    u.color_in.base = color
    u.sputter_in.base = sputter
    u.rate_in.base = rate
    n = int(seconds * SR / BLOCK)
    y = np.zeros(n * BLOCK)
    for b in range(n):
        u.render(BLOCK)
        y[b*BLOCK:(b+1)*BLOCK] = u.out.array(BLOCK)
    return y[SR//4:]

def _rms(y):
    return float(np.sqrt(np.mean(y * y)))

def _centroid(y):
    m = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    return float((m * f).sum() / m.sum())

ny = run_noise()
check('noise steady hiss bounded',
      np.isfinite(ny).all() and np.max(np.abs(ny)) < 2.0,
      f'rms={_rms(ny):.3f}')

nd, nb = run_noise(color=0.2), run_noise(color=1.0)
check('noise color tilts spectrum', _centroid(nb) > 4.0 * _centroid(nd),
      f'{_centroid(nd):.0f} -> {_centroid(nb):.0f} Hz')
check('noise color holds loudness', 0.6 < _rms(nb) / _rms(nd) < 1.7,
      f'ratio={_rms(nb)/_rms(nd):.2f}')

check('noise pressure is steep',
      _rms(run_noise(pressure=1.6)) > 8.0 * _rms(run_noise(pressure=0.4)))

nu = sc.NoiseUnit(SR)
nu.pressure_in.base = 0.0
for _ in range(40):
    nu.render(BLOCK)
check('noise zero pressure is constant silence',
      nu.out.constant and nu.out.array(BLOCK).max() == 0.0)

def _dropouts(y):
    win = max(1, int(SR / 1000))
    env = np.convolve(np.abs(y), np.ones(win) / win, mode='valid')
    low = env < 0.25 * np.median(env)
    return float(np.mean(low)), int(np.sum(low[1:] & ~low[:-1]))

frac0, _ = _dropouts(run_noise(sputter=0.0))
frac1, _ = _dropouts(run_noise(sputter=0.9, rate=8.0, seconds=4.0))
check('noise sputter drops out', frac0 < 0.01 and frac1 > 0.10,
      f'{frac0*100:.1f}% -> {frac1*100:.1f}%')

nsp = run_noise(sputter=0.9, rate=8.0, seconds=4.0)
npl = run_noise(sputter=0.0)
check('noise reopen spits',
      np.max(np.abs(nsp)) / _rms(nsp) > 1.4 * np.max(np.abs(npl)) / _rms(npl))

_, slow_eps = _dropouts(run_noise(sputter=0.9, rate=3.0, seconds=4.0))
_, fast_eps = _dropouts(run_noise(sputter=0.9, rate=30.0, seconds=4.0))
check('noise rate is flutter tempo', fast_eps > 2.5 * slow_eps,
      f'{slow_eps} at 3 Hz, {fast_eps} at 30 Hz')

# ---------------------------------------------------- bounce~ and drum~
MEMBRANE6 = [(1.0, 1.0, 1.0), (1.59, 0.7, 0.8), (2.14, 0.55, 0.65),
             (2.30, 0.45, 0.6), (2.65, 0.4, 0.5), (2.92, 0.35, 0.45)]

def run_bounce(bounce=0.7, press=0.0, gravity=0.3, seconds=4.0):
    u = sc.BounceUnit(SR)
    u.gravity_in.base = gravity
    u.bounce_in.base = bounce
    u.press_in.base = press
    u.hardness_in.base = 0.9
    s = sc.Signal()
    u.drop_in.sources.append(s)
    n = int(seconds * SR / BLOCK)
    y = np.zeros(n * BLOCK)
    for b in range(n):
        t0 = b * BLOCK / SR
        s.data[:BLOCK] = 1.0 if t0 < 0.01 else 0.0
        s.constant = False
        u.render(BLOCK)
        y[b*BLOCK:(b+1)*BLOCK] = u.out.array(BLOCK)
    return y, u, s

by, bu_unit, bsig = run_bounce()
benv = np.abs(by)
bpeaks = []
bi = 0
thresh_pk = 0.2 * benv.max()
while bi < len(benv):
    if benv[bi] > thresh_pk:
        j = min(bi + int(0.01*SR), len(benv))
        bpeaks.append(bi + int(np.argmax(benv[bi:j])))
        bi = j + int(0.005*SR)
    else:
        bi += 1
biv = np.diff(np.array(bpeaks) / SR)
biv = biv[biv > 0.008]
bratios = biv[1:] / biv[:-1]
check('bounce cadence is geometric (gravity, not a pattern)',
      len(bratios) >= 3 and 0.55 < np.median(bratios[:5]) < 0.85,
      f'ratios {[f"{r:.2f}" for r in bratios[:4]]} target ~0.7')
bsig.constant = True
bsig.value = 0.0
for _ in range(40):
    bu_unit.render(BLOCK)
check('bounce comes to rest and goes constant', bu_unit.out.constant)

def roll_active(press):
    y, _, _ = run_bounce(bounce=0.85, press=press, seconds=5.0)
    nz = np.nonzero(np.abs(y) > 1e-4)[0]
    return nz[-1] / SR if len(nz) else 0.0
check('bounce press shortens the roll',
      roll_active(0.7) < 0.6 * roll_active(0.0))

def drum_hit(height, tension=0.9, snares=0.0):
    d = sc.DrumUnit(SR)
    d.set_modes(MEMBRANE6)
    d.frequency_in.base = 170.0
    d.decay_in.base = 0.7
    d.tension_in.base = tension
    d.snares_in.base = snares
    d.hardness_in.base = 0.75
    trig = sc.Signal()
    d.trigger_in.sources.append(trig)
    trig.data[:BLOCK] = 0.0
    trig.data[4] = height
    trig.constant = False
    d.render(BLOCK)
    trig.constant = True
    trig.value = 0.0
    n = int(1.5*SR/BLOCK)
    y = np.zeros(n*BLOCK)
    y[:BLOCK] = d.out.array(BLOCK)
    for b in range(1, n):
        d.render(BLOCK)
        y[b*BLOCK:(b+1)*BLOCK] = d.out.array(BLOCK)
    return y

def _dom_low(seg):
    m = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))**2
    f = np.fft.rfftfreq(len(seg), 1.0/SR)
    band = (f > 60) & (f < 1200)
    return f[band][np.argmax(m[band])]

dy = drum_hit(1.0)
d_early = _dom_low(dy[:int(0.05*SR)])
d_late = _dom_low(dy[int(0.5*SR):int(1.1*SR)])
check('drum tension: full hit lands sharp and bends down',
      d_early > 1.12 * d_late, f'{d_early:.0f} -> {d_late:.0f} Hz')
dsoft = drum_hit(0.3)
check('drum tension is quadratic in the hit',
      _dom_low(dsoft[:int(0.05*SR)]) < d_early)
dflat = drum_hit(1.0, tension=0.0)
# a 150 ms window: the 50 ms one's FFT bins are coarser than the
# tolerance being asserted
check('drum tension 0 is flat',
      abs(_dom_low(dflat[:int(0.15*SR)])
          - _dom_low(dflat[int(0.5*SR):int(1.1*SR)])) < 0.05 * d_late)

dsn = drum_hit(1.0, tension=0.0, snares=0.9)
def _hf(seg):
    m = np.abs(np.fft.rfft(seg))**2
    f = np.fft.rfftfreq(len(seg), 1.0/SR)
    return m[f > 2000].sum() / m.sum()
check('drum snares rattle with the ring and die with it',
      _hf(dsn[:int(0.15*SR)]) > 4 * _hf(dflat[:int(0.15*SR)])
      and np.sqrt(np.mean(dsn[int(1.2*SR):]**2)) < 0.01)

bu2 = sc.BounceUnit(SR)
bu2.gravity_in.base = 0.15
bu2.bounce_in.base = 0.88
bu2.press_in.base = 0.5
bs2 = sc.Signal()
bu2.drop_in.sources.append(bs2)
dr2 = sc.DrumUnit(SR)
dr2.set_modes(MEMBRANE6)
dr2.frequency_in.base = 185.0
dr2.decay_in.base = 0.18
dr2.snares_in.base = 0.85
dr2.excite_in.sources.append(bu2.out)
nroll = int(3.0*SR/BLOCK)
ry = np.zeros(nroll*BLOCK)
for b in range(nroll):
    t = (b*BLOCK)/SR
    bs2.data[:BLOCK] = 0.6 if (t % 0.4) < 0.02 else 0.0
    bs2.constant = False
    bu2.render(BLOCK)
    dr2.render(BLOCK)
    ry[b*BLOCK:(b+1)*BLOCK] = dr2.out.array(BLOCK)
check('bounce~ into drum~ makes a sustained bounded roll',
      np.isfinite(ry).all() and 0.005 < np.sqrt(np.mean(ry[SR:]**2)) < 1.0
      and np.max(np.abs(ry)) < 2.0,
      f'rms {np.sqrt(np.mean(ry[SR:]**2)):.3f}')

# The same excitation should arrive at about the same loudness whatever
# it is driving. A stick on a bass string and a stick on a drum head are
# not thirty decibels apart. drum~ used to be impulse-normalized on the
# one buffer it had -- right for a mallet, and it multiplies anything
# SUSTAINED by the mode's Q, so bowing a drum came out thirty-two
# decibels over bowing modal~ and clipped at 2.74. The mallet has its
# own buffer and gain now, so the strike keeps the impulse convention
# and the excite input gets the same sqrt(1-r) modal~ uses.
_xr = np.random.default_rng(4).normal(size=int(1.5 * SR)) * 0.1


def _bank_response(unit):
    sig = sc.Signal()
    unit.excite_in.sources.append(sig)
    got = []
    for _i in range(0, len(_xr) - BLOCK, BLOCK):
        sig.data[:BLOCK] = _xr[_i:_i + BLOCK]
        sig.constant = False
        unit.render(BLOCK)
        got.append(unit.out.array(BLOCK).copy())
    y = np.concatenate(got)
    return float(np.sqrt(np.mean(y ** 2))), float(np.max(np.abs(y)))


_xd = sc.DrumUnit(SR)
_xd.frequency_in.base = 90.0
_xd.decay_in.base = 0.18
_xm = sc.ModalUnit(SR)
_xm.set_modes([(1.0, 1.0, 1.0), (1.594, 0.7, 0.7), (2.136, 0.5, 0.5),
               (2.296, 0.45, 0.45)])
_xm.frequency_in.base = 90.0
_xm.decay_in.base = 0.18
_xdr, _xdp = _bank_response(_xd)
_xmr, _xmp = _bank_response(_xm)
_xgap = 20 * np.log10(max(_xdr, 1e-12) / max(_xmr, 1e-12))
check('the same drive is about as loud into drum~ as into modal~',
      abs(_xgap) < 12.0,
      f'{_xgap:+.1f} dB apart (it was +32.0)')
check('a bowed drum does not clip',
      _xdp < 1.0, f'peak {_xdp:.3f} (it was 2.74)')

# The exciters should be in the same company too: a dropped mallet is
# not a twentieth of a bow.
_xb = sc.BounceUnit(SR)
_xbs = sc.Signal()
_xb.drop_in.sources.append(_xbs)
_xby = []
for _i in range(int(2.0 * SR / BLOCK)):
    _xbs.data[:BLOCK] = 1.0 if (_i * BLOCK / SR) > 0.05 else 0.0
    _xbs.constant = False
    _xb.render(BLOCK)
    _xby.append(_xb.out.array(BLOCK).copy())
_xbp = float(np.max(np.abs(np.concatenate(_xby))))
_xw = sc.BowUnit(SR)
_xws = sc.Signal()
_xw.velocity_in.sources.append(_xws)
_xwy = []
for _i in range(int(2.0 * SR / BLOCK)):
    _xws.data[:BLOCK] = 0.6
    _xws.constant = False
    _xw.render(BLOCK)
    _xwy.append(_xw.out.array(BLOCK).copy())
_xwp = float(np.max(np.abs(np.concatenate(_xwy))))
check('bounce~ strikes as hard as bow~ bows',
      abs(20 * np.log10(max(_xbp, 1e-12) / max(_xwp, 1e-12))) < 8.0,
      f'bounce~ {_xbp:.3f} against bow~ {_xwp:.3f} '
      f'({20 * np.log10(max(_xbp, 1e-12) / max(_xwp, 1e-12)):+.1f} dB; '
      f'it was -20.7)')

# A maraca is not shaken OR rolled with two separate gestures -- you
# cannot shake one while you are rolling it. There is one agitation and
# what changes is the ANGLE it meets the shell at: head on a bean stops
# dead against the wall and rings it, tangential it keeps its speed
# along the wall and drags. Everything between is both.
#
# Two earlier goes at this were wrong. The first pinned the beans to the
# wall by centripetal force and slid them round it -- a friction
# mechanism, and they tumble. The second made rolling a second gesture
# with a rate and a surge of its own, which is a shape a hand makes and
# has no business being generated in here.
def run_shaker(shake=0.8, swirl=0.0, hard=0.7, seconds=3.0, mode=1):
    u = sc.ShakerUnit(SR)
    u.shake_mode = mode
    u.hardness_in.base = hard
    u.swirl_in.base = swirl
    sig = sc.Signal()
    u.shake_in.sources.append(sig)
    got = []
    for _ in range(int(seconds * SR / BLOCK)):
        sig.data[:BLOCK] = shake
        sig.constant = False
        u.render(BLOCK)
        got.append(u.out.array(BLOCK).copy())
    return np.concatenate(got)[int(0.6 * SR):]


def _shaker_level(y):
    return float(np.sqrt(np.mean(y ** 2)))


def _shaker_shape(y):
    y = y - y.mean()
    v = np.mean(y ** 2)
    return float(np.mean(y ** 4) / max(v * v, 1e-30))


_mk_head = run_shaker(swirl=0.0)
_mk_tang = run_shaker(swirl=1.0)
check('head on it ticks, tangential it grazes',
      _shaker_shape(_mk_head) > 9.0 > _shaker_shape(_mk_tang),
      f'kurtosis {_shaker_shape(_mk_head):.1f} head on, '
      f'{_shaker_shape(_mk_tang):.1f} tangential (gaussian noise is 3)')
check('and opening the angle moves the sound rather than adding to it',
      -8.0 < 20 * np.log10(_shaker_level(_mk_tang)
                           / _shaker_level(_mk_head)) < -1.0,
      f'{20 * np.log10(_shaker_level(_mk_tang) / _shaker_level(_mk_head)):+.1f} dB '
      f'across the whole continuum -- a roll sits a little under a '
      f'shake, as it does in a hand')
check('the angle is not a gesture: on its own it makes nothing',
      _shaker_level(run_shaker(shake=0.0, swirl=1.0)) < 1e-5,
      'no agitation, no sound, whatever the angle')

# Nothing in here should wobble. A roll surges as the heap comes round,
# but that is the hand's shape to make.
_mk_env = np.abs(sig_hilbert(run_shaker(swirl=0.7, seconds=4.0)))[::64]
_mk_c = _mk_env - _mk_env.mean()
_mk_sp = np.abs(np.fft.rfft(_mk_c * np.hanning(len(_mk_c))))
_mk_f = np.fft.rfftfreq(len(_mk_c), 64.0 / SR)
_mk_band = (_mk_f > 0.3) & (_mk_f < 20.0)
check('and there is no oscillator hidden in it',
      np.max(_mk_sp[_mk_band]) < 6.0 * np.mean(_mk_sp[_mk_band]),
      f'the envelope has no line in it: strongest is '
      f'{np.max(_mk_sp[_mk_band]) / np.mean(_mk_sp[_mk_band]):.1f}x the '
      f'floor, and a surge put there deliberately measured 20x')


# Two ways to mean a gesture, as spin~ has. THROWN, it is a stroke and
# the beans carry on by themselves. HELD, it is how agitated they are
# right now. Pumping from the LEVEL gives the same steady state and the
# same tail either way, which is two names for one behaviour.
def _shaker_mode(mode, seconds=3.0):
    u = sc.ShakerUnit(SR)
    u.shake_mode = mode
    sig = sc.Signal()
    u.shake_in.sources.append(sig)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        sig.data[:BLOCK] = 0.8 if (b * BLOCK / SR) > 0.3 else 0.0
        sig.constant = False
        u.render(BLOCK)
        got.append(u.out.array(BLOCK).copy())
    y = np.concatenate(got)
    return (_shaker_level(y[int(0.32 * SR):int(0.55 * SR)]),
            _shaker_level(y[int(2.0 * SR):int(2.9 * SR)]))


_th_e, _th_l = _shaker_mode(0)
_hd_e, _hd_l = _shaker_mode(1)
check('a held gesture keeps the beans going; a thrown one is a stroke',
      _hd_l > 20.0 * max(_th_l, 1e-12) and _th_e > 1e-4,
      f'a second later: {_th_l:.5f} thrown, {_hd_l:.5f} held')
check('but a stroke still speaks when it arrives',
      _th_e > 0.2 * _hd_e,
      f'as it lands: {_th_e:.5f} thrown against {_hd_e:.5f} held')


# Shaking is back AND forth. Taking only the rise threw the beans on
# half the strokes and let them settle through the other half.
def _shaker_edges(invert):
    u = sc.ShakerUnit(SR)
    u.shake_mode = 0
    sig = sc.Signal()
    u.shake_in.sources.append(sig)
    got = []
    for b in range(int(2.0 * SR / BLOCK)):
        high = (b * BLOCK / SR) % 0.5 < 0.25
        sig.data[:BLOCK] = (0.0 if high else 0.9) if invert else (
            0.9 if high else 0.0)
        sig.constant = False
        u.render(BLOCK)
        got.append(u.out.array(BLOCK).copy())
    return _shaker_level(np.concatenate(got)[int(0.4 * SR):])


check('a thrown gesture agitates on the way back too',
      abs(20 * np.log10(_shaker_edges(False)
                        / max(_shaker_edges(True), 1e-12))) < 1.5,
      f'{_shaker_edges(False):.5f} leading with the rise, '
      f'{_shaker_edges(True):.5f} leading with the fall')

# A shake is ONE DIMENSIONAL -- back and forth -- so the hand stops dead
# at every turnaround and the agitation pulses. That is what makes it a
# rhythm. A swirl is sine AND cosine: the speed never passes through
# zero, so there are no troughs to fall into and its peaks are subtler.
# Opening the angle should therefore fill the troughs in, which is the
# same lever as 'settle' -- reaching for settle to get a roll is the
# right instinct and this is that instinct built in.
def _shaker_troughs(swirl):
    u = sc.ShakerUnit(SR)
    u.shake_mode = 1
    u.swirl_in.base = swirl
    sig = sc.Signal()
    u.shake_in.sources.append(sig)
    got = []
    for b in range(int(4.0 * SR / BLOCK)):
        # a softened sawtooth, which is how one is actually driven
        phase = ((b * BLOCK / SR) * 2.5) % 1.0
        sig.data[:BLOCK] = 0.9 * (phase ** 0.6)
        sig.constant = False
        u.render(BLOCK)
        got.append(u.out.array(BLOCK).copy())
    y = np.concatenate(got)[int(0.8 * SR):]
    env = np.abs(sig_hilbert(y))
    width = int(0.02 * SR) | 1
    env = np.convolve(env, np.ones(width) / width, mode='same')
    env = env[width:-width]
    return float(np.std(env) / max(np.mean(env), 1e-12))


_tr_shake = _shaker_troughs(0.0)
_tr_roll = _shaker_troughs(1.0)
check('a shake pulses with the gesture; a roll fills the troughs in',
      _tr_roll < 0.6 * _tr_shake,
      f'the level swings {_tr_shake:.2f} of its mean shaken, '
      f'{_tr_roll:.2f} rolled -- a swirl has no turnaround to stop at')


# Rolling changes how the strokes JOIN, not what each one is worth.
# Thrown, the strokes keep arriving while the beans hold their energy
# longer, so without scaling the stroke against that the sound piles up
# instead of smoothing: ten decibels by half travel and the peak
# agitation more than tripled. And a plain cosine reaches zero at the
# top of the knob, so the last tenth of the travel fell away to almost
# nothing -- a maraca rolled flat out still has beans in it.
def _shaker_continuum(mode, seconds=10.0):
    """Level across the angle, measured long enough to mean something.

    Each unit seeds its own collisions, so a four-second reading of this
    carries five decibels of noise -- three identical runs gave -5.6,
    -2.7 and -0.9 at full roll. Ten seconds brings that under one, which
    is the difference between a measurement and a coincidence.
    """
    out = []
    for swirl in (0.0, 0.5, 1.0):
        u = sc.ShakerUnit(SR)
        u.shake_mode = mode
        u.swirl_in.base = swirl
        sig = sc.Signal()
        u.shake_in.sources.append(sig)
        got = []
        for b in range(int(seconds * SR / BLOCK)):
            phase = ((b * BLOCK / SR) * 2.5) % 1.0
            sig.data[:BLOCK] = 0.9 * (phase ** 0.6)
            sig.constant = False
            u.render(BLOCK)
            got.append(u.out.array(BLOCK).copy())
        out.append(_shaker_level(np.concatenate(got)[int(0.8 * SR):]))
    return out


for _mode, _name in ((0, 'thrown'), (1, 'held')):
    _cn = _shaker_continuum(_mode)
    _swing = 20 * np.log10(max(_cn) / max(min(_cn), 1e-12))
    check(f'the angle does not run away with the level ({_name})',
          _swing < 6.0,
          f'{_swing:.1f} dB from head on to fully tangential '
          f'(unscaled, thrown, it was 17)')
    check(f'and the far end still speaks ({_name})',
          _cn[-1] > 0.4 * max(_cn),
          f'{20 * np.log10(_cn[-1] / max(_cn)):+.1f} dB at full roll -- '
          f'a plain cosine put it at minus infinity')

# Settle must not reach back into beans that are already moving.
def _shaker_settle_step():
    u = sc.ShakerUnit(SR)
    u.shake_mode = 1
    u.settle_in.base = 0.9
    sig = sc.Signal()
    u.shake_in.sources.append(sig)
    got = []
    for b in range(int(2.0 * SR / BLOCK)):
        now = b * BLOCK / SR
        sig.data[:BLOCK] = 0.8 if now < 0.8 else 0.0
        sig.constant = False
        if now >= 0.9:
            u.settle_in.base = 0.05
        u.render(BLOCK)
        got.append(u.out.array(BLOCK).copy())
    y = np.concatenate(got)
    w = int(0.02 * SR)
    before = float(np.sqrt(np.mean(y[int(0.88 * SR):int(0.88 * SR) + w] ** 2)))
    after = float(np.sqrt(np.mean(y[int(0.92 * SR):int(0.92 * SR) + w] ** 2)))
    return before, after


_st_b, _st_a = _shaker_settle_step()
check('and settle glides rather than cutting the ring where it stands',
      _st_a > 0.25 * _st_b,
      f'{_st_b:.5f} before the drop, {_st_a:.5f} just after -- it '
      f'arrives over a tenth of a second, because no hand can stop a '
      f'shaker where it stands')

# ------------------------------------------------------------ rattle~
# Particles in a container, actually simulated. The point of it is that
# the gesture stops needing translating: shaking along a LINE and
# swirling round a CIRCLE are the same simulation given a line and a
# circle, and the difference between them is not modelled anywhere.
def run_rattle(motion='line', shape=0, count=48, seconds=4.0, amp=3.0,
               spin=0.0, spin_axis='z'):
    u = sc.RattleUnit(SR)
    u.shape = shape
    u.set_count(count)
    sx, sy, wz = sc.Signal(), sc.Signal(), sc.Signal()
    u.shake_x_in.sources.append(sx)
    u.shake_y_in.sources.append(sy)
    (u.turn_x_in if spin_axis == 'x' else u.turn_z_in).sources.append(wz)
    out, knock, scrape = [], [], []
    for b in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + b * BLOCK) / SR
        sx.data[:BLOCK] = amp * np.sin(2 * np.pi * 2.5 * t)
        sx.constant = False
        sy.data[:BLOCK] = (amp * np.cos(2 * np.pi * 2.5 * t)
                           if motion == 'circle' else 0.0)
        sy.constant = False
        # 'turn' is an angle, so a steady turn is a ramp.
        wz.data[:BLOCK] = np.degrees(spin) * t
        wz.constant = False
        u.render(BLOCK)
        out.append(u.out.array(BLOCK).copy())
        knock.append(u.knock.array(BLOCK).copy())
        scrape.append(u.scrape.array(BLOCK).copy())
    keep = slice(int(SR), None)
    return (np.concatenate(out)[keep], np.concatenate(knock)[keep],
            np.concatenate(scrape)[keep])


def _rattle_swing(y):
    env = np.abs(sig_hilbert(y))
    width = int(0.02 * SR) | 1
    env = np.convolve(env, np.ones(width) / width, mode='same')[width:-width]
    return float(np.std(env) / max(np.mean(env), 1e-12))


_ra_line, _ra_lk, _ra_ls = run_rattle('line')
_ra_circ, _ra_ck, _ra_cs = run_rattle('circle')
check('a line pulses and a circle does not, with nothing modelled to say so',
      _rattle_swing(_ra_circ) < 0.5 * _rattle_swing(_ra_line),
      f'envelope swing {_rattle_swing(_ra_line):.2f} driven along a line, '
      f'{_rattle_swing(_ra_circ):.2f} round a circle -- a circle never '
      f'stops the way a line does at each end')
_ra_lr = (np.sqrt(np.mean(_ra_ls ** 2))
          / max(np.sqrt(np.mean(_ra_lk ** 2)), 1e-12))
_ra_cr = (np.sqrt(np.mean(_ra_cs ** 2))
          / max(np.sqrt(np.mean(_ra_ck ** 2)), 1e-12))
check('and a circle glances where a line strikes',
      _ra_cr > 1.2 * _ra_lr,
      f'scrape against knock {_ra_lr:.2f} along a line, {_ra_cr:.2f} '
      f'round a circle')

# Stated the other way up from how it used to be. The old form asked
# for MORE scrape against knock in the box, which contradicts its own
# sentence, and it only ever passed because tumble was wobbling every
# particle's radius and inventing collisions out of nothing.
# Driven hard enough that the box unambiguously knocks. Near the
# threshold whether it knocks at all depends on where the handful
# happened to be scattered, which is not what this is asking about.
_ra_box = run_rattle('circle', shape=1, amp=20.0)
_ra_bk = (np.sqrt(np.mean(_ra_box[1] ** 2))
          / max(np.sqrt(np.mean(_ra_box[2] ** 2)), 1e-12))
_ra_sph = run_rattle('circle', shape=0, amp=20.0)
_ra_ck = (np.sqrt(np.mean(_ra_sph[1] ** 2))
          / max(np.sqrt(np.mean(_ra_sph[2] ** 2)), 1e-12))
check('flat walls take a bean head on where a curved one lets it glance',
      _ra_bk > 5.0 * _ra_ck,
      f'knock against scrape {_ra_ck:.3f} in a sphere, {_ra_bk:.3f} in a '
      f'box -- a curved wall under a circling gesture is never actually '
      f'struck, so it does not knock at all; shape is only a boundary '
      f'test, so this costs nothing')

_ra_counts = [float(np.sqrt(np.mean(run_rattle(count=n)[0] ** 2)))
              for n in (8, 48, 256)]
check('how many things are in there changes the texture, not the level',
      max(_ra_counts) < 1.4 * min(_ra_counts),
      f'{_ra_counts[0]:.4f} / {_ra_counts[1]:.4f} / {_ra_counts[2]:.4f} '
      f'for 8, 48 and 256')

# Turning it does something on its own -- the centrifugal push, the
# Coriolis deflection, the Euler shove. Without them a rotated container
# would sit there.
# About an axis ACROSS the pull of gravity, not along it. Turned about
# the pull itself nothing is dragged anywhere -- everything settles at
# the bottom and stays held there, and the container is properly
# silent. That silence is the support angle working, not a fault, but
# it says nothing about whether turning is a gesture.
_ra_still = float(np.sqrt(np.mean(run_rattle(amp=0.0, spin=0.0)[0] ** 2)))
_ra_spun = float(np.sqrt(np.mean(run_rattle(amp=0.0, spin=40.0,
                                            spin_axis='x')[0] ** 2)))
check('turning the container is a gesture in itself',
      _ra_spun > 3.0 * _ra_still,
      f'{_ra_still:.4f} sitting still, {_ra_spun:.4f} turning at 40 '
      f'radians a second, with no shaking at all')

# Identical particles in one shared field keep STEP. They differ only in
# where they started, so they arrive together and a hundred of them
# sound like eight -- events per particle collapsing from 14 for one
# alone to 0.5 for a hundred and twenty-eight. Real grains differ in
# size, and being irregular they scatter off a wall rather than
# reflecting off it, since which face they present depends on how they
# are tumbling.
def _rattle_events(count, variety, seconds=3.0):
    u = sc.RattleUnit(SR)
    u.set_count(count)
    u.variety_in.base = variety
    sig = sc.Signal()
    u.shake_x_in.sources.append(sig)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + b * BLOCK) / SR
        sig.data[:BLOCK] = 3.0 * np.sin(2 * np.pi * 2.5 * t)
        sig.constant = False
        u.render(BLOCK)
        got.append(u.out.array(BLOCK).copy())
    y = np.concatenate(got)[int(0.5 * SR):]
    y = y[:(len(y) // 64) * 64]
    rms = float(np.sqrt(np.mean(y ** 2)))
    # Counted against the NOISE FLOOR, not against the loudest thing in
    # the run. A threshold set from the maximum moves whenever the
    # crest moves, so it measures the threshold rather than the sound,
    # and it reported this effect with the wrong sign.
    peaks, _ = sig_find_peaks(np.abs(y), height=1.5 * rms,
                              distance=int(0.001 * SR))
    env = np.abs(y).reshape(-1, 64).max(axis=1)
    return (len(peaks) / (len(y) / SR),
            float(np.std(env) / max(np.mean(env), 1e-12)))


_rv_same, _rc_same = _rattle_events(128, 0.0)
_rv_var, _rc_var = _rattle_events(128, 1.0)
# Variety genuinely spreads the SIZES now. It never did before: the
# spread was drawn once at full width whatever variety said, and the
# control only ever reached the rebound scattering -- so turning it up
# piled on knocks instead of spreading the trajectories out, which is
# the opposite of what it is for.
check('unalike things stop marching in step',
      _rv_var >= 0.95 * _rv_same,
      f'{_rv_same:.0f} events a second from identical spheres, '
      f'{_rv_var:.0f} from a mixed handful -- it must not go DOWN, '
      f'which is what a big rebound tilt used to do by flinging them '
      f'across the middle to fly a long way between contacts')
check('and the sound stops arriving in lumps',
      _rc_var < 0.92 * _rc_same,
      f'envelope swing {_rc_same:.2f} identical, {_rc_var:.2f} mixed')

# ------------------------------------------------ the support angle
# A thing resting on a wall is HELD while the slope under it is
# shallower than the friction can support, and slides or lets go when
# it is not. Without that there is no held state at all -- only
# bookkeeping at the instant of contact, which can say how much speed
# was lost but not whether anything is being held up. The symptom was
# that a slow turn on smooth glass came out thirty to one knocks over
# slide, which is backwards: a slow turn on a fairly smooth surface
# should slide, and sliding is a continuous sound.
#
# Two things vary the angle. TEXTURE is the shell's, and differs from
# place to place, so a rough shell catches, carries, lets go and
# catches again somewhere else -- and each letting go is a tap. GRIP is
# the coefficient itself. A smooth shell holds the same everywhere, so
# once a thing starts sliding it goes on sliding, and no taps.
def _rattle_contact(texture, grip, spin=6.0, seconds=6.0):
    u = sc.RattleUnit(SR)
    u.set_count(48)
    u.texture_in.base = texture
    u.friction_in.base = grip
    wx = sc.Signal()
    u.turn_x_in.sources.append(wx)
    knock, scrape = [], []
    for b in range(int(seconds * SR / BLOCK)):
        wx.data[:BLOCK] = np.degrees(spin) * ((np.arange(BLOCK)
                                               + b * BLOCK) / SR)
        wx.constant = False
        u.render(BLOCK)
        knock.append(u.knock.array(BLOCK).copy())
        scrape.append(u.scrape.array(BLOCK).copy())
    keep = slice(int(SR), None)
    k = float(np.sqrt(np.mean(np.concatenate(knock)[keep] ** 2)))
    sr_ = float(np.sqrt(np.mean(np.concatenate(scrape)[keep] ** 2)))
    return k, sr_


_sa_smooth_k, _sa_smooth_s = _rattle_contact(0.0, 0.15)
check('a slow turn on a smooth shell slides, and does not knock at all',
      _sa_smooth_k == 0.0 and _sa_smooth_s > 0.002,
      f'knock {_sa_smooth_k:.5f}, slide {_sa_smooth_s:.5f} -- nothing '
      f'lets go, because there is nowhere for it to catch')

_sa_rough_k, _sa_rough_s = _rattle_contact(0.8, 0.15)
check('roughening the same shell is what makes it tick',
      _sa_rough_k > 10.0 * max(_sa_smooth_k, 1e-4),
      f'knock {_sa_smooth_k:.5f} smooth -> {_sa_rough_k:.5f} rough, on '
      f'the same slow turn, with the slide barely moved '
      f'({_sa_smooth_s:.5f} -> {_sa_rough_s:.5f})')

_sa_mids = [_rattle_contact(t, 0.15)[0] for t in (0.0, 0.25, 0.5, 0.8)]
# And it goes on rising all the way. It used to peak and FALL BACK,
# because roughness was carried into what holds a thing up at full
# strength -- so a rougher shell held everything harder and let go less
# often, and roughening it past about a fifth made it quieter.
check('and it comes in by degrees, not at a threshold',
      _sa_mids[0] == 0.0
      and all(b > a for a, b in zip(_sa_mids, _sa_mids[1:])),
      'knock ' + ' -> '.join(f'{x:.4f}' for x in _sa_mids)
      + ' as the shell roughens')

# Grip on a shell with no roughness at all buys more SLIDE and still
# no knock, however hard it grips. A thing held harder is carried
# further up before the slope stops supporting it, but on a smooth wall
# there is nothing there to fall off, so letting go is silent. This
# used to assert the opposite and passed on an artefact: judged by
# normal speed, anything sliding faster than about a metre a second
# reported a fresh arrival every step from the curvature alone.
_sa_grippy_k, _sa_grippy_s = _rattle_contact(0.0, 0.3)
check('gripping a smooth shell harder buys more slide, and still no knock',
      _sa_grippy_k == 0.0 and _sa_grippy_s > 1.25 * _sa_smooth_s,
      f'knock {_sa_smooth_k:.5f} -> {_sa_grippy_k:.5f} and slide '
      f'{_sa_smooth_s:.5f} -> {_sa_grippy_s:.5f} as grip goes 0.15 -> 0.3')

# Roughness resists as well as catching, so it works where there is no
# friction coefficient at all -- which it could not before, because a
# thing only came to rest when its speed fell under grip*press*dt, so
# with no grip it never rested, never caught, and its support was never
# drawn. Texture moved knock by a tenth across its whole range.
_sa_slick_k, _sa_slick_s = _rattle_contact(0.0, 0.0)
_sa_rasp_k, _sa_rasp_s = _rattle_contact(1.0, 0.0)
check('roughness holds and rasps even on a shell with no friction at all',
      _sa_rasp_s > 0.002 and _sa_rasp_s > 100.0 * _sa_slick_s,
      f'with friction at zero: slide {_sa_slick_s:.8f} -> '
      f'{_sa_rasp_s:.5f} '
      f'and knock {_sa_slick_k:.5f} -> {_sa_rasp_k:.5f} as the shell '
      f'roughens')

# Held means held. With nothing shaking it and nothing turning it,
# everything comes to rest against the wall and STAYS there.
_sa_dead = sc.RattleUnit(SR)
_sa_dead.set_count(48)
for _ in range(int(6.0 * SR / BLOCK)):
    _sa_dead.render(BLOCK)
_sa_quiet = float(np.max(np.abs(_sa_dead.out.array(BLOCK))))
check('left alone it settles and stays settled',
      _sa_quiet == 0.0 and float(np.sum(_sa_dead._held)) > 40.0,
      f'{np.sum(_sa_dead._held):.0f} of 48 held against the wall, and '
      f'peak {_sa_quiet:.6f} out')

# Shaken hard it still knocks, whatever the shell is like -- the
# support angle governs resting contact, not arrival.
_sa_shaken = float(np.sqrt(np.mean(run_rattle(amp=3.0)[1] ** 2)))
# Not by much, mind, and it should not be by much: a rough shell turned
# slowly is a rattle in its own right. A rainstick is exactly that, and
# a rainstick is not quiet.
check('none of which stops a hard shake from knocking',
      _sa_shaken > 0.95 * _sa_rough_k and _sa_shaken > 0.02,
      f'knock {_sa_shaken:.4f} shaken against {_sa_rough_k:.4f} for the '
      f'roughest slow turn -- neck and neck, which is right')


# What comes out of an exciter is a force, and a wall only ever
# pushes, so a contact really is one-sided. As a SIGNAL that is just an
# offset. Added one way only, a slide put out one bump per control step
# for ever and they summed to a constant: mean +0.0083 against an rms
# of 0.0084, never once below zero, every spectral component under
# 10 Hz. It was not a sound, it was a force level -- and the level
# stepping as things caught and let go is what made the output jump to
# random offsets. What radiates from a sliding contact is the part that
# FLUCTUATES.
def _rattle_slide(spin=6.0, seconds=5.0, hardness=1.0):
    u = sc.RattleUnit(SR)
    u.set_count(48)
    u.texture_in.base = 0.0
    u.friction_in.base = 0.3
    u.hardness_in.base = hardness
    wx = sc.Signal()
    u.turn_x_in.sources.append(wx)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        wx.data[:BLOCK] = np.degrees(spin) * ((np.arange(BLOCK)
                                               + b * BLOCK) / SR)
        wx.constant = False
        u.render(BLOCK)
        got.append(u.scrape.array(BLOCK).copy())
    return np.concatenate(got)[int(SR):]


_rs = _rattle_slide()
_rs_rms = float(np.sqrt(np.mean(_rs ** 2)))
# Counted over the samples that carry signal. A rub is laid down as
# separate contacts, so plenty of samples between them are silence, and
# silence is neither up nor down.
_rs_live = _rs[np.abs(_rs) > 1e-12]
check('a slide comes out as a sound and not as a force level',
      abs(float(np.mean(_rs))) < 0.05 * _rs_rms
      and 0.4 < float(np.mean(_rs_live < 0.0)) < 0.6,
      f'mean {np.mean(_rs):+.6f} against rms {_rs_rms:.5f}, and of the '
      f'samples that are not silence {100.0 * np.mean(_rs_live < 0.0):.0f}% '
      f'are below zero')


def _rattle_centroid(x):
    w = 1 << 15
    mag = np.abs(np.fft.rfft(x[:w] * np.hanning(w)))
    freq = np.fft.rfftfreq(w, 1.0 / SR)
    return float((freq * mag).sum() / max(mag.sum(), 1e-12))


_rs_slow = _rattle_centroid(_rattle_slide(spin=1.0))
_rs_fast = _rattle_centroid(_rattle_slide(spin=25.0))
check('and a faster slide comes out brighter, without being told to',
      _rs_fast > 2.0 * _rs_slow,
      f'centroid {_rs_slow:.0f} Hz turning at 1 radian a second, '
      f'{_rs_fast:.0f} Hz at 25 -- a rub lasts one bump of the surface '
      f'going by, so its width is the spacing over the speed')

# But only up to what the CONTACT allows. A rub is a run of tiny
# impacts and each of them is still a contact, so the same stiffness
# that stops a blow being sharper stops a rub too. Floored at two
# samples instead of at the contact, a rub at speed went white -- finer
# than the hardest blow the model can make -- and came out at 5900 Hz
# against that blow's 432 Hz, from what is meant to be one material.
_rs_soft = (_rattle_centroid(_rattle_slide(spin=25.0, hardness=0.25))
            / max(_rattle_centroid(_rattle_slide(spin=1.0,
                                                 hardness=0.25)), 1e-9))
check('but no brighter than the contact it is rubbing through',
      _rs_soft < 0.6 * (_rs_fast / max(_rs_slow, 1e-9)),
      f'speed buys {_rs_fast / max(_rs_slow, 1e-9):.1f}x of brightness '
      f'through a hard contact and {_rs_soft:.1f}x through a soft one')

_rk_shaken = run_rattle(amp=1.0)[1]
check('and a blow does not sit on an offset either',
      abs(float(np.mean(_rk_shaken)))
      < 0.05 * float(np.sqrt(np.mean(_rk_shaken ** 2))),
      f'mean {np.mean(_rk_shaken):+.6f} against rms '
      f'{np.sqrt(np.mean(_rk_shaken ** 2)):.5f}')


# Touching means AT the wall, not past it. Asking for strictly past it,
# anything that had come to rest sat exactly on the line and counted as
# not touching: held and the contact were BOTH cleared, every other
# step, for ever. Nothing downstream of that could work -- the support
# was redrawn twice a control period instead of once per catch, so
# roughness came out as fast noise rather than catch-and-release, and
# resting contacts kept re-registering as new arrivals. A gentle rock
# that should have been a pure slide came out five to one knocks.
def _rattle_rock(spin=30.0, seconds=16.0):
    u = sc.RattleUnit(SR)
    u.set_count(128)
    u.size_in.base = 0.5
    u.grain_in.base = 0.159
    u.friction_in.base = 0.28
    u.texture_in.base = 0.0
    u.bounce_in.base = 0.0
    wx = sc.Signal()
    u.turn_x_in.sources.append(wx)
    knock, scrape, held = [], [], []
    for b in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + b * BLOCK) / SR
        wx.data[:BLOCK] = spin * np.sin(2 * np.pi * 0.06 * t)
        wx.constant = False
        u.render(BLOCK)
        knock.append(u.knock.array(BLOCK).copy())
        scrape.append(u.scrape.array(BLOCK).copy())
        held.append(float(np.sum(u._held[:128])))
    keep = slice(int(5 * SR), None)
    return (float(np.sqrt(np.mean(np.concatenate(knock)[keep] ** 2))),
            float(np.sqrt(np.mean(np.concatenate(scrape)[keep] ** 2))),
            float(np.mean(held[int(5 * SR / BLOCK):])))


_rr_k, _rr_s, _rr_h = _rattle_rock()
check('a thing that has come to rest stays in contact with the wall',
      _rr_h > 80.0,
      f'{_rr_h:.0f} of 128 held through a slow rock -- asking for '
      f'strictly past the wall this flapped every other step')
check('so a gentle rock is a slide, with no knocks in it at all',
      _rr_k == 0.0 and _rr_s > 0.005,
      f'knock {_rr_k:.5f}, slide {_rr_s:.5f} rocking 30 degrees at '
      f'0.06 Hz')


# A rub is a FORCE, and it has to be emitted as one. Emitted as the
# speed it took off the particle in a single step, it carried the step
# SIZE into the sound -- 0.0070 / 0.0098 / 0.0140 at decimation
# 4 / 8 / 16, a root two per doubling -- so how finely the thing was
# integrated set how loud it rubbed. The same class of fault as the
# per-control-step rates in spin~, and it hides just as well.
_rd_was = sc.RattleUnit.CONTROL_DECIM
_rd = []
for _rd_n in (4, 8, 16):
    sc.RattleUnit.CONTROL_DECIM = _rd_n
    _rd.append(_rattle_contact(0.0, 0.3)[1])
sc.RattleUnit.CONTROL_DECIM = _rd_was
check('how loud a rub is does not depend on how finely it is integrated',
      max(_rd) < 1.08 * min(_rd),
      f'{_rd[0]:.5f} / {_rd[1]:.5f} / {_rd[2]:.5f} at decimation '
      f'4 / 8 / 16')


# A blow and a rub are one material answering two ways, so their
# balance has to come from the physics rather than from two constants
# tuned apart. Both are forces at the same contact now, through one
# gain, and the ratio holds still while the gesture changes completely.
def _rattle_balance(shake=0.0, spin=0.0, texture=0.25, seconds=10.0):
    u = sc.RattleUnit(SR)
    u.set_count(48)
    u.texture_in.base = texture
    u.friction_in.base = 0.3
    wx, sx = sc.Signal(), sc.Signal()
    u.turn_x_in.sources.append(wx)
    u.shake_x_in.sources.append(sx)
    knock, scrape = [], []
    for b in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + b * BLOCK) / SR
        wx.data[:BLOCK] = np.degrees(spin) * t
        wx.constant = False
        sx.data[:BLOCK] = shake * np.sin(2 * np.pi * 2.5 * t)
        sx.constant = False
        u.render(BLOCK)
        knock.append(u.knock.array(BLOCK).copy())
        scrape.append(u.scrape.array(BLOCK).copy())
    keep = slice(int(2 * SR), None)
    k = float(np.sqrt(np.mean(np.concatenate(knock)[keep] ** 2)))
    s_ = float(np.sqrt(np.mean(np.concatenate(scrape)[keep] ** 2)))
    return k / max(s_, 1e-12)


_rb = [_rattle_balance(shake=1.0), _rattle_balance(shake=2.0),
       _rattle_balance(spin=6.0), _rattle_balance(spin=10.0)]
check('a blow and a rub stay in proportion, whatever the gesture',
      max(_rb) < 2.4 * min(_rb),
      'blow against rub ' + ' / '.join(f'{x:.2f}' for x in _rb)
      + ' shaken at 1 g, at 2 g, turned slowly, turned faster')


# One blow laid down once. The working variables were cleared once a
# CONTROL STEP while the laying-down happens once a PARTICLE, so the
# first thing to register a blow left it standing and everything after
# it in that step laid the same blow down again. It hides well: the
# sound is plausible, and what it looks like is blows standing out of
# all proportion to rubs -- which is what it did. Independent blows add
# in power, so a handful of them must come out as the ROOT of the
# count. With the fault in they went as the count itself.
def _rattle_blow_energy(count, seconds=8.0):
    u = sc.RattleUnit(SR)
    u.set_count(count)
    u.variety_in.base = 0.0
    sig = sc.Signal()
    u.shake_x_in.sources.append(sig)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + b * BLOCK) / SR
        sig.data[:BLOCK] = 2.0 * np.sin(2 * np.pi * 2.5 * t)
        sig.constant = False
        u.render(BLOCK)
        got.append(u.knock.array(BLOCK).copy())
    y = np.concatenate(got)[int(2 * SR):]
    rms = float(np.sqrt(np.mean(y ** 2)))
    # Undo the density gain, leaving how the raw sum actually grew.
    gain = (sc.RattleUnit.DENSITY_REF
            / max(1.0, count) ** sc.RattleUnit.DENSITY_LAW)
    return rms / gain


_be_n = (4, 16, 64, 256)
_be = [_rattle_blow_energy(n) for n in _be_n]
_be_law = float(np.polyfit(np.log(_be_n), np.log(_be), 1)[0])
check('a blow is laid down once, not once for every thing in there',
      0.40 < _be_law < 0.62,
      f'blows grow as the count to the {_be_law:.2f} -- a root, which '
      f'is independent things adding in power. Laid down once per '
      f'particle per step instead of once each it went as the count '
      f'itself, and a full container multiplied every blow by a '
      f'hundred and twenty-eight')


# What a blow hands a resonator is an IMPULSE. How hard the contact is
# decides how that impulse is spread in time -- its colour -- and not
# how much of it there is. A Hann hump of width W and peak A carries
# A*W/2, so the peak has to go as 1/W to hold the impulse still. Going
# as one over the ROOT of the width, as it did, the impulse grew as the
# root of the width instead: a soft contact handed over five times the
# momentum of a hard one. Since a mode below the contact bandwidth
# answers the impulse and not the shape, that made hardness a fourteen
# decibel loudness control on everything low -- softer being LOUDER,
# and duller with it.
def _rattle_into_modal(hardness, freq=320.0, shake=8.0, seconds=6.0):
    u = sc.RattleUnit(SR)
    u.set_count(48)
    u.shape = 1
    u.texture_in.base = 0.0
    u.bounce_in.base = 0.8
    u.hardness_in.base = hardness
    sx = sc.Signal()
    u.shake_x_in.sources.append(sx)
    m = sc.ModalUnit(SR)
    m.frequency_in.base = freq
    m.decay_in.base = 1.2
    drive = sc.Signal()
    m.excite_in.sources.append(drive)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + b * BLOCK) / SR
        sx.data[:BLOCK] = shake * np.sin(2 * np.pi * 2.0 * t)
        sx.constant = False
        u.render(BLOCK)
        drive.data[:BLOCK] = u.knock.array(BLOCK)
        drive.constant = False
        m.render(BLOCK)
        got.append(m.out.array(BLOCK).copy())
    y = np.concatenate(got)[int(2 * SR):]
    w = 1 << 15
    mag = np.abs(np.fft.rfft(y[:w] * np.hanning(w)))
    freqs = np.fft.rfftfreq(w, 1.0 / SR)
    return (float(np.sqrt(np.mean(y ** 2))),
            float((freqs * mag).sum() / max(mag.sum(), 1e-12)))


# At a mode well inside every one of these contact bandwidths, so that
# what is being measured is the WEIGHT and not the roll-off. At 320 Hz
# a hardness of 0.5 is already at the edge of what a 1.6 ms contact can
# reach, and the roll-off there is real physics rather than a fault.
_ri = [_rattle_into_modal(h, freq=120.0) for h in (1.0, 0.75, 0.5)]
_ri_lv = [x[0] for x in _ri]
_ri_ct = [x[1] for x in _ri]
check('softening the contact changes a blow\'s colour, not its weight',
      max(_ri_lv) < 1.8 * min(_ri_lv),
      'ringing a 120 Hz resonator: '
      + ' / '.join(f'{x:.4f}' for x in _ri_lv)
      + ' at hardness 1.0 / 0.75 / 0.5, within '
      + f'{20 * np.log10(max(_ri_lv) / min(_ri_lv)):.1f} dB')
# Measured on the DRIVE. Through the resonator its own mode dominates
# the centroid and hides what the drive is doing.
def _rattle_knock_colour(hardness, shake=8.0, seconds=6.0):
    u = sc.RattleUnit(SR)
    u.set_count(48)
    u.shape = 1
    u.texture_in.base = 0.0
    u.bounce_in.base = 0.8
    u.hardness_in.base = hardness
    sx = sc.Signal()
    u.shake_x_in.sources.append(sx)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + b * BLOCK) / SR
        sx.data[:BLOCK] = shake * np.sin(2 * np.pi * 2.0 * t)
        sx.constant = False
        u.render(BLOCK)
        got.append(u.knock.array(BLOCK).copy())
    y = np.concatenate(got)[int(2 * SR):]
    w = 1 << 15
    mag = np.abs(np.fft.rfft(y[:w] * np.hanning(w)))
    freqs = np.fft.rfftfreq(w, 1.0 / SR)
    return float((freqs * mag).sum() / max(mag.sum(), 1e-12))


_ri_ct = [_rattle_knock_colour(h) for h in (1.0, 0.75, 0.5, 0.0)]
check('and the colour is what it does change',
      _ri_ct[0] > 4.0 * _ri_ct[-1],
      'the blows themselves run '
      + ' / '.join(f'{x:.0f}' for x in _ri_ct)
      + ' Hz at hardness 1.0 / 0.75 / 0.5 / 0')

# Softer still and the mode stops answering at all, which is right: an
# eight millisecond contact has no energy left up at 320 Hz. A very
# soft mallet does not ring a bell.
_ri_soft = _rattle_into_modal(0.0)[0]
check('and a contact too soft to reach a mode does not ring it',
      _ri_soft < 0.2 * _ri_lv[0],
      f'{_ri_soft:.4f} at hardness 0 against {_ri_lv[0]:.4f} at 1.0, '
      f'{20 * np.log10(_ri_soft / _ri_lv[0]):.0f} dB down')


# The two outlets exist to separate two behaviours, so nothing that
# follows hardness may appear on the rubbing one. A rub lasts one bump
# of the surface going by; hardness is how long a CONTACT lasts, and
# the two have nothing to do with each other. The tangential impulse of
# a glancing blow was going to the scrape outlet -- real enough as
# physics, but it is impulsive, it lasts the contact, and it therefore
# moved with hardness. The result was a blow-shaped, hardness-following
# pulse sitting in the one signal meant to be nothing but rubbing:
# audible, and plain on a scope. Both impulses of one collision are one
# blow now, combined in quadrature the way perpendicular things are.
def _rattle_scrape_only(hardness, shake=8.0, seconds=10):
    u = sc.RattleUnit(SR)
    u.set_count(48)
    u.shape = 1
    u.texture_in.base = 0.0
    u.bounce_in.base = 0.8
    u.hardness_in.base = hardness
    sx = sc.Signal()
    u.shake_x_in.sources.append(sx)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + b * BLOCK) / SR
        sx.data[:BLOCK] = shake * np.sin(2 * np.pi * 2.0 * t)
        sx.constant = False
        u.render(BLOCK)
        got.append(u.scrape.array(BLOCK).copy())
    y = np.concatenate(got)[int(4 * SR):]
    rms = float(np.sqrt(np.mean(y ** 2)))
    return rms, float(np.max(np.abs(y)) / max(rms, 1e-12))


# Below the top of hardness. Up there the rub is CONTACT-limited and
# nearly white, and a white rub built out of two-sample impulses is
# legitimately spikier than a dark one built of long overlapping ones.
# What must not move with hardness is the LEVEL, which is the check
# above and holds across the whole range.
_so = [_rattle_scrape_only(h) for h in (0.5, 0.25, 0.0)]
_so_lv = [x[0] for x in _so]
_so_cr = [x[1] for x in _so]
check('hardness does not reach the rubbing at all',
      max(_so_lv) < 1.5 * min(_so_lv),
      'rub level ' + ' / '.join(f'{x:.4f}' for x in _so_lv)
      + ' at hardness 0.5 / 0.25 / 0 while things are striking the walls'
      + f' -- {20 * np.log10(max(_so_lv) / min(_so_lv)):.1f} dB, and it '
      + 'was 6.2')
check('and no blow-shaped spike rides in on it',
      max(_so_cr) < 1.6 * min(_so_cr),
      'crest ' + ' / '.join(f'{x:.1f}' for x in _so_cr)
      + ' over the same three -- it was 12.7 / 7.9 / 6.5, following '
      + 'hardness because a blow was leaking in')

# The settle is a settle, not a crash. Scattered over a fixed four
# tenths of a metre they landed ten times outside a default-sized
# shell: every one of them started embedded and was clamped out.
_rz = sc.RattleUnit(SR)
_rz.set_count(48)
_rz_r = np.linalg.norm(_rz._pos[:144].reshape(48, 3), axis=1)
check('they are dropped in inside the container they are in',
      float(_rz_r.max()) < _rz.size_in.base,
      f'furthest starts at {_rz_r.max():.4f} m in a '
      f'{_rz.size_in.base:.3f} m shell')


# One material answers a blow and a rub with the same stiffness, so
# their colours have to move together. They did not: a rub's width was
# floored at two samples, so at speed it went white, finer than the
# hardest blow the model can make -- 5900 Hz of rub against 432 Hz of
# blow at the same setting. A rub is a run of tiny impacts and every
# one of them is still a contact.
def _rattle_colours(hardness, shake=8.0, seconds=10):
    u = sc.RattleUnit(SR)
    u.set_count(48)
    u.shape = 1
    u.hardness_in.base = hardness
    u.bounce_in.base = 0.8
    sx = sc.Signal()
    u.shake_x_in.sources.append(sx)
    knock, scrape = [], []
    for b in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + b * BLOCK) / SR
        sx.data[:BLOCK] = shake * np.sin(2 * np.pi * 2.0 * t)
        sx.constant = False
        u.render(BLOCK)
        knock.append(u.knock.array(BLOCK).copy())
        scrape.append(u.scrape.array(BLOCK).copy())
    out = []
    for got in (knock, scrape):
        y = np.concatenate(got)[int(4 * SR):]
        w = 1 << 15
        mag = np.abs(np.fft.rfft(y[:w] * np.hanning(w)))
        freqs = np.fft.rfftfreq(w, 1.0 / SR)
        out.append(float((freqs * mag).sum() / max(mag.sum(), 1e-12)))
    return out


_cl = [_rattle_colours(h) for h in (1.0, 0.5, 0.0)]
_cl_r = [c[1] / max(c[0], 1e-9) for c in _cl]
check('a blow and a rub are the same material, so they sit in the '
      'same register',
      all(0.3 < r < 1.8 for r in _cl_r),
      'rub against blow '
      + ' / '.join(f'{r:.2f}' for r in _cl_r)
      + ' at hardness 1.0 / 0.5 / 0 -- '
      + ' and '.join(f'{c[0]:.0f} vs {c[1]:.0f} Hz' for c in _cl)
      + '. At 0.5 it used to be 432 against 5900')

# And the vessel's own surface has a say in it. How far apart the bumps
# are grows with how rough the shell is, so a polished one hisses and a
# coarse one rasps lower. Roughness used to be a fixed constant here,
# so 'texture' changed how much the wall CAUGHT and nothing whatever
# about what it sounded like.
def _rattle_rasp(texture, seconds=10):
    u = sc.RattleUnit(SR)
    u.set_count(48)
    u.texture_in.base = texture
    u.hardness_in.base = 1.0
    wx = sc.Signal()
    u.turn_x_in.sources.append(wx)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + b * BLOCK) / SR
        wx.data[:BLOCK] = np.degrees(6.0) * t
        wx.constant = False
        u.render(BLOCK)
        got.append(u.scrape.array(BLOCK).copy())
    return _rattle_centroid(np.concatenate(got)[int(4 * SR):])


_rp = [_rattle_rasp(t) for t in (0.0, 0.25, 0.5, 1.0)]
# Not strictly step by step at the very top of hardness: down there the
# bumps are so close together that a crossing is shorter than a CONTACT,
# and the contact is what limits it until the shell is rough enough for
# the spacing to take over. That is the right way round.
check('a polished shell hisses and a coarse one rasps lower',
      _rp[0] > 2.2 * _rp[-1]
      and all(b < 1.3 * a for a, b in zip(_rp, _rp[1:])),
      'rub centroid ' + ' / '.join(f'{x:.0f}' for x in _rp)
      + ' Hz as the shell roughens')


# A small hard shaker hisses. The top of hardness has to reach a rub
# that is nearly white, and it could not: the range stopped where the
# rest of the rack's MALLET range stops, a third of a millisecond, and
# since a rub cannot be sharper than a contact that put a floor under
# it. Nothing in here is a mallet -- contact time falls with mass, and
# a light bead on a stiff shell rings far shorter than the hardest
# mallet head -- so the range runs on down to two samples.
_wh = _rattle_colours(1.0)
check('the hardest setting hisses, the way a small shaker does',
      _wh[1] > 2500.0 and _wh[0] > 4000.0,
      f'rub {_wh[1]:.0f} Hz against blow {_wh[0]:.0f} Hz at hardness 1 '
      f'-- it stopped at 1855 Hz of rub while the range stopped where a '
      f'mallet does')

_wh_contact = sc.RattleUnit(SR)
_wh_contact.hardness_in.base = 1.0
_wh_contact.render(BLOCK)
_wh_w = _wh_contact._window_width
check('and the hardest contact really is as short as this can carry',
      _wh_w <= 3,
      f'{_wh_w} samples, {1000.0 * _wh_w / SR:.3f} ms')


# How long a container is against how wide, which is only the boundary
# test and so costs nothing. Pinned between two close walls things
# rattle constantly; given a length to travel they cross it and arrive
# less often, harder. The level should not care either way.
def _rattle_aspect(aspect, shape=1, shake=2.0, seconds=12):
    u = sc.RattleUnit(SR)
    u.set_count(48)
    u.shape = shape
    u.aspect_in.base = aspect
    sx, sz = sc.Signal(), sc.Signal()
    u.shake_x_in.sources.append(sx)
    u.shake_z_in.sources.append(sz)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + b * BLOCK) / SR
        sx.data[:BLOCK] = shake * np.sin(2 * np.pi * 2.5 * t)
        sx.constant = False
        sz.data[:BLOCK] = shake * np.sin(2 * np.pi * 2.5 * t)
        sz.constant = False
        u.render(BLOCK)
        got.append(u.out.array(BLOCK).copy())
    y = np.concatenate(got)[int(4 * SR):]
    rms = float(np.sqrt(np.mean(y ** 2)))
    peaks, _ = sig_find_peaks(np.abs(y), height=1.5 * rms,
                              distance=int(0.001 * SR))
    return len(peaks) / (len(y) / SR), rms


_ap = [_rattle_aspect(a) for a in (0.2, 0.5, 1.0, 2.5, 5.0)]
_ap_ev = [x[0] for x in _ap]
_ap_lv = [x[1] for x in _ap]
# Not step by step, and it should not be: a cube is the roomiest of
# these for its half-width, so it is the quietest of them, and both
# squashing and stretching it shut things in. What is flat rattles most.
# Being FLAT is what tells: pinned between two close walls things
# rattle constantly. Past a cube it hardly matters how much longer it
# gets -- the cross-section is what confines them, and that does not
# change.
check('a flat box rattles where a roomier one lets things travel',
      _ap_ev[0] > 1.15 * max(_ap_ev[1:]),
      'contacts a second ' + ' / '.join(f'{x:.0f}' for x in _ap_ev)
      + ' from a flat slab through a cube to a long tube')
check('and how long it is does not decide how loud it is',
      max(_ap_lv) < 1.6 * min(_ap_lv),
      'level ' + ' / '.join(f'{x:.4f}' for x in _ap_lv)
      + ' over the same five')


# A cylinder: curved round the barrel, flat at the two ends. Things
# glance off the side the way they do in a sphere and are taken head on
# by the caps the way they are in a box, so which of the two you get
# depends on which way it is shaken -- which no single-surface shape
# can do.
def _rattle_tube(shape, axis, aspect=3.0, shake=2.0, seconds=12):
    u = sc.RattleUnit(SR)
    u.set_count(48)
    u.shape = shape
    u.aspect_in.base = aspect
    sig = sc.Signal()
    getattr(u, f'shake_{axis}_in').sources.append(sig)
    knock, scrape = [], []
    for b in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + b * BLOCK) / SR
        sig.data[:BLOCK] = shake * np.sin(2 * np.pi * 2.5 * t)
        sig.constant = False
        u.render(BLOCK)
        knock.append(u.knock.array(BLOCK).copy())
        scrape.append(u.scrape.array(BLOCK).copy())
    keep = slice(int(4 * SR), None)
    k = float(np.sqrt(np.mean(np.concatenate(knock)[keep] ** 2)))
    s_ = float(np.sqrt(np.mean(np.concatenate(scrape)[keep] ** 2)))
    return k / max(s_, 1e-12)


_tu_ends = _rattle_tube(3, 'z')
_tu_side = _rattle_tube(3, 'x')
_sp_ends = _rattle_tube(0, 'z')
_sp_side = _rattle_tube(0, 'x')
check('a tube is taken head on at its ends and glanced off its side',
      _tu_ends > 20.0 * _tu_side,
      f'blow against rub {_tu_ends:.0f} shaken along it into the flat '
      f'caps, {_tu_side:.2f} shaken across it into the curved barrel')
# Which is the whole point of it, and the thing no single-surface shape
# can do. Across it a tube glances about as freely as a sphere -- a
# little MORE freely, in fact, since a stretched sphere narrows towards
# its ends where a barrel stays at full width all the way along.
check('so which way you shake a tube decides what it is',
      (_tu_ends / max(_tu_side, 1e-12))
      > 20.0 * (_sp_ends / max(_sp_side, 1e-12)),
      f'a tube goes {_tu_side:.2f} across to {_tu_ends:.0f} along; a '
      f'sphere only {_sp_side:.2f} to {_sp_ends:.2f}')


# What a rub radiates is a share of the WORK friction is doing, and that
# work is the force times how fast the thing is being dragged -- so the
# amplitude goes as the root of the two together. Taken from the force
# alone, a thing pressed hard against a wall and barely creeping rasped
# exactly as loudly as one skating across it, and the noise followed how
# hard it was PRESSED instead of how fast it was MOVING. Shaken, the
# press swings with the gesture, so the sound wheezed in and out with
# it, like a bicycle pump.
def _rattle_creep(spin, seconds=10):
    u = sc.RattleUnit(SR)
    u.set_count(48)
    u.texture_in.base = 0.0
    u.friction_in.base = 0.9
    wx = sc.Signal()
    u.turn_x_in.sources.append(wx)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + b * BLOCK) / SR
        wx.data[:BLOCK] = np.degrees(spin) * t
        wx.constant = False
        u.render(BLOCK)
        got.append(u.scrape.array(BLOCK).copy())
    y = np.concatenate(got)[int(4 * SR):]
    return float(np.sqrt(np.mean(y ** 2)))


_cr = [_rattle_creep(w) for w in (3.0, 1.0, 0.3)]
check('a thing pressed hard and barely moving hardly rubs at all',
      _cr[0] > 2.5 * _cr[-1]
      and all(b < a for a, b in zip(_cr, _cr[1:])),
      'rub ' + ' / '.join(f'{x:.5f}' for x in _cr)
      + ' as a gripped shell is turned at 3, 1 and 0.3 radians a second '
      + '-- it used to hold up whatever the speed, because it was taken '
      + 'from the pressing force alone')


# An irregular thing presents a different face every time it lands, its
# contact sits off to one side of its middle, and how much of the blow
# goes into turning it rather than lifting it changes with each one. So
# how bouncy a LANDING is varies, not just how bouncy the grain is.
#
# Without that, a handful driven along one axis onto a flat wall never
# comes out of step: with no collisions between them, grains alike in
# bounciness all leave and land together, and 128 of them arrived inside
# a fifth of a millisecond -- one enormous thud a cycle instead of a
# rattle. The direction scatter cannot do this job, and turning it up
# makes it worse, because the normal part of a tilted unit vector is its
# cosine and so scattering the direction can only ever REMOVE lift.
def _rattle_lockstep(variety, shake=2.0, freq=5.0, seconds=14):
    u = sc.RattleUnit(SR)
    u.set_count(128)
    u.shape = 3
    u.aspect_in.base = 2.4
    u.size_in.base = 0.15
    u.grain_in.base = 0.043
    u.bounce_in.base = 0.3
    u.variety_in.base = variety
    sz = sc.Signal()
    u.shake_z_in.sources.append(sz)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + b * BLOCK) / SR
        sz.data[:BLOCK] = shake * np.sin(2 * np.pi * freq * t)
        sz.constant = False
        u.render(BLOCK)
        got.append(u.knock.array(BLOCK).copy())
    y = np.concatenate(got)[int(4 * SR):]
    cycle = int(SR / freq)
    n = (len(y) // cycle) * cycle
    prof = (y[:n] ** 2).reshape(-1, cycle).mean(axis=0)
    cum = np.cumsum(prof) / max(prof.sum(), 1e-30)
    lo = int(np.searchsorted(cum, 0.1))
    hi = int(np.searchsorted(cum, 0.9))
    return 1000.0 * (hi - lo) / SR, float(np.sqrt(np.mean(y ** 2)))


_ls_same, _ls_lv0 = _rattle_lockstep(0.0)
_ls_var, _ls_lv1 = _rattle_lockstep(1.0)
check('unalike things do not all land at once, even on a flat end',
      _ls_var > 20.0 * _ls_same,
      f'a cycle\'s blows arrive over {_ls_same:.1f} ms from identical '
      f'grains and {_ls_var:.1f} ms from a mixed handful, of a 200 ms '
      f'cycle -- shaken along a tube, onto its flat cap, which is the '
      f'worst case there is for keeping step')
check('and that costs nothing in level',
      max(_ls_lv0, _ls_lv1) < 1.3 * min(_ls_lv0, _ls_lv1),
      f'{_ls_lv0:.4f} identical against {_ls_lv1:.4f} mixed')


# A rough wall is not a plane. A thing resting on one sits on a SLOPE,
# so when the wall drives it away it leaves along the local normal
# rather than the mean one -- a little faster or slower than its
# neighbour. That is the only thing that can break a handful out of
# step when the gesture is along one axis, the wall it lands on is
# flat, and nothing bounces.
#
# Nothing else can, and it is worth being plain about why: with no
# collisions between them the grains are EXACTLY identical. They leave
# together carrying the wall's own speed, fall in a field that is the
# same everywhere, and land dead. Spreading their sizes cannot part
# them, because a thing that starts higher also lands higher and the
# fall cancels. Spreading their bounciness cannot either, because with
# no bounce nothing about a landing survives it. 128 of them arrived
# inside a fifth of a millisecond, which is one click, not a shaker.
def _rattle_flatwall(texture, variety=1.0, shake=2.0, freq=5.0,
                     seconds=14):
    u = sc.RattleUnit(SR)
    u.set_count(128)
    u.shape = 3
    u.aspect_in.base = 2.74
    u.size_in.base = 0.092
    u.grain_in.base = 0.0
    u.bounce_in.base = 0.0
    u.friction_in.base = 0.15
    u.texture_in.base = texture
    u.variety_in.base = variety
    sz = sc.Signal()
    u.shake_z_in.sources.append(sz)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + b * BLOCK) / SR
        sz.data[:BLOCK] = shake * np.sin(2 * np.pi * freq * t)
        sz.constant = False
        u.render(BLOCK)
        got.append(u.knock.array(BLOCK).copy())
    y = np.concatenate(got)[int(4 * SR):]
    cycle = int(SR / freq)
    n = (len(y) // cycle) * cycle
    prof = (y[:n] ** 2).reshape(-1, cycle).mean(axis=0)
    cum = np.cumsum(prof) / max(prof.sum(), 1e-30)
    lo = int(np.searchsorted(cum, 0.1))
    hi = int(np.searchsorted(cum, 0.9))
    rms = float(np.sqrt(np.mean(y ** 2)))
    return (1000.0 * (hi - lo) / SR,
            float(np.max(np.abs(y)) / max(rms, 1e-12)), rms)


_fw_rough = _rattle_flatwall(0.22)
_fw_smooth = _rattle_flatwall(0.0)
_fw_same = _rattle_flatwall(0.22, variety=0.0)
check('a rough wall throws things off it unevenly, and so breaks step',
      _fw_rough[0] > 20.0 * _fw_smooth[0]
      and _fw_rough[1] < 0.4 * _fw_smooth[1],
      f'a cycle\'s blows arrive over {_fw_rough[0]:.1f} ms against '
      f'{_fw_smooth[0]:.1f} ms off a smooth one, crest {_fw_rough[1]:.0f} '
      f'against {_fw_smooth[1]:.0f} -- with nothing bouncing and every '
      f'grain a point, this is all there is')
check('and it takes unalike things to sit on a rough wall unalike',
      _fw_same[0] < 0.05 * _fw_rough[0],
      f'{_fw_same[0]:.1f} ms from identical grains on the same rough '
      f'wall against {_fw_rough[0]:.1f} ms from a mixed handful')
check('none of which changes how loud it is',
      max(_fw_rough[2], _fw_smooth[2]) < 1.35 * min(_fw_rough[2],
                                                    _fw_smooth[2]),
      f'{_fw_smooth[2]:.4f} smooth against {_fw_rough[2]:.4f} rough')


# ------------------------------------------------- a deliberate hit
# A CONTACT STIFFENS AS IT COMPRESSES. Press a ball against a plate and
# the harder you press the stiffer it gets, since more of it is
# touching: Hertz's law, force going as the squash to the power three
# halves. What follows is that a harder blow is a SHORTER one -- the
# contact time falls as the fifth root of the speed, the peak rises as
# the six fifths -- so hitting harder does not merely make a thing
# louder, it makes it brighter. That is most of what dynamics on a
# struck instrument are, and with a fixed contact time none of it
# happens: everything just gets louder, evenly, the way a sampler does.
def _strike(style='tap', force=1.0, hardness=0.6, spread=0.4,
            scatter=0.0, seconds=0.6):
    u = sc.StrikeUnit(SR)
    u.style = sc.StrikeUnit.STYLES.index(style)
    u.force_in.base = force
    u.hardness_in.base = hardness
    u.spread_in.base = spread
    u.scatter_in.base = scatter
    trig = sc.Signal()
    u.trigger_in.sources.append(trig)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        trig.data[:BLOCK] = 0.0
        if b == 2:
            trig.data[10] = 1.0
        trig.constant = False
        u.render(BLOCK)
        got.append(u.out.array(BLOCK).copy())
    return np.concatenate(got)


def _blow(y):
    """Impulse, peak, and how long the contact lasted."""
    live = np.flatnonzero(np.abs(y) > 1e-12)
    if live.size == 0:
        return 0.0, 0.0, 0.0, 0
    runs = np.split(live, np.flatnonzero(np.diff(live) > 1) + 1)
    return (float(np.abs(y).sum()), float(np.abs(y).max()),
            1000.0 * len(runs[0]) / SR, len(runs))


_hz = [_blow(_strike(force=f)) for f in (0.25, 0.5, 1.0, 2.0)]
_hz_imp = [x[0] for x in _hz]
_hz_dur = [x[2] for x in _hz]
_hz_pk = [x[1] for x in _hz]
check('a harder blow puts in more, exactly in proportion',
      abs(_hz_imp[3] / _hz_imp[0] - 8.0) < 0.02 * 8.0,
      'impulse ' + ' / '.join(f'{x:.3f}' for x in _hz_imp)
      + ' for force a quarter, a half, one and two')
check('and spends it FASTER, as Hertz says: the fifth root of it',
      abs(_hz_dur[3] / _hz_dur[0] - 8.0 ** -0.2) < 0.05 * 8.0 ** -0.2,
      'contact ' + ' / '.join(f'{x:.3f}' for x in _hz_dur)
      + ' ms -- eight times the blow over ' + f'{_hz_dur[3] / _hz_dur[0]:.3f}'
      + ' of the time, against the eight to the minus a fifth '
      + f'({8.0 ** -0.2:.3f}) a stiffening contact gives')
check('so it is brighter as well as louder, which is what dynamics are',
      abs(_hz_pk[3] / _hz_pk[0] - 8.0 ** 1.2) < 0.08 * 8.0 ** 1.2,
      f'peak up {_hz_pk[3] / _hz_pk[0]:.1f} times for eight times the '
      f'blow, against the {8.0 ** 1.2:.1f} that follows -- a fixed '
      f'contact would have given eight, and only loudness')

# Hardness colours a blow rather than weighing it: the momentum handed
# over is the same, spent over a longer or shorter contact.
_hd = [_blow(_strike(hardness=h)) for h in (0.0, 0.5, 1.0)]
check('hardness colours a blow and does not weigh it',
      max(x[0] for x in _hd) < 1.02 * min(x[0] for x in _hd)
      and _hd[0][2] > 8.0 * _hd[2][2],
      'impulse ' + ' / '.join(f'{x[0]:.3f}' for x in _hd)
      + ' while the contact runs ' + ' / '.join(f'{x[2]:.3f}' for x in _hd)
      + ' ms, softest to hardest')

# Each style is a different number of contacts laid out differently, and
# that is all a style is.
_st = {name: _blow(_strike(style=name)) for name in sc.StrikeUnit.STYLES}
check('every style is a different figure of contacts',
      _st['tap'][3] == 1 and _st['mallet'][3] == 1
      and _st['stick'][3] == 2 and _st['flam'][3] == 2
      and _st['drag'][3] == 4 and _st['brush'][3] > 8,
      ', '.join(f'{k} {v[3]}' for k, v in _st.items())
      + ' contacts')
check('and a mallet is a longer contact than a tap',
      _st['mallet'][2] > 2.0 * _st['tap'][2],
      f'{_st["tap"][2]:.3f} ms for a tap, {_st["mallet"][2]:.3f} for a '
      f'mallet')

# A brush is ONE stroke divided among its hairs, so it hands over what a
# tap does. A flam is two strokes and a drag is four, and those really
# do put in more -- which is what they are for.
check('a brush is one stroke shared out, not many strokes',
      abs(_st['brush'][0] - _st['tap'][0]) < 0.05 * _st['tap'][0]
      and _st['flam'][0] > 1.3 * _st['tap'][0]
      and _st['drag'][0] > 1.8 * _st['tap'][0],
      f'impulse {_st["tap"][0]:.2f} tap, {_st["brush"][0]:.2f} brush, '
      f'{_st["flam"][0]:.2f} flam, {_st["drag"][0]:.2f} drag')

# 'spread' is the time the figure is laid over, and means nothing at all
# to a style with one contact in it.
_sp_near = _blow(_strike(style='flam', spread=0.0))
_sp_far = _blow(_strike(style='flam', spread=1.0))
check('spread opens out a figure and leaves a single hit alone',
      _sp_far[3] == _sp_near[3]
      and abs(_st['tap'][0] - _blow(_strike(spread=1.0))[0]) < 1e-9,
      'a flam still two contacts either way, and a tap unmoved by it')

# The trigger's own height carries the dynamics, so one cord does both.
# Both above the threshold, or the quieter one never fires at all --
# the trigger has to CROSS to be a strike.
_tv = []
for _tv_h in (0.6, 1.2):
    _tv_u = sc.StrikeUnit(SR)
    _tv_t = sc.Signal()
    _tv_u.trigger_in.sources.append(_tv_t)
    _tv_got = []
    for _tv_b in range(40):
        _tv_t.data[:BLOCK] = 0.0
        if _tv_b == 2:
            _tv_t.data[10] = _tv_h
        _tv_t.constant = False
        _tv_u.render(BLOCK)
        _tv_got.append(_tv_u.out.array(BLOCK).copy())
    _tv.append(_blow(np.concatenate(_tv_got)))
check('and how tall the trigger is, is how hard it hits',
      _tv[0][0] > 0.0
      and abs(_tv[1][0] / max(_tv[0][0], 1e-12) - 2.0) < 0.05 * 2.0,
      f'impulse {_tv[0][0]:.3f} from a trigger of 0.6 and '
      f'{_tv[1][0]:.3f} from one of 1.2')


# And it has to arrive at a useful LEVEL, which an honest impulse does
# not on its own. A blow of area one is the right convention, but a
# resonator's excite inlet is not calibrated for impulses -- it is
# calibrated for SIGNALS, scaling each mode by the root of one minus its
# pole radius so that a sustained drive sounds the same however long the
# decay. That is right for a bow, and it leaves a bare unit impulse some
# forty decibels under what the same impulse does through the trigger,
# which is normalised the other way about. Two sensible conventions that
# do not meet, and the gap is not even a constant: because of that
# normalisation an impulse gets weaker the longer the decay, so the same
# blow wants anywhere from eight to two hundred times over a spread of
# settings. A unit blow is worth the middle of that, and 'level' reaches
# four times it.
def _strike_against_trigger(freq=220.0, decay=1.2, hardness=0.6,
                            seconds=1.5):
    table = [(1.0, 1.0, 1.0), (2.32, 0.8, 0.8), (4.25, 0.6, 0.6),
             (6.1, 0.4, 0.5)]

    def bank():
        m = sc.ModalUnit(SR)
        m.set_modes(table)
        m.frequency_in.base = freq
        m.decay_in.base = decay
        m.hardness_in.base = hardness
        return m

    ring = bank()
    trig = sc.Signal()
    ring.trigger_in.sources.append(trig)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        trig.data[:BLOCK] = 0.0
        if b == 2:
            trig.data[10] = 1.0
        trig.constant = False
        ring.render(BLOCK)
        got.append(ring.out.array(BLOCK).copy())
    button = float(np.sqrt(np.mean(np.concatenate(got) ** 2)))

    hammer = sc.StrikeUnit(SR)
    hammer.hardness_in.base = hardness
    fire = sc.Signal()
    hammer.trigger_in.sources.append(fire)
    ring2 = bank()
    drive = sc.Signal()
    ring2.excite_in.sources.append(drive)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        fire.data[:BLOCK] = 0.0
        if b == 2:
            fire.data[10] = 1.0
        fire.constant = False
        hammer.render(BLOCK)
        drive.data[:BLOCK] = hammer.out.array(BLOCK)
        drive.constant = False
        ring2.render(BLOCK)
        got.append(ring2.out.array(BLOCK).copy())
    return button, float(np.sqrt(np.mean(np.concatenate(got) ** 2)))


_sv = [_strike_against_trigger(freq=f, decay=d)
       for f in (110.0, 660.0) for d in (0.3, 4.0)]
_sv_db = [20.0 * math.log10(b / max(a, 1e-12)) for a, b in _sv]
check('a hit lands where a resonator\'s own trigger lands',
      all(-11.0 < x < 8.0 for x in _sv_db),
      'strike~ against the trigger button: '
      + ' / '.join(f'{x:+.1f}' for x in _sv_db)
      + ' dB over low and high pitches, short and long decays -- it was '
      + 'thirty-nine decibels under, which is a unit impulse arriving at '
      + 'an inlet calibrated for signals')
check('and level reaches far enough to close what is left',
      sc.StrikeUnit(SR).level_in.eval(1) is not None
      and _sv_db and max(_sv_db) - min(_sv_db) < 20.0,
      f'the spread across those four is '
      f'{max(_sv_db) - min(_sv_db):.1f} dB, and level reaches four times '
      f'-- the spread is the excite path making an impulse weaker the '
      f'longer the decay, which no single gain can answer')


# ------------------------------------------------ a strip and its mute
# A mute is not the fader pulled down. The whole point is that the HANDLE
# stays where it was set, so the balance you had is the balance you come
# back to -- and it has to ARRIVE at silence, not merely approach it.
def _fader_run(mute_at=None, unmute_at=None, blocks=60, unit=None):
    u = unit if unit is not None else sc.FaderUnit(SR)
    sig = sc.Signal()
    u.signal_in.sources.append(sig)
    got = []
    for b in range(blocks):
        t = (np.arange(BLOCK) + b * BLOCK) / SR
        sig.data[:BLOCK] = 0.5 * np.sin(2 * np.pi * 220.0 * t)
        sig.constant = False
        if mute_at is not None and b == mute_at:
            u.muted = True
        if unmute_at is not None and b == unmute_at:
            u.muted = False
        u.render(BLOCK)
        got.append(u.out.array(BLOCK).copy())
    return np.concatenate(got), u


_fm, _fu = _fader_run(mute_at=20, unmute_at=40)
_seg = lambda a, b: float(np.sqrt(np.mean(_fm[a * BLOCK:b * BLOCK] ** 2)))
check('a mute reaches silence, and does not merely approach it',
      _seg(25, 39) == 0.0 and _seg(10, 19) > 0.3,
      f'{_seg(10, 19):.4f} open, {_seg(25, 39):.6f} muted -- chased by a '
      f'fraction of the way each block it was still 46 dB short of '
      f'silence forty milliseconds in, so the channel never went quiet')
check('and unmuting comes back to the balance it had',
      abs(_seg(50, 59) - _seg(10, 19)) < 0.01 * _seg(10, 19)
      and abs(_fu.position_in.base - sc.FaderUnit.UNITY_POSITION) < 1e-9,
      f'{_seg(10, 19):.4f} before, {_seg(50, 59):.4f} after, with the '
      f'handle still at {_fu.position_in.base:.3f}')
# The step into silence has to be a ramp, or it is a click: a 220 Hz sine
# at 0.5 already steps 0.0157 between samples of its own accord, and the
# mute must not beat that.
check('and it fades rather than steps, so it does not click',
      float(np.abs(np.diff(_fm)).max()) < 1.5 * 0.5 * 2.0 * math.pi
      * 220.0 / SR,
      f'biggest step between samples {np.abs(np.diff(_fm)).max():.4f}, '
      f'against the sine\'s own {0.5 * 2 * math.pi * 220.0 / SR:.4f} -- '
      f'a hard switch would have shown about 0.5')

# The strip is a fader and a socket in one, and it must be a SINK to the
# compiler, which finds them by what they are rather than by asking.
_strip = sc.FaderOutUnit(SR)
check('a strip that ends at the device is a terminus to the compiler',
      isinstance(_strip, sc.AudioOutUnit),
      'the compiler collects sinks by isinstance, so a subclass of the '
      'socket is one without the compiler being told anything')
_sm, _su = _fader_run(mute_at=15, blocks=30, unit=_strip)
_mix = np.zeros((BLOCK, 2), dtype=np.float32)
_strip.mix_into(_mix, BLOCK)
check('and it carries the fader whole rather than a copy of it',
      _strip.levels is _strip.fader.levels
      and _strip.current_db() == _strip.fader.current_db()
      and float(np.abs(_mix).max()) == 0.0,
      'meters, dB and mute all read through to the contained fader, so '
      'the taper and the pan law cannot drift from fader~\'s; muted, '
      'nothing reaches the device')


# ------------------------------------------- modes from a shape
# A table solved for, rather than looked up. What has to be right is the
# SHAPE: modal~'s table is ratios and its 'frequency' sets the pitch, so
# an error in the overall scale costs nothing, and the elastic constants
# only matter in their proportions.
from dpg_system import modal_shape as msh

BAR_BOOK = (1.0, 2.7565, 5.4039, 8.9330)


def _bar_ratios(sweep_mode, detail=24, length=1.0, half=0.025):
    stations = detail if sweep_mode != 'mirror' else detail // 2
    profile = [half] * (stations + 1)
    section = (msh.disc_section(2, 2) if sweep_mode == 'revolve'
               else msh.rect_section(3, 3))
    nodes, hexes = msh.sweep(profile, length, sweep_mode, 2.0 * half,
                             section)
    freq, shape = msh.solve_modes(nodes, hexes, 'steel', want=18)
    table = msh.table_from(nodes, freq, shape, length, strike=1.0,
                           damping=0.0)
    return [row[0] for row in table], len(nodes)


_bx, _bn = _bar_ratios('extrude')
check('a bar solved from its shape rings where a bar rings',
      all(abs(g - b) < 0.06 * b for g, b in zip(_bx[:2], BAR_BOOK[:2])),
      'ratios ' + '  '.join(f'{x:.3f}' for x in _bx[:4])
      + ' from ' + str(_bn) + ' nodes, against the free-free bar\'s '
      + '  '.join(f'{x:.3f}' for x in BAR_BOOK))

# The same bar built three different ways -- extruded on a square grid,
# mirrored, and revolved on rings round a square core. The BENDING
# ratios must not care, since the shape of the section cancels out of
# them: a round rod bends in the same proportions as a square one. What
# does differ further up is TORSION, and it has to: twisting stiffness
# depends very much on the section, so a square bar and a round one are
# not the same thing to twist.
_bm, _ = _bar_ratios('mirror')
_br, _ = _bar_ratios('revolve')
check('and it does not matter which way the mesh was swept',
      all(abs(a - b) < 0.03 * a for a, b in zip(_bx[:2], _bm[:2]))
      and all(abs(a - b) < 0.03 * a for a, b in zip(_bx[:2], _br[:2])),
      'bending ' + ' '.join(f'{x:.3f}' for x in _bx[:2])
      + ' extruded, ' + ' '.join(f'{x:.3f}' for x in _bm[:2])
      + ' mirrored, ' + ' '.join(f'{x:.3f}' for x in _br[:2])
      + ' revolved -- three meshes, one shape')

# Refining must settle, or the answer is the mesh talking rather than
# the shape.
_bc = [_bar_ratios('extrude', detail=d)[0][1] for d in (12, 24, 36)]
check('and refining the mesh settles instead of wandering',
      abs(_bc[2] - _bc[1]) < 0.5 * abs(_bc[1] - _bc[0]) + 0.01,
      'second ratio ' + ' -> '.join(f'{x:.3f}' for x in _bc)
      + ' as the mesh is refined')

# Where it is struck decides what is heard. A free-free bar has a node
# at its middle in the second mode, so a strike there cannot wake it --
# and the mode should be ABSENT, not quiet.
_st_nodes, _st_hexes = msh.sweep([0.025] * 25, 1.0, 'extrude', 0.05,
                                 msh.rect_section(3, 3))
_st_f, _st_s = msh.solve_modes(_st_nodes, _st_hexes, 'steel', want=18)
_st_end = [r[0] for r in msh.table_from(_st_nodes, _st_f, _st_s, 1.0,
                                        strike=1.0, damping=0.0)]
_st_mid = [r[0] for r in msh.table_from(_st_nodes, _st_f, _st_s, 1.0,
                                        strike=0.5, damping=0.0)]
check('a strike at a mode\'s node does not wake that mode',
      any(abs(x - BAR_BOOK[1]) < 0.06 * BAR_BOOK[1] for x in _st_end)
      and not any(abs(x - BAR_BOOK[1]) < 0.06 * BAR_BOOK[1]
                  for x in _st_mid),
      'struck at the end ' + ' '.join(f'{x:.3f}' for x in _st_end[:3])
      + ', struck dead centre ' + ' '.join(f'{x:.3f}' for x in _st_mid[:3])
      + ' -- the 2.76 mode has a node at the middle and is simply gone')

# And which WAY it is struck. A bar hit on its face wakes the modes that
# bend it that way, not the ones that bend it sideways or twist it, so a
# weight taken from how far a mode moves without asking which way fills
# the table with modes the mallet never reached.
_dir_face = msh.table_from(_st_nodes, _st_f, _st_s, 1.0, strike=1.0,
                           direction=(0.0, 0.0, 1.0))
_dir_all = msh.table_from(_st_nodes, _st_f, _st_s, 1.0, strike=1.0,
                          direction=(1.0, 1.0, 1.0))
check('and which way it is struck thins the table to what it can reach',
      len(_dir_face) < len(_dir_all),
      f'{len(_dir_face)} modes reachable striking the face, '
      f'{len(_dir_all)} striking every way at once')

# The point of the whole thing: the shape decides the tuning. A marimba
# bar is undercut until its second mode lands two octaves up, and the
# cut is what puts it there.
def _undercut(cut, detail=24):
    x = np.linspace(-1.0, 1.0, detail + 1)
    profile = [0.03 * (1.0 - cut * math.cos(v * math.pi / 2.0) ** 2)
               for v in x]
    nodes, hexes = msh.sweep(profile, 1.0, 'extrude', 0.05,
                             msh.rect_section(3, 3))
    freq, shape = msh.solve_modes(nodes, hexes, 'wood', want=18)
    table = msh.table_from(nodes, freq, shape, 1.0, strike=1.0,
                           damping=0.0)
    return [row[0] for row in table]


_uc = [_undercut(c)[1] for c in (0.0, 0.15, 0.3, 0.45)]
check('cutting the middle away tunes the second mode up, as it should',
      all(b > a for a, b in zip(_uc, _uc[1:])) and _uc[-1] > 1.2 * _uc[0],
      'second ratio ' + ' -> '.join(f'{x:.3f}' for x in _uc)
      + ' as the middle is cut away -- a marimba bar is cut until this '
      + 'reaches 4.000')

# Three columns, in the shape modal~ eats.
_tb = msh.mode_table([0.03, 0.028, 0.028, 0.03], length=0.4,
                     sweep_mode='extrude', depth=0.02, material='wood',
                     count=12)
check('what comes out is a table modal~ can eat',
      len(_tb) >= 3 and all(len(r) == 3 for r in _tb)
      and abs(_tb[0][0] - 1.0) < 1e-9
      and all(0.0 < r[1] <= 1.0 for r in _tb)
      and all(r[0] > 0.0 for r in _tb),
      f'{len(_tb)} rows of [ratio, weight, decay], first ratio '
      f'{_tb[0][0]:.3f}, weights {_tb[0][1]:.2f} down to '
      f'{min(r[1] for r in _tb):.2f}')

# A mesh turned inside out would hand back negative stiffness and a
# spectrum of nonsense, so it says so instead.
try:
    msh.sweep([0.03, -0.01, 0.03], 0.4, 'extrude')
    _guard = False
except ValueError:
    _guard = True
check('and a profile that cannot be swept is refused, not solved',
      _guard,
      'a negative half-width raises rather than returning a spectrum '
      'of nonsense')


# ------------------------------------------------- every unit renders
# A block of spin~'s NaN-recovery once got pasted into ShakerUnit's
# render, referring to outlets that class does not have. It shipped,
# because nothing here had ever rendered a shaker -- the suite tests the
# units it has laws for, and says nothing at all about the rest. This
# says the cheapest possible thing about all of them: that they run, and
# that what comes out is a number.
def _every_unit():
    made, broken = 0, []
    table = [(1.0, 1.0, 1.0), (2.32, 0.5, 0.75), (4.25, 0.3, 0.5)]
    drives = ('shake_in', 'excite_in', 'velocity_in', 'pressure_in',
              'speed_in', 'drop_in', 'spin_in', 'bounce_in', 'breath_in',
              'trigger_in', 'force_in', 'fill_in')
    for name, cls in sorted(vars(sc).items()):
        if not (isinstance(cls, type) and issubclass(cls, sc.Unit)
                and cls is not sc.Unit):
            continue
        try:
            u = cls(SR)
            if hasattr(u, 'set_modes'):
                u.set_modes(table)
            sig = sc.Signal()
            for port in drives:
                if hasattr(u, port):
                    getattr(u, port).sources.append(sig)
                    break
            got = []
            for _ in range(int(0.3 * SR / BLOCK)):
                sig.data[:BLOCK] = 0.6
                sig.constant = False
                u.render(BLOCK)
                if hasattr(u, 'out'):
                    got.append(u.out.array(BLOCK).copy())
            made += 1
            if got and not np.isfinite(np.concatenate(got)).all():
                broken.append(f'{name}: not finite')
        except Exception as problem:
            broken.append(f'{name}: {type(problem).__name__}: {problem}')
    return made, broken


_units_made, _units_broken = _every_unit()
check('every unit renders a block without falling over',
      not _units_broken and _units_made > 30,
      f'{_units_made} units rendered'
      + (f'; broken: {_units_broken}' if _units_broken else ''))

# --------------------------------------------------------- vessel~
# Water in a ringing vessel does three separate things and they do not
# agree. Filling takes the pitch DOWN, weighted to the fifth power of
# the fill because the wall hardly moves at the base and most at the
# rim. Tipping hardly moves the pitch at all -- it makes the thing BEAT,
# by loading one side more than the other and splitting mode pairs that
# were degenerate. And moving the tip sets it sloshing.
GLASSV = [(1.0, 1.0, 1.0), (2.32, 0.5, 0.75), (4.25, 0.3, 0.5),
          (6.63, 0.2, 0.32), (9.38, 0.1, 0.2)]


def run_vessel(fill=0.0, tip=0.0, seconds=3.0, decay=4.0, freq=800.0):
    u = sc.VesselUnit(SR)
    u.set_modes(GLASSV)
    u.frequency_in.base = freq
    u.decay_in.base = decay
    u.fill_in.base = fill
    u.tip_in.base = tip
    sig = sc.Signal()
    u.trigger_in.sources.append(sig)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        sig.data[:BLOCK] = 0.0
        if b == 1:
            sig.data[0] = 1.0
        sig.constant = False
        u.render(BLOCK)
        got.append(u.out.array(BLOCK).copy())
    return np.concatenate(got)


def _vessel_pitch(y):
    seg = y[int(0.2 * SR):int(0.2 * SR) + 32768]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    spec[:20] = 0
    return int(np.argmax(spec)) * SR / 32768


_vp = {f: _vessel_pitch(run_vessel(fill=f)) for f in (0.0, 0.25, 0.7, 1.0)}
check('filling a vessel takes its pitch down about ten semitones',
      -11.5 < 12 * np.log2(_vp[1.0] / _vp[0.0]) < -9.0,
      f'{12 * np.log2(_vp[1.0] / _vp[0.0]):.2f} semitones empty to full')
check('and does almost nothing until it is well up the wall',
      abs(12 * np.log2(_vp[0.25] / _vp[0.0])) < 0.3
      and 12 * np.log2(_vp[0.7] / _vp[0.0]) < -2.0,
      f'{12 * np.log2(_vp[0.25] / _vp[0.0]):+.2f} st at a quarter, '
      f'{12 * np.log2(_vp[0.7] / _vp[0.0]):+.2f} st at seven tenths')


def _vessel_beat(fill, tip):
    """How deeply the ring wavers, with the decay divided out.

    Taking the envelope's spectrum raw does not work: a plain decay is
    one big low-frequency component and scores HIGHER than a beat. So
    the trend is removed first -- a one-second moving average is the
    decay -- and what is left is the beating.
    """
    y = run_vessel(fill=fill, tip=tip, seconds=8.0, decay=25.0)
    env = np.abs(sig_hilbert(y[int(0.5 * SR):]))[::64]
    sr = SR / 64.0
    width = int(sr * 1.0) | 1
    trend = np.convolve(env, np.ones(width) / width, mode='same')
    keep = slice(width, len(env) - width)
    resid = env[keep] / np.maximum(trend[keep], 1e-12) - 1.0
    spec = np.abs(np.fft.rfft((resid - resid.mean()) * np.hanning(len(resid))))
    f = np.fft.rfftfreq(len(resid), 1.0 / sr)
    band = (f > 0.15) & (f < 12.0)
    return float(np.std(resid)), float(f[band][int(np.argmax(spec[band]))])


_d_level, _ = _vessel_beat(0.5, 0.0)
_d_far, _f_far = _vessel_beat(0.5, 45.0)
_d_some, _ = _vessel_beat(0.5, 20.0)
_d_warble, _f_warble = _vessel_beat(0.5, 30.0)
check('tipping a vessel makes it beat; standing level it does not',
      _d_far > 10.0 * _d_level and _d_level < 0.03,
      f'waver {_d_level:.4f} level, {_d_far:.4f} tipped right over')
check('the beat has a threshold in it, as the geometry does',
      _d_some < 0.05 < _d_warble,
      f'{_d_some:.4f} at twenty degrees, {_d_warble:.4f} at thirty -- '
      f'a tilted surface is the wrong shape to split a pair until the '
      f'water line reaches the base or the rim')
check('and it beats faster the further it goes over',
      _f_far > 3.0 * _f_warble,
      f'{_f_warble:.2f} Hz at thirty, {_f_far:.2f} Hz at forty-five')

# Tipping splits the ring in two; where the blow lands against where
# the water is decides which of them answers. On a belly of the pattern
# only one does and there is no beat at all -- which is also why the
# simulation of a static tilt showed ONE line, not two, until it was
# struck off-axis.
def _vessel_turn(turn):
    u = sc.VesselUnit(SR)
    u.set_modes(GLASSV)
    u.frequency_in.base = 800.0
    u.decay_in.base = 25.0
    u.fill_in.base = 0.5
    u.tip_in.base = 35.0
    u.turn_in.base = turn
    sig = sc.Signal()
    u.trigger_in.sources.append(sig)
    got = []
    for b in range(int(8.0 * SR / BLOCK)):
        sig.data[:BLOCK] = 0.0
        if b == 1:
            sig.data[0] = 1.0
        sig.constant = False
        u.render(BLOCK)
        got.append(u.out.array(BLOCK).copy())
    y = np.concatenate(got)
    env = np.abs(sig_hilbert(y[int(0.5 * SR):]))[::64]
    sr = SR / 64.0
    width = int(sr * 1.0) | 1
    trend = np.convolve(env, np.ones(width) / width, mode='same')
    keep = slice(width, len(env) - width)
    resid = env[keep] / np.maximum(trend[keep], 1e-12) - 1.0
    return float(np.std(resid)), float(np.sqrt(np.mean(y ** 2)))


_t_on, _l_on = _vessel_turn(0.0)
_t_mid, _l_mid = _vessel_turn(22.5)
_t_back, _l_back = _vessel_turn(45.0)
_t_round, _l_round = _vessel_turn(90.0)
check('turning the vessel decides whether the beat is heard at all',
      _t_on < 0.05 < _t_mid and _t_back < 0.05,
      f'{_t_on:.4f} on a belly, {_t_mid:.4f} between, {_t_back:.4f} on '
      f'the next')
check('and it comes round every ninety degrees',
      abs(_t_round - _t_on) < 0.02,
      f'{_t_on:.4f} at nought, {_t_round:.4f} at ninety')
check('turning moves the sound between the pair, it does not fade it',
      max(_l_on, _l_mid, _l_back) < 1.06 * min(_l_on, _l_mid, _l_back),
      f'level within {20 * np.log10(max(_l_on, _l_mid, _l_back) / min(_l_on, _l_mid, _l_back)):.2f} dB '
      f'across the turn (sharing the weight instead of the amplitude '
      f'cost 3 dB)')

# A swirl is not a tilt that moves -- it is a different sound. The split
# lives in the frame of whatever is loading the vessel, so if that frame
# turns, the pattern turns with it and a fixed listener hears the
# bellies go past. Sidebands either side of every mode, spaced at four
# times the swirl, rather than a beat.
def _vessel_lines(swirl, quiet_water=True):
    """The lines a swirled vessel shows.

    With the water held still this is the rotating pickup alone, and it
    puts one sideband either side at four times the swirl. Let the swirl
    push the water as well and the slosh modulates the pitch on top of
    that, which fills in a comb -- true, and a separate thing, so it is
    switched off here rather than measured through.
    """
    was = sc.VesselUnit.SWIRL_DRIVE
    try:
        if quiet_water:
            sc.VesselUnit.SWIRL_DRIVE = 0.0
        u = sc.VesselUnit(SR)
        u.set_modes([(1.0, 1.0, 1.0)])
        u.frequency_in.base = 800.0
        u.decay_in.base = 30.0
        u.fill_in.base = 0.5
        u.tip_in.base = 35.0
        u.turn_in.base = 0.0
        u.swirl_in.base = swirl
        sig = sc.Signal()
        u.trigger_in.sources.append(sig)
        got = []
        for b in range(int(6.0 * SR / BLOCK)):
            sig.data[:BLOCK] = 0.0
            if b == 1:
                sig.data[0] = 1.0
            sig.constant = False
            u.render(BLOCK)
            got.append(u.out.array(BLOCK).copy())
    finally:
        sc.VesselUnit.SWIRL_DRIVE = was
    y = np.concatenate(got)[int(0.5 * SR):]
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    band = (f > 700.0) & (f < 900.0)
    f, spec = f[band], spec[band]
    return sorted(f[i] for i in range(2, len(spec) - 2)
                  if spec[i] > spec[i-1] and spec[i] > spec[i+1]
                  and spec[i] > 0.15 * spec.max())


_sw_still = _vessel_lines(0.0)
_sw_one = _vessel_lines(1.0)
_sw_two = _vessel_lines(2.0)
check('swirling moves the line into a pair either side of where it was',
      len(_sw_still) == 1 and len(_sw_one) == 2
      and min(_sw_one) < _sw_still[0] < max(_sw_one),
      f'{len(_sw_still)} line held still, {len(_sw_one)} swirled -- and '
      f'the original is GONE from between them, which is what a pickup '
      f'that goes right round does: the node sweeps fully past, so what '
      f'is left is the two sidebands and no carrier')
check('and they sit four times the swirl rate out',
      abs((max(_sw_one) - _sw_still[0]) - 4.0) < 0.5
      and abs((max(_sw_two) - _sw_still[0]) - 8.0) < 0.5,
      f'{max(_sw_one) - _sw_still[0]:.2f} Hz out at one turn a second, '
      f'{max(_sw_two) - _sw_still[0]:.2f} at two')
check('and letting the swirl push the water fills that in',
      len(_vessel_lines(1.0, quiet_water=False)) > len(_sw_one),
      f'{len(_sw_one)} lines with the water still, '
      f'{len(_vessel_lines(1.0, quiet_water=False))} with it sloshing')


def _vessel_slosh(swirl):
    u = sc.VesselUnit(SR)
    u.set_modes([(1.0, 1.0, 1.0), (2.32, 0.5, 0.75)])
    u.frequency_in.base = 800.0
    u.decay_in.base = 8.0
    u.fill_in.base = 0.5
    u.tip_in.base = 30.0
    u.swirl_in.base = swirl
    sig = sc.Signal()
    u.trigger_in.sources.append(sig)
    seen = []
    for b in range(int(6.0 * SR / BLOCK)):
        # It has to be sounding: a silent bank with a silent input skips
        # its own render, and then nothing sloshes because nothing runs.
        sig.data[:BLOCK] = 0.0
        if b == 1:
            sig.data[0] = 1.0
        sig.constant = False
        u.render(BLOCK)
        seen.append(u._slosh_x)
    return float(np.std(np.array(seen[len(seen) // 2:])))


check('swirling near the sloshing rate builds the slop up',
      _vessel_slosh(3.5) > 3.0 * _vessel_slosh(1.0)
      and _vessel_slosh(3.5) > 3.0 * _vessel_slosh(7.0),
      f'slop {_vessel_slosh(1.0):.4f} slow, {_vessel_slosh(3.5):.4f} at '
      f'the rate, {_vessel_slosh(7.0):.4f} fast (resonance is 3.6 Hz)')

check('tipping barely moves the pitch, unlike filling',
      abs(12 * np.log2(_vessel_pitch(run_vessel(fill=0.5, tip=30.0))
                       / _vessel_pitch(run_vessel(fill=0.5)))) < 1.0,
      'under a semitone at thirty degrees')

# ------------------------------------------------------------- motor~
def run_motor(speed=0.6, load=0.3, parts=4, tone=0.4, throb=0.35,
              grind=0.4, rate=45.0, seconds=2.0):
    u = sc.MotorUnit(SR)
    u.speed_in.base = speed
    u.load_in.base = load
    u.parts_in.base = parts
    u.tone_in.base = tone
    u.throb_in.base = throb
    u.grind_in.base = grind
    u.rate_in.base = rate
    n = int(seconds * SR / BLOCK)
    y = np.zeros(n * BLOCK)
    for b in range(n):
        u.render(BLOCK)
        y[b*BLOCK:(b+1)*BLOCK] = u.out.array(BLOCK)
    return y[SR//2:]

def _firing_peak(y):
    m = np.abs(np.fft.rfft(y * np.hanning(len(y))))**2
    f = np.fft.rfftfreq(len(y), 1.0/SR)
    band = (f > 20) & (f < 2000)
    return f[band][np.argmax(m[band])]

mfp = _firing_peak(run_motor(speed=0.6, throb=0.0, grind=0.0, load=0.0))
check('motor pitch is linear in speed at rate x parts',
      abs(mfp - 108.0) < 16.0, f'{mfp:.0f} Hz (expect 108)')
check('motor parts set the harmonic spacing',
      _firing_peak(run_motor(parts=8, throb=0.0, grind=0.0, load=0.0))
      > 3.0 * _firing_peak(run_motor(parts=2, throb=0.0, grind=0.0,
                                     load=0.0)))

my = run_motor(speed=0.6, throb=0.9, grind=0.0, load=0.2, seconds=4.0)
menv = np.abs(my)
mwin = int(SR/200)
msm = np.convolve(menv, np.ones(mwin)/mwin, mode='valid')
msm = msm - msm.mean()
mem = np.abs(np.fft.rfft(msm * np.hanning(len(msm))))**2
mef = np.fft.rfftfreq(len(msm), 1.0/SR)
mband = (mef > 5) & (mef < 100)
mlope = mef[mband][np.argmax(mem[mband])]
# which multiple of the rev rate dominates depends on the instance's
# cylinder draw (an opposite pair beats at 2/rev); any low multiple is
# the lope
_k = max(1, round(mlope / 27.0))
check('motor throb lopes at a multiple of the revolution',
      _k <= 4 and abs(mlope - _k * 27.0) < 7.0,
      f'{mlope:.1f} Hz = {_k}x rotation')

check('motor load is punch',
      np.sqrt(np.mean(run_motor(load=0.95)**2))
      > 1.5 * np.sqrt(np.mean(run_motor(load=0.05)**2)))

def _hf_motor(y):
    m = np.abs(np.fft.rfft(y))**2
    f = np.fft.rfftfreq(len(y), 1.0/SR)
    return m[f > 1500].sum() / m.sum()
check('motor grind leans on load',
      _hf_motor(run_motor(load=0.9, grind=1.0, tone=1.0, throb=0.0))
      > 1.5 * _hf_motor(run_motor(load=0.1, grind=1.0, tone=1.0,
                                  throb=0.0)))

mu = sc.MotorUnit(SR)
mu.speed_in.base = 0.0
for _ in range(40):
    mu.render(BLOCK)
check('motor stillness is silent-constant', mu.out.constant)

mflat = run_motor(speed=1.2, load=1.0, throb=1.0, grind=1.0)
check('motor flat out is bounded',
      np.isfinite(mflat).all() and np.max(np.abs(mflat)) < 3.0,
      f'peak {np.max(np.abs(mflat)):.2f}')

# ----------------------------------------------------------- bubbles~
def run_bubbles(flow=0.8, size=0.5, spread=0.4, chirp=0.6, gulp=0.0,
                regular=0.0, density=80.0, seconds=4.0):
    u = sc.BubblesUnit(SR)
    u.flow_in.base = flow
    u.size_in.base = size
    u.spread_in.base = spread
    u.chirp_in.base = chirp
    u.gulp_in.base = gulp
    u.regular_in.base = regular
    u.density_in.base = density
    n = int(seconds * SR / BLOCK)
    y = np.zeros(n * BLOCK)
    for b in range(n):
        u.render(BLOCK)
        y[b*BLOCK:(b+1)*BLOCK] = u.out.array(BLOCK)
    return y[SR//4:]

def _one_bubble(chirp):
    # size 0.35 keeps the tails short enough for quiet gaps to exist
    y = run_bubbles(flow=0.4, size=0.35, density=3.0, spread=0.0,
                    chirp=chirp, seconds=8.0)
    env = np.abs(y)
    q = int(0.05*SR)
    guard = int(0.01*SR)
    i = q + guard
    while i < len(env) - int(0.1*SR):
        if env[i] > 0.05 and env[i-q-guard:i-guard].max() < 0.008:
            return y[i:]
        i += 1
    return None

def _zc(seg):
    sgn = np.signbit(seg)
    return np.sum(sgn[1:] != sgn[:-1]) / (2 * len(seg) / SR)

bub = _one_bubble(1.0)
if bub is None:
    check('bubble pitch rises as it dies', False, 'no isolated bubble')
else:
    b_early = _zc(bub[:int(0.01*SR)])
    b_late = _zc(bub[int(0.05*SR):int(0.06*SR)])
    check('bubble pitch rises as it dies', b_late > 1.1 * b_early,
          f'{b_early:.0f} -> {b_late:.0f} Hz')
ping = _one_bubble(0.0)
if ping is None:
    check('chirp 0 is a pure submerged ping', False, 'no isolated bubble')
else:
    p_early = _zc(ping[:int(0.01*SR)])
    p_late = _zc(ping[int(0.05*SR):int(0.06*SR)])
    check('chirp 0 is a pure submerged ping',
          abs(p_late - p_early) < 0.08 * p_early)

def _cent(y):
    m = np.abs(np.fft.rfft(y))**2
    f = np.fft.rfftfreq(len(y), 1.0/SR)
    return (m*f).sum()/m.sum()
check('bubble size runs fizz to glug',
      _cent(run_bubbles(size=0.1)) > 4.0 * _cent(run_bubbles(size=0.95)))

def _low_frac(y, f_center=548.0):
    m = np.abs(np.fft.rfft(y))**2
    f = np.fft.rfftfreq(len(y), 1.0/SR)
    return m[(f > 100) & (f < f_center*0.55)].sum() / m.sum()
check('gulp is a low onset presence',
      _low_frac(run_bubbles(gulp=1.0, spread=0.0, density=8.0))
      > 3.0 * _low_frac(run_bubbles(gulp=0.0, spread=0.0, density=8.0)))

check('bubble flow is rate (power rises)',
      np.sqrt(np.mean(run_bubbles(flow=1.2)**2))
      > 1.5 * np.sqrt(np.mean(run_bubbles(flow=0.3)**2)))

bu3 = sc.BubblesUnit(SR)
bu3.flow_in.base = 0.0
for _ in range(40):
    bu3.render(BLOCK)
check('bubbles stillness is silent-constant', bu3.out.constant)

btor = run_bubbles(flow=1.5, density=400.0, spread=1.0, gulp=1.0)
check('bubbles full torrent bounded',
      np.isfinite(btor).all() and np.max(np.abs(btor)) < 4.0,
      f'peak {np.max(np.abs(btor)):.2f}')

def _spawn_cv(regular):
    u = sc.BubblesUnit(SR)
    u.flow_in.base = 0.8
    u.spread_in.base = 0.0
    u.regular_in.base = regular
    u.decay_in.base = 0.3
    u.density_in.base = 8.0
    spawns = []
    prev = u._amp.copy()
    for b in range(int(10.0*SR/BLOCK)):
        u.render(BLOCK)
        if (u._amp > prev + 1e-6).any():
            spawns.append(b)
        prev = u._amp.copy()
    iv = np.diff(spawns)
    return np.std(iv)/np.mean(iv), len(spawns)

bcv0, bn0 = _spawn_cv(0.0)
bcv1, bn1 = _spawn_cv(1.0)
check('bubbles regular: boil to metronome at the same rate',
      bcv0 > 0.5 and bcv1 < 0.15 and 0.6 < bn0/bn1 < 1.6,
      f'CV {bcv0:.2f} -> {bcv1:.2f}')

def _ring_len(decay):
    u = sc.BubblesUnit(SR)
    u.flow_in.base = 0.8
    u.spread_in.base = 0.0
    u.chirp_in.base = 0.0
    u.decay_in.base = decay
    u.density_in.base = 2.0
    n = int(8.0*SR/BLOCK)
    y = np.zeros(n*BLOCK)
    for b in range(n):
        u.render(BLOCK)
        y[b*BLOCK:(b+1)*BLOCK] = u.out.array(BLOCK)
    win = int(0.002*SR)
    env = np.convolve(np.abs(y), np.ones(win)/win, mode='valid')
    q = int(0.05*SR)
    guard = int(0.01*SR)
    i = q + guard
    while i < len(env) - int(1.5*SR):
        if env[i] > 0.04 and env[i-q-guard:i-guard].max() < 0.008:
            seg = env[i:i+int(1.4*SR)]
            below = np.nonzero(seg < 0.002)[0]
            return below[0]/SR if len(below) else 1.4
        i += 1
    return None
_bshort = _ring_len(0.0)
_blong = _ring_len(1.0)
def _spawn_cv_gulp(gulp):
    u = sc.BubblesUnit(SR)
    u.flow_in.base = 0.8
    u.spread_in.base = 0.0
    u.regular_in.base = 1.0
    u.gulp_in.base = gulp
    u.decay_in.base = 0.3
    u.density_in.base = 8.0
    spawns = []
    prev = u._amp.copy()
    for b in range(int(10.0*SR/BLOCK)):
        u.render(BLOCK)
        if (u._amp > prev + 1e-6).any():
            spawns.append(b)
        prev = u._amp.copy()
    iv = np.diff(spawns)
    return np.std(iv) / np.mean(iv)

check('gulp leaves the timing alone', _spawn_cv_gulp(0.9) < 0.15)

check('bubbles decay: dry drip to cave',
      _bshort is not None and _blong is not None
      and _blong > 6.0 * _bshort,
      f'{(_bshort or 0)*1000:.0f} -> {(_blong or 0)*1000:.0f} ms')

# ---------------------------------------------------------- brass mute
def _blow_muted(mute, wah):
    u = sc.BrassUnit(SR)
    u.frequency_in.base = 110.0
    u.lip_in.base = 0.3
    u.pressure_in.base = 0.55
    u.mute_in.base = mute
    u.wah_in.base = wah
    n = int(3.0*SR/BLOCK)
    y = np.zeros(n*BLOCK)
    for b in range(n):
        u.render(BLOCK)
        y[b*BLOCK:(b+1)*BLOCK] = u.out.array(BLOCK)
    return y[SR:]

def _dom_brass(y):
    m = np.abs(np.fft.rfft(y * np.hanning(len(y))))**2
    f = np.fft.rfftfreq(len(y), 1.0/SR)
    band = (f > 40) & (f < 2000)
    return f[band][np.argmax(m[band])]

def _cent_brass(y):
    m = np.abs(np.fft.rfft(y))**2
    f = np.fft.rfftfreq(len(y), 1.0/SR)
    return (m*f).sum()/m.sum()

bm_dom = _dom_brass(_blow_muted(0.85, 0.5))
check('brass speaks harmonics through the mute',
      abs(bm_dom - round(bm_dom/110.0)*110.0) < 15 and bm_dom > 200,
      f'locks at {bm_dom:.0f} Hz')
def _tilt_brass(y):
    # the vowel's band, excluding the constant 2.6 kHz buzz bed
    m = np.abs(np.fft.rfft(y))**2
    f = np.fft.rfftfreq(len(y), 1.0/SR)
    return (m[(f > 1000) & (f < 2100)].sum()
            / m[(f > 300) & (f < 800)].sum())
bw_dark = _tilt_brass(_blow_muted(0.85, 0.05))
bw_bright = _tilt_brass(_blow_muted(0.85, 0.95))
check('brass wah moves the vowel over the bright bed',
      bw_bright > 2.5 * bw_dark,
      f'tilt {bw_dark:.2f} -> {bw_bright:.2f}')
bopen = _blow_muted(0.0, 0.5)
check('brass mute 0 is the open horn, bounded',
      np.isfinite(bopen).all() and np.max(np.abs(bopen)) < 4.0
      and _dom_brass(bopen) > 200)

# --------------------------------------------------------------- spin~
# A settling disc. Everything rests on one law -- the contact point races
# round the rim at a rate going as one over the square root of the tilt --
# so most of these check that law arriving at the outlets, plus the three
# things that make it a disc rather than a sweep: a continuous contact
# whose LOAD is what varies, two rotations moving opposite ways, and a
# flop that is the grinding gone sharp rather than any kind of blow.
def run_spin(seconds=5.0, spin=1.0, hold=0.02, **kw):
    """Defaults to a cleanly-cast coin.

    Most of the laws below describe a disc that ROLLS -- a monotone
    settle, a contact that never breaks, a held gesture holding a steady
    rate. A badly cast one does none of that by design, so the rolling
    laws are measured where they apply and the cast has its own checks.
    """
    kw.setdefault('twist', 1.0)
    u = sc.SpinUnit(SR)
    for name, value in kw.items():
        getattr(u, name + '_in').base = value
    s = sc.Signal()
    u.spin_in.sources.append(s)
    n = int(seconds * SR / BLOCK)
    got = {k: np.zeros(n * BLOCK)
           for k in ('out', 'grind', 'landing', 'rate', 'face')}
    for b in range(n):
        s.data[:BLOCK] = spin if (b * BLOCK / SR) < hold else 0.0
        s.constant = False
        u.render(BLOCK)
        for k in got:
            got[k][b*BLOCK:(b+1)*BLOCK] = getattr(u, k).array(BLOCK)
    got['unit'] = u
    return got


def _spin_rate(radius, tilt):
    """The precession rate the unit is built on, in Hz."""
    return np.sqrt(4.0 * 9.80665 / (radius * tilt)) / (2.0 * np.pi)


def _spin_blows(sig):
    """How many discrete events are in an impact outlet."""
    return int((np.diff((sig != 0.0).astype(int)) > 0).sum())


def _spin_brightness(seg):
    """Where a segment's energy sits: high over total."""
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1.0 / SR)
    total = mag.sum()
    return mag[f > 4000].sum() / total if total > 0 else 0.0


def _spin_cycles(seg):
    """Rate of a bipolar control signal, in Hz, by zero crossings."""
    flips = (np.diff(np.signbit(seg).astype(int)) != 0).sum()
    return flips / (len(seg) / SR) / 2.0


sg = run_spin()
slive = sg['rate'][sg['rate'] > 0]
check('spin sounds', np.max(np.abs(sg['out'])) > 0.01,
      f'peak={np.max(np.abs(sg["out"])):.3f}')
check('spin starts at the full-lean precession rate',
      abs(slive[0] - _spin_rate(0.012, 0.5)) < 0.1,
      f'{slive[0]:.2f} Hz, law says {_spin_rate(0.012, 0.5):.2f}')
SPIN_FLAT = 0.5 * 10.0 ** -(0.7 + 2.3 * 0.7)
check('spin ends at the rate its polish allows',
      abs(slive[-1] - _spin_rate(0.012, SPIN_FLAT)) < 1.0,
      f'{slive[-1]:.2f} Hz, law says {_spin_rate(0.012, SPIN_FLAT):.2f}')
check('spin accelerates, never the other way',
      np.all(np.diff(slive[::64]) >= -1e-9))
s_warble = run_spin(twist=0.0, seconds=4.0)['rate']
s_warble = s_warble[s_warble > 0]
s_smooth = np.convolve(s_warble, np.ones(2048)/2048, mode='valid')
check('a badly cast disc warbles in pitch but still trends up',
      np.any(np.diff(s_warble[::64]) < -1e-9)
      and s_smooth[-1] > s_smooth[0],
      f'{s_smooth[0]:.1f} -> {s_smooth[-1]:.1f} Hz through the warble')
check('spin settles in its settle time', abs(len(slive)/SR - 3.0) < 0.05,
      f'{len(slive)/SR:.2f} s of a 3 s settle')

# Size is the rate scale: rate goes as one over the square root of the
# radius, so a tenfold disc turns at a third the speed.
sbig = run_spin(size=0.12, seconds=4.0)
check('spin size scales every rate as one over root radius',
      abs(sbig['rate'][sbig['rate'] > 0][0] - _spin_rate(0.12, 0.5)) < 0.1,
      f'{sbig["rate"][sbig["rate"]>0][0]:.2f} Hz at 120 mm '
      f'vs {slive[0]:.2f} at 12 mm')

# The contact is continuous and the load is what varies, so the sound is
# one voice modulated by the rotation. That modulation is the pitch:
# the grinding's own spectral peak has to sit on the rate outlet.
sp = run_spin(wobble=0.4, scrape=0.3, rush=0.0, seconds=5.0)
splive = np.nonzero(sp['rate'] > 0)[0]
s_tracked = []
for _frac in (0.1, 0.5, 0.9):
    _a = splive[0] + int((splive[-1] - splive[0]) * _frac)
    _seg = sp['grind'][_a:_a+8192]
    _mag = np.abs(np.fft.rfft(_seg * np.hanning(len(_seg))))
    _f = np.fft.rfftfreq(len(_seg), 1.0/SR)
    _band = (_f > 5) & (_f < 2000)
    s_tracked.append((sp['rate'][_a], _f[_band][np.argmax(_mag[_band])]))
check('spin\'s continuous voice is pitched at the precession rate',
      all(abs(pk - rt) < 0.25 * rt for rt, pk in s_tracked),
      ' '.join(f'{rt:.0f}->{pk:.0f}Hz' for rt, pk in s_tracked))

# Everything scales as the square root of the tilt, so a settling disc
# thins as it quickens. This is the half a bouncing model cannot reach:
# bounces only ever get faster, never smaller.
squiet_g = run_spin(scrape=0.0)
sspan = np.nonzero(squiet_g['rate'] > 0)[0]
seighths = np.array_split(squiet_g['out'][sspan[0]:sspan[-1]], 8)
speaks = [np.abs(e).max() for e in seighths]
check('spin thins as the rattle rises', speaks[-1] < 0.25 * speaks[0],
      f'peak {speaks[0]:.3f} -> {speaks[-1]:.3f} across the tail')
check('spin thinning is monotone-ish',
      sum(speaks[i+1] < speaks[i] for i in range(7)) >= 6,
      f'{[f"{v:.3f}" for v in speaks]}')

# Two rotations, moving opposite ways. The face turns at the tilt times
# the precession rate, so as the rattle accelerates the face SLOWS -- the
# counter-motion you can watch on a spinning disc, and the reason 'face'
# is worth an outlet: nothing else in the rack decelerates.
sfg = run_spin(wobble=0.8, scrape=0.3, seconds=5.0)
sflive = np.nonzero(sfg['rate'] > 0)[0]
sf_face = sfg['face'][sflive[0]:sflive[-1]]
sf_rate = sfg['rate'][sflive[0]:sflive[-1]]
_fifth = len(sf_face) // 5
s_face_early = _spin_cycles(sf_face[:_fifth])
s_face_late = _spin_cycles(sf_face[-_fifth:])
s_rate_early = sf_rate[:_fifth].mean()
s_rate_late = sf_rate[-_fifth:].mean()
check('spin rattle accelerates while its face slows',
      s_rate_late > 4.0 * s_rate_early and s_face_late < 0.5 * s_face_early,
      f'precession {s_rate_early:.0f}->{s_rate_late:.0f} Hz while the face '
      f'turns {s_face_early:.2f}->{s_face_late:.2f} Hz')
check('spin face turns at the tilt times the precession rate',
      abs(s_face_early - s_rate_early * 0.5) < 0.25 * s_rate_early * 0.5,
      f'{s_face_early:.2f} Hz measured, {s_rate_early*0.5:.2f} predicted')
check('spin face stays a usable control', -1.05 < sf_face.min()
      and sf_face.max() < 1.05,
      f'{sf_face.min():.2f} .. {sf_face.max():.2f}')

# Tested against real coins: a settling disc hardly ever leaves the
# table. So there is exactly ONE impact in the life of a spin -- the face
# landing flat at the end -- however violently it wobbles on the way.
s_blow_counts = [(w, _spin_blows(run_spin(wobble=w, scrape=0.3)['landing']))
                 for w in (0.0, 0.4, 0.7, 1.0)]
check('a rolling disc never leaves the table, however untrue the coin',
      all(n == 1 for w, n in s_blow_counts),
      f'{[(w, n) for w, n in s_blow_counts]} -- one landing apiece')

# The flop of an off-kilter coin is the grinding gone sharp: a loaded
# contact engages more surface and stiffens, so deeper wobble makes the
# grind both LOUDER and BRIGHTER. Both halves matter -- louder alone
# would be a swell, and it is the brightening that reads as a flop.
def _spin_grind(wob, seconds=4.0):
    g = run_spin(wobble=wob, scrape=0.6, hardness=0.9, seconds=seconds)
    live = np.nonzero(g['rate'] > 0)[0]
    return g['grind'][live[0]:live[-1]]


s_true, s_kilter = _spin_grind(0.15), _spin_grind(1.0)
check('spin wobble drives the grind harder',
      np.sqrt(np.mean(s_kilter**2)) > 2.0 * np.sqrt(np.mean(s_true**2)),
      f'rms {np.sqrt(np.mean(s_true**2)):.4f} true -> '
      f'{np.sqrt(np.mean(s_kilter**2)):.4f} off-kilter')
# The sharpening lives IN the cycle: the coin bites while its heavy side
# is down and merely rolls while it is up, so what deep wobble buys is
# CONTRAST between the loud part of a turn and the quiet part -- not a
# uniformly brighter sound.
def _spin_contrast(wob):
    g = run_spin(wobble=wob, scrape=0.6, hardness=0.9, seconds=4.0)
    live = np.nonzero(g['rate'] > 0)[0]
    seg = g['grind'][live[0]:live[-1]]
    w = int(0.01 * SR)
    wins = [seg[a:a+w] for a in range(0, len(seg) - w, w)]
    lvl = np.array([np.sqrt(np.mean(x**2)) for x in wins])
    order = np.argsort(lvl)
    fifth = max(1, len(order) // 5)
    loud = np.concatenate([wins[i] for i in order[-fifth:]])
    quiet = np.concatenate([wins[i] for i in order[:fifth]])
    return _spin_brightness(loud) / max(_spin_brightness(quiet), 1e-9)


s_c_true, s_c_kilter = _spin_contrast(0.15), _spin_contrast(1.0)
check('spin is sharper where it presses hardest',
      s_c_kilter > 1.3 and s_c_kilter > 1.15 * s_c_true,
      f'loud-to-quiet brightness {s_c_true:.2f}x true -> '
      f'{s_c_kilter:.2f}x off-kilter')

# 'hardness' is how sharply the contact answers that load. Soft leans
# into it, hard cuts -- same load, different edge.
def _spin_hard(h):
    g = run_spin(wobble=0.9, scrape=0.6, hardness=h, seconds=4.0)
    live = np.nonzero(g['rate'] > 0)[0]
    return _spin_brightness(g['grind'][live[0]:live[-1]])


check('spin hardness sets how sharply a flop cuts',
      _spin_hard(0.95) > 1.3 * _spin_hard(0.05),
      f'high-frequency share {_spin_hard(0.05):.3f} soft -> '
      f'{_spin_hard(0.95):.3f} hard')

# And the sharpening comes and goes on the SLOW turn, not the fast one:
# the coin bites while its heavy side is down. Measured as the depth of
# the grind envelope's swing at the face rate.
sbg = run_spin(wobble=1.0, scrape=0.6, hardness=0.9, seconds=5.0)
sblive = np.nonzero(sbg['rate'] > 0)[0]
sbseg = np.abs(sbg['grind'][sblive[0]:sblive[-1]])
S_WIN = int(0.02 * SR)
s_env = np.array([sbseg[a:a+S_WIN].mean()
                  for a in range(0, len(sbseg) - S_WIN, S_WIN)])
s_env = s_env[len(s_env)//5:]
check('spin flop comes and goes on the slow turn',
      s_env.max() > 2.0 * np.median(s_env),
      f'grind envelope peaks {s_env.max()/max(np.median(s_env),1e-12):.1f}x '
      f'its median')

# A badly spun coin does not trace a circle: its contact runs an eccentric
# orbit, racing through the tight side and dawdling through the wide one,
# so a pulse symmetric in ANGLE lands lopsided in TIME. The flops get
# harder and less even as the coin goes off-kilter.
def _spin_pulsing(wob):
    g = run_spin(wobble=wob, scrape=0.7, hardness=0.85, seconds=4.0)
    live = np.nonzero(g['rate'] > 0)[0]
    seg = g['grind'][live[0]:live[-1]]
    w = int(0.01 * SR)
    lvl = np.array([np.sqrt(np.mean(seg[a:a+w]**2))
                    for a in range(0, len(seg) - w, w)])
    o = np.argsort(lvl)
    fifth = max(1, len(o) // 5)
    return lvl[o[-fifth:]].mean() / max(lvl[o[:fifth]].mean(), 1e-12)


s_pulse = [(w, _spin_pulsing(w)) for w in (0.0, 0.5, 1.0)]
check('spin flops harder as it goes off-kilter',
      s_pulse[0][1] < s_pulse[1][1] < s_pulse[2][1]
      and s_pulse[2][1] > 2.5 * s_pulse[0][1],
      ' -> '.join(f'{v:.1f}x' for _, v in s_pulse))

# A settling disc runs into a real singularity, and no amount of
# clamping proves that no corner of the parameter space steps over it.
# What must be true is that stepping over it is survivable: one bad block
# is a dropped block, not a voice that is silent for the rest of the
# session. Both models are checked, because the guard has to be in both.
for _model in (0, 1):
    _u = sc.SpinUnit(SR)
    _u.model = _model
    _sig = sc.Signal()
    _u.spin_in.sources.append(_sig)
    for _b in range(20):
        _sig.data[:BLOCK] = 1.0 if _b < 2 else 0.0
        _sig.constant = False
        _u.render(BLOCK)
    _u._d_q2 = float('nan')
    _u._tilt = float('inf')
    _sig.data[:BLOCK] = 0.0
    _sig.constant = False
    _u.render(BLOCK)
    _clean = all(np.isfinite(getattr(_u, _n).array(BLOCK)).all()
                 for _n in ('out', 'grind', 'landing', 'rate', 'face'))
    _sounded = 0.0
    _finite = True
    for _b in range(80):
        _sig.data[:BLOCK] = 1.0 if _b < 2 else 0.0
        _sig.constant = False
        _u.render(BLOCK)
        _y = _u.out.array(BLOCK)
        _finite = _finite and np.isfinite(_y).all()
        _sounded = max(_sounded, float(np.abs(_y).max()))
    check(f'spin survives a poisoned state ({sc.SpinUnit.MODELS[_model]})',
          _clean and _finite and _sounded > 1e-5,
          f'block clean={_clean}, recovers and sounds again '
          f'(peak {_sounded:.4f})')

# --- the cast, which is the continuum this node is really about ------
# A coin spun true on its edge rolls: the contact drifts round the rim,
# the lean falls smoothly, nothing ever leaves the table. A coin merely
# pushed over never rolls at all -- the lean swings past level every
# cycle, the face slaps, and it rattles to a stop. 'twist' is how much
# spin was in the fall, and everything between those is a real coin.
# Counted per second of sounding, not per settle: a bad cast also has a
# much SHORTER tail, so the total says less than the rate does.
def _spin_contact_rate(tw):
    g = run_spin(twist=tw, seconds=5.0)
    sounding = (g['rate'] > 0).sum() / SR
    return _spin_blows(g['landing']) / max(sounding, 1e-9), sounding


s_cast = [(t,) + _spin_contact_rate(t) for t in (1.0, 0.6, 0.3, 0.0)]
check('twist runs from a coin that rolls to one that only rattles',
      s_cast[0][1] < 1.0 and s_cast[-1][1] > 10.0
      and all(a[1] <= b[1] for a, b in zip(s_cast, s_cast[1:])),
      ' -> '.join(f'twist {t}: {r:.1f}/s' for t, r, _ in s_cast))
check('a bad cast also has a shorter tail',
      s_cast[-1][2] < 0.35 * s_cast[0][2],
      ' -> '.join(f'{d:.2f} s' for _, _, d in s_cast))

# The lean oscillating is what a bad cast IS, and because the rate goes
# as one over the square root of the lean, that oscillation is heard as
# a warble in pitch -- not merely as unevenness in loudness.
def _spin_warble(tw):
    r = run_spin(twist=tw, seconds=4.0)['rate']
    r = r[r > 0]
    a, b = int(len(r)*0.05), int(len(r)*0.30)
    seg = r[a:b]
    trend = np.convolve(seg, np.ones(1024)/1024, mode='valid')
    ripple = seg[:len(trend)] - trend
    return np.sqrt(np.mean(ripple**2)) / seg.mean()


s_w_clean, s_w_cast = _spin_warble(1.0), _spin_warble(0.2)
check('a bad cast warbles the pitch, a clean one does not',
      s_w_clean < 0.02 and s_w_cast > 5.0 * s_w_clean,
      f'rate ripple {s_w_clean*100:.2f}% clean -> {s_w_cast*100:.1f}% cast')

# And a disc settles into rolling only because its SPIN holds it there,
# so with no twist at all it never settles -- it flops until it stops.
s_late_clean = _spin_contact_rate(0.55)[0]
s_late_none = _spin_contact_rate(0.0)[0]
check('without spin there is nothing to settle into',
      s_late_none > 5.0 * max(s_late_clean, 0.2),
      f'{s_late_clean:.1f} contacts/s at twist 0.55, '
      f'{s_late_none:.1f}/s at 0')

# The excitation itself has to change KIND, not just level. A contact is
# a stream of micro-impacts; a rolling rim meets thousands a second and
# they fuse into a continuous sound, a face coming down delivers one. So
# the grains thin out and grow as twist falls, at constant power -- which
# shows up as a crest factor (peak over rms) that climbs steeply while
# the level does not.
def _spin_crest(tw):
    """Impulsiveness by kurtosis, not by crest.

    Crest stopped separating these once the grains were given the
    power-law sizes a self-affine surface really produces: the tail sets
    the peak whatever the density is. Kurtosis still reads the thing
    that matters -- whether the energy arrives in many small pieces or
    a few large ones.
    """
    g = run_spin(twist=tw, seconds=5.0)
    live = np.nonzero(g['rate'] > 0)[0]
    seg = g['grind'][live[0]:live[-1]]
    r = np.sqrt(np.mean(seg**2))
    kurt = np.mean(seg**4) / max(np.mean(seg**2)**2, 1e-18)
    return kurt, r


s_crest = [(t,) + _spin_crest(t) for t in (1.0, 0.5, 0.0)]
# The ends, not three points in a strict order: with power-law grains
# the middle sits within seed noise of either side, so ordering it would
# be testing the draw rather than the law.
check('the contact goes from a hiss to separate hits',
      s_crest[2][1] > 3.0 * s_crest[0][1],
      ' -> '.join(f'kurtosis {c:.0f}' for _, c, _ in s_crest))
check('and it does so at roughly constant power, not by getting louder',
      s_crest[2][2] < s_crest[0][2],
      ' -> '.join(f'rms {r:.4f}' for _, _, r in s_crest))

# But the lurch must not RETUNE it. The orbit's rate factor is normalized
# over a turn, so however lopsided the motion becomes the mean precession
# is untouched -- otherwise wobble would double as a pitch control, and
# every law above that rests on the rate would shift under it.
s_tunings = []
for _w in (0.0, 0.35, 0.7, 1.0):
    _live = run_spin(wobble=_w, seconds=4.0)['rate']
    _live = _live[_live > 0]
    s_tunings.append((_live[0], _live[-1], len(_live)))
check('spin wobble does not retune the disc',
      all(abs(t[0] - s_tunings[0][0]) < 1e-6
          and abs(t[1] - s_tunings[0][1]) < 1e-6
          and t[2] == s_tunings[0][2] for t in s_tunings),
      f'{s_tunings[0][0]:.3f} Hz start and {s_tunings[0][1]:.3f} Hz end '
      f'at every wobble')

# A node out of the box has to put something on every outlet it has.
sdef = run_spin(seconds=6.0, twist=sc.SpinUnit(SR).twist_in.base)
check("spin's default settings reach every outlet",
      _spin_blows(sdef['landing']) >= 1
      and np.abs(sdef['grind']).max() > 1e-3
      and sdef['rate'].max() > 1.0
      and np.abs(sdef['face']).max() > 0.1,
      f'landing fires {_spin_blows(sdef["landing"])}x, '
      f'grind peak {np.abs(sdef["grind"]).max():.3f}, '
      f'rate to {sdef["rate"].max():.0f} Hz, '
      f'face swing {np.abs(sdef["face"]).max():.2f}')

# The outlets are a split of one sound, not three renderings of it.
ssum = run_spin(wobble=0.9, scrape=0.5)
check('spin out is exactly grind plus landing',
      np.abs(ssum['out'] - (ssum['grind'] + ssum['landing'])).max() < 1e-6,
      f'{np.abs(ssum["out"] - (ssum["grind"]+ssum["landing"])).max():.2e}')
sloud = run_spin(wobble=0.9, scrape=0.5, level=1.7)
# Compared by rms, not by peak: the grains are power-law sized, so a
# peak is whichever rare large one a given run happened to draw.
_q_loud = np.sqrt(np.mean(sloud['out']**2))
_q_ref = np.sqrt(np.mean(ssum['out']**2))
check('spin outlets stay split when level is not unity',
      np.abs(sloud['out'] - (sloud['grind'] + sloud['landing'])).max() < 1e-6
      and _q_loud > 1.3 * _q_ref,
      f'{_q_loud/_q_ref:.2f}x louder by rms')

# 'rush' is which loss dominates. Measured as the fraction of the tail
# spent below the sweep's geometric midpoint: an even glide sits at a
# half, the viscous-air law spends the whole tail below it and then does
# all of the climbing at once.
def _spin_lean(rush):
    live = run_spin(rush=rush, seconds=4.0)['rate']
    live = live[live > 0]
    return (live < np.sqrt(live[0] * live[-1])).sum() / len(live)


s_even, s_default, s_cliff = _spin_lean(0.0), _spin_lean(0.4), _spin_lean(1.0)
check('spin rush 0 spreads the climb evenly across the tail',
      abs(s_even - 0.5) < 0.08,
      f'{s_even*100:.0f}% of the tail below the midpoint')
check('spin rush leans the climb towards the end',
      s_even < s_default < s_cliff and s_cliff > 0.97,
      f'{s_even*100:.0f}% -> {s_default*100:.0f}% -> {s_cliff*100:.0f}%')


# Polish is where flat is, so it sets both the top of the sweep and how
# much lean is left to land with: a rough surface stops a disc with a
# clack, a polished one lets it whisper away to nothing.
def _spin_landing(polish):
    g = run_spin(polish=polish, scrape=0.0, seconds=4.0)
    land = np.nonzero(g['rate'] > 0)[0][-1]
    return g['rate'][land], np.abs(g['landing'][land:land+400]).max()


s_rough_hz, s_rough_slap = _spin_landing(0.0)
s_glass_hz, s_glass_slap = _spin_landing(1.0)
check('spin polish sets how high the whir climbs',
      s_glass_hz > 10.0 * s_rough_hz,
      f'{s_rough_hz:.0f} Hz rough -> {s_glass_hz:.0f} Hz polished')
check('spin lands with a clack when rough, a whisper when polished',
      s_rough_slap > 4.0 * s_glass_slap,
      f'final landing {s_rough_slap:.4f} rough vs {s_glass_slap:.4f} '
      f'polished, {s_rough_slap/max(s_glass_slap,1e-9):.1f}x')

# The gesture only ever adds energy, so a smaller one leans the disc less:
# it starts higher and has less to fall through. The tail is bought by the
# movement, which is the whole reason for patching one in.
s_starts, s_tails = [], []
for _level in (1.0, 0.5, 0.25):
    live = run_spin(spin=_level, seconds=4.0)['rate']
    live = live[live > 0]
    s_starts.append(live[0])
    s_tails.append(len(live) / SR)
check('spin gesture size sets the pitch it starts from',
      s_starts[0] < s_starts[1] < s_starts[2],
      f'{[f"{v:.0f} Hz" for v in s_starts]}')
check('spin gesture size buys the length of the tail',
      s_tails[0] > s_tails[1] > s_tails[2],
      f'{[f"{v:.2f} s" for v in s_tails]}')

# Held movement holds the sound: a disc cannot settle while a hand keeps
# leaning it over, and the tail begins where the holding stops.
shu = sc.SpinUnit(SR)
shu.twist_in.base = 1.0
shs = sc.Signal()
shu.spin_in.sources.append(shs)
_n = int(8.0*SR/BLOCK)
sheld = np.zeros(_n * BLOCK)
for b in range(_n):
    shs.data[:BLOCK] = 0.8 if (b*BLOCK/SR) < 3.0 else 0.0
    shs.constant = False
    shu.render(BLOCK)
    sheld[b*BLOCK:(b+1)*BLOCK] = shu.rate.array(BLOCK)
s_during = sheld[SR//2:int(2.9*SR)]
s_after = np.nonzero(sheld[int(3.0*SR):] > 0)[0]
check('spin holds its sound while the gesture holds',
      s_during.max() - s_during.min() < 0.01,
      f'{s_during.min():.2f}-{s_during.max():.2f} Hz over 2.4 s')
check('spin tail starts when the gesture stops', len(s_after)/SR > 2.0,
      f'{len(s_after)/SR:.2f} s of tail after release')

# A landed disc set down again starts from the lean it is given, not from
# the top of the sweep it just finished.
sru = sc.SpinUnit(SR)
sru.twist_in.base = 1.0
sru.settle_in.base = 0.4
srs = sc.Signal()
sru.spin_in.sources.append(srs)
_n = int(2.0*SR/BLOCK)
sagain = np.zeros(_n * BLOCK)
for b in range(_n):
    _t = b * BLOCK / SR
    srs.data[:BLOCK] = 1.0 if (_t < 0.02 or 1.0 <= _t < 1.02) else 0.0
    srs.constant = False
    sru.render(BLOCK)
    sagain[b*BLOCK:(b+1)*BLOCK] = sru.rate.array(BLOCK)
s_relaunch = sagain[int(1.0*SR):int(1.05*SR)]
check('spin relaunches from the lean it is given',
      abs(s_relaunch[s_relaunch > 0][0] - _spin_rate(0.012, 0.5)) < 0.5,
      f'{s_relaunch[s_relaunch>0][0]:.2f} Hz')

# Going flat must not cut the grinding off between two samples: a noise
# silenced in one step is a click, so the contact loses its grip over a
# couple of milliseconds. Checked as a level that decays rather than ends.
sgg = run_spin(scrape=1.0, wobble=0.0, polish=1.0, seconds=4.0)
s_land = np.nonzero(sgg['rate'] > 0)[0][-1]
S_WIN = int(0.0005*SR)


def _spin_win_rms(offset):
    seg = sgg['grind'][s_land + offset*S_WIN:s_land + (offset+1)*S_WIN]
    return np.sqrt(np.mean(seg**2))


check('spin fades its grinding out at landing rather than cutting it',
      _spin_win_rms(14) < 0.3 * _spin_win_rms(0)
      and _spin_win_rms(1) > 0.2 * _spin_win_rms(0),
      f'rms {_spin_win_rms(0):.4f} -> {_spin_win_rms(1):.4f} -> '
      f'{_spin_win_rms(14):.4f} over 7 ms')

# The quiet path: a landed disc with nothing patched costs nothing. The
# face is the exception -- it holds where it stopped, because a control
# that jumped to zero on landing would step whatever it drives.
squ = sc.SpinUnit(SR)
squ.settle_in.base = 0.5
sqs = sc.Signal()
squ.spin_in.sources.append(sqs)
for b in range(int(0.1*SR/BLOCK)):
    sqs.data[:BLOCK] = 1.0
    sqs.constant = False
    squ.render(BLOCK)
sqs.constant = True
sqs.value = 0.0
for b in range(int(3.0*SR/BLOCK)):
    squ.render(BLOCK)
squ.render(BLOCK)
check('spin goes quiet-constant once landed',
      squ.out.constant and squ.grind.constant and squ.landing.constant
      and squ.rate.constant and squ.rate.value == 0.0)
check('spin face holds where it stopped rather than stepping to zero',
      squ.face.constant and abs(squ.face.value) <= 1.05,
      f'holding {squ.face.value:+.3f}')

# Bounded everywhere it can be driven.
s_bad = []
for _size in (0.004, 0.6):
    for _rush in (0.0, 1.0):
        for _settle in (0.05, 60.0):
            for _polish in (0.0, 1.0):
                for _wob in (0.0, 1.0):
                    _g = run_spin(seconds=1.5, size=_size, rush=_rush,
                                  settle=_settle, polish=_polish,
                                  hardness=1.0, wobble=_wob, scrape=1.0)
                    if (not np.isfinite(_g['out']).all()
                            # Eight, not four: the grains are power-law
                            # sized on purpose, so a rare large one is
                            # the design. Measured across sixteen seeds
                            # the worst corner runs 2.0 to 3.8, and a
                            # tighter bound tests which seed came up.
                            or np.abs(_g['out']).max() > 8.0):
                        s_bad.append((_size, _rush, _settle, _polish, _wob))
check('spin is bounded at every corner', not s_bad, f'{s_bad}')

# And into a resonator, which is what it is for. A coin's modes are up at
# two or three kilohertz, where a bank rings small -- hence the drive.
PLATE8 = [(1.0, 1.0, 1.0), (1.73, 0.85, 0.9), (2.33, 0.7, 0.75),
          (3.91, 0.55, 0.55), (4.06, 0.5, 0.5), (5.94, 0.35, 0.35),
          (8.72, 0.22, 0.25), (11.75, 0.15, 0.18)]
scoin = sc.SpinUnit(SR)
scs = sc.Signal()
scoin.spin_in.sources.append(scs)
mcoin = sc.ModalUnit(SR)
mcoin.set_modes(PLATE8)
mcoin.frequency_in.base = 2600.0
mcoin.decay_in.base = 0.25
mcoin.sensitivity_in.base = 2.0
mcoin.excite_in.sources.append(scoin.out)
_n = int(5.0*SR/BLOCK)
scy = np.zeros(_n * BLOCK)
for b in range(_n):
    scs.data[:BLOCK] = 1.0 if (b*BLOCK/SR) < 0.02 else 0.0
    scs.constant = False
    scoin.render(BLOCK)
    mcoin.render(BLOCK)
    scy[b*BLOCK:(b+1)*BLOCK] = mcoin.out.array(BLOCK)
# Angles must stay wrapped however long a gesture is held. A contact can
# sweep more than a full turn in one control step, so wrapping by
# subtracting a single turn cannot keep up -- the angle grows without
# bound and the cosine of it loses its precision long before it reaches
# anything a NaN check would notice. It shows up as the 'face' outlet
# degenerating from a cosine into a stepped square, with nothing else
# obviously wrong.
_hu = sc.SpinUnit(SR)
_hu.model = 0
_hu.size_in.base = 0.044
_hu.settle_in.base = 3.0
_hu.rush_in.base = 0.0
_hu.polish_in.base = 1.0
_hu.hardness_in.base = 1.0
_hs = sc.Signal()
_hu.spin_in.sources.append(_hs)
_face_clean = True
for _b in range(int(30.0 * SR / BLOCK)):
    _hs.data[:BLOCK] = 1.0
    _hs.constant = False
    _hu.render(BLOCK)
    _f = _hu.face.array(BLOCK)
    if not np.isfinite(_f).all() or np.abs(_f).max() > 1.0001:
        _face_clean = False
        break
check('spin keeps its angles wrapped under a long held gesture',
      _face_clean and abs(_hu._d_q3) < 7.0 and abs(_hu._d_q1) < 7.0,
      f'after 30 s held: |q3|={abs(_hu._d_q3):.3f} |q1|={abs(_hu._d_q1):.3f} '
      f'(a turn is 6.283), face clean={_face_clean}')

# A coin always stops. Whatever surface it is on and however it was
# cast, the sound has to end -- and end near the settle time it was
# asked for. Landing once waited on the ROLL falling away as well as the
# bouncing, which a rough surface never satisfies: its flat limit is a
# large lean, and the steady roll at that lean is still fast, so the
# coin arrived at its limit already too fast to be judged stopped and
# rolled there for ever.
_forever = []
for _pol in (0.0, 0.5, 1.0):
    for _tw in (0.0, 0.3, 0.7, 1.0):
        _g = run_spin(seconds=9.0, twist=_tw, polish=_pol, settle=3.0,
                      rush=1.0, size=0.028, wobble=0.0, scrape=0.55,
                      hardness=1.0)
        _nz = np.nonzero(np.abs(_g['out']) > 1e-5)[0]
        _dur = (_nz[-1] - _nz[0]) / SR if len(_nz) else 0.0
        # still sounding at the very end means it never stopped
        if _dur > 6.0 or np.abs(_g['out'][-SR//4:]).max() > 1e-5:
            _forever.append((_pol, _tw, round(_dur, 2)))
check('spin always stops, on any surface and from any cast',
      not _forever, f'{_forever}' if _forever else
      'every polish and twist ends inside a 3 s settle')

# The node must never stop answering the hand. 'spin' works by change,
# and the change was accumulated even while the coin lay still -- so
# releasing a gesture AFTER the coin had already stopped drove the
# accumulator to minus one, and the next throw, however hard, only
# brought it back to zero. The node went silent and stayed silent.
_again = sc.SpinUnit(SR)
_again.model = 0
_again.size_in.base = 0.028
_again.settle_in.base = 3.0
_again.rush_in.base = 1.0
_again.twist_in.base = 0.4
_again.polish_in.base = 0.7
_ags = sc.Signal()
_again.spin_in.sources.append(_ags)
_throws = []
for _k in range(6):
    _peak = 0.0
    for _b in range(int(6.0 * SR / BLOCK)):
        # thrown, left alone long enough to stop, then released
        _ags.data[:BLOCK] = 1.0 if (_b * BLOCK / SR) < 4.0 else 0.0
        _ags.constant = False
        _again.render(BLOCK)
        _peak = max(_peak, float(np.abs(_again.out.array(BLOCK)).max()))
    _throws.append(_peak)
check('spin answers every throw, however often it is released',
      all(t > 1e-4 for t in _throws),
      'peaks ' + ' '.join(f'{t:.2f}' for t in _throws))

# How the hand moves decides how steady the coin is, and that falls out
# of one asymmetry: a RISE both speeds the roll and balances it, while a
# FALL only takes roll away. So a sustained smooth rise keeps balancing
# and produces a steady spin, whereas a spike throws the coin and then
# stops -- leaving whatever wobble the cast imparted -- and a rough rise
# keeps interrupting its own balancing with little drains. Worth holding
# on to: it is what makes the control feel like throwing something.
def _spin_wobbliness(gfn, seconds=8.0):
    u = sc.SpinUnit(SR)
    u.model = 0
    u.size_in.base = 0.028
    u.settle_in.base = 3.0
    u.rush_in.base = 1.0
    u.twist_in.base = 0.5
    u.polish_in.base = 0.7
    sig = sc.Signal()
    u.spin_in.sources.append(sig)
    y = []
    for b in range(int(seconds * SR / BLOCK)):
        sig.data[:BLOCK] = float(np.clip(gfn(b * BLOCK / SR), 0.0, 1.0))
        sig.constant = False
        u.render(BLOCK)
        y.append(u.out.array(BLOCK).copy())
    y = np.concatenate(y)
    nz = np.nonzero(np.abs(y) > 1e-5)[0]
    if len(nz) < 2:
        return 0.0
    seg = np.abs(y[nz[0]:nz[-1]])
    w = int(0.01 * SR)
    env = np.array([seg[a:a+w].mean() for a in range(0, len(seg) - w, w)])
    env = env[env > 0]
    return float(env.max() / np.median(env)) if len(env) else 0.0


_w_spike = _spin_wobbliness(lambda t: 1.0 if t > 0.2 else 0.0)
_w_gentle = _spin_wobbliness(
    lambda t: min(1.0, max(0.0, (t - 0.2) / 1.2)))
check('a thrown coin wobbles, a wound-up one runs steady',
      _w_spike > 2.0 * _w_gentle,
      f'envelope swing {_w_spike:.0f}x for a spike, '
      f'{_w_gentle:.0f}x for a sustained rise')

# A throw has to have range in it: how hard you throw must change what
# you get. The cast lean once saturated, so every gesture past halfway
# cast an identical coin. Note the direction -- a bigger throw stands
# the coin up, and the rate goes as one over the root of the lean, so a
# hard throw starts SLOWER and sweeps further.
def _spin_throw(amp):
    u = sc.SpinUnit(SR)
    u.model = 0
    u.size_in.base = 0.028
    u.settle_in.base = 3.0
    u.rush_in.base = 1.0
    u.twist_in.base = 0.6
    u.polish_in.base = 0.7
    sig = sc.Signal()
    u.spin_in.sources.append(sig)
    y, r = [], []
    for b in range(int(8.0 * SR / BLOCK)):
        sig.data[:BLOCK] = amp if (b * BLOCK / SR) > 0.2 else 0.0
        sig.constant = False
        u.render(BLOCK)
        y.append(u.out.array(BLOCK).copy())
        r.append(u.rate.array(BLOCK).copy())
    y, r = np.concatenate(y), np.concatenate(r)
    live = r[r > 0]
    nz = np.nonzero(np.abs(y) > 1e-5)[0]
    return (live[0] if len(live) else 0.0,
            live.max() if len(live) else 0.0,
            (nz[-1] - nz[0]) / SR if len(nz) > 1 else 0.0)


_soft, _mid, _hard = _spin_throw(0.25), _spin_throw(0.5), _spin_throw(1.0)
check('a harder throw is a different coin, not the same one louder',
      _hard[0] < _mid[0] < _soft[0] and _hard[2] > _mid[2] > _soft[2],
      f'starts {_soft[0]:.1f}/{_mid[0]:.1f}/{_hard[0]:.1f} Hz, '
      f'lasts {_soft[2]:.2f}/{_mid[2]:.2f}/{_hard[2]:.2f} s')

# Surging again and again without letting the coin settle must not wind
# it up without limit. Excess spin is what stands a coin up, and the
# lean is capped -- so the spin is capped with it. Left uncapped, every
# surge added roll the lean had nowhere to put and the coin simply got
# faster each time.
_sg = sc.SpinUnit(SR)
_sg.model = 0
_sg.size_in.base = 0.028
_sg.settle_in.base = 3.0
_sg.rush_in.base = 1.0
_sg.twist_in.base = 0.6
_sg.polish_in.base = 0.7
_sgs = sc.Signal()
_sg.spin_in.sources.append(_sgs)
_casts, _was = [], 1.0
for _b in range(int(16.0 * SR / BLOCK)):
    _t = _b * BLOCK / SR
    _sgs.data[:BLOCK] = 1.0 if (_t % 0.6) < 0.15 else 0.0
    _sgs.constant = False
    _sg.render(BLOCK)
    if _was > 0.5 and _sg._landed < 0.5:
        _casts.append(_sg._d_u3)
    _was = _sg._landed
check('surging again and again does not wind the coin up',
      len(_casts) >= 2 and max(_casts) < 1.15 * min(_casts),
      f'roll at each re-throw spans {min(_casts):.2f} to {max(_casts):.2f}')

# Two readings of the gesture, and they have to differ: 'hold' sustains
# while the hand is held, 'throw' does not care what the level is.
def _spin_sounds(mode, gfn, seconds=10.0):
    u = sc.SpinUnit(SR)
    u.model = 0
    u.spin_mode = mode
    u.size_in.base = 0.028
    u.settle_in.base = 3.0
    u.rush_in.base = 1.0
    u.twist_in.base = 0.5
    u.polish_in.base = 0.7
    sig = sc.Signal()
    u.spin_in.sources.append(sig)
    y = []
    for b in range(int(seconds * SR / BLOCK)):
        sig.data[:BLOCK] = float(np.clip(gfn(b * BLOCK / SR), 0.0, 1.0))
        sig.constant = False
        u.render(BLOCK)
        y.append(u.out.array(BLOCK).copy())
    y = np.concatenate(y)
    nz = np.nonzero(np.abs(y) > 1e-5)[0]
    return (nz[-1] - nz[0]) / SR if len(nz) > 1 else 0.0


_held = lambda t: 1.0 if t > 0.2 else 0.0
_throw_len = _spin_sounds(0, _held)
_hold_len = _spin_sounds(1, _held)
check('hold sustains a held gesture, throw does not',
      _hold_len > 2.0 * _throw_len,
      f'{_throw_len:.2f} s in throw, {_hold_len:.2f} s in hold')

# The spin cap belongs to the COIN, not to one reading of the gesture.
# Held up in hold mode the pump feeds the roll every control step, so
# without a cap it intensifies for as long as the gesture is held --
# which is a runaway, not an instrument. Checked in both modes and at
# both ends of twist.
_runaway = []
for _mode in (0, 1):
    for _tw in (0.132, 1.0):
        _hu = sc.SpinUnit(SR)
        _hu.model = 0
        _hu.spin_mode = _mode
        _hu.size_in.base = 0.028
        _hu.settle_in.base = 3.0
        _hu.rush_in.base = 1.0
        _hu.twist_in.base = _tw
        _hu.polish_in.base = 0.7
        _hus = sc.Signal()
        _hu.spin_in.sources.append(_hus)
        _worst = 0.0
        for _b in range(int(20.0 * SR / BLOCK)):
            _hus.data[:BLOCK] = 1.0
            _hus.constant = False
            _hu.render(BLOCK)
            _worst = max(_worst, float(np.abs(_hu.out.array(BLOCK)).max()),
                         abs(_hu._d_u3) / 100.0)
        if _worst > 4.0:
            _runaway.append((sc.SpinUnit.SPIN_MODES[_mode], _tw,
                             round(_worst, 2)))
check('a held gesture does not wind the coin up without limit',
      not _runaway,
      f'{_runaway}' if _runaway else
      'twenty seconds held, in both modes, stays bounded')

# Held in hold mode, a cleanly cast coin must sit where it is held --
# not settle all the way out and start again. The drain and the pump act
# on a lagging measurement against a coin with its own dynamics, so with
# no dead zone between them they take turns overshooting and the lean
# swings over half its range while its goal stands still. A badly cast
# coin SHOULD swing, because that is nutation; a true one should not.
def _spin_hold_swing(tw, mode=1):
    u = sc.SpinUnit(SR)
    u.model = 0
    u.spin_mode = mode
    u.size_in.base = 0.028
    u.settle_in.base = 3.0
    u.rush_in.base = 1.0
    u.twist_in.base = tw
    u.polish_in.base = 0.7
    # A true rim, so what is measured is the controller and not the
    # coin's own out-of-roundness.
    u.wobble_in.base = 0.1
    sig = sc.Signal()
    u.spin_in.sources.append(sig)
    leans = []
    for b in range(int(14.0 * SR / BLOCK)):
        sig.data[:BLOCK] = 1.0
        sig.constant = False
        u.render(BLOCK)
        leans.append(np.pi/2 - u._d_q2)
    leans = np.array(leans[200:])
    return leans.max() - leans.min(), float(np.mean(leans))


_swing_true, _mean_true = _spin_hold_swing(1.0)
check('a true coin held open runs steady, and stays where it is held',
      _swing_true < 0.15 and abs(_mean_true - 0.5) < 0.1,
      f'lean swings {_swing_true:.3f} about {_mean_true:.3f}')

# (A check that a bad cast swings the lean further stood here. It is
# ill-posed: over a whole settle the lean range is dominated by the
# settle itself, and a badly cast coin lands early so most of its
# window is silence. What twist actually does is already covered, and
# better, by the contact rate and the pitch warble above.)

# The coin must never reach a state nothing can stop. It once could:
# the drain's dead zone stayed open all the way down, so at the end --
# goal on the floor, coin a fraction above it, gesture released -- the
# drain declined to act and the coin rolled at full speed for ever.
# Nothing in the interface could touch it. The dead zone now closes as
# the goal reaches the floor, where there is no pump left to fight.
_unstoppable = []
for _mode in (0, 1):
    for _rush in (0.0, 1.0):
        for _hard in (0.279, 1.0):
            for _tw in (1.0, 0.5):
                _u = sc.SpinUnit(SR)
                _u.model = 0
                _u.spin_mode = _mode
                _u.size_in.base = 0.028
                _u.settle_in.base = 3.0
                _u.rush_in.base = _rush
                _u.twist_in.base = _tw
                _u.wobble_in.base = 0.0
                _u.scrape_in.base = 0.548
                _u.hardness_in.base = _hard
                _u.polish_in.base = 1.0
                _u.level_in.base = 2.0
                _us = sc.Signal()
                _u.spin_in.sources.append(_us)
                _y = []
                for _b in range(int(20.0 * SR / BLOCK)):
                    _us.data[:BLOCK] = 1.0 if (_b * BLOCK / SR) < 6.0 else 0.0
                    _us.constant = False
                    _u.render(BLOCK)
                    _y.append(_u.out.array(BLOCK).copy())
                _y = np.concatenate(_y)
                # fourteen seconds after release it must be silent
                if np.abs(_y[-int(3.0 * SR):]).max() > 1e-5:
                    _unstoppable.append(
                        (sc.SpinUnit.SPIN_MODES[_mode], _rush, _hard, _tw))
check('a released coin always comes to rest, from every corner',
      not _unstoppable,
      f'{_unstoppable}' if _unstoppable else
      'sixteen corners, all silent well before the end')

# However fast or slow the control is MOVED to a value, the coin it
# leaves behind must be the same one. This is the axis that kept
# breaking: a feedback loop on the coin's energy was tuned four times,
# and every gain that tracked one ramp time overshot or lagged another,
# because the coin's own dynamics sit inside the loop. Hold mode now
# says what it means -- the gesture IS the lean -- and carries the disc
# there directly. Swept from a millisecond to eight seconds, because a
# hand on a slider produces every one of them.
def _spin_ramp_lean(ramp, seconds=20.0):
    u = sc.SpinUnit(SR)
    u.model = 0
    u.spin_mode = 1
    u.size_in.base = 0.028
    u.settle_in.base = 3.0
    u.rush_in.base = 0.0
    u.twist_in.base = 1.0
    u.wobble_in.base = 0.0
    u.hardness_in.base = 0.279
    u.polish_in.base = 1.0
    sig = sc.Signal()
    u.spin_in.sources.append(sig)
    leans = []
    for b in range(int(seconds * SR / BLOCK)):
        t = b * BLOCK / SR
        sig.data[:BLOCK] = float(min(1.0, t / ramp))
        sig.constant = False
        u.render(BLOCK)
        leans.append(np.pi/2 - u._d_q2)
    k = int(max(1.0, ramp * 1.5) * SR / BLOCK)
    leans = np.array(leans[k:])
    return leans.min(), leans.max()


_ramps = [(r,) + _spin_ramp_lean(r)
          for r in (0.001, 0.02, 0.1, 0.5, 1.0, 4.0, 8.0)]
_wobbly = [(r, round(hi - lo, 3)) for r, lo, hi in _ramps if hi - lo > 0.05]
check('however the control is moved, it leaves the same coin',
      not _wobbly,
      f'{_wobbly}' if _wobbly else
      'ramps from a millisecond to eight seconds all settle at the same lean')

# 'twist' must keep working in BOTH readings of the gesture. Hold mode
# once carried the coin to the balanced steady roll and damped the lean
# rate on the way there -- which is exactly what twist 1 means, so it
# overwrote the control every step and every held coin sounded
# perfectly spun whatever twist said.
def _spin_hold_twist(tw):
    u = sc.SpinUnit(SR)
    u.model = 0
    u.spin_mode = 1
    u.size_in.base = 0.028
    u.settle_in.base = 3.0
    u.rush_in.base = 0.0
    u.twist_in.base = tw
    u.wobble_in.base = 0.0
    u.hardness_in.base = 0.279
    u.polish_in.base = 1.0
    sig = sc.Signal()
    u.spin_in.sources.append(sig)
    rolls = []
    for b in range(int(14.0 * SR / BLOCK)):
        sig.data[:BLOCK] = float(min(1.0, (b * BLOCK / SR) / 0.5))
        sig.constant = False
        u.render(BLOCK)
        rolls.append(abs(u._d_u3))
    return float(np.mean(rolls[int(4.0 * SR / BLOCK):]))


_r_true, _r_half, _r_bad = (_spin_hold_twist(1.0), _spin_hold_twist(0.5),
                            _spin_hold_twist(0.132))
check('twist still shapes the coin when the gesture is held',
      _r_true > 1.5 * _r_half > 2.0 * _r_bad,
      f'roll held at {_r_true:.1f} / {_r_half:.1f} / {_r_bad:.1f} '
      f'for twist 1 / 0.5 / 0.13')

# Grains are sized by a power law on purpose, but they were all fired
# as ONE SAMPLE whatever their size -- so the rare huge ones came out as
# perfect impulses, flat to Nyquist, and were heard as clicks that had
# nothing to do with the grinding around them. A self-affine asperity is
# as wide as it is tall and the contact crosses it at a finite speed, so
# a big grain must be LONG and therefore LOW; and nothing can be sharper
# than the contact itself can answer.
def _spin_grain(scr=0.55, hard=0.279, seconds=2.5):
    u = sc.SpinUnit(SR)
    u.model = 0
    u.size_in.base = 0.028
    u.settle_in.base = 3.0
    u.rush_in.base = 0.0
    u.twist_in.base = 1.0
    u.wobble_in.base = 0.0
    u.scrape_in.base = scr
    u.hardness_in.base = hard
    u.polish_in.base = 1.0
    u.level_in.base = 1.0
    sig = sc.Signal()
    u.spin_in.sources.append(sig)
    got = []
    for b in range(int(seconds * SR / BLOCK)):
        t = b * BLOCK / SR
        sig.data[:BLOCK] = float(min(1.0, t / 0.3)) if t < 0.4 else 0.0
        sig.constant = False
        u.render(BLOCK)
        got.append(u.grind.array(BLOCK).copy())
    # The steady grinding, before the runaway: the load spikes near the
    # singularity on its own, and that is a different question.
    return np.concatenate(got)[int(0.5 * SR):int(1.5 * SR)]


def _spin_centroid(x):
    S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    return float((f * S).sum() / max(S.sum(), 1e-30))


def _spin_local_crest(seg):
    """Peak against the sound immediately AROUND it, not against the
    whole window -- the grind swells and fades by design, and a crest
    taken over a second measures that instead of the grains."""
    w = int(0.005 * SR)
    loc, pk = [], []
    for i in range(0, len(seg), 64):
        cut = seg[max(0, i - w):i + w]
        loc.append(np.sqrt(np.mean(cut ** 2)))
        pk.append(np.abs(cut).max())
    return float(np.percentile(np.array(pk) / np.maximum(loc, 1e-12), 99))


_gr_soft = _spin_grain(hard=0.279)
_gr_hard = _spin_grain(hard=0.95)
check('no grain stands out as an impulse',
      _spin_local_crest(_gr_soft) < 16.0,
      f'local crest {_spin_local_crest(_gr_soft):.1f}')
check('a soft contact cannot grind sharply, whatever it runs over',
      _spin_centroid(_gr_hard) > 2.0 * _spin_centroid(_gr_soft),
      f'centroid {_spin_centroid(_gr_soft):.0f}Hz soft -> '
      f'{_spin_centroid(_gr_hard):.0f}Hz hard')
check('shaping the grains left scrape a roughness, not a mute',
      _spin_centroid(_spin_grain(scr=1.0)) > 0.0
      and np.sqrt(np.mean(_spin_grain(scr=1.0) ** 2))
      > np.sqrt(np.mean(_gr_soft ** 2)))

# The load on the contact used to run into a hard ceiling and sit
# there, and was then raised to a power that reached 1.9 with hardness
# -- so a five-fold load became a twenty-fold burst with a square edge
# on it, tens of milliseconds long. Heard as a glitch, and it got worse
# the harder the contact was set.
def _spin_bursts(hard):
    g = _spin_grain(hard=hard, seconds=3.0)
    w = int(0.01 * SR)
    env = np.array([np.sqrt(np.mean(g[i*w:(i+1)*w] ** 2))
                    for i in range(len(g) // w)])
    med = np.median(env[env > 0])
    runs, run = [], 0
    for hot in env > 6.0 * med:
        if hot:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return env.max() / max(med, 1e-12), (max(runs) if runs else 0) * 10


_bp_soft, _bl_soft = _spin_bursts(0.3)
_bp_hard, _bl_hard = _spin_bursts(0.9)
check('a flop is a swell, not a burst',
      _bp_hard < 30.0 and _bl_hard <= 30,
      f'loudest {_bp_hard:.0f}x the median, longest run {_bl_hard}ms')
check('a harder contact does not make the flops more violent',
      _bp_hard < 1.6 * _bp_soft,
      f'{_bp_soft:.0f}x soft -> {_bp_hard:.0f}x hard')
check('friction answers the load in proportion, per Amontons',
      sc.SpinUnit.LOAD_EXP <= 1.0)

# The kernel integrates every few samples and reads between, and the
# rates it applies there -- the drain, the hold tracking, the lean it
# averages -- all ran once per STEP. So the integration rate quietly set
# how fast the coin gave up its energy: the same settings ran 1.84 s at
# a decimation of eight and 0.75 s at one. They are per unit time now,
# and the step can be made finer for accuracy without changing the coin.
def _spin_settle_at(decim, settle):
    was = sc.SpinUnit.CONTROL_DECIM
    try:
        sc.SpinUnit.CONTROL_DECIM = decim
        u = sc.SpinUnit(SR)
        u.model = 0
        u.size_in.base = 0.028
        u.settle_in.base = settle
        u.rush_in.base = 0.0
        u.twist_in.base = 1.0
        u.wobble_in.base = 0.0
        u.polish_in.base = 1.0
        sig = sc.Signal()
        u.spin_in.sources.append(sig)
        got = []
        for b in range(int(6.0 * SR / BLOCK)):
            t = b * BLOCK / SR
            sig.data[:BLOCK] = float(min(1.0, t / 0.3)) if t < 0.4 else 0.0
            sig.constant = False
            u.render(BLOCK)
            got.append(u.out.array(BLOCK).copy())
        y = np.concatenate(got)
        alive = np.nonzero(np.abs(y) > 1e-5)[0]
        return alive[-1] / SR if len(alive) else 0.0
    finally:
        sc.SpinUnit.CONTROL_DECIM = was


_st = [_spin_settle_at(d, 3.0) for d in (4, 2, 1)]
check('the integration rate does not decide how long a coin spins',
      max(_st) < 1.05 * min(_st),
      f'{_st[0]:.3f}s / {_st[1]:.3f}s / {_st[2]:.3f}s at decimation 4/2/1')

# A settling coin gets LOUDER. The steady family says why: as the lean
# closes from twenty degrees to a tenth the roll falls, 29.9 to 2.2, but
# the precession rises fourteenfold and the contact sweep fifteenfold,
# so the grind -- riding the square root of that sweep -- gains about
# twelve decibels on the way down. A real coin recorded settling gains
# nine, brightens by a third, and is loudest at the very end. The model
# used to FADE by twenty-eight decibels instead, because the drain
# scaled the roll away rather than letting the lean carry it: the coin
# was left with a tenth of the roll its lean called for, could not
# precess, and merely rocked.
def _spin_arc(**kw):
    u = sc.SpinUnit(SR)
    u.model = 0
    base = dict(size=0.028, settle=3.0, rush=0.0, twist=1.0, wobble=0.0,
                scrape=0.55, hardness=0.279, polish=1.0, level=1.0)
    base.update(kw)
    for name, value in base.items():
        getattr(u, name + '_in').base = value
    sig = sc.Signal()
    u.spin_in.sources.append(sig)
    got, rate = [], []
    for b in range(int(8.0 * SR / BLOCK)):
        t = b * BLOCK / SR
        sig.data[:BLOCK] = float(min(1.0, t / 0.3)) if t < 0.4 else 0.0
        sig.constant = False
        u.render(BLOCK)
        got.append(u.out.array(BLOCK).copy())
        rate.append(u.rate.array(BLOCK).copy())
    y, r = np.concatenate(got), np.concatenate(rate)
    w = int(0.25 * SR)
    env = np.array([np.sqrt(np.mean(y[i*w:(i+1)*w] ** 2))
                    for i in range(len(y) // w)])
    rr = np.array([r[i*w:(i+1)*w].mean() for i in range(len(y) // w)])
    live = env > env.max() * 0.02
    e, rl = env[live], rr[live]
    return (20*np.log10(e[-1]/e[0]), int(np.argmax(e)) / max(1, len(e)-1),
            rl[0], rl[-1])


_rise, _where, _r0, _r1 = _spin_arc()
check('a settling coin gets louder, not quieter',
      _rise > 6.0, f'{_rise:+.1f} dB from first quarter-second to last')
check('and is loudest near the end, where the real one is',
      _where > 0.8, f'loudest {100*_where:.0f}% of the way through')
check('the contact rate accelerates into the finish',
      _r1 > 6.0 * _r0, f'{_r0:.0f} Hz -> {_r1:.0f} Hz')

# A rolling coin meets its OWN rim again every revolution, so whatever
# is uneven about that rim repeats while the table underneath stays
# fresh. The rim used to enter as a single cosine -- a coin bent once,
# swelling smoothly under the contact -- which gives no rhythm and no
# jump. Measured on a recording of real rolling coins, the excitation
# (inverse-filtered to take the coin's own ringing out) runs at a
# kurtosis of 23 against 3 for gaussian noise: impacts, not noise.
def _spin_kurtosis(wob, harmonics):
    was = sc.SpinUnit.RIM_HARMONICS
    try:
        sc.SpinUnit.RIM_HARMONICS = harmonics
        u = sc.SpinUnit(SR)
        u.model = 0
        for name, value in dict(size=0.028, settle=3.0, rush=0.0, twist=1.0,
                                wobble=wob, scrape=0.55, hardness=0.279,
                                polish=1.0, level=1.0).items():
            getattr(u, name + '_in').base = value
        sig = sc.Signal()
        u.spin_in.sources.append(sig)
        got = []
        for b in range(int(3.0 * SR / BLOCK)):
            t = b * BLOCK / SR
            sig.data[:BLOCK] = float(min(1.0, t / 0.3)) if t < 0.4 else 0.0
            sig.constant = False
            u.render(BLOCK)
            got.append(u.grind.array(BLOCK).copy())
        seg = np.concatenate(got)[int(0.5 * SR):int(1.5 * SR)]
        seg = seg - seg.mean()
        v = np.mean(seg ** 2)
        return float(np.mean(seg ** 4) / max(v * v, 1e-30))
    finally:
        sc.SpinUnit.RIM_HARMONICS = was


_k_cos = _spin_kurtosis(0.35, 1)
_k_rim = _spin_kurtosis(0.35, 8)
check('an uneven rim makes the contact jump, not swell',
      _k_rim > 1.3 * _k_cos and _k_rim > 8.0,
      f'kurtosis {_k_cos:.1f} for one cosine -> {_k_rim:.1f} for a rim '
      f'(gaussian noise is 3, the real coin 23)')
check('the rim is a fixed profile, so it repeats every revolution',
      len(sc.SpinUnit(SR)._rim_amp) == sc.SpinUnit.RIM_HARMONICS
      and abs(float((sc.SpinUnit(SR)._rim_amp ** 2).sum()) - 1.0) < 1e-9,
      'unit-power profile, phases fixed per coin')

# A resonator must not pass its own drive. Every mode had a feedforward
# term straight to the output, and at lag zero those add COHERENTLY
# across the bank while the rings they excite dephase within a sample --
# so a bank of eight passed eight times the excitation against one
# mode's worth of tone. On the plate bank the leak measured 100.6% of
# the whole ring peak: the drive as loud as the resonance, unfiltered
# and so flat, and audible with the dry mix at zero.
_lk = sc.ModalUnit(SR)
_lk.set_modes(PLATE8)
_lk.frequency_in.base = 2600.0
_lk.decay_in.base = 0.25
_lk.sensitivity_in.base = 2.0
_lks = sc.Signal()
_lk.excite_in.sources.append(_lks)
_lky = []
for _b in range(int(0.5 * SR / BLOCK)):
    _lks.data[:BLOCK] = 0.0
    if _b == 0:
        _lks.data[0] = 1.0
    _lks.constant = False
    _lk.render(BLOCK)
    _lky.append(_lk.out.array(BLOCK).copy())
_lky = np.concatenate(_lky)
_ring = float(np.abs(_lky[1:]).max())
check('modal~ rings its drive rather than passing it',
      abs(float(_lky[0])) < 1e-9 * max(_ring, 1e-12) and _ring > 1e-4,
      f'lag-zero feed-through {abs(float(_lky[0])):.3e} against a ring '
      f'peak of {_ring:.5f}')

check('spin~ into modal~ rings a coin, bounded',
      np.isfinite(scy).all() and np.max(np.abs(scy)) < 2.0
      and np.sqrt(np.mean(scy[:int(3*SR)]**2)) > 1e-4,
      f'peak {np.max(np.abs(scy)):.3f} '
      f'rms {np.sqrt(np.mean(scy[:int(3*SR)]**2)):.4f}')

# ------------------------------------------------- excite sensitivity
# 'sensitivity' is the gain on what arrives at an excite inlet -- what was
# modal~'s 'drive', renamed for the passive side of the transaction and
# moved under the inlet it scales. Two laws: it scales the excite path,
# and it leaves the unit's OWN mallet alone, or turning an inlet down
# would quietly soften a strike the node makes itself.
def _excite_through(unit_maker, level, seconds=1.0, settled=False):
    """RMS out of a unit with noise into its excite inlet.

    Sensitivity glides rather than steps -- one factor per block would be
    a staircase, and a staircase on a sustained excitation is a zipper --
    so 'settled' measures the tail, after the glide has arrived.
    """
    u = unit_maker()
    u.sensitivity_in.base = level
    src = sc.Signal()
    u.excite_in.sources.append(src)
    n = int(seconds * SR / BLOCK)
    y = np.zeros(n * BLOCK)
    rng = np.random.default_rng(4)
    for b in range(n):
        src.data[:BLOCK] = rng.standard_normal(BLOCK) * 0.05
        src.constant = False
        u.render(BLOCK)
        y[b*BLOCK:(b+1)*BLOCK] = u.out.array(BLOCK)
    if settled:
        y = y[3 * len(y) // 4:]
    return np.sqrt(np.mean(y**2))


def _strike_alone(unit_maker, level, seconds=1.0):
    u = unit_maker()
    u.sensitivity_in.base = level
    u.fire()
    n = int(seconds * SR / BLOCK)
    y = np.zeros(n * BLOCK)
    for b in range(n):
        u.render(BLOCK)
        y[b*BLOCK:(b+1)*BLOCK] = u.out.array(BLOCK)
    return np.sqrt(np.mean(y**2))


def _make_drum():
    u = sc.DrumUnit(SR)
    u.set_modes(MEMBRANE6)
    u.frequency_in.base = 140.0
    u.decay_in.base = 0.4
    return u


def _make_string():
    u = sc.StringUnit(SR)
    u.frequency_in.base = 180.0
    u.decay_in.base = 0.8
    return u


def _make_modal():
    u = sc.ModalUnit(SR)
    u.set_modes(MEMBRANE6)
    u.frequency_in.base = 300.0
    u.decay_in.base = 0.6
    return u


for _nm, _mk, _unity in (('drum', _make_drum, 1.0),
                         ('string', _make_string, 1.0),
                         ('modal', _make_modal, 0.7)):
    _quiet_e = _excite_through(_mk, _unity * 0.5)
    _loud_e = _excite_through(_mk, _unity * 2.0)
    check(f'{_nm} sensitivity scales the excite path',
          _loud_e > 3.0 * _quiet_e,
          f'{_quiet_e:.5f} at half -> {_loud_e:.5f} at double')
    # Long enough for the glide to arrive AND for the body's own ring to
    # die: a string is a delay loop, so what it was fed before the inlet
    # closed is still sounding well after.
    check(f'{_nm} sensitivity 0 shuts the excite inlet',
          _excite_through(_mk, 0.0, seconds=4.0, settled=True) < 1e-6,
          'measured once the glide has arrived and the ring has died')
    check(f'{_nm} sensitivity glides rather than stepping',
          _excite_through(_mk, 0.0) > _excite_through(_mk, 0.0, settled=True),
          'a step would be a zipper on a sustained excitation')
    _s_low = _strike_alone(_mk, _unity * 0.25)
    _s_high = _strike_alone(_mk, _unity * 4.0)
    check(f'{_nm} sensitivity leaves its own strike alone',
          abs(_s_high - _s_low) < 1e-9,
          f'{_s_low:.6f} vs {_s_high:.6f}')

# modal~'s default is still 0.7, the value it carried as 'drive', so the
# rename cannot have changed how any saved patch sounds -- and drum~ and
# string~ default to unity, which is the identity they always had.
check('modal sensitivity keeps drive\'s old default',
      abs(sc.ModalUnit(SR).sensitivity_in.base - 0.7) < 1e-12,
      f'{sc.ModalUnit(SR).sensitivity_in.base}')
check('drum and string sensitivity default to unity',
      sc.DrumUnit(SR).sensitivity_in.base == 1.0
      and sc.StringUnit(SR).sensitivity_in.base == 1.0)
check('sensitivity reaches past the old ceiling of 2.0',
      min(sc.ModalUnit(SR).sensitivity_in.max,
          sc.DrumUnit(SR).sensitivity_in.max,
          sc.StringUnit(SR).sensitivity_in.max) >= 8.0)

print()
if failures:
    print('FAILURES:', failures)
    sys.exit(1)
print('all checks passed')
