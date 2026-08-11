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
import numpy as np

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

print()
if failures:
    print('FAILURES:', failures)
    sys.exit(1)
print('all checks passed')
