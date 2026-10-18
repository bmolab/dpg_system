"""torch signal processing: one page per node.

Spectra (t.fft, t.ifft, t.rfft, t.irfft), the ten t.window.* windows, and the
torch filters (t.ESEKF, t.quat_ESEKF, the smart-clamp / persistence / hybrid /
jerk-aware quaternion Kalman filters, t.filter_bank, t.rotation_band_diff,
t.sav_gol_filter). These were family pages (t.fft, t.window, t.ESEKF,
t.smart_clamp_kf, t.filter_bank); each node now has its own page whose demo
shows what it does to a signal you can change while you watch.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT


# ------------------------------------------------------------------ helpers
def HM(width, height, count, miny=0.0, maxy=1.0, fmt='%.2f', color='viridis'):
    """Property block for a heat_map that draws a whole array as one picture."""
    return {'color': color, 'width': width, 'height': height, 'sample count': count,
            'min y': miny, 'max y': maxy, 'update_mode': 'heat_map',
            'number format': fmt}


def see_also(y, stems, x=30, right=600):
    """A SEE ALSO row: one help_link per related page, left to right, wrapping
    onto a second row rather than running under the prose."""
    row = []
    x0 = x
    for s in stems:
        w = int(len(s) * 7.7) + 30
        if x > x0 and x + w > right:
            x, y = x0, y + 44
        row.append({'key': 'sa_' + s, 'init': 'help_link ' + s, 'pos': (x, y),
                    'w': w, 'h': 30})
        x += w + 14
    if x + 200 > right:
        x, y = x0, y + 44
    row.append({'key': 'sa_c', 'comment': True, 'text': 'SEE ALSO - click to open',
                'pos': (x, y)})
    return row


def MULTI_PLOT(miny=-1.0, maxy=1.0, count=200, width=300, height=128):
    """A plot that draws each element of a list (or each row) as its own line."""
    return dict(PLOT(miny, maxy, count),
                **{'width': width, 'height': height,
                   'update style': 'input is multi-channel sample'})


def ARRAY_PLOT(miny, maxy, count, width=300, height=128, style='line'):
    """A plot that draws one whole 1-D array as a curve, first value on the left."""
    return dict(PLOT(miny, maxy, count, style),
                **{'width': width, 'height': height,
                   'update style': 'buffer holds one sample of input'})


def SLIDER(v, lo, hi, fmt='%.2f', width=200):
    return {'': float(v), 'min': float(lo), 'max': float(hi), 'format': fmt,
            'width': width}


METRO = lambda ms: {'on': True, 'period': float(ms), 'units': 'milliseconds'}


def text_h(body):
    """Height of the prose block: 15px per line at text size 30, plus a margin."""
    return (body.count('\n') + 1) * 15 + 30


TORCH_PROPS = {'device': 'cpu', 'dtype': 'float32', 'requires_grad': False}


# ====================================================================== SPECTRA
TWO_PI_256 = 6.283185307179586 / 256

# Inside make_signal: 256 samples of two sines, re-made ten times a second.
# One sine's frequency comes in from the page (cycles per 256 samples); the
# other is fixed at 40 cycles and half as loud.
SIGNAL_SUB = {
    'name': 'make_signal',
    'host': 'mk',
    'demo': [
        {'key': 'in', 'init': 'in freq', 'pos': (420, 40), 'w': 150, 'h': 80},
        {'key': 'k', 'init': '* %.7f' % TWO_PI_256, 'pos': (420, 140), 'w': 150, 'h': 46,
         'props': {'operand': TWO_PI_256}},
        {'key': 'ck', 'init': 'comment cycles per 256 samples -> radians per sample',
         'pos': (420, 200), 'w': 420, 'h': 30},
        {'key': 'met', 'init': 'metro 100', 'pos': (40, 40), 'w': 129, 'h': 70,
         'props': METRO(100)},
        {'key': 'cm', 'init': 'comment ten times a second', 'pos': (190, 50),
         'w': 220, 'h': 30},
        {'key': 'ls', 'init': 't.linspace 0 255 256', 'pos': (40, 130), 'w': 200, 'h': 150,
         'props': dict(TORCH_PROPS, **{'start': 0.0, 'stop': 255.0, 'steps': 256})},
        {'key': 'cs', 'init': 'comment sample numbers 0 to 255', 'pos': (40, 290),
         'w': 280, 'h': 30},
        {'key': 'b', 'init': '* %.7f' % (10 * TWO_PI_256), 'pos': (40, 330), 'w': 150, 'h': 46,
         'props': {'operand': 10 * TWO_PI_256}},
        {'key': 's1', 'init': 'sin', 'pos': (40, 400), 'w': 120, 'h': 70,
         'props': {'degrees': False}},
        {'key': 'c', 'init': '* %.7f' % (40 * TWO_PI_256), 'pos': (260, 330), 'w': 150, 'h': 46,
         'props': {'operand': 40 * TWO_PI_256}},
        {'key': 'cc', 'init': 'comment 40 cycles, fixed', 'pos': (420, 340), 'w': 220, 'h': 30},
        {'key': 's2', 'init': 'sin', 'pos': (260, 400), 'w': 120, 'h': 70,
         'props': {'degrees': False}},
        {'key': 'h', 'init': '* 0.5', 'pos': (260, 490), 'w': 120, 'h': 46,
         'props': {'operand': 0.5}},
        {'key': 'ch', 'init': 'comment at half the size', 'pos': (400, 500), 'w': 220, 'h': 30},
        {'key': 'add', 'init': '+ 0.0', 'pos': (40, 560), 'w': 120, 'h': 46,
         'props': {'operand': 0.0}},
        {'key': 'out', 'init': 'out signal', 'pos': (40, 630), 'w': 150, 'h': 80},
    ],
    'links': [('in', 'freq', 'k', 'in'), ('k', 'result', 'b', 'operand'),
              ('met', '', 'ls', '###input'),
              ('ls', 'linspace out', 'c', 'in'), ('ls', 'linspace out', 'b', 'in'),
              ('c', 'result', 's2', 'in'), ('s2', 'out', 'h', 'in'),
              ('h', 'result', 'add', 'operand'),
              ('b', 'result', 's1', 'in'), ('s1', 'out', 'add', 'in'),
              ('add', 'result', 'out', 'signal')],
}


def signal_source(y=62):
    """The frequency slider and the make_signal subpatcher, with the signal drawn."""
    return [
        {'key': 'sl', 'init': 'slider', 'pos': (30, y), 'w': 220, 'h': 60,
         'props': SLIDER(10.0, 1.0, 60.0, '%.1f')},
        {'key': 'csl', 'comment': True, 'text': 'drag: the first sine, cycles per 256 samples',
         'pos': (270, y + 10)},
        {'key': 'mk', 'init': 'p make_signal', 'pos': (30, y + 80), 'w': 200, 'h': 70},
        {'key': 'cmk', 'comment': True,
         'text': '256 samples: that sine plus a fixed one at 40\n'
                 'cycles, half as loud - open it to see how',
         'pos': (250, y + 85)},
        {'key': 'psig', 'init': 'plot', 'pos': (30, y + 170), 'w': 520, 'h': 150,
         'props': ARRAY_PLOT(-1.6, 1.6, 256, width=500, height=110)},
        {'key': 'cps', 'comment': True, 'text': 'the signal: 256 samples, left to right',
         'pos': (30, y + 330)},
    ]


SIGNAL_LINKS = [('sl', '', 'mk', 'freq'), ('mk', 'signal', 'psig', 'y')]

FFT_COMMON = """dim (an int box, only when you give an argument):
t.{name} 0 transforms along dimension 0 instead of the last one. With no
argument the node has no dim box and always uses the last dimension, which
is what you want for one signal, or for a stack of signals one per row.

norm (menu):
Where the 1/n scaling goes. 'backward' (the default) leaves the forward
transform unscaled and divides on the way back, so a sine of size 1 over
256 samples makes a peak of 128. 'forward' divides on the way forward
(that peak becomes 0.5 - half the sine's size, the other half sits in the
mirror bin). 'ortho' divides both ways by the square root of n. A forward
and an inverse node must use the same setting or the round trip comes back
the wrong size."""

FFT_TENSOR = """A list or a NumPy array is turned into a float32 tensor first. The
transform runs on whatever device the tensor is on."""

# ------------------------------------------------------------------ t.rfft
body = """Takes the Fourier transform of a real signal: how much of each frequency
the signal contains.

Feed it a window of samples (a sound frame, a stretch of motion data) and
it hands back one number per frequency, from 0 (the average) up to half
the sample count. Use it to find the pitch of a tone, the tempo of a
gesture, or to see whether a movement is slow sway or fast tremor.

The output is COMPLEX: each value carries a size (how much of that
frequency) and a phase (where in its cycle it starts). For a picture you
want the size - put abs after it, as the demo does. t.real and t.imag
take the two parts apart.

WHY rfft AND NOT fft:
For a real signal (all ordinary data) the full Fourier transform is a
mirror image: the top half repeats the bottom half backwards and carries
nothing new. t.rfft computes only the bottom half, n/2 + 1 values for n
samples, so 256 samples give 129 frequencies. It is the one to reach for
unless the input is itself complex - then use t.fft.

SYNTAX:
t.rfft <dim: int>
With no argument it transforms along the last dimension.

EXAMPLE:
t.rfft

INPUTS and PARAMETERS:

tensor in:
The samples. Each tensor that arrives is transformed and sent at once.
""" + FFT_TENSOR + """

""" + FFT_COMMON.format(name='rfft') + """

OUTPUTS:

fft tensor out:
A complex tensor. Along the transformed dimension it has n/2 + 1 values:
value k is the part of the signal that makes k full cycles across the
window. Other dimensions are unchanged.

READING THE DEMO:
The two sines make two spikes: one at the slider's number (10 cycles at
first) and one at 40, half as tall because that sine is half the size.
At a whole number of cycles a spike is one bin wide. Drag the slider to
10.5 and the spike smears into its neighbours - LEAKAGE, because the
window no longer holds a whole number of cycles and its ends do not
meet. Multiplying the samples by a window (t.window.hann) before the
transform is the cure; that is what the window nodes are for.

RELATED:
t.irfft turns this back into samples. t.fft is the full transform for
complex input. spectrum and band_pass split a live stream into bands
instead of transforming a block. t.cwt shows how frequency content
changes over time."""

demo = signal_source() + [
    {'key': 'ft', 'init': 't.rfft', 'pos': (30, 440), 'w': 160, 'h': 70,
     'props': {'norm': 'backward'}},
    {'key': 'cft', 'comment': True, 'text': '129 complex values: one per frequency',
     'pos': (210, 450)},
    {'key': 'ab', 'init': 'abs', 'pos': (30, 530), 'w': 100, 'h': 46},
    {'key': 'cab', 'comment': True, 'text': 'the size of each one', 'pos': (150, 540)},
    {'key': 'pft', 'init': 'plot', 'pos': (30, 600), 'w': 520, 'h': 170,
     'props': ARRAY_PLOT(0.0, 140.0, 129, width=500, height=130)},
    {'key': 'cpf', 'comment': True,
     'text': 'the spectrum: 0 cycles on the left, 128 on the right.\n'
             'two spikes - at the slider\'s number and at 40, half as tall',
     'pos': (30, 780)},
] + see_also(840, ['t.irfft', 't.fft', 't.window.hann', 'spectrum', 't.cwt'])
links = SIGNAL_LINKS + [('mk', 'signal', 'ft', 'tensor in'),
                        ('ft', 'fft tensor out', 'ab', 'in'),
                        ('ab', 'result', 'pft', 'y')]
print(build('t.rfft', 't.rfft - the spectrum of a real signal',
            body, demo, links, demo_width=620, text_width=760,
            text_height=text_h(body), subpatch=SIGNAL_SUB))

# ------------------------------------------------------------------- t.fft
body = """Takes the full Fourier transform of a signal: how much of each frequency
it contains, positive and negative.

Use it when the input is complex (the output of another transform, a pair
of signals packed as real and imaginary parts), or when you need every
frequency bin for an operation that expects the full set. For ordinary
real data t.rfft gives the same information in half the space.

The output is COMPLEX, one value per input sample: a size (how much of
that frequency) and a phase. Put abs after it to see the sizes.

THE MIRROR:
With n samples in you get n frequencies out. Bins 0 to n/2 are the
frequencies 0 to n/2 cycles across the window; bins above n/2 are the
NEGATIVE frequencies, counting back down. For a real signal the negative
half is the positive half reflected, which is why the demo's spectrum is
symmetrical: the spikes at 10 and 40 reappear at 256 - 10 = 246 and
256 - 40 = 216. That second half is the part t.rfft leaves out.

SYNTAX:
t.fft <dim: int>
With no argument it transforms along the last dimension.

EXAMPLE:
t.fft

INPUTS and PARAMETERS:

tensor in:
The samples, real or complex. Each tensor that arrives is transformed and
sent at once. """ + FFT_TENSOR + """

""" + FFT_COMMON.format(name='fft') + """

OUTPUTS:

fft tensor out:
A complex tensor the same shape as the input.

READING THE DEMO:
Drag the slider and watch the two spikes on the left half move one way
while their reflections on the right move the other. At a fraction of a
cycle (10.5) the spikes smear into their neighbours - leakage; see the
t.rfft page.

RELATED:
t.ifft turns the spectrum back into samples. t.rfft is the half-size
transform for real signals. t.window.hann tapers a frame before
transforming it."""

demo = signal_source() + [
    {'key': 'ft', 'init': 't.fft', 'pos': (30, 440), 'w': 160, 'h': 70,
     'props': {'norm': 'backward'}},
    {'key': 'cft', 'comment': True, 'text': '256 complex values: one per input sample',
     'pos': (210, 450)},
    {'key': 'ab', 'init': 'abs', 'pos': (30, 530), 'w': 100, 'h': 46},
    {'key': 'cab', 'comment': True, 'text': 'the size of each one', 'pos': (150, 540)},
    {'key': 'pft', 'init': 'plot', 'pos': (30, 600), 'w': 520, 'h': 170,
     'props': ARRAY_PLOT(0.0, 140.0, 256, width=500, height=130)},
    {'key': 'cpf', 'comment': True,
     'text': 'all 256 bins: the left half is the spectrum, the right\n'
             'half its mirror image (the negative frequencies)',
     'pos': (30, 780)},
] + see_also(840, ['t.ifft', 't.rfft', 't.window.hann', 't.complex'])
links = SIGNAL_LINKS + [('mk', 'signal', 'ft', 'tensor in'),
                        ('ft', 'fft tensor out', 'ab', 'in'),
                        ('ab', 'result', 'pft', 'y')]
print(build('t.fft', 't.fft - the full Fourier transform',
            body, demo, links, demo_width=620, text_width=760,
            text_height=text_h(body), subpatch=SIGNAL_SUB))

# ------------------------------------------------------------------ t.ifft
body = """Takes the inverse Fourier transform: turns a spectrum back into samples.

Whatever t.fft took apart, t.ifft puts back together. The reason to go
there and back is to do something in between: silence a band of
frequencies, boost one, shift the phases, or multiply two spectra
together (which is convolution, done the fast way).

The output is COMPLEX even when the original signal was real. For a real
signal that came through t.fft untouched the imaginary part is only
rounding error (around 0.0000001); t.real keeps the real part, as in the
demo. If you edited the spectrum and the imaginary part is NOT tiny, the
edit broke the mirror symmetry a real signal needs - edit the matching
negative-frequency bin too, or work with t.rfft / t.irfft, which keep
the symmetry for you.

SYNTAX:
t.ifft <dim: int>
With no argument it transforms along the last dimension.

EXAMPLE:
t.ifft

INPUTS and PARAMETERS:

tensor in:
A spectrum, as t.fft makes it: complex, one value per sample. Each tensor
that arrives is transformed and sent at once. A real tensor is accepted
too and treated as a spectrum with no imaginary part.

""" + FFT_COMMON.format(name='ifft') + """

OUTPUTS:

fft tensor out:
A complex tensor the same shape as the input: the samples.

READING THE DEMO:
The top plot is the signal going in; the bottom one is what comes back
after t.fft and t.ifft. Drag the slider: the two always match, because
the transform loses nothing.

RELATED:
t.fft is the forward transform. t.irfft is the inverse for the half
spectrum t.rfft makes, and returns real samples directly. t.real and
t.imag take a complex result apart."""

demo = signal_source() + [
    {'key': 'ft', 'init': 't.fft', 'pos': (30, 440), 'w': 160, 'h': 70,
     'props': {'norm': 'backward'}},
    {'key': 'cft', 'comment': True, 'text': 'into frequencies', 'pos': (210, 450)},
    {'key': 'ift', 'init': 't.ifft', 'pos': (30, 530), 'w': 160, 'h': 70,
     'props': {'norm': 'backward'}},
    {'key': 'cif', 'comment': True, 'text': 'and back into samples (still complex)',
     'pos': (210, 540)},
    {'key': 're', 'init': 't.real', 'pos': (30, 620), 'w': 130, 'h': 46},
    {'key': 'cre', 'comment': True, 'text': 'keep the real part', 'pos': (180, 630)},
    {'key': 'prt', 'init': 'plot', 'pos': (30, 690), 'w': 520, 'h': 150,
     'props': ARRAY_PLOT(-1.6, 1.6, 256, width=500, height=110)},
    {'key': 'cpr', 'comment': True, 'text': 'the round trip: the same signal as the top plot',
     'pos': (30, 850)},
] + see_also(900, ['t.fft', 't.irfft', 't.rfft', 't.complex'])
links = SIGNAL_LINKS + [('mk', 'signal', 'ft', 'tensor in'),
                        ('ft', 'fft tensor out', 'ift', 'tensor in'),
                        ('ift', 'fft tensor out', 're', 'tensor in'),
                        ('re', 'real tensor', 'prt', 'y')]
print(build('t.ifft', 't.ifft - from a spectrum back to samples',
            body, demo, links, demo_width=620, text_width=760,
            text_height=text_h(body), subpatch=SIGNAL_SUB))

# ----------------------------------------------------------------- t.irfft
body = """Takes the inverse of t.rfft: turns a half spectrum back into real samples.

t.rfft keeps only the non-negative frequencies of a real signal; t.irfft
rebuilds the samples from them, filling in the mirror half itself, so the
result is REAL - no t.real needed. Go there and back to work on a signal
in frequency terms: zero the bins above a cut-off for a brick-wall low
pass, scale bins to reshape the tone, or multiply spectra to convolve.

LENGTH:
m frequency values come back as 2 x (m - 1) samples. 129 values (from 256
samples) give 256 again. An ODD number of samples does not survive the
trip: 255 samples make 128 values, which come back as 254. Keep frame
lengths even.

SYNTAX:
t.irfft <dim: int>
With no argument it transforms along the last dimension.

EXAMPLE:
t.irfft

INPUTS and PARAMETERS:

tensor in:
A half spectrum, as t.rfft makes it: complex, n/2 + 1 values. Each tensor
that arrives is transformed and sent at once. The imaginary parts of the
first value (and, for an even length, the last) cannot be represented in
a real signal and are ignored.

""" + FFT_COMMON.format(name='irfft') + """

OUTPUTS:

fft tensor out:
A REAL tensor of 2 x (m - 1) samples along the transformed dimension.

READING THE DEMO:
The top plot is the signal going in; the bottom one is what comes back
after t.rfft and t.irfft - the same 256 samples. Drag the slider and the
two stay identical.

RELATED:
t.rfft is the forward transform. t.ifft is the inverse for the full
spectrum t.fft makes, and returns complex samples."""

demo = signal_source() + [
    {'key': 'ft', 'init': 't.rfft', 'pos': (30, 440), 'w': 160, 'h': 70,
     'props': {'norm': 'backward'}},
    {'key': 'cft', 'comment': True, 'text': 'into 129 frequencies', 'pos': (210, 450)},
    {'key': 'ift', 'init': 't.irfft', 'pos': (30, 530), 'w': 160, 'h': 70,
     'props': {'norm': 'backward'}},
    {'key': 'cif', 'comment': True, 'text': 'and back into 256 real samples',
     'pos': (210, 540)},
    {'key': 'prt', 'init': 'plot', 'pos': (30, 620), 'w': 520, 'h': 150,
     'props': ARRAY_PLOT(-1.6, 1.6, 256, width=500, height=110)},
    {'key': 'cpr', 'comment': True, 'text': 'the round trip: the same signal as the top plot',
     'pos': (30, 780)},
] + see_also(830, ['t.rfft', 't.ifft', 't.fft', 't.window.hann'])
links = SIGNAL_LINKS + [('mk', 'signal', 'ft', 'tensor in'),
                        ('ft', 'fft tensor out', 'ift', 'tensor in'),
                        ('ift', 'fft tensor out', 'prt', 'y')]
print(build('t.irfft', 't.irfft - from a half spectrum back to samples',
            body, demo, links, demo_width=620, text_width=760,
            text_height=text_h(body), subpatch=SIGNAL_SUB))


# ====================================================================== WINDOWS
WINDOW_COMMON_TOP = """SYNTAX:
{name}
No arguments: set the length and the rest on the node.

EXAMPLE:
{name}

INPUTS and PARAMETERS:

(button, the first inlet):
Makes the window and sends it. Anything arriving here triggers it; in the
demo a metro bangs it ten times a second so the plot follows your edits.
Nothing else on the node sends by itself.

type (menu):
Which window. Choosing another one turns this node into that window - the
title changes and the parameter boxes change to suit - so one node can
audition them all.

m (int):
The length: how many values. Default 256, smallest 1. Match it to the
number of samples in the frame you are going to multiply."""

WINDOW_COMMON_BOTTOM = """sym (checkbox, default on):
On: SYMMETRIC - the window rises and falls the same way and both ends are
the edge value. Right for designing filters. Off: PERIODIC - computed as
if one sample longer and the last sample dropped, so a row of them laid
end to end repeats exactly. That is the version to use before an FFT.

device / dtype / requires_grad:
Where the tensor is made (cpu by default), its number type (float32 by
default) and whether gradients flow through it.

OUTPUTS:

window tensor out:
A 1-D tensor of m values, at most 1. Multiply a frame by it (a * node
does tensors) before t.rfft.

WHAT A WINDOW IS FOR:
Cutting a frame out of a longer signal leaves ragged ends: unless the
frame happens to hold a whole number of cycles, the last sample does not
meet the first, and t.rfft reads that jump as extra frequencies smeared
around every real one (LEAKAGE - drag the slider on the t.rfft page to
10.5 cycles to see it). Multiplying by a window fades the frame in and
out so the ends meet at or near zero. The price is that each spectral
spike gets wider. Every window is a different bargain between a narrow
spike (the main lobe) and quiet surroundings (the side lobes)."""

WINDOWS = [
    # name, title, first-sentence, shape paragraph, params[(name, lo, hi, val, fmt, text)],
    # demo note, see also
    ('hann', 'a raised cosine that falls to zero',
     'Makes a Hann window: one hump of a cosine, rising from 0 to 1 and back to 0.',
     """THE SHAPE:
0.5 - 0.5 cos: a smooth hump that touches exactly zero at both ends, so
the frame starts and ends in silence. It is the usual first choice for
spectral analysis - a good, plain bargain between spike width and
leakage, and the window most code reaches for by default. Its side lobes
fall away quickly with distance from a peak.""",
     [], 'drag m on the node, or untick sym',
     ['t.window.hamming', 't.window.blackman', 't.window.general_hamming', 't.rfft']),
    ('hamming', 'a raised cosine that stops short of zero',
     'Makes a Hamming window: a cosine hump that rises from 0.08 to 1 and back.',
     """THE SHAPE:
0.54 - 0.46 cos. Almost the Hann hump, but lifted so the ends stop at
0.08 instead of zero. Those weights were chosen to cancel the nearest
side lobe: the leakage right next to a peak is lower than Hann's, but
far from the peak it dies away more slowly. Choose it when a weak tone
close to a strong one must not be swamped.""",
     [], 'drag m on the node, or untick sym',
     ['t.window.hann', 't.window.general_hamming', 't.window.blackman', 't.rfft']),
    ('blackman', 'three cosines, very quiet side lobes',
     'Makes a Blackman window: a narrower hump built from three cosines, zero at the ends.',
     """THE SHAPE:
0.42 - 0.5 cos + 0.08 cos(2x). The extra term pulls the shoulders in, so
the hump is slimmer than Hann's and reaches zero at both ends. Leakage is
much lower (side lobes about 58 dB down, against Hann's 31) and the
price is a spike about half as wide again. Use it when you need to see a
quiet tone next to a loud one.""",
     [], 'drag m on the node, or untick sym',
     ['t.window.hann', 't.window.nuttall', 't.window.kaiser', 't.rfft']),
    ('bartlett', 'a straight-sided triangle',
     'Makes a Bartlett window: a triangle, rising in a straight line from 0 to the middle and back.',
     """THE SHAPE:
Straight lines up to a point in the middle, zero at both ends. The
corner at the top and the corners at the ends make its side lobes fall
off slowly, so it leaks more than the cosine windows. Use it for a
simple linear fade (overlap-adding frames that cross-fade), or as the
textbook reference.""",
     [], 'drag m on the node, or untick sym',
     ['t.window.hann', 't.window.cosine', 't.window.hamming']),
    ('cosine', 'half a sine wave',
     'Makes a cosine (sine) window: one half cycle of a sine, from nearly 0 up to 1 and back.',
     """THE SHAPE:
sin(pi (n + 0.5) / m): the top half of a sine wave. It does not quite
reach zero at the ends (0.006 for m 256) and its sides are steeper than
Hann's, so the spike stays narrower but leakage is higher. Its square is
the Hann window, which is why it is used in overlapped audio frames that
are windowed twice, once before analysis and once after resynthesis.""",
     [], 'drag m on the node, or untick sym',
     ['t.window.hann', 't.window.bartlett', 't.window.kaiser']),
    ('nuttall', 'four cosines, the quietest of the fixed windows',
     'Makes a Nuttall window: a slim hump built from four cosines, almost zero at the ends.',
     """THE SHAPE:
Four cosine terms whose weights are tuned for the lowest side lobes of
any window here (well below Blackman's). The hump is the slimmest of the
fixed windows and stops at 0.0004 at the ends. Use it when leakage must
be as low as possible - measuring a very quiet partial - and you can
afford a wide spike.""",
     [], 'drag m on the node, or untick sym',
     ['t.window.blackman', 't.window.kaiser', 't.window.hann']),
    ('gaussian', 'a bell curve of adjustable width',
     'Makes a Gaussian window: a bell curve centred on the middle of the frame.',
     """THE SHAPE:
exp(-0.5 (n / std) squared), with n counted from the centre. std is the
width of the bell in samples: small std is a narrow spike in the middle
and near-silence elsewhere; large std approaches a flat frame with
softened edges. The bell never reaches zero, so with a wide bell the
ends are cut off abruptly. A Gaussian's transform is another Gaussian,
with no side lobes at all apart from those the truncated ends add.""",
     [('std', 1.0, 128.0, 32.0, '%.1f',
       """std (float, also an inlet):
The width of the bell, in samples (one standard deviation). Default 1.0,
which for m 256 is a spike only a few samples wide - almost nothing gets
through. Something near m / 8 (32 for 256) is a useful start. Must be
greater than 0; anything else prints an error and sends nothing.""")],
     'drag std: narrow spike to nearly flat',
     ['t.window.kaiser', 't.window.exponential', 't.window.hann', 't.dist.normal']),
    ('general_hamming', 'the Hann / Hamming family on one dial',
     'Makes a generalised Hamming window: alpha - (1 - alpha) cos, a raised cosine whose lift you set.',
     """THE SHAPE:
A cosine hump that ends at 2 x alpha - 1. alpha 0.5 is exactly the Hann
window (ends at zero); 0.54, the default, is exactly the Hamming window
(ends at 0.08); 1.0 is flat, the same as no window. Between, it trades
the side lobe next to a peak against those further away.""",
     [('alpha', 0.0, 1.0, 0.54, '%.2f',
       """alpha (float, also an inlet):
The weight of the constant term. Default 0.54 (Hamming). 0.5 is Hann,
1.0 is flat. Below 0.5 the ends go negative, which is rarely useful.""")],
     'drag alpha: 0.5 is hann, 0.54 hamming, 1 flat',
     ['t.window.hamming', 't.window.hann', 't.window.kaiser']),
    ('kaiser', 'one dial from flat to very narrow',
     'Makes a Kaiser window: a hump whose shape is set by one number, beta.',
     """THE SHAPE:
Built from a Bessel function: beta 0 is flat (a rectangular window, no
taper at all); as beta grows the hump narrows and the side lobes fall.
Around 5 it resembles Hamming, around 8.6 Blackman. It is the window to
pick when you want to tune the bargain rather than choose from a list -
a filter-design rule of thumb sets beta from the leakage you can
tolerate.""",
     [('beta', 0.0, 30.0, 12.0, '%.1f',
       """beta (float, also an inlet):
How strongly the window tapers. Default 12.0 (narrow, very low leakage).
0 is flat. Must not be negative; a negative beta prints an error and
sends nothing.""")],
     'drag beta: 0 is flat, higher is narrower',
     ['t.window.gaussian', 't.window.blackman', 't.window.general_hamming']),
    ('exponential', 'a sharp peak with exponential sides',
     'Makes an exponential (Poisson) window: exp(-|n - centre| / tau), a peak with straight-sided fall-offs on a log scale.',
     """THE SHAPE:
A point at the centre, falling away exponentially on both sides: tau is
how many samples it takes to fall to about a third. Unlike the cosine
windows it never reaches zero and has a sharp peak rather than a rounded
top. Use it where recent samples should count more than distant ones,
or for damped (decaying) signals.""",
     [('tau', 1.0, 200.0, 40.0, '%.1f',
       """center (float, also an inlet):
Where the peak sits, in samples. Used ONLY when sym is off: with sym on
the peak is always in the middle and this box is ignored. Default 1.0,
which with sym off puts the peak at the second sample.

tau (float, also an inlet):
How fast the sides fall, in samples. Default 1.0, which for m 256 is a
spike only a couple of samples wide; 40 or so gives a visible shape.
Must be greater than 0; anything else prints an error and sends
nothing.""")],
     'drag tau: how fast the sides fall',
     ['t.window.gaussian', 't.window.kaiser', 't.window.hann']),
]

for (wname, wtitle, first, shape, params, note, see) in WINDOWS:
    name = 't.window.' + wname
    param_text = '\n\n'.join(p[5] for p in params)
    body = (first + """

Use it to taper a frame of samples before a Fourier transform (t.rfft),
or as a smoothing kernel or fade shape: it is a list of m weights, at
most 1, that rise towards the middle.

""" + shape + """

""" + WINDOW_COMMON_TOP.format(name=name) + ('\n\n' + param_text if params else '') +
            """

""" + WINDOW_COMMON_BOTTOM + """

READING THE DEMO:
The plot draws the window, first value on the left; the strip under it is
the same values as shades (dark 0, bright 1). """ +
            ('Drag the slider and the shape follows.' if params else
             'Drag m on the node to change the length, or untick sym.') + """

RELATED:
""" + ', '.join(see[:-1]) + ' and ' + see[-1] + """ are on their own pages.
t.rfft is what a window usually feeds; the type menu on this node
switches to any other window.""")
    sl = []
    sl_links = []
    y0 = 62
    if params:
        p = params[0]
        sl = [{'key': 'sl', 'init': 'slider', 'pos': (30, 62), 'w': 220, 'h': 60,
               'props': SLIDER(p[3], p[1], p[2], p[4])},
              {'key': 'csl', 'comment': True, 'text': note, 'pos': (270, 72)}]
        sl_links = [('sl', '', 'w', p[0])]
        y0 = 150
    props = dict(TORCH_PROPS, **{'type': wname, 'm': 256, 'sym': True})
    for p in params:
        props[p[0]] = p[3]
    demo = sl + [
        {'key': 'met', 'init': 'metro 100', 'pos': (30, y0), 'w': 129, 'h': 70,
         'props': METRO(100)},
        {'key': 'cmet', 'comment': True, 'text': 'remakes the window ten times a second',
         'pos': (180, y0 + 10)},
        {'key': 'w', 'init': name, 'pos': (30, y0 + 90), 'w': 220, 'h': 200,
         'props': props},
    ] + ([] if params else [
        {'key': 'cw', 'comment': True, 'text': note, 'pos': (270, y0 + 100)}]) + [
        {'key': 'pw', 'init': 'plot', 'pos': (30, y0 + 310), 'w': 520, 'h': 170,
         'props': ARRAY_PLOT(-0.05, 1.05, 256, width=500, height=130)},
        {'key': 'hw', 'init': 'heat_map', 'pos': (30, y0 + 490), 'w': 520, 'h': 70,
         'props': HM(500, 40, 1, 0.0, 1.0, fmt='none', color='viridis')},
        {'key': 'cpw', 'comment': True,
         'text': 'the window as a curve, and as shades (dark 0, bright 1)',
         'pos': (30, y0 + 570)},
    ] + see_also(y0 + 620, [s for s in see])
    links = sl_links + [('met', '', 'w', ''), ('w', 'window tensor out', 'pw', 'y'),
                        ('w', 'window tensor out', 'hw', 'y')]
    print(build(name, name + ' - ' + wtitle, body, demo, links, demo_width=620,
                text_width=760, text_height=text_h(body)))


# ====================================================================== FILTERS
# Inside make_noisy: a slow sine (one cycle in 4 seconds, -1 to 1) with a
# little jitter on every frame and, now and then, a one-frame spike of 1.5 -
# the kind of glitch a tracker or sensor throws.
NOISY_SUB = {
    'name': 'make_noisy',
    'host': 'mk',
    'demo': [
        {'key': 'sig', 'init': 'signal 4.0 sin', 'pos': (40, 40), 'w': 129, 'h': 78,
         'props': SIG('sin', 4.0, 1.0)},
        {'key': 'csig', 'init': 'comment the movement: one slow cycle every 4 seconds',
         'pos': (190, 50), 'w': 420, 'h': 30},
        {'key': 'tk', 'init': 'tick', 'pos': (420, 110), 'w': 33, 'h': 42,
         'props': {'': True}},
        {'key': 'ctk', 'init': 'comment every frame', 'pos': (470, 115), 'w': 160, 'h': 30},
        {'key': 'g', 'init': 'random.gauss 0.0 0.04', 'pos': (240, 170), 'w': 175, 'h': 100},
        {'key': 'cg', 'init': 'comment jitter', 'pos': (240, 280), 'w': 120, 'h': 30},
        {'key': 'r', 'init': 'random 1.0', 'pos': (440, 170), 'w': 140, 'h': 80,
         'props': {'range': 1.0, 'bipolar': False}},
        {'key': 'gt', 'init': '> 0.985', 'pos': (440, 270), 'w': 120, 'h': 46,
         'props': {'operand': 0.985, 'output_type': 'float'}},
        {'key': 'sz', 'init': '* 1.5', 'pos': (440, 340), 'w': 120, 'h': 46,
         'props': {'operand': 1.5}},
        {'key': 'csz', 'init': 'comment about one frame in 70: a spike of 1.5',
         'pos': (440, 400), 'w': 360, 'h': 30},
        {'key': 'sq', 'init': 'signal 5.0 square', 'pos': (40, 170), 'w': 129, 'h': 78,
         'props': SIG('square', 5.0, 0.4)},
        {'key': 'csq', 'init': 'comment a sudden step that stays: up 0.8, down 0.8',
         'pos': (40, 260), 'w': 180, 'h': 30},
        {'key': 'a0', 'init': '+ 0.0', 'pos': (40, 310), 'w': 120, 'h': 46,
         'props': {'operand': 0.0}},
        {'key': 'a1', 'init': '+ 0.0', 'pos': (40, 380), 'w': 120, 'h': 46,
         'props': {'operand': 0.0}},
        {'key': 'a2', 'init': '+ 0.0', 'pos': (40, 450), 'w': 120, 'h': 46,
         'props': {'operand': 0.0}},
        {'key': 'out', 'init': 'out noisy', 'pos': (40, 530), 'w': 150, 'h': 80},
    ],
    'links': [('tk', '', 'g', 'trigger'), ('tk', '', 'r', 'trigger'),
              ('g', 'out', 'a1', 'operand'),
              ('r', 'out', 'gt', 'in'), ('gt', 'result', 'sz', 'in'),
              ('sz', 'result', 'a2', 'operand'),
              ('sq', '', 'a0', 'operand'),
              ('sig', '', 'a0', 'in'), ('a0', 'result', 'a1', 'in'),
              ('a1', 'result', 'a2', 'in'),
              ('a2', 'result', 'out', 'noisy')],
}

# Inside make_orientation: an orientation turning slowly back and forth
# (60 degrees each way, one cycle in 6 seconds) about one axis, with a degree
# or so of jitter on all three axes and, now and then, a one-frame glitch of
# 45 degrees - what a mocap joint looks like when a marker or sensor slips.
# Sent as a [1, 4] tensor: one stream, w x y z.
ORIENT_SUB = {
    'name': 'make_orientation',
    'host': 'mk',
    'demo': [
        {'key': 'sig', 'init': 'signal 6.0 sin', 'pos': (40, 40), 'w': 129, 'h': 78,
         'props': SIG('sin', 6.0, 60.0)},
        {'key': 'csig', 'init': 'comment the turn: 60 degrees each way, one cycle in 6 seconds',
         'pos': (190, 50), 'w': 500, 'h': 30},
        {'key': 'tk', 'init': 'tick', 'pos': (460, 110), 'w': 33, 'h': 42,
         'props': {'': True}},
        {'key': 'ctk', 'init': 'comment every frame', 'pos': (510, 115), 'w': 160, 'h': 30},
        {'key': 'gz', 'init': 'random.gauss 0.0 1.0', 'pos': (240, 170), 'w': 175, 'h': 100},
        {'key': 'cgz', 'init': 'comment a degree of jitter', 'pos': (240, 280), 'w': 220, 'h': 30},
        {'key': 'r', 'init': 'random 1.0', 'pos': (460, 170), 'w': 140, 'h': 80,
         'props': {'range': 1.0, 'bipolar': False}},
        {'key': 'gt', 'init': '> 0.985', 'pos': (460, 270), 'w': 120, 'h': 46,
         'props': {'operand': 0.985, 'output_type': 'float'}},
        {'key': 'sz', 'init': '* 45.0', 'pos': (460, 340), 'w': 120, 'h': 46,
         'props': {'operand': 45.0}},
        {'key': 'csz', 'init': 'comment about one frame in 70: a 45 degree glitch',
         'pos': (460, 400), 'w': 400, 'h': 30},
        {'key': 'sq', 'init': 'signal 5.0 square', 'pos': (40, 170), 'w': 129, 'h': 78,
         'props': SIG('square', 5.0, 15.0)},
        {'key': 'csq', 'init': 'comment a quick 30 degree turn that stays',
         'pos': (40, 260), 'w': 180, 'h': 30},
        {'key': 'a0', 'init': '+ 0.0', 'pos': (40, 310), 'w': 120, 'h': 46,
         'props': {'operand': 0.0}},
        {'key': 'a1', 'init': '+ 0.0', 'pos': (40, 380), 'w': 120, 'h': 46,
         'props': {'operand': 0.0}},
        {'key': 'a2', 'init': '+ 0.0', 'pos': (40, 450), 'w': 120, 'h': 46,
         'props': {'operand': 0.0}},
        {'key': 'gx', 'init': 'random.gauss 0.0 1.0', 'pos': (240, 520), 'w': 175, 'h': 100},
        {'key': 'gy', 'init': 'random.gauss 0.0 1.0', 'pos': (440, 520), 'w': 175, 'h': 100},
        {'key': 'cgx', 'init': 'comment and jitter on the other two axes',
         'pos': (640, 540), 'w': 340, 'h': 30},
        {'key': 'pk', 'init': 'pack 3', 'pos': (40, 650), 'w': 160, 'h': 110},
        {'key': 'e2q', 'init': 'euler_to_quaternion', 'pos': (40, 790), 'w': 220, 'h': 80,
         'props': {'degrees': True, 'order': 'zxy'}},
        {'key': 'ce2q', 'init': 'comment angles in degrees -> a quaternion, w first',
         'pos': (280, 805), 'w': 420, 'h': 30},
        {'key': 'rs', 'init': 't.reshape 1 4', 'pos': (40, 900), 'w': 180, 'h': 90,
         'props': {'shape': '1 4'}},
        {'key': 'crs', 'init': 'comment one stream: a [1, 4] tensor',
         'pos': (240, 915), 'w': 300, 'h': 30},
        {'key': 'out', 'init': 'out quaternion', 'pos': (40, 1020), 'w': 170, 'h': 80},
    ],
    'links': [('tk', '', 'gz', 'trigger'), ('tk', '', 'r', 'trigger'),
              ('tk', '', 'gx', 'trigger'), ('tk', '', 'gy', 'trigger'),
              ('gz', 'out', 'a1', 'operand'),
              ('r', 'out', 'gt', 'in'), ('gt', 'result', 'sz', 'in'),
              ('sz', 'result', 'a2', 'operand'),
              ('gx', 'out', 'pk', 'in 2'), ('gy', 'out', 'pk', 'in 3'),
              ('sq', '', 'a0', 'operand'),
              ('sig', '', 'a0', 'in'), ('a0', 'result', 'a1', 'in'),
              ('a1', 'result', 'a2', 'in'),
              ('a2', 'result', 'pk', 'in 1'),
              ('pk', 'out', 'e2q', 'xyz rotation'),
              ('e2q', 'quaternion rotation', 'rs', 'tensor in'),
              ('rs', 'output', 'out', 'quaternion')],
}

# Inside read_turn: both orientations back to angles, and the turn angle of
# each packed into one pair for the plot - filtered first, input second. The
# input must arrive first (it is the cold inlet of the pack).
READ_SUB = {
    'name': 'read_turn',
    'host': 'rd',
    'demo': [
        {'key': 'ii', 'init': 'in input', 'pos': (300, 40), 'w': 150, 'h': 80},
        {'key': 'if', 'init': 'in filtered', 'pos': (40, 40), 'w': 150, 'h': 80},
        {'key': 'q1', 'init': 'quaternion_to_euler', 'pos': (300, 140), 'w': 220, 'h': 80},
        {'key': 'q2', 'init': 'quaternion_to_euler', 'pos': (40, 140), 'w': 220, 'h': 80},
        {'key': 'cq', 'init': 'comment each orientation as three angles, in degrees',
         'pos': (40, 230), 'w': 420, 'h': 30},
        {'key': 'v1', 'init': 'np.ravel', 'pos': (300, 280), 'w': 120, 'h': 46},
        {'key': 'v2', 'init': 'np.ravel', 'pos': (40, 280), 'w': 120, 'h': 46},
        {'key': 'u1', 'init': 'unpack 3', 'pos': (300, 350), 'w': 160, 'h': 80},
        {'key': 'u2', 'init': 'unpack 3', 'pos': (40, 350), 'w': 160, 'h': 80},
        {'key': 'cu', 'init': 'comment the third angle is the turn',
         'pos': (40, 440), 'w': 300, 'h': 30},
        {'key': 'pk', 'init': 'pack 2', 'pos': (40, 490), 'w': 140, 'h': 90},
        {'key': 'cpk', 'init': 'comment filtered, input', 'pos': (200, 510), 'w': 200, 'h': 30},
        {'key': 'out', 'init': 'out turn', 'pos': (40, 610), 'w': 150, 'h': 80},
    ],
    'links': [('ii', 'input', 'q1', 'quaternion'), ('if', 'filtered', 'q2', 'quaternion'),
              ('q1', 'euler angles', 'v1', 'input'), ('q2', 'euler angles', 'v2', 'input'),
              ('v1', 'flattened array', 'u1', ''), ('v2', 'flattened array', 'u2', ''),
              ('u1', 'out 3', 'pk', 'in 2'), ('u2', 'out 3', 'pk', 'in 1'),
              ('pk', 'out', 'out', 'turn')],
}




# ------------------------------------------------------------ demo builders
COL2 = 380          # x of the right-hand column (plots beside the tall filter)


def scalar_demo(init, props, inlet, out='filtered', fh=420, note='', alphas=None):
    """make_noisy -> t float float -> t.reshape 1 1 -> the filter; the raw value
    and the filtered one meet in a pack 2 and are drawn as two lines on one
    plot. Returns demo, links and the y below the lowest node."""
    demo = [
        {'key': 'mk', 'init': 'p make_noisy', 'pos': (30, 62), 'w': 190, 'h': 70},
        {'key': 'cmk', 'comment': True,
         'text': 'a slow sine with jitter, a step now and then,\n'
                 'and the odd one-frame spike - open it to see how',
         'pos': (240, 62)},
        {'key': 'tr', 'init': 't float float', 'pos': (30, 160), 'w': 150, 'h': 46},
        {'key': 'ctr', 'comment': True, 'text': 'right first: the raw value to the plot, then',
         'pos': (200, 165)},
        {'key': 'rs', 'init': 't.reshape 1 1', 'pos': (30, 230), 'w': 180, 'h': 90,
         'props': {'shape': '1 1'}},
        {'key': 'crs', 'comment': True, 'text': 'one stream of one value: a [1, 1] tensor',
         'pos': (230, 245)},
        {'key': 'f', 'init': init, 'pos': (30, 340), 'w': 320, 'h': fh, 'props': props},
        {'key': 'tf', 'init': 't float', 'pos': (COL2, 340), 'w': 100, 'h': 46},
        {'key': 'pk', 'init': 'pack 2', 'pos': (COL2, 400), 'w': 140, 'h': 90},
        {'key': 'pl', 'init': 'plot', 'pos': (COL2, 510), 'w': 400, 'h': 200,
         'props': MULTI_PLOT(-1.6, 2.6, 300, 380, 160)},
        {'key': 'cpl', 'comment': True,
         'text': 'line 1: filtered.  line 2: the input.\nfive seconds, newest on the right',
         'pos': (COL2, 720)},
    ]
    if note:
        demo.append({'key': 'cf', 'comment': True, 'text': note, 'pos': (COL2, 290)})
    links = [('mk', 'noisy', 'tr', ''),
             ('tr', 'float', 'pk', 'in 2', 1), ('tr', 'float', 'rs', 'tensor in', 0),
             ('rs', 'output', 'f', inlet),
             ('f', out, 'tf', ''), ('tf', 'float', 'pk', 'in 1'),
             ('pk', 'out', 'pl', 'y')]
    bottom = max(340 + fh, 760)
    if alphas:
        demo += [
            {'key': 'pa', 'init': 'plot', 'pos': (COL2, 780), 'w': 400, 'h': 160,
             'props': MULTI_PLOT(-0.05, 1.05, 300, 380, 120)},
            {'key': 'cpa', 'comment': True,
             'text': 'the alphas: 1 damping, 2 responsive, 3 error -\n'
                     'which mode the filter is in, moment to moment',
             'pos': (COL2, 950)},
        ]
        links.append(('f', alphas, 'pa', 'y'))
        bottom = max(bottom, 1000)
    return demo, links, bottom


def quat_demo(init, props, inlet, fh=420, note='', alphas=None):
    """make_orientation -> t array array -> the filter; read_turn turns the
    input and the output back into the turn angle, drawn as two lines."""
    demo = [
        {'key': 'mk', 'init': 'p make_orientation', 'pos': (30, 62), 'w': 220, 'h': 70},
        {'key': 'cmk', 'comment': True,
         'text': 'a joint turning slowly back and forth, with jitter,\n'
                 'a quick 30 degree turn now and then and the odd\n'
                 'one-frame 45 degree glitch: a [1, 4] tensor, w x y z',
         'pos': (270, 62)},
        {'key': 'tr', 'init': 't array array', 'pos': (30, 160), 'w': 150, 'h': 46},
        {'key': 'ctr', 'comment': True, 'text': 'right first: the raw orientation, then',
         'pos': (200, 165)},
        {'key': 'f', 'init': init, 'pos': (30, 240), 'w': 320, 'h': fh, 'props': props},
        {'key': 'rd', 'init': 'p read_turn', 'pos': (COL2, 240), 'w': 200, 'h': 90},
        {'key': 'crd', 'comment': True, 'text': 'both orientations as a turn angle,\nin degrees',
         'pos': (COL2, 340)},
        {'key': 'pl', 'init': 'plot', 'pos': (COL2, 400), 'w': 400, 'h': 200,
         'props': MULTI_PLOT(-100.0, 100.0, 300, 380, 160)},
        {'key': 'cpl', 'comment': True,
         'text': 'line 1: filtered.  line 2: the input.\nfive seconds, newest on the right',
         'pos': (COL2, 610)},
    ]
    if note:
        demo.append({'key': 'cf', 'comment': True, 'text': note, 'pos': (COL2, 180)})
    links = [('mk', 'quaternion', 'tr', ''),
             ('tr', 'array', 'rd', 'input', 1), ('tr', 'array', 'f', inlet, 0),
             ('f', 'filtered', 'rd', 'filtered'), ('rd', 'turn', 'pl', 'y')]
    bottom = max(240 + fh, 660)
    if alphas:
        demo += [
            {'key': 'pa', 'init': 'plot', 'pos': (COL2, 680), 'w': 400, 'h': 160,
             'props': MULTI_PLOT(-0.05, 1.05, 300, 380, 120)},
            {'key': 'cpa', 'comment': True,
             'text': 'the alphas: 1 damping, 2 responsive, 3 error -\n'
                     'which mode the filter is in, moment to moment',
             'pos': (COL2, 850)},
        ]
        links.append(('f', alphas, 'pa', 'y'))
        bottom = max(bottom, 900)
    return demo, links, bottom


def slider_into(param, v, lo, hi, fmt, text, x=COL2, y=62):
    return [{'key': 'sl', 'init': 'slider', 'pos': (x, y), 'w': 220, 'h': 60,
             'props': SLIDER(v, lo, hi, fmt)},
            {'key': 'csl', 'comment': True, 'text': text, 'pos': (x + 240, y + 10)}], \
           [('sl', '', 'f', param)]


PARALLEL = """MANY STREAMS AT ONCE:
The input is a tensor of shape [streams, values]: one row per stream, all
of them one moment in time. Every row gets its own filter state, so a
whole body - 22 joints, one row each - is filtered in one operation. The
first frame (and any change in the number of rows) builds the state
afresh."""

QUAT_IN = """The quaternions are w x y z, w first (as euler_to_quaternion makes
them), one row per joint: shape [joints, 4]."""

TRADE = """THE GLITCH OR THE MOVEMENT:
A tracker throws two kinds of sudden change: a GLITCH, one bad frame
that comes straight back, and a real QUICK MOVEMENT, a change that stays.
On the frame it happens they look the same. Any filter that refuses the
glitch will also hesitate on the quick movement; the filters in this
family differ in how they make that bet. The demo input has both, so you
can see each filter's choice."""

KF_FAMILY = """THE FAMILY:
t.ESEKF            ordinary numbers, three modes blended by motion
t.quat_ESEKF       the same for quaternions (error-state)
t.smart_clamp_kf   ordinary numbers, the correction per frame capped
t.smart_clamp_quat_kf      the same for quaternions, plus an acceleration cap
t.persistence_quat_kf      acceleration cap that gives way if the change persists
t.hybrid_quat_kf   three-mode blending driving a smart-clamped filter
t.jerk_aware_quat_kf       spike detection from jerk; extrapolates through spikes"""


# ---------------------------------------------------------------------- t.ESEKF
body = """Smooths streams of ordinary numbers with a Kalman filter that shifts
between three behaviours - damping, responsive and error-correcting -
according to how fast the input moves and how far it strays.

Use it on positions, joint angles or any measured values that jitter
when still and must keep up when they move. A fixed smoother has to pick
one compromise; this one relaxes its smoothing as the motion speeds up
and tightens it again at rest.

HOW IT WORKS:
Underneath is a constant-velocity Kalman filter: it keeps a position and
a velocity per value, predicts the next frame, and corrects towards the
measurement. How hard it corrects depends on two noise settings - how
much it distrusts the measurement (Meas Noise) and how much it expects
the velocity to change (Vel Noise). The node holds three sets of these:

   damping            for stillness: distrust the input, smooth hard
   responsive         for motion: trust the input, follow quickly
   error correction   for when the estimate is far off: snap back

Every frame it measures the input's speed (the distance between this
frame and the last, per second) and the error (how far the input is
from where the filter predicted). Speed between Min and Max Motion
fades in the responsive set; error between Min and Max Error fades in
the error set; the rest stays damping. The three weights (the ALPHAS)
glide towards their targets at Blending Speed, so the filter never
switches abruptly. Despite the name it is not an error-state filter -
that is its quaternion sibling, t.quat_ESEKF. This is the same
three-mode design for numbers that can simply be added.

""" + PARALLEL + """ A 1-D input of n values is one stream of n values;
it comes back as [1, n]. A tensor with more than two dimensions is
treated as [first dimension, everything else] and reshaped back.

SYNTAX:
t.ESEKF
No arguments.

EXAMPLE:
t.ESEKF

INPUTS and PARAMETERS:

input:
The values, [streams, values]. Each arrival runs one filter step and
sends. A list is turned into a tensor.

dt (sec) (float, default 1/60):
The time between frames. Speeds are measured with it, so it must match
the real frame rate. Zero or less: nothing is sent.

Blending Speed (float, 0.01 to 1, default 0.1):
How far the alphas move towards their targets each frame. 1 jumps at
once; 0.1 takes some tens of frames to change mode.

reset (button):
The next input becomes the filter's state outright (velocity zero) -
use it when the data jumps on purpose, such as a new take.

Accel Rejection Factor (float, default 0, off):
When the filter's estimated acceleration reverses direction from one
frame to the next, multiply the measurement noise by (1 + factor) and
divide the velocity noise by it - stiffer just when the input chatters.

Damp Meas Noise / Damp Vel Noise (defaults 0.1, 0.01):
Responsive Meas Noise / Resp Vel Noise (defaults 0.01, 1.0):
Err Correct Meas Noise / Err Correct Vel Noise (defaults 0.001, 10.0):
The three behaviours. Meas Noise is the expected size of the jitter,
in the input's own units: larger smooths more. Vel Noise is how much
the velocity may change per second: larger follows changes faster.

Min / Max Motion (units/s) (defaults 1, 5):
Speeds over which the responsive set fades in.

Min / Max Error (units) (defaults 0.5, 2):
Distances from the prediction over which the error set fades in.

OUTPUTS:

filtered:
The filtered values, [streams, values].

alphas:
[streams, 3]: the weights of damping, responsive and error correction,
summing to 1. Watch it while tuning: it says which mode the filter is
actually in.

READING THE DEMO:
Jitter alone moves the input fast enough to count as motion (0.04 of
jitter per frame at 60 frames a second is over 2 units a second), so
the alphas wander between damping and responsive. A spike is cut to
about half and fades over five frames; a step is followed within one
or two frames. Drag the slider (Damp Meas Noise) up: at 0.8 the jitter
is three times smaller again and spikes are cut to a fifth, but each
step now takes about five frames.

""" + KF_FAMILY + """

RELATED:
kalman_filter is a plain one-dimensional Kalman filter for single
numbers. one_euro_filter and filter are simpler smoothers."""

demo, links, bottom = scalar_demo('t.ESEKF', {'Damp Meas Noise': 0.1}, 'input',
                                  fh=520, alphas='alphas')
sl, sl_links = slider_into('Damp Meas Noise', 0.1, 0.01, 1.0, '%.2f',
                           'drag: Damp Meas Noise', y=160)
demo += sl
demo += see_also(bottom + 30, ['t.quat_ESEKF', 't.smart_clamp_kf', 'kalman_filter',
                               'one_euro_filter', 'filter'])
print(build('t.ESEKF', 't.ESEKF - a Kalman filter with three moods',
            body, demo, links + sl_links, demo_width=840, text_width=760,
            text_height=text_h(body), subpatch=NOISY_SUB))

# ----------------------------------------------------------------- t.quat_ESEKF
body = """Smooths streams of quaternions (joint orientations) with an error-state
Kalman filter that shifts between damping, responsive and
error-correcting behaviour according to how fast the joint turns and
how far the input strays from the prediction.

Use it on mocap joint rotations that jitter at rest but must keep up
with real turns. It is the quaternion sibling of t.ESEKF: the same three
modes and the same settings, but working in the rotation's own terms so
the result is always a valid rotation.

WHY 'ERROR-STATE':
Quaternions cannot be averaged or subtracted like ordinary numbers: the
average of two rotations' four numbers is not a rotation, and q and -q
are the same rotation. So the filter keeps its best orientation and an
angular velocity as they are, and runs the Kalman arithmetic only on the
SMALL difference between its prediction and the measurement - three
numbers, a rotation vector, which do behave like ordinary numbers. The
correction is then applied as a small rotation. This works the same at
any orientation, whichever way the performer faces.

THE THREE MODES:
   damping            for stillness: distrust the input, smooth hard
   responsive         for turning: trust the input, follow quickly
   error correction   for when the estimate is far off: snap back
Speed is the angle between this input and the last, per second; error is
the angle between the input and the filter's prediction. Speed between
Min and Max Motion fades in the responsive set, error between Min and
Max Error the error set. The weights (the ALPHAS) glide towards their
targets at Blending Speed.

""" + PARALLEL + """
""" + QUAT_IN + """ A single quaternion [4] is accepted and
comes back as [4] (and its alphas as [3]).

SYNTAX:
t.quat_ESEKF
No arguments.

EXAMPLE:
t.quat_ESEKF

INPUTS and PARAMETERS:

quaternions:
[joints, 4], w first, or one quaternion [4]. Anything else is ignored.
Each arrival runs one filter step and sends.

dt (sec) (float, default 1/60):
The time between frames; turning speeds are measured with it.

Blending Speed (float, 0.01 to 1, default 0.1):
How far the alphas move towards their targets each frame. 1 jumps at
once; 0.1 takes some tens of frames to change mode.

reset (button):
The next input becomes the filter's orientation outright, at rest. The
filter also starts this way, from its first input.

Accel Rejection Factor (float, default 0, off):
When the estimated angular acceleration reverses from one frame to the
next, the measurement noise is multiplied by (1 + factor) and the
angular velocity noise divided by it: stiffer when the input chatters.

Damp Meas Noise / Damp AngVel Noise (defaults 0.5, 0.1):
Resp Meas Noise / Resp AngVel Noise (defaults 0.05, 1.0):
Err Correct Meas Noise / Err Correct AngVel Noise (defaults 0.005, 10.0):
The three behaviours. Meas Noise is the expected jitter, in radians
(0.05 is about 3 degrees): larger smooths more. AngVel Noise is how much
the turning speed may change, radians a second: larger follows changes
faster.

Min / Max Motion (deg/sec) (defaults 15, 90):
Turning speeds over which the responsive set fades in.

Min / Max Error (deg) (defaults 5, 45):
Angles from the prediction over which the error set fades in.

OUTPUTS:

filtered:
The filtered quaternions, [joints, 4], unit length.

alphas (damp, resp, err):
[joints, 3]: the weight of each mode, summing to 1.

READING THE DEMO:
Measured on this input: the quick 30 degree turns are followed within
a frame - the error mode sees to that - and about half of each glitch
gets through, a bump that rings back over three or four frames. A degree or two of jitter between frames is already 60 to 140
degrees a second, so with the default Min Motion of 15 the alphas sit
mostly in the responsive mode even when the joint is barely turning: raise Min and
Max Motion above the jitter's speed for more smoothing at rest. Drag the
slider (Resp Meas Noise) up: at 0.3 the jitter is a third of what it
was, glitches are cut to about a third, and turns take three frames.

""" + TRADE + """

""" + KF_FAMILY + """

RELATED:
t.smart_clamp_quat_kf rejects glitches more firmly. adaptive_quaternion_filter
is a simpler adaptive smoother for quaternions."""

demo, links, bottom = quat_demo('t.quat_ESEKF', {'Resp Meas Noise': 0.05}, 'quaternions',
                                fh=560, alphas='alphas (damp, resp, err)')
sl, sl_links = slider_into('Resp Meas Noise', 0.05, 0.01, 0.5, '%.3f',
                           'drag: Resp Meas Noise', y=120)
demo += sl
demo += see_also(bottom + 30, ['t.ESEKF', 't.smart_clamp_quat_kf', 't.hybrid_quat_kf',
                               't.jerk_aware_quat_kf', 'adaptive_quaternion_filter'])
print(build('t.quat_ESEKF', 't.quat_ESEKF - an error-state Kalman filter for rotations',
            body, demo, links + sl_links, demo_width=840, text_width=760,
            text_height=text_h(body), subpatch=[ORIENT_SUB, READ_SUB]))

# ------------------------------------------------------------- t.smart_clamp_kf
body = """Smooths streams of ordinary numbers with a Kalman filter that will not
let any one frame pull it further than a set distance.

Use it on positions or values from a tracker that occasionally throws a
wild reading. A one-frame spike moves the output by no more than 'max
jump', so it barely shows; steady change still comes through.

HOW IT WORKS:
A constant-velocity Kalman filter: it keeps a position and a velocity
per value, predicts the next frame, and corrects towards the
measurement. The correction is the INNOVATION - the measurement minus
the prediction. Smart clamping caps the length of that innovation
(across all of a stream's values together) at 'max jump' before it is
used. A wild reading is treated as if it were only max jump away.

The cost is the other half of the bet: a real step bigger than max jump
is also taken a bounded bite at a time, so the output walks to the new
level over several frames instead of jumping. Set max jump just above
the largest honest change between two frames.

""" + PARALLEL + """ A 1-D input of n values is one stream and comes back
1-D. A tensor with more dimensions is treated as rows of its last
dimension and reshaped back.

SYNTAX:
t.smart_clamp_kf
No arguments.

EXAMPLE:
t.smart_clamp_kf

INPUTS and PARAMETERS:

input:
The values. Each arrival runs one filter step and sends.

dt (float, default 1/60):
The time between frames.

responsiveness (float, default 1.0):
The process noise: how freely the filter lets the velocity change. Higher
follows changes sooner and smooths less.

smoothness (float, default 0.1):
The measurement noise: how much the filter distrusts each reading.
Higher smooths more and lags more.

max jump (clamp) (float, default 0.1):
The largest innovation used in one frame, in the input's own units. 0
turns clamping off.

accel_rejection (float, default 0, off):
When the estimated acceleration reverses from one frame to the next,
divide the process noise by (1 + 100 x this): much stiffer on chatter.

max_accel (deg/s^2):
Has no effect on this node - only the quaternion filters use it.

reset (button):
The next input becomes the state outright, at rest. The filter also
starts this way, from its first input, so there is no glide in from zero
when the patch opens.

OUTPUTS:

filtered:
The filtered values, the same shape as the input.

READING THE DEMO:
Measured on this input at the default 0.1: the spikes are almost gone
(about 4 percent of each gets through), the jitter is six times smaller,
and each step of 0.8 is reached in about a dozen frames. Drag max jump
on the node: at 0.5 the steps take about four frames but a tenth of
each spike shows; at 0.02 the output can no longer keep up even with
the slow sine.

""" + TRADE + """

""" + KF_FAMILY + """

RELATED:
kalman_filter is a plain Kalman filter for single numbers; one_euro_filter
adapts its smoothing to speed instead of clamping."""

demo, links, bottom = scalar_demo('t.smart_clamp_kf', {'max jump (clamp)': 0.1}, 'input',
                                  fh=330, note='drag max jump on the node')
demo += see_also(bottom + 30, ['t.smart_clamp_quat_kf', 't.ESEKF', 'kalman_filter',
                               'one_euro_filter', 't.sav_gol_filter'])
print(build('t.smart_clamp_kf', 't.smart_clamp_kf - a Kalman filter that caps each correction',
            body, demo, links, demo_width=840, text_width=760,
            text_height=text_h(body), subpatch=NOISY_SUB))

# -------------------------------------------------------- t.smart_clamp_quat_kf
SMART_QUAT_PARAMS = """input:
[joints, 4], w first; a single quaternion [4] is accepted and comes back
as [1, 4]. Anything else is ignored. Each arrival runs one step and sends.

responsiveness (float, default 1.0):
The process noise: how freely the turning speed may change. Higher
follows changes sooner and smooths less.

smoothness (float, default 0.1):
The measurement noise: how much each reading is distrusted. Higher
smooths more and lags more.

max jump (deg) (float, default 5):
The largest correction used in one frame, in degrees. 0 turns this cap
off.

accel_rejection (float, default 0, off):
When the estimated angular acceleration reverses from one frame to the
next, divide the process noise by (1 + 100 x this).

max_accel (deg/s^2) (float, default 50000):
The second cap: the angular acceleration a correction may imply."""

body = """Smooths streams of quaternions with a Kalman filter that caps how far
any one frame can turn it.

Use it on mocap joint rotations that are mostly clean but now and then
throw a frame that is wildly wrong. The glitch moves the output by no
more than 'max jump' degrees, so it barely shows.

HOW IT WORKS:
It works in the rotation's own terms (an error-state filter, like
t.quat_ESEKF): it keeps an orientation and an angular velocity per
joint, predicts the next frame, and measures the difference from the
input as a small rotation - the INNOVATION. Two caps are applied before
the correction is used:

  max jump      the innovation is shortened to at most this many degrees
  max_accel     the whole correction is scaled down if the angular
                acceleration it implies is above this. The limit is
                relaxed up to 50 times while the joint is already turning
                fast (from about 60 to 85 degrees a second), so it is
                strict at rest, where a sudden jump is most likely a
                glitch, and lenient in motion.

A real quick turn bigger than max jump is followed a bounded bite at a
time, so it takes several frames. Unlike t.quat_ESEKF there are no
modes: the same settings always apply.

""" + PARALLEL + """
""" + QUAT_IN + """

SYNTAX:
t.smart_clamp_quat_kf
No arguments.

EXAMPLE:
t.smart_clamp_quat_kf

INPUTS and PARAMETERS:

""" + SMART_QUAT_PARAMS + """ With the default the
size cap does almost all the work; lower this (to a few thousand) to
make the acceleration cap matter.

reset (button):
The next input becomes the orientation outright, at rest. The filter
also starts this way, from its first input.

dt (float, default 1/60):
The time between frames. Zero or less is taken as 1/60.

OUTPUTS:

filtered:
The filtered quaternions, [joints, 4], unit length.

READING THE DEMO:
Measured on this input at the default 5 degrees: a 45 degree glitch moves
the output by only a degree or two; the jitter is about a sixth of the
input's; each 30 degree quick turn takes around ten frames to be
followed. Drag max jump on the node: at 20 the turns are followed in
three or four frames but an eighth of each glitch shows; at 2 a turn
takes twenty frames and the output trails even the slow turning.

""" + TRADE + """

""" + KF_FAMILY + """

RELATED:
t.persistence_quat_kf is this filter with a different acceleration cap:
it lets a change through once it has lasted a few frames. t.smart_clamp_kf
is the version for ordinary numbers."""

demo, links, bottom = quat_demo('t.smart_clamp_quat_kf', {'max jump (deg)': 5.0}, 'input',
                                fh=360, note='drag max jump (deg) on the node')
demo += see_also(bottom + 30, ['t.persistence_quat_kf', 't.smart_clamp_kf', 't.quat_ESEKF',
                               't.hybrid_quat_kf', 't.jerk_aware_quat_kf'])
print(build('t.smart_clamp_quat_kf', 't.smart_clamp_quat_kf - capped corrections for rotations',
            body, demo, links, demo_width=840, text_width=760,
            text_height=text_h(body), subpatch=[ORIENT_SUB, READ_SUB]))

# -------------------------------------------------------- t.persistence_quat_kf
body = """Smooths streams of quaternions with a Kalman filter that refuses a
sudden change for its first frame or two, then accepts it if it is still
there.

Use it on mocap rotations where a glitch lasts one frame and a real fast
movement lasts longer. Persistence is the test: a change that keeps
arriving is believed.

HOW IT WORKS:
It is t.smart_clamp_quat_kf with the acceleration cap changed. Each
frame the correction towards the input implies an angular acceleration.
When that is above max_accel the frame is a VIOLATION:

  the first threshold_frames - 1 violations in a row are scaled down
  to max_accel (held back, as the smart-clamp filter would);
  once violations have lasted threshold_frames frames in a row, the
  correction is used in full - the change has persisted, so it is real.

A frame without a violation resets the count. There is no relaxing of
the cap with speed as in t.smart_clamp_quat_kf: it is the same at rest
and in motion, and only duration decides.

The max jump cap still applies to every frame, persistent or not, and at
its default of 5 degrees it does most of the limiting - which hides the
persistence test. The demo therefore sets max jump to 0 (off) and
max_accel to 3000, so the acceleration cap and its persistence rule do
the work.

""" + PARALLEL + """
""" + QUAT_IN + """

SYNTAX:
t.persistence_quat_kf
No arguments.

EXAMPLE:
t.persistence_quat_kf

INPUTS and PARAMETERS:

""" + SMART_QUAT_PARAMS + """ Above this a frame
counts as a violation.

reset (button):
The next input becomes the orientation outright, at rest, and the count
is cleared. The filter also starts this way.

dt (float, default 1/60):
The time between frames. Zero or less is taken as 1/60.

threshold_frames (int, 1 to 10, default 2):
How many violating frames in a row it takes to believe the change. At 1
every change is believed at once (no cap at all); at 2 a one-frame
glitch is held back and anything longer is let through.

OUTPUTS:

filtered:
The filtered quaternions, [joints, 4], unit length.

READING THE DEMO:
Measured on this input (max jump 0, max_accel 3000): at threshold_frames
2 almost nothing of a glitch gets through (about 4 percent) and the
quick 30 degree turns are followed in three or four frames. At 4,
glitches shrink to 2 percent and turns take about six frames; at 1 more
than a quarter of every glitch passes and turns take three. Drag
threshold_frames on the node. With the node's own defaults (max jump 5,
max_accel 50000) it behaves like t.smart_clamp_quat_kf: turns take
about ten frames.

""" + TRADE + """

""" + KF_FAMILY + """

RELATED:
t.smart_clamp_quat_kf caps by size and by speed instead of by duration;
t.jerk_aware_quat_kf decides from the jerk of the input."""

demo, links, bottom = quat_demo('t.persistence_quat_kf',
                                {'max jump (deg)': 0.0, 'max_accel (deg/s^2)': 3000.0,
                                 'threshold_frames': 2}, 'input',
                                fh=400, note='drag threshold_frames on the node')
demo += see_also(bottom + 30, ['t.smart_clamp_quat_kf', 't.jerk_aware_quat_kf',
                               't.hybrid_quat_kf', 't.quat_ESEKF'])
print(build('t.persistence_quat_kf', 't.persistence_quat_kf - believe a change that lasts',
            body, demo, links, demo_width=840, text_width=760,
            text_height=text_h(body), subpatch=[ORIENT_SUB, READ_SUB]))

# ------------------------------------------------------------- t.hybrid_quat_kf
body = """Smooths streams of quaternions with a smart-clamped Kalman filter whose
smoothing is steered by three blended modes - damping, responsive and
error-correcting - as in t.quat_ESEKF.

Use it when you want both things: the mode blending that relaxes the
smoothing while a joint turns and tightens it at rest, and the caps that
stop a one-frame glitch from yanking the output.

HOW IT WORKS:
Each frame it measures the turning speed (the angle between this input
and the last, per second) and the error (the angle between the input
and the filter's prediction), fades between three sets of settings with
the same rules as t.quat_ESEKF, and hands the blend to the filter inside
t.smart_clamp_quat_kf as its smoothness and responsiveness. That filter
then applies its max jump and max_accel caps. 'alphas' reports the
blend. q and -q count as the same rotation in both measures, so a sign
flip in the input is not mistaken for a turn.

""" + PARALLEL + """
""" + QUAT_IN + """ A single quaternion [4] is accepted and
comes back as [1, 4].

SYNTAX:
t.hybrid_quat_kf
No arguments.

EXAMPLE:
t.hybrid_quat_kf

INPUTS and PARAMETERS:

input:
[joints, 4], w first. Each arrival runs one step and sends.

reset (button):
The next input becomes the orientation outright, and the alphas return
to damping. The filter also starts this way.

dt (sec) (float, default 1/60):
The time between frames; turning speeds and accelerations are measured
with it, so it must match the real frame rate. Zero or less: nothing is
sent.

Damp Smoothness / Damp Responsiveness (defaults 0.1, 1.0):
Resp Smoothness / Resp Responsiveness (defaults 0.01, 100):
Err Correct Smoothness / Err Correct Responsiveness (defaults 0.5, 50):
The three modes, as smoothness (measurement noise: higher smooths more)
and responsiveness (process noise: higher follows changes sooner).

Min / Max Motion (deg/sec) (defaults 10, 90):
Turning speeds over which the responsive mode fades in.

Min / Max Error (deg) (defaults 5, 45):
Angles from the prediction over which the error mode fades in.

Blending Speed (float, default 0.1):
How far the alphas move towards their targets each frame.

max jump (clamp) (float, default 5):
The largest correction per frame, in DEGREES. 0 turns it off.

accel_rejection (float, default 0, off):
On a reversal of the estimated angular acceleration, divide the process
noise by (1 + 100 x this).

max_accel (deg/s^2) (float, default 50000):
The acceleration cap, as on t.smart_clamp_quat_kf; relaxed up to 50
times while the joint is already turning fast. At the default the max
jump cap does almost all the work; lower it (to a few thousand) to make
the acceleration cap hold back sudden jumps at rest as well.

OUTPUTS:

filtered:
The filtered quaternions, [joints, 4].

alphas (damp, resp, err):
[joints, 3]: the weight of each mode, summing to 1.

READING THE DEMO:
Measured at the node's defaults: glitches are cut to about a tenth,
quick 30 degree turns are followed in about three frames (overshooting
by a few degrees), and the jitter is a little over half the input's. The jitter keeps the alphas
mostly in the responsive mode, as on t.quat_ESEKF. Drag the slider
(Blending Speed) towards 1 and the alphas jump from mode to mode; towards
0.01 they barely move.

""" + TRADE + """

""" + KF_FAMILY + """

RELATED:
t.quat_ESEKF has the mode blending without the caps;
t.smart_clamp_quat_kf has the caps without the modes."""

demo, links, bottom = quat_demo('t.hybrid_quat_kf',
                                {'Blending Speed': 0.1},
                                'input', fh=600, alphas='alphas (damp, resp, err)',
                                note='')
sl, sl_links = slider_into('Blending Speed', 0.1, 0.01, 1.0, '%.2f',
                           'drag: Blending Speed', y=120)
demo += sl
demo += see_also(bottom + 30, ['t.quat_ESEKF', 't.smart_clamp_quat_kf',
                               't.persistence_quat_kf', 't.jerk_aware_quat_kf'])
print(build('t.hybrid_quat_kf', 't.hybrid_quat_kf - mode blending plus capped corrections',
            body, demo, links + sl_links, demo_width=840, text_width=760,
            text_height=text_h(body), subpatch=[ORIENT_SUB, READ_SUB]))

# --------------------------------------------------------- t.jerk_aware_quat_kf
body = """Follows streams of quaternions closely, but recognises one-frame spikes
by their jerk and coasts through them on the joint's recent turning
speed instead of following them.

Use it on mocap joint rotations where you want almost no lag in normal
movement and only the glitches removed. It is not a Kalman filter: in
normal tracking it simply moves most of the way (max_responsiveness)
towards each new input, so it smooths very little - its work is all in
deciding what is a spike.

HOW IT WORKS:
From the RAW input it computes, per joint, the turning speed, its change
(angular acceleration) and the change of that (JERK). Each frame each
joint is classed as:

  dynamic motion   the joint's smoothed turning speed is above
                   velocity_threshold: track it, never call it a spike.
                   It stays in this class until the speed falls below 30
                   percent of the threshold for spike_recovery_frames.
  noise spike      the jerk is above jerk_threshold (or the acceleration
                   reverses sharply above accel_limit) while the joint is
                   not moving fast: a glitch.
  normal           everything else: track it.

During a spike the output carries on from the last good orientation at
the smoothed turning speed - extrapolating, not freezing. When the spike
ends it blends back to the input over spike_recovery_frames, with its
turning speed limited so the return itself does not jerk.

FOR WHOLE BODIES:
With 22 or 24 rows (SMPL joints) or 52 or 55 (with fingers), the jerk
threshold is scaled per joint: the pelvis is judged four times more
strictly, the wrists twice as leniently, fingers four times. And if at
least glitch_joint_threshold of all the joints spike in the same frame
while the body is not moving fast, it is taken as a whole-body jump (a
mocap recalibration): every joint glides to the new pose over
glitch_transition_frames instead of being held.

""" + PARALLEL + """
""" + QUAT_IN + """ A single quaternion [4] is accepted and
comes back as [1, 4].

SYNTAX:
t.jerk_aware_quat_kf
No arguments.

EXAMPLE:
t.jerk_aware_quat_kf

INPUTS and PARAMETERS:

input:
[joints, 4], w first. Each arrival runs one step and sends.

dt (float, default 1/60):
The time between frames. The thresholds are scaled for it (jerk by the
cube of the frame rate, acceleration by the square), so they mean the
same at any rate.

reset (button):
Forget all history; the next input is taken as it is.

jerk_threshold (rad/s cubed) (default 2000):
Jerk above this, at low speed, marks a spike. Lower catches more.

accel_limit (rad/s squared) (default 50):
A sharp reversal of angular acceleration above half this also marks a
spike.

velocity_threshold (rad/s) (default 5, about 290 degrees a second):
Above this the joint is in dynamic motion and nothing is called a spike.

min_responsiveness (default 0.1):
Has no effect - nothing in the filter reads it.

max_responsiveness (0.5 to 1, default 0.95):
How far towards each new input the output moves in normal tracking.
0.95 is nearly the raw input; lower smooths (and lags) more.

spike_recovery_frames (default 3):
How long a spike is held after it ends, and how long the blend back
takes.

glitch_joint_threshold (default 0.3) / glitch_transition_frames (default 5):
The whole-body jump rule above.

OUTPUTS:

filtered:
The filtered quaternions, [joints, 4], unit length.

READING THE DEMO:
Measured on this one-joint input: glitches mostly vanish (about 6
percent shows); the jitter is only roughly halved (the default moves 95
percent of the way to each frame); a quick 30 degree turn - a jump that
stays - is first treated as a spike and then recovered, taking seven to
nine frames. Drag max_responsiveness on the node down to 0.5: the
jitter drops to a third and the output starts to lag the turning.

""" + TRADE + """

""" + KF_FAMILY + """

RELATED:
t.persistence_quat_kf decides by how long a change lasts;
t.smart_clamp_quat_kf caps every correction."""

demo, links, bottom = quat_demo('t.jerk_aware_quat_kf', {'max_responsiveness': 0.95},
                                'input', fh=480,
                                note='drag max_responsiveness on the node')
demo += see_also(bottom + 30, ['t.persistence_quat_kf', 't.smart_clamp_quat_kf',
                               't.quat_ESEKF', 't.hybrid_quat_kf'])
print(build('t.jerk_aware_quat_kf', 't.jerk_aware_quat_kf - coast through spikes',
            body, demo, links, demo_width=840, text_width=760,
            text_height=text_h(body), subpatch=[ORIENT_SUB, READ_SUB]))

# ------------------------------------------------------------- t.sav_gol_filter
body = """Smooths streams by fitting a curve to the most recent stretch of samples
(a Savitzky-Golay filter), which keeps the shape of peaks that a moving
average would flatten.

Use it on a trace whose peaks matter - their height, their timing - such
as a speed you will look for maxima in. Over each window of samples it
fits a low-order polynomial and reports the fitted value, so a peak fits
a curve with a peak in it and keeps most of its height, while jitter
does not fit and is removed.

THE DELAY:
The value reported is the fitted curve at the MIDDLE of the window, so
the output is late by half the window: (window length - 1) / 2 frames -
15 frames, a quarter of a second at 60 frames a second, at the default
31. The demo's filtered line runs visibly behind the input. A shorter
window is less late and smooths less.

The fit also assumes the data is smooth across the window: at a real
step it overshoots a little on either side.

""" + PARALLEL + """

SYNTAX:
t.sav_gol_filter
No arguments.

EXAMPLE:
t.sav_gol_filter

INPUTS and PARAMETERS:

signal:
[streams, values], exactly 2-D. A single value must be shaped [1, 1]
(as the demo's t.reshape does); anything that is not 2-D prints a
message in the console and sends nothing. Each arrival is one step.

reset (button):
Empties the history. The next input fills the whole window, as if the
signal had always been at that value, so the output starts AT it rather
than climbing from zero: a steady 1.0 comes out as 1.0 from the first
frame. The filter starts the same way from its first input.

window length (int, 3 to 1001, default 31):
How many recent samples each fit spans. An even number is raised by one.

polynomial order (int, 1 to 8, default 3):
The degree of the fitted curve; must be below the window length (it is
lowered if not). Higher follows sharper shapes and smooths less. Order
1 is a straight line - a plain moving average.

normalize output (checkbox, default ON):
Scales each row of the result to length 1. That is right for quaternion
rows - a smoothed quaternion must be unit length to be a rotation - and
WRONG for anything else: a single value becomes +1 or -1. Untick it for
ordinary numbers, as in the demo. (Quaternions are smoothed as four
plain numbers, so a sign flip between q and -q in the input will make a
mess of them - keep the signs consistent first.)

Changing any of these starts the filter again with an empty history.

OUTPUTS:

filtered:
[streams, values], the fitted values half a window ago.

READING THE DEMO:
Drag window length on the node: at 31 the line is smooth but a quarter
of a second late and the spikes are spread into small bumps; at 7 it is
nearly on time but smooths less and lets a third of each spike
through.

RELATED:
filter and one_euro_filter are moving-average style smoothers with
less delay. t.smart_clamp_kf rejects spikes instead of spreading them."""

demo, links, bottom = scalar_demo('t.sav_gol_filter',
                                  {'window length': 31, 'polynomial order': 3,
                                   'normalize output': False}, 'signal',
                                  fh=240, note='drag window length on the node')
demo += see_also(bottom + 30, ['filter', 'one_euro_filter', 't.smart_clamp_kf', 't.ESEKF'])
print(build('t.sav_gol_filter', 't.sav_gol_filter - smoothing by fitting a curve',
            body, demo, links, demo_width=840, text_width=760,
            text_height=text_h(body), subpatch=NOISY_SUB))

# ---------------------------------------------------------------- t.filter_bank
# Inside make_mix: three sines summed - slow (0.5 Hz), middle (1.7 Hz) and
# fast (6 Hz), each smaller than the last.
MIX_SUB = {
    'name': 'make_mix',
    'host': 'mk',
    'demo': [
        {'key': 's1', 'init': 'signal 2.0 sin', 'pos': (40, 40), 'w': 129, 'h': 78,
         'props': SIG('sin', 2.0, 1.0)},
        {'key': 's2', 'init': 'signal 0.6 sin', 'pos': (220, 40), 'w': 129, 'h': 78,
         'props': SIG('sin', 0.6, 0.5)},
        {'key': 's3', 'init': 'signal 0.1667 sin', 'pos': (400, 40), 'w': 129, 'h': 78,
         'props': SIG('sin', 0.16667, 0.3)},
        {'key': 'cs', 'init': 'comment 0.5 Hz size 1, 1.7 Hz size 0.5, 6 Hz size 0.3',
         'pos': (40, 140), 'w': 460, 'h': 30},
        {'key': 'a1', 'init': '+ 0.0', 'pos': (40, 200), 'w': 120, 'h': 46,
         'props': {'operand': 0.0}},
        {'key': 'a2', 'init': '+ 0.0', 'pos': (40, 270), 'w': 120, 'h': 46,
         'props': {'operand': 0.0}},
        {'key': 'out', 'init': 'out mix', 'pos': (40, 350), 'w': 150, 'h': 80},
    ],
    'links': [('s2', '', 'a1', 'operand'), ('s3', '', 'a2', 'operand'),
              ('s1', '', 'a1', 'in'), ('a1', 'result', 'a2', 'in'),
              ('a2', 'result', 'out', 'mix')],
}

body = """Splits each incoming stream into frequency bands with a bank of IIR
filters (Butterworth or Chebyshev), sample by sample - slow movement in
one band, fast in another.

Use it to ask WHICH timescale a movement is on, not just how much there
is: a slow sway and a fast tremor become separate numbers instead of one.
Feed it one sample per frame; every value in the input gets its own
complete bank.

THE BANDS:
low and high bound the range; 'number of bands' is the number of band
EDGES spread between them, so it makes one band FEWER: the default 8
makes 7 bands. With log scaling (the default) the edges are evenly
spaced in octaves; with linear, evenly in Hz. The output lists the bands
from the HIGHEST down. 'overlap fraction' widens each band past its
edges by that fraction of a band's width. Send the message report_bands
to print each band's low, centre and high frequency in the console.

What each band passes depends on filter type:
   bandpass   between its two edges (the default)
   bandstop   everything but its two edges' span
   lowpass    everything below its upper edge
   highpass   everything above its lower edge

SAMPLE FREQ MUST BE RIGHT:
The bands are in Hz only if 'sample freq' is the rate the samples really
arrive at. The demo assumes the patch runs at 60 frames a second; at
another rate every band lands in the wrong place, with no warning. high
is held below half the sample frequency.

SYNTAX:
t.filter_bank
No arguments.

EXAMPLE:
t.filter_bank

INPUTS and PARAMETERS:

signal:
A 2-D tensor (or array or list), [rows, columns]: one frame of values.
A single value must be shaped [1, 1]; a 1-D input prints an error in
the console and sends nothing. Each arrival is one sample for every
value.

reset (button):
On the next frame, fills all the filters' memory with that frame's
values, as if the signal had always been there. Use it to avoid the
start-up surge when a stream begins far from zero.

filter type (menu): bandpass, lowpass, highpass, bandstop. See above.

filter design (menu): butter (flat, the default), cheby1 (steeper, with
1 dB of ripple in the passband), cheby2 (steeper, with ripple in the
stopband, which is held at least 40 dB down - a hundredth of the size;
for cheby2 the band edges are where that rejection starts, so a band
passes less of its own range than with butter).

order (int, 1 to 8, default 1):
How steeply each band cuts off. Higher separates the bands more cleanly
and takes longer to settle.

low / high (Hz, defaults 0.01 and 12):
The range the bands cover. low must be above 0.

number of bands (int, default 8): the number of band edges - one more
than the number of bands.

band scaling (menu): log or linear.

overlap fraction (float, default 0).

sample freq (Hz, default 60).

Changing any setting rebuilds the filters (their memory starts empty).

device / dtype / requires_grad (options):
Where the filters run and their number type (float32).

OUTPUTS:

filtered:
[rows, columns, bands] - every value, once per band, highest band first.

READING THE DEMO:
The input is three sines added together: 0.5 Hz, 1.7 Hz and 6 Hz. With
low 0.3, high 12, 4 band edges (3 bands) and order 2 the bands are
about 3.5 to 12 Hz, 1 to 3.5 Hz and 0.3 to 1 Hz, and each picks out
one sine: measured, line 1 swings at 6 Hz, line 2 at 1.7 Hz, line 3 at
0.5 Hz, with a seventh to a fifth of the neighbouring sines leaking in.
Change filter type to lowpass and each line becomes everything below its
band's top: line 1 is then nearly the whole input. With highpass each
line is everything above its band's bottom, so it is line 3 that carries
nearly the whole input and line 1 only the 6 Hz sine. Change filter
design to cheby2 and the separation is much cleaner - the other sines
are held to about 1 percent - but the band edges then mark where that
rejection begins, so at order 2 each band passes its own sine at only
about 0.6 of its size.

RELATED:
spectrum and band_pass do this for plain numbers; t.rotation_band_diff
applies the same banding to joint rotations; t.rfft analyses a whole
block at once instead of sample by sample."""

demo = [
    {'key': 'mk', 'init': 'p make_mix', 'pos': (30, 62), 'w': 170, 'h': 70},
    {'key': 'cmk', 'comment': True,
     'text': 'three sines added: 0.5, 1.7 and 6 Hz\n(at 60 frames a second) - open it',
     'pos': (220, 62)},
    {'key': 'pin', 'init': 'plot', 'pos': (COL2, 62), 'w': 400, 'h': 170,
     'props': MULTI_PLOT(-2.0, 2.0, 300, 380, 130)},
    {'key': 'cin', 'comment': True, 'text': 'the input', 'pos': (COL2, 240)},
    {'key': 'rs', 'init': 't.reshape 1 1', 'pos': (30, 160), 'w': 180, 'h': 90,
     'props': {'shape': '1 1'}},
    {'key': 'crs', 'comment': True, 'text': 'one value: a [1, 1] tensor', 'pos': (230, 175)},
    {'key': 'f', 'init': 't.filter_bank', 'pos': (30, 280), 'w': 320, 'h': 330,
     'props': dict(TORCH_PROPS, **{'filter type': 'bandpass', 'filter design': 'butter',
                                   'order': 2, 'low': 0.3, 'high': 12.0,
                                   'number of bands': 4, 'band scaling': 'log',
                                   'overlap fraction': 0.0, 'sample freq': 60.0})},
    {'key': 'cf', 'comment': True, 'text': 'try filter type, order, number of bands',
     'pos': (COL2, 290)},
    {'key': 'pl', 'init': 'plot', 'pos': (COL2, 330), 'w': 400, 'h': 200,
     'props': MULTI_PLOT(-2.0, 2.0, 300, 380, 160)},
    {'key': 'cpl', 'comment': True,
     'text': 'the bands: 1 fastest (3.5-12 Hz), 2 middle,\n3 slowest (0.3-1 Hz)',
     'pos': (COL2, 540)},
] + see_also(660, ['spectrum', 'band_pass', 't.rotation_band_diff', 't.rfft', 'filter_bank'])
links = [('mk', 'mix', 'rs', 'tensor in'), ('mk', 'mix', 'pin', 'y'),
         ('rs', 'output', 'f', 'signal'), ('f', 'filtered', 'pl', 'y')]
print(build('t.filter_bank', 't.filter_bank - split streams into frequency bands',
            body, demo, links, demo_width=840, text_width=760,
            text_height=text_h(body), subpatch=MIX_SUB))

# --------------------------------------------------------- t.rotation_band_diff
body = """Breaks each joint's movement into timescales: for every joint and every
band, how much that joint turned this frame once the turning is smoothed
down to that band.

Use it on a whole pose stream to tell slow movement from fast in one
step - the swing of an arm and the tremor in it come out as different
numbers. One frame in gives [joints, bands, 3] out.

HOW IT WORKS:
Each joint's rotation is turned into a form that can be filtered (the
first two rows of its rotation matrix) and passed through a bank of
LOWPASS filters, one per band, each cutting off at its band's upper
edge. So band 1 keeps everything below about 8 Hz (nearly the raw
movement) and the last keeps only the slowest drift. Each filtered copy
is turned back into a rotation, and the node reports the rotation from
last frame's copy to this frame's, as an axis-angle vector (its length
is the angle in radians), multiplied by that band's GAIN. Because the
bands are lowpass, a fast tremor shows only in the first bands, while a
slow sway shows in all of them.

The gains make the slow bands, which move very little per frame, big
enough to see: by default 2.25, 2.25, 3.375, 5.06, 7.59, 11.39, 17.09
from the fastest band to the slowest (1.5 times more per band). Send
the message band_gains followed by one number per band to set your
own. The band edges follow t.filter_bank's rules: 'number of bands'
counts edges, so the default 8 gives 7 bands, highest first; report_bands
prints them.

SYNTAX:
t.rotation_band_diff
No arguments.

EXAMPLE:
t.rotation_band_diff

INPUTS and PARAMETERS:

rotations:
[joints, 3] axis-angle or [joints, 4] quaternions (w first). A flat
list of joints is split by 'input format'. Each arrival is one frame.
The first frame (and the first after reset) sends nothing: there is no
previous frame to compare with.

reset (button):
Empties the filters and forgets the previous frame. Unlike
t.filter_bank's reset it does not pre-fill the filters.

input format (menu: auto, axis-angle, quaternion):
How a FLAT input is split into joints: every 3 or every 4 numbers. auto
decides from the length, and gives up (with a console message) when the
length divides by both 3 and 4 - set it explicitly for SMPL poses of 72
values.

sample freq (Hz, default 60): the rate frames really arrive at.

number of bands (default 8): band edges, one more than the bands.

low / high (Hz, defaults 0.01 and 8): the range of the band edges.

order (1 to 8, default 1): how steeply each lowpass cuts.

band scaling (log or linear), overlap fraction (default 0): as on
t.filter_bank.

Changing a setting rebuilds the bank (and the gains return to default
if the number of bands no longer matches them).

OUTPUTS:

banded rotation diffs:
[joints, bands, 3]: per joint and band, this frame's turn as an
axis-angle vector, times the band's gain.

READING THE DEMO:
np.linalg.norm (axis 2) turns each vector into its length - how much the
joint turned in that band this frame - and heat_scroll draws the 7
bands as 7 rows scrolling left, the fastest band at the top. The jitter
and the glitches light up the top rows; the slow turning back and forth
glows in every row and fades where the turn reverses; the quick turns
flash through all of them.

RELATED:
t.filter_bank is the general band splitter for ordinary values.
quaternion_diff_and_axis measures frame-to-frame rotation without bands."""

demo = [
    {'key': 'mk', 'init': 'p make_orientation', 'pos': (30, 62), 'w': 220, 'h': 70},
    {'key': 'cmk', 'comment': True,
     'text': 'one joint turning back and forth, with jitter,\nquick turns and glitches',
     'pos': (270, 62)},
    {'key': 'f', 'init': 't.rotation_band_diff', 'pos': (30, 160), 'w': 320, 'h': 330,
     'props': dict(TORCH_PROPS, **{'input format': 'auto', 'sample freq': 60.0,
                                   'number of bands': 8, 'low': 0.01, 'high': 8.0,
                                   'order': 1, 'band scaling': 'log',
                                   'overlap fraction': 0.0})},
    {'key': 'nm', 'init': 'np.linalg.norm', 'pos': (COL2, 160), 'w': 170, 'h': 100,
     'props': {'axis': '2'}},
    {'key': 'cnm', 'comment': True, 'text': 'axis 2: the length of each\nvector - [1, 7]',
     'pos': (COL2 + 190, 170)},
    {'key': 'hs', 'init': 'heat_scroll', 'pos': (COL2, 290), 'w': 400, 'h': 200,
     'props': {'color': 'viridis', 'width': 380, 'height': 160, 'sample count': 200,
               'min y': 0.0, 'max y': 0.2, 'update_mode': 'heat_scroll',
               'number format': 'none'}},
    {'key': 'chs', 'comment': True,
     'text': '7 bands, fastest at the top; time scrolls left',
     'pos': (COL2, 500)},
] + see_also(560, ['t.filter_bank', 'quaternion_diff_and_axis', 't.quat_ESEKF',
                   'diff_filter'])
links = [('mk', 'quaternion', 'f', 'rotations'),
         ('f', 'banded rotation diffs', 'nm', 'input'), ('nm', 'norm', 'hs', 'y')]
print(build('t.rotation_band_diff', 't.rotation_band_diff - joint turning by timescale',
            body, demo, links, demo_width=840, text_width=760,
            text_height=text_h(body), subpatch=ORIENT_SUB))
