"""Regression tests for four node bugs found while writing the help patches.

Each one was silent: the patch loaded, validated and looked right, and the demo
simply sat at zero or threw downstream. Run in the project's conda env:

    python3 dpg_system/help/tools/test_node_fixes.py
"""
import sys, os

REPO = '/Users/drokeby/dpg_system'
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np
import dearpygui.dearpygui as dpg
from dpg_system.dpg_app import App

try:
    import torch
except ImportError:
    torch = None

FAILS = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label:56} {got!r}")
    if not ok:
        FAILS.append(f"{label}: got {got!r}, wanted {want!r}")


app = App()
app.register_nodes()
app.start()
for _ in range(3):
    dpg.render_dearpygui_frame()


def make(init):
    parts = init.split(' ')
    n = app.create_node_by_name_from_file(parts[0], [10, 10], parts[1:])
    for _ in range(2):
        dpg.render_dearpygui_frame()
    return n


def feed(node, value, inlet=None):
    """Push one value at a node's triggering inlet and return what it sent."""
    out = []
    orig = node.output.send
    node.output.send = lambda v, *a, **k: (out.append(v), orig(v, *a, **k))[1]
    (inlet or node.input).receive_data(value)
    node.execute()
    node.output.send = orig
    return out[-1] if out else None


# ---------------------------------------------------------------------------
print("ArithmeticNode: a float operand must survive an integer input")
# Narrowing it ran 0.5 through any_to_int, so '* 0.5' became '* 0' for good.
n = make('* 0.5')
check('* 0.5 <- int 4    output', feed(n, 4), 2.0)
check('* 0.5 <- int 4    operand kept', n.operand, 0.5)
check('* 0.5 <- int 5    output (not decayed)', feed(n, 5), 2.5)
check('* 0.25 <- np.int64 8', feed(make('* 0.25'), np.int64(8)), 2.0)
check('+ 0.5 <- bool True  operand kept', (lambda m: (feed(m, True), m.operand)[1])(make('+ 0.5')), 0.5)

print("\nArithmeticNode: widening and containers still conform")
m = make('* 2')
check('* 2 <- float 3.5  output', feed(m, 3.5), 7.0)
m = make('* 0.5')
check('* 0.5 <- ndarray  output', list(feed(m, np.array([2.0, 4.0]))), [1.0, 2.0])
check('* 0.5 <- ndarray  operand conformed', type(m.operand).__name__, 'ndarray')
if torch is not None:
    m = make('* 0.5')
    check('* 0.5 <- tensor   output', feed(m, torch.tensor([2.0, 6.0])).tolist(), [1.0, 3.0])
m = make('* 3')
check('* 3 <- int 4      stays integer', type(feed(m, 4)).__name__, 'int')

# ---------------------------------------------------------------------------
print("\nSignalNode: the first thing it emits must be a float")
# The 'on' inlet triggers execution, so an int 0 here reached arithmetic nodes
# before any real sample did, and narrowed their operands.
s = make('signal')
check('signal.signal_value type', type(s.signal_value).__name__, 'float')

# ---------------------------------------------------------------------------
print("\nDifferentiateNode: the first sample must not be a 0-d array")
# np.zeros_like(0.5) has shape (), and plot does incoming.shape[0] on it.
d = make('diff')
first = feed(d, 1.5)
check('diff first output is scalar', np.ndim(first), 0)
check('diff first output is not ndarray', isinstance(first, np.ndarray), False)
check('diff second output', feed(d, 2.0), 0.5)

# ---------------------------------------------------------------------------
print("\nrandom.*: each parameter must read its own argument")
# arg_as_number defaults to index=0, so every parameter read the FIRST argument:
# 'random.gauss 0 1' was gauss(0, 0) and emitted 0.0 forever.
g = make('random.gauss 0.0 1.0')
check('random.gauss mean inlet', g.mean(), 0.0)
check('random.gauss dev inlet (not 0)', g.dev(), 1.0)
vals = []
_orig = g.output.send
g.output.send = lambda v, *a, **k: (vals.append(v), _orig(v, *a, **k))[1]
for _ in range(60):
    g.trigger_input.receive_data('bang')
    g.execute()
g.output.send = _orig
check('random.gauss emits something', len(vals), 60)
check('random.gauss is not stuck at one value', len(set(vals)) > 1, True)
check('random.gauss is not all zero', any(v != 0 for v in vals), True)

t3 = make('random.triangular -1.0 1.0 0.0')
check('random.triangular low', t3.low(), -1.0)
check('random.triangular high', t3.high(), 1.0)
check('random.triangular mode', t3.mode(), 0.0)

gm = make('random.gammavariate 2.0 0.5')
check('random.gammavariate alpha', gm.alpha(), 2.0)
check('random.gammavariate beta', gm.beta(), 0.5)

# ---------------------------------------------------------------------------
print("\nknob: restoring a changed option must not crash it out of the patch")
# ValueNode gives knobs no width option (DPG's knob_float is fixed size), but
# options_changed called width_option() regardless -- so any saved knob whose
# min/max/format had been touched raised during load and was dropped.
k = make('knob 0.5')
container = {}
k.save(container, 0)
for pc in container.get('properties', {}).values():
    if pc.get('name') == 'max':
        pc['value'] = 2.0            # force a real (non no-op) restore
k2 = make('knob 0.5')
try:
    k2.load(container)
    check('knob survives an option restore', True, True)
except Exception as e:
    check(f'knob survives an option restore ({e})', False, True)

# ---------------------------------------------------------------------------
print("\nquaternion_to_matrix must accept a NumPy quaternion")
# Its guard let ndarray through unconverted while the branch below handled only
# torch.Tensor, so a quaternion straight out of euler_to_quaternion vanished.
import numpy as _np
eq = make('euler_to_quaternion')
q = feed(eq, [30.0, 0.0, 0.0])
check('euler_to_quaternion emits ndarray', isinstance(q, _np.ndarray), True)
qm = make('quaternion_to_matrix')
check('quaternion_to_matrix <- ndarray', feed(qm, q) is not None, True)
check('quaternion_to_matrix <- list', feed(qm, [1.0, 0.0, 0.0, 0.0]) is not None, True)

# ---------------------------------------------------------------------------
print("\ngl_align must be creatable")
# Its __init__ set self.axis AFTER super().__init__(), but initialize() runs
# inside that call and reads it -- so the node raised and could not be made.
check('gl_align creates', make('gl_align') is not None, True)

# ---------------------------------------------------------------------------
print("\nsmpl_body must survive betas arriving with no model loaded")
# receive_betas dereferenced self.smpl_model unconditionally; a beta editor
# wired into it failed the whole patch load with AttributeError.
sb = make('smpl_body')
try:
    sb.betas.receive_data([0.0] * 16)
    check('smpl_body accepts betas with no model', True, True)
except Exception as e:
    check(f'smpl_body accepts betas with no model ({e})', False, True)

# ---------------------------------------------------------------------------
print("\nmidi nodes must be creatable with no MIDI hardware attached")
# in_port is None when no input port exists. add_client() was called on it
# unconditionally, so these three could not be created away from the hardware.
for _label in ('midi_device', 'mpd218', 'blue_board'):
    check(f'{_label} creates with no ports', make(_label) is not None, True)

# ---------------------------------------------------------------------------
print("\nkornia nodes must agree with each other about dtype and rank")
import torch as _t
# k.rgb_to_grayscale was the only node in that file not calling .float(), so a
# uint8 image came back uint8 with the weighted sum truncated.
_gs = make('k.rgb_to_grayscale')
_u8 = (_t.rand(3, 16, 16) * 255).to(_t.uint8)
_r = feed(_gs, _u8)
check('k.rgb_to_grayscale floats a uint8 image',
      _r is not None and _r.dtype == _t.float32, True)
# k.apply_colormap alone leaked the batch dim kornia adds, sending 4-D where
# every sibling sends CHW.
_cm = make('k.apply_colormap')
_r = feed(_cm, _t.rand(1, 16, 16))
check('k.apply_colormap sends CHW, not a batch',
      _r is not None and tuple(_r.shape) == (3, 16, 16), True)

# ---------------------------------------------------------------------------
print("\nultracwt must build its wavelet bank")
# Both nodes built their bank from scipy.signal.morlet2, which SciPy REMOVED in
# 1.15. The constructor guards caught it, so the nodes stayed silent and
# produced nothing at all rather than raising. A local morlet2 restores them.
for _label in ('t.ultracwt', 'ultracwt'):
    _u = make(_label)
    check(f'{_label} has a wavelet bank',
          _u is not None and getattr(_u, 'cwt', None) is not None, True)
# and a list of scales must not raise -- it used to go into re.findall
_u = make('t.ultracwt')
try:
    [i for i in _u.inputs if i.get_label() == 'scales'][0].receive_data([2.0, 4.0, 8.0])
    check('t.ultracwt accepts a list of scales', True, True)
except Exception as _e:
    check(f't.ultracwt accepts a list of scales ({type(_e).__name__})', False, True)

# ---------------------------------------------------------------------------
print("\ntranslate must work without the Google Cloud SDK")
# The Cloud imports were at module level, so a machine without the SDK lost the
# whole module -- including 'translate', which needs nothing but requests.
check('translate creates without the Cloud SDK', make('translate') is not None, True)

# ---------------------------------------------------------------------------
print("\nplot must answer 'dump' like its siblings")
# HeatMapNode and ProfileNode both handed their buffer out on 'dump'; PlotNode
# did not, so plot's outlet could never fire at all.
for _label, _want in (('plot', 200), ('heat_scroll', 200)):
    _p = make(_label)
    _dumped = []
    _p.output.send = lambda v, *a, **k: _dumped.append(v)
    for _i in range(5):
        _p.input.receive_data(float(_i)); _p.active_input = _p.input; _p.execute()
    check(f'{_label} sends nothing unasked', len(_dumped), 0)
    _p.input.receive_data('dump'); _p.active_input = _p.input; _p.execute()
    check(f'{_label} dumps its buffer',
          bool(_dumped) and len(_dumped[-1]) == _want, True)

# ---------------------------------------------------------------------------
print("\ncolor_source must build, and send only what changed")
# Three separate faults: it did not inherit OSCBase, so OSCSender.__init__ could
# not reach osc_manager and 'color_source 7' raised AttributeError; it wired a
# widget to a nonexistent address_changed; and its dirty flags were named after
# the callback methods, so before a slider had moved the flag WAS the bound
# method -- truthy -- and the first change sent all five parameters, intensity 0
# included, which blacks out the channel.
_cs = make('color_source 7')
check('color_source builds with a channel argument', _cs is not None, True)
if _cs is not None:
    _msgs = []
    class _FakeTarget:
        def send_message(self, addr, val): _msgs.append((addr, val))
    _cs.target = _FakeTarget()
    _cs.red_input.set(80)
    _cs.red_changed()
    _cs.frame_task()
    check('moving red sends exactly one message', len(_msgs), 1)
    check('and it is the red one on the right path',
          _msgs[0][0] if _msgs else None, '/eos/user/99/chan/7/param/red')

# ---------------------------------------------------------------------------
print("\nt.cross_entropy_loss must accept class indices")
# The target was coerced to the input's dtype via match_tensor, but class
# indices must be int64 -- so the normal way to use this loss always failed
# with 'expected scalar type Long but found Float'.
_ce = make('t.cross_entropy_loss')
_ce_got = []
_ce.loss_output.send = lambda v, *a, **k: _ce_got.append(v)
_ce.target_input.receive_data(0)
_ce.input.receive_data([2.0, 0.5, 0.1])
_ce.active_input = _ce.input
_ce.execute()
check('cross entropy returns a loss for a class index',
      bool(_ce_got) and isinstance(_ce_got[-1], float) and _ce_got[-1] > 0, True)

# ---------------------------------------------------------------------------
print("\nvision_describe nodes must create, and SmolVLM must find its model class")
for _label in ('vision_describe', 'vision_describe_smol',
               'vision_describe_qwen', 'vision_describe_gemma'):
    check(f'{_label} creates', make(_label) is not None, True)
# SmolVLM imported AutoModelForVision2Seq, which transformers 5 removed -- every
# inference died with ImportError. The node now falls back across both names.
try:
    try:
        from transformers import AutoModelForImageTextToText as _VLMClass
    except ImportError:
        from transformers import AutoModelForVision2Seq as _VLMClass
    check('a vision2seq model class is importable', _VLMClass is not None, True)
except ImportError as _e:
    check(f'a vision2seq model class is importable ({_e})', False, True)

# ---------------------------------------------------------------------------
print("\nbuffer must survive a resize while something holds its output")
# BufferNode called ndarray.resize() in place. numpy refuses that once another
# object references the array -- and the node SENDS the buffer itself, so any
# downstream connection is such a reference. Changing sample count or update
# style then raised ValueError and the node stopped working.
_bf = make('buffer 8')
_held = []
_bf.output.send = lambda v, *a, **k: _held.append(v)   # a consumer, as a patch would have
_bf.update_style_option.set('input is stream of samples')
_bf.update_style_changed()
_bf.output_style_option.set('output buffer on every input')
_bf.output_style_changed()
_bf.input.receive_data([1.0, 2.0, 3.0])
try:
    _bf.execute()
    _bf.sample_count_option.set(32)      # force the resize path, output held
    _bf.input.receive_data([4.0, 5.0])
    _bf.execute()
    check('buffer resizes with its output held', _held[-1].shape[0], 32)
except Exception as _e:
    check(f'buffer resizes with its output held ({type(_e).__name__})', False, True)

# ---------------------------------------------------------------------------
print("\npc_crop must attach its volume to the frame")
# The whole point-cloud chain depends on this: pc_crop turns a raw (N,3) array
# into a cloud frame carrying the crop, and every grid node downstream uses that
# volume rather than its own fallback widgets.
import numpy as _np2
_cr = make('pc_crop')
_cr.min_input.receive_data([-1.0, -1.0, 1.0])
_cr.max_input.receive_data([1.0, 1.0, 3.0])
_pts = _np2.array([[0, 0, 2.0], [0, 0, 5.0], [0.5, 0.5, 2.5]], dtype=_np2.float32)
_out = feed(_cr, _pts)
check('pc_crop emits a cloud frame dict', isinstance(_out, dict), True)
check('pc_crop attaches the crop volume',
      isinstance(_out, dict) and 'crop' in _out, True)
check('pc_crop drops the out-of-box point',
      isinstance(_out, dict) and _out['point_cloud'].shape[0] == 2, True)

# ---------------------------------------------------------------------------
print("\ntrigger must not silently drop booleans")
# The scalar branch tested `type(data) in [float, np.double, int, np.int64]`.
# type(True) is bool, which is not in that list even though bool subclasses
# int, so a boolean fell through every branch and produced nothing -- which is
# how speech_envelope's 'onset' reached trigger and did nothing at all.
_tr = make('trigger')
_fired = []
_tr.output.send = lambda v, *a, **k: _fired.append(v)
for _v in (False, False, True, False):
    _tr.input.receive_data(_v)
    _tr.execute()
check('trigger fires on a boolean', len(_fired) > 0, True)

# ---------------------------------------------------------------------------
print("\ntorchvision adjust nodes must all dispatch to a real op")
# The if/elif chain that picked self.op compared self.label against
# 'torchvision.*' while the nodes register as 'tv.*', so no branch ever matched
# and every one of these crashed on construction.
for _label in ('tv.adjust_hue', 'tv.adjust_saturation', 'tv.adjust_sharpness',
               'tv.adjust_contrast', 'tv.adjust_brightness'):
    _n = make(_label)
    check(f'{_label} creates with an op',
          _n is not None and getattr(_n, 'op', None) is not None, True)

# ---------------------------------------------------------------------------
print("\nsocket nodes must be creatable")
# socket_nodes used os.environ but never imported os, so process_group raised
# NameError; udp_numpy_send passed width= to add_input, which takes
# widget_width, so width collided in **kwargs and it raised TypeError.
for _label in ('udp_numpy_send', 'udp_numpy_receive'):
    check(f'{_label} creates', make(_label) is not None, True)

# ---------------------------------------------------------------------------
print()
if FAILS:
    print(f"{len(FAILS)} FAILURES")
    for f in FAILS:
        print('  -', f)
    sys.exit(1)
print("all node fix regressions pass")
