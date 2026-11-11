"""neural_synth: the 2018 NeuralSynth feature visualiser as a live stream.

The synth grows an image of what one feature channel of a network layer
responds to, octave by octave, and outputs a frame as it develops (every
'frame every' gradient steps, plus the end of every octave), so the
development itself is the output.

The work runs on a worker thread (dpg_system.neural_synth.engine); frames
are rendered there and handed to the main thread, which sends them from
frame_task. With 'drop frames' off the worker waits for each frame to be sent,
so a recording downstream gets every step.

Targets: 'layer' and 'feature' set the next target. 'octave schedule' takes a
list that sets the target per octave (the last entry holds): a list of ints is
features on the current layer; a list of [layer, feature] pairs (or a flat
list alternating layer names and features) sets both. An empty list or
'clear' returns to the single layer / feature. When a new target takes effect
is 'apply': next image, next octave, or next step. While holding a finished
image, a new target starts a new image.

'at end': hold the finished image; repeat the target; or advance to the next
feature of the layer.
"""
import collections
import threading
import time
import traceback

import numpy as np
import torch

from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.neural_synth.engine import SynthEngine, default_settings, best_device
from dpg_system.neural_synth.models import MODEL_NAMES, DEFAULT_EXIT, model_exits


def register_neural_synth_nodes():
    Node.app.register_node('neural_synth', NeuralSynthNode.factory)


def _device_list():
    devices = ['cpu']
    try:
        if torch.backends.mps.is_available():
            devices.append('mps')
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            devices.append('cuda')
    except Exception:
        pass
    return devices


def _parse_floats(text, n=5):
    try:
        vals = [float(v) for v in any_to_string(text).replace(',', ' ').split()]
    except ValueError:
        return None
    return tuple(vals[:n]) if len(vals) >= n else None


class NeuralSynthNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        return NeuralSynthNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.settings = default_settings()
        model = 'inception5h'
        if args:
            for a in args:
                a = any_to_string(a)
                if a in MODEL_NAMES:
                    model = a
        self.settings.model = model
        self.exit_names, self.exit_counts = model_exits(model)
        s = self.settings

        self.engine = SynthEngine(self.settings)
        self.schedule = None              # per-octave target list, or None

        # worker state
        self._cond = threading.Condition()
        self._alive = True
        self._running = False
        self._restart = False
        self._idle = True                 # no image in progress (holding / never started)
        self._wake = False                # a new target or restart while idle
        self._frames = collections.deque()
        self._frame_counter = 0
        self._thread = None

        self.run_input = self.add_input('run', widget_type='checkbox', default_value=False, callback=self.run_changed)
        self.restart_input = self.add_input('restart', widget_type='button', callback=self.restart_pressed)
        self.layer_input = self.add_input('layer', widget_type='combo', widget_width=220,
                                          default_value=DEFAULT_EXIT[model], callback=self.target_changed)
        self.layer_input.widget.combo_items = list(self.exit_names)
        self.feature_input = self.add_input('feature', widget_type='drag_int', default_value=0, min=0,
                                            callback=self.target_changed)
        self.schedule_input = self.add_input('octave schedule', callback=self.schedule_received)
        self.apply_input = self.add_input('apply', widget_type='combo', widget_width=110,
                                          default_value='next image', callback=self.apply_changed)
        self.apply_input.widget.combo_items = ['next image', 'next octave', 'next step']
        self.at_end_input = self.add_input('at end', widget_type='combo', widget_width=110,
                                           default_value='next feature', callback=self.at_end_changed)
        self.at_end_input.widget.combo_items = ['hold', 'repeat', 'next feature']
        self.status_label = self.add_label('')

        self.frame_output = self.add_output('frame')
        self.status_output = self.add_output('status')
        self.done_output = self.add_output('done')

        # options
        self.model_option = self.add_option('model', widget_type='combo', width=110, default_value=model,
                                            callback=self.model_changed)
        self.model_option.widget.combo_items = list(MODEL_NAMES)
        self.device_option = self.add_option('device', widget_type='combo', width=80, default_value=best_device(),
                                             callback=self.settings_changed)
        self.device_option.widget.combo_items = _device_list()
        self.frame_every_option = self.add_option('frame every', widget_type='drag_int', default_value=1, min=1)
        self.drop_frames_option = self.add_option('drop frames', widget_type='checkbox', default_value=False)

        opt = self._add_setting_option
        # image (fixed when an image starts)
        opt('width', 'drag_int', 'width', 1)
        opt('height', 'drag_int', 'height', 1)
        opt('octaves', 'drag_int', 'octaves', 1)
        opt('iterations', 'drag_int', 'iter_base', 1)
        opt('iteration factor', 'drag_float', 'iter_factor', 0.01)
        opt('seed', 'drag_int', 'seed', None)
        self.seed_mode_option = opt('seed mode', 'combo', 'seed_mode', None, ['fixed', 'random'])
        opt('noise amplitude', 'drag_float', 'noise_amplitude', 0.0)
        # gradient
        opt('objective', 'combo', 'objective', None, ['sum', 'square', 'all'])
        opt('roll', 'drag_float', 'roll', 0.0)
        opt('roll mode', 'combo', 'roll_mode', None, ['zero', 'wrap'])
        opt('pre tanh gain', 'drag_float', 'pre_tanh_gain', 0.0)
        opt('lrn size', 'drag_int', 'lrn', 1)
        # shaping
        opt('band levels', 'drag_int', 'lap_levels', 0)
        opt('band padding', 'combo', 'band_padding', None, ['zero', 'replicate'])
        opt('desaturate', 'combo', 'desaturate', None, ['row mean', 'luma'])
        opt('octave scaler', 'drag_float', 'octave_scaler', 0.01)
        opt('sat elbow', 'drag_float', 'sat_elbow', 0.0)
        opt('base sat', 'drag_float', 'base_sat', 0.0)
        opt('band weights start', 'text_input', 'lap_weights_start', None)
        opt('band weights end', 'text_input', 'lap_weights_end', None)
        opt('lerp band weights', 'checkbox', 'lerp_lap_weights', None)
        # dual path
        opt('dual path', 'checkbox', 'dual_path', None)
        opt('out octave scaler', 'drag_float', 'out_octave_scaler', 0.01)
        opt('out sat elbow', 'drag_float', 'out_sat_elbow', 0.0)
        opt('out base sat', 'drag_float', 'out_base_sat', 0.0)
        opt('out band weights start', 'text_input', 'out_lap_weights_start', None)
        opt('out band weights end', 'text_input', 'out_lap_weights_end', None)
        opt('out lerp band weights', 'checkbox', 'out_lerp_lap_weights', None)
        # display
        opt('norm scale', 'drag_int', 'norm_scale', 1)
        opt('norm smoothing', 'drag_float', 'norm_sigma', 0.0)
        opt('norm gamma', 'drag_float', 'norm_gamma', 0.01)
        opt('norm min range', 'drag_float', 'norm_min_range', 0.0)
        opt('octave fade', 'drag_int', 'octave_fade', 0)

        self.engine.set_target([DEFAULT_EXIT[model]], [0])

    # ---- settings options ---------------------------------------------
    def _add_setting_option(self, label, widget, attr, minimum, items=None):
        """An option bound to one engine setting."""
        value = getattr(self.settings, attr)
        if attr == 'lrn':
            value = value[0]
        elif isinstance(value, tuple):
            value = ' '.join('%g' % v for v in value)
        kwargs = dict(widget_type=widget, default_value=value, callback=self.settings_changed)
        if minimum is not None and widget in ('drag_int', 'drag_float'):
            kwargs['min'] = minimum
        o = self.add_option(label, **kwargs)
        if items is not None:
            o.widget.combo_items = items
        o._ns_attr = attr
        if not hasattr(self, '_setting_options'):
            self._setting_options = []
        self._setting_options.append(o)
        return o

    def settings_changed(self):
        s = self.settings
        for o in getattr(self, '_setting_options', []):
            v = o()
            if v is None:
                continue
            attr = o._ns_attr
            cur = getattr(s, attr)
            if attr == 'lrn':
                s.lrn = (max(1, any_to_int(v)),) + tuple(s.lrn[1:])
            elif isinstance(cur, tuple):
                parsed = _parse_floats(v)
                if parsed is not None:
                    setattr(s, attr, parsed)
            elif isinstance(cur, bool):
                setattr(s, attr, any_to_bool(v))
            elif isinstance(cur, int):
                setattr(s, attr, any_to_int(v))
            elif isinstance(cur, float):
                setattr(s, attr, any_to_float(v))
            else:
                setattr(s, attr, any_to_string(v))
        device = self.device_option()
        if device:
            s.device = any_to_string(device)

    # ---- lifecycle ------------------------------------------------------
    def custom_create(self, from_file):
        self.settings_changed()
        self.add_frame_task()
        self._thread = threading.Thread(target=self._worker, name='neural_synth', daemon=True)
        self._thread.start()
        if any_to_bool(self.run_input()):
            self.run_changed()

    def post_load_callback(self):
        self.model_changed()
        self.settings_changed()
        self.target_changed()
        self.run_changed()

    def custom_cleanup(self):
        with self._cond:
            self._alive = False
            self._cond.notify_all()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)

    # ---- controls -------------------------------------------------------
    def run_changed(self):
        if self.in_loading_process:
            # restore order is not settings-first: starting here could begin
            # an image at the default size before width/height/octaves arrive.
            # post_load_callback starts it once everything is restored.
            return
        with self._cond:
            self._running = any_to_bool(self.run_input())
            if self._running and self._idle and self.engine.info is None:
                self._wake = True        # first start
            self._cond.notify_all()

    def restart_pressed(self):
        with self._cond:
            self._restart = True
            self._wake = True
            self._cond.notify_all()

    def apply_changed(self):
        v = self.apply_input()
        if v:
            self.settings.apply = any_to_string(v)

    def at_end_changed(self):
        pass

    def model_changed(self):
        model = any_to_string(self.model_option())
        if model not in MODEL_NAMES:
            return
        self.exit_names, self.exit_counts = model_exits(model)
        self.layer_input.widget.combo_items = list(self.exit_names)
        try:
            import dearpygui.dearpygui as dpg
            if dpg.does_item_exist(self.layer_input.widget.uuid):
                dpg.configure_item(self.layer_input.widget.uuid, items=self.exit_names)
        except Exception:
            pass
        if self.layer_input() not in self.exit_counts:
            self.layer_input.set(DEFAULT_EXIT[model])
        self.settings.model = model
        self.target_changed()

    def target_changed(self):
        layer = any_to_string(self.layer_input())
        if layer not in self.exit_counts:
            layer = DEFAULT_EXIT[self.settings.model]
        feature = any_to_int(self.feature_input())
        if self.schedule:
            layers = [l if l in self.exit_counts else layer for l, _ in self.schedule]
            features = [f for _, f in self.schedule]
        else:
            layers, features = [layer], [feature]
        self.engine.set_target(layers, features)
        if not self.in_loading_process:
            with self._cond:
                if self._idle:
                    self._wake = True
                self._cond.notify_all()

    def schedule_received(self):
        data = self.schedule_input()
        layer = any_to_string(self.layer_input())
        sched = []
        if isinstance(data, str):
            data = data.split()
        if isinstance(data, np.ndarray):
            data = data.tolist()
        if isinstance(data, (list, tuple)):
            if len(data) == 1 and isinstance(data[0], str) and data[0] == 'clear':
                data = []
            pending_layer = layer
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    sched.append((any_to_string(item[0]), any_to_int(item[1])))
                    pending_layer = any_to_string(item[0])
                elif isinstance(item, str) and not item.lstrip('-').isdigit():
                    pending_layer = item
                else:
                    sched.append((pending_layer, any_to_int(item)))
        elif data is not None:
            sched = [(layer, any_to_int(data))]
        self.schedule = sched or None
        self.target_changed()

    # ---- worker thread --------------------------------------------------
    def _wait(self, predicate):
        with self._cond:
            while self._alive and not predicate():
                self._cond.wait(0.25)
            return self._alive

    def _set_status(self, text):
        self._status_text = text

    def _worker(self):
        engine = self.engine
        while self._alive:
            # wait for something to do
            if not self._wait(lambda: self._running and self._wake):
                return
            with self._cond:
                self._wake = False
                self._restart = False
                self._idle = False
            completed = False
            try:
                self._set_status('loading ' + self.settings.model + ' on ' + self.settings.device)
                for info in engine.run_image():
                    if not self._wait(lambda: self._running or self._restart):
                        return
                    if self._restart:
                        break
                    every = max(1, any_to_int(self.frame_every_option()))
                    self._frame_counter += 1
                    if info.octave_end or self._frame_counter >= every:
                        self._frame_counter = 0
                        frame = engine.render_frame()
                        self._push_frame(frame, info)
                    if info.image_end:
                        completed = True
            except Exception as e:
                traceback.print_exc()
                self._set_status('error: ' + str(e))
                with self._cond:
                    self._idle = True
                    self._running = False
                continue

            with self._cond:
                if self._restart:
                    self._wake = True
                    continue
                if not completed:
                    self._idle = True
                    continue
                mode = any_to_string(self.at_end_input())
                if engine.pending is not None:
                    self._wake = True            # a queued target starts the next image
                elif mode == 'repeat':
                    self._wake = True
                elif mode == 'next feature':
                    engine.advance_feature(1)
                    self._wake = True
                else:
                    self._idle = True

    def _push_frame(self, frame, info):
        item = (frame, info)
        drop = any_to_bool(self.drop_frames_option())
        with self._cond:
            if drop:
                # keep the newest frame, but never lose an image's last frame
                while self._frames and not self._frames[-1][1].image_end:
                    self._frames.pop()
                self._frames.append(item)
                return
            while self._alive and not self._restart and len(self._frames) >= 2:
                self._cond.wait(0.25)
            self._frames.append(item)

    # ---- main thread ----------------------------------------------------
    def frame_task(self):
        with self._cond:
            if not self._frames:
                item = None
            else:
                item = self._frames.popleft()
                self._cond.notify_all()
            idle = self._idle
        if item is not None:
            frame, info = item
            self.status_label.set('image %d  octave %d/%d  step %d/%d  %s #%d  %dx%d' % (
                info.image, info.octave + 1, info.octaves, info.total_step, info.total_steps,
                info.layer, info.feature, info.size[1], info.size[0]))
            self.status_output.send([info.image, info.octave, info.octaves, info.step, info.steps,
                                     info.total_step, info.total_steps, info.layer, info.feature,
                                     info.size[1], info.size[0]])
            self.frame_output.send(frame)
            if info.image_end:
                self.done_output.send('bang')
        else:
            text = getattr(self, '_status_text', None)
            if text is not None:
                self._status_text = None
                self.status_label.set(text)

    def execute(self):
        pass
