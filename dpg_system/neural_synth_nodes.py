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

'direction' takes an activation-atlas direction (from neural_atlas, or a raw
list with one weight per channel of the current layer) and grows that
instead of a single channel. Touching 'layer' or 'feature' goes back to
channels.
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
from dpg_system.neural_synth.atlas import Direction, LAYERS as ATLAS_LAYERS, GRIDS as ATLAS_GRIDS, load_atlas


def register_neural_synth_nodes():
    Node.app.register_node('neural_synth', NeuralSynthNode.factory)
    Node.app.register_node('neural_atlas', NeuralAtlasNode.factory)


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
        self.direction = None             # atlas Direction replacing layer / feature
        self._start_after_load = False

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
                                          default_value=DEFAULT_EXIT[model], callback=self.channel_changed)
        self.layer_input.widget.combo_items = list(self.exit_names)
        self.feature_input = self.add_input('feature', widget_type='drag_int', default_value=0, min=0,
                                            callback=self.channel_changed)
        self.schedule_input = self.add_input('octave schedule', callback=self.schedule_received)
        self.direction_input = self.add_input('direction', callback=self.direction_received)
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
        opt('direction cossim', 'drag_float', 'direction_cossim', 0.0)
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
        # start on the first frame after the load, so that targets other
        # nodes send from their own post_load_callback (neural_atlas) are
        # in place before the first image begins
        self._start_after_load = True

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

    def channel_changed(self):
        self.direction = None
        self.target_changed()

    def direction_received(self):
        data = self.direction_input()
        if isinstance(data, Direction):
            d = data
        else:
            if isinstance(data, str) and data.strip() in ('', 'clear'):
                self.direction = None
                self.target_changed()
                return
            try:
                v = np.asarray(any_to_array(data), dtype=np.float32).reshape(-1)
            except Exception:
                return
            layer = any_to_string(self.layer_input())
            if self.exit_counts.get(layer) != v.shape[0]:
                self.status_label.set('direction has %d values; %s has %s channels' % (
                    v.shape[0], layer, self.exit_counts.get(layer)))
                return
            n = float(np.linalg.norm(v))
            if n == 0:
                return
            d = Direction(layer, v / n, layer + ' direction')
        if d.layer not in self.exit_counts or self.exit_counts[d.layer] != d.vector.shape[0]:
            self.status_label.set('direction is for %s, not this model' % d.layer)
            return
        self.direction = d
        self.target_changed()

    def target_changed(self):
        layer = any_to_string(self.layer_input())
        if layer not in self.exit_counts:
            layer = DEFAULT_EXIT[self.settings.model]
        feature = any_to_int(self.feature_input())
        if self.schedule:
            layers = [l if l in self.exit_counts else layer for l, _ in self.schedule]
            features = [f for _, f in self.schedule]
        elif self.direction is not None:
            layers, features = [self.direction.layer], [self.direction]
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
                if isinstance(item, Direction):
                    sched.append((item.layer, item))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
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
        if self._start_after_load and not self.in_loading_process:
            self._start_after_load = False
            self.run_changed()
        with self._cond:
            if not self._frames:
                item = None
            else:
                item = self._frames.popleft()
                self._cond.notify_all()
            idle = self._idle
        if item is not None:
            frame, info = item
            if isinstance(info.feature, Direction):
                target, feature = info.feature.name, info.feature.name
            else:
                target, feature = '%s #%d' % (info.layer, info.feature), info.feature
            self.status_label.set('image %d  octave %d/%d  step %d/%d  %s  %dx%d' % (
                info.image, info.octave + 1, info.octaves, info.total_step, info.total_steps,
                target, info.size[1], info.size[0]))
            self.status_output.send([info.image, info.octave, info.octaves, info.step, info.steps,
                                     info.total_step, info.total_steps, info.layer, feature,
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


class NeuralAtlasNode(Node):
    """neural_atlas: pick directions to grow from an activation atlas.

    An atlas maps 100,000 activations of one Inception5h layer, taken at
    random spots of ImageNet photographs, onto a grid: similar activations
    land in nearby cells, and each cell's average is a direction neural_synth
    can grow (see neural_synth/atlas.py). Cells are chosen by position
    ('cell'), by class name ('find'), or by stepping ('next', as set by
    'walk'). The chosen cell's direction goes out of 'direction' for
    neural_synth's 'direction' input.
    """
    WALKS = ['neighbour', 'snake', 'random', 'matches']

    @staticmethod
    def factory(name, data, args=None):
        return NeuralAtlasNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        layer = 'mixed4c'
        if args:
            for a in args:
                a = any_to_string(a)
                if a in ATLAS_LAYERS:
                    layer = a
        self.atlas = None
        self.k = None                    # current cell index
        self.matches = []
        self.match_pos = 0
        self.match_text = None
        self.visited = collections.deque(maxlen=12)
        self.rng = np.random.default_rng()

        self.cell_input = self.add_input('cell', callback=self.cell_received)
        self.find_input = self.add_input('find', widget_type='text_input', widget_width=160,
                                         default_value='', callback=self.find_received)
        self.next_input = self.add_input('next', widget_type='button', callback=self.next_pressed)
        self.layer_input = self.add_input('layer', widget_type='combo', widget_width=110,
                                          default_value=layer, callback=self.layer_changed)
        self.layer_input.widget.combo_items = list(ATLAS_LAYERS)
        self.grid_input = self.add_input('grid', widget_type='combo', widget_width=60,
                                         default_value=str(ATLAS_GRIDS[1]), callback=self.grid_changed)
        self.grid_input.widget.combo_items = [str(g) for g in ATLAS_GRIDS]
        self.walk_input = self.add_input('walk', widget_type='combo', widget_width=110,
                                         default_value='neighbour')
        self.walk_input.widget.combo_items = list(self.WALKS)
        self.status_label = self.add_label('')

        self.direction_output = self.add_output('direction')
        self.cell_output = self.add_output('cell')
        self.classes_output = self.add_output('classes')

        self.whiten_option = self.add_option('whiten', widget_type='checkbox', default_value=True,
                                             callback=self.whiten_changed)
        self.current_option = self.add_option('current cell', widget_type='text_input', width=80,
                                              default_value='', callback=self.current_edited)

    # ---- atlas ----------------------------------------------------------
    def _grid(self):
        g = any_to_int(self.grid_input())
        return g if g in ATLAS_GRIDS else ATLAS_GRIDS[1]

    def _load(self):
        layer = any_to_string(self.layer_input())
        if self.atlas is not None and self.atlas.layer == layer:
            return self.atlas
        try:
            self.atlas = load_atlas(layer)
        except FileNotFoundError:
            self.atlas = None
            self.status_label.set('no atlas for %s: python -m dpg_system.neural_synth.atlas %s'
                                  % (layer, layer))
        except Exception as e:
            self.atlas = None
            self.status_label.set('atlas error: ' + str(e))
        return self.atlas

    def _cell_xy(self):
        return self._cell_xy_on(self._grid())

    def _cell_xy_on(self, grid):
        """Position of the current cell; `grid` is the grid it was chosen on."""
        if self.k is None or self.atlas is None or grid not in self.atlas.cells:
            return None
        xy = self.atlas.cells[grid]['xy']
        if self.k >= len(xy):
            return None
        return int(xy[self.k][0]), int(xy[self.k][1])

    def _select(self, k, send=True):
        a = self.atlas
        if a is None or k is None:
            return
        g = self._grid()
        self.k = int(k)
        self.visited.append(self.k)
        x, y = self._cell_xy()
        tops = a.top_classes(g, self.k, 3)
        classes = ', '.join(name.split(',')[0] for name, _ in tops)
        n = int(a.cells[g]['count'][self.k])
        self.status_label.set('%s %d:%d,%d  %s  (%d samples)' % (a.layer, g, x, y, classes, n))
        self.current_option.set('%d %d' % (x, y))
        if not send:
            return
        self.classes_output.send(classes)
        self.cell_output.send([x, y])
        self.direction_output.send(a.direction(g, self.k, any_to_bool(self.whiten_option())))

    def _reselect_at(self, xy, send=True):
        """After a layer or grid change: the cell at the same place on the map."""
        a = self._load()
        if a is None:
            return
        g = self._grid()
        if xy is None:
            xy = (g // 2, g // 2)
        self.matches, self.match_text = [], None
        self._select(a.nearest(g, *xy), send)

    # ---- inputs ---------------------------------------------------------
    def cell_received(self):
        a = self._load()
        if a is None:
            return
        data = any_to_list(self.cell_input())
        if len(data) < 2:
            return
        self._select(a.nearest(self._grid(), any_to_int(data[0]), any_to_int(data[1])))

    def find_received(self):
        a = self._load()
        if a is None:
            return
        text = any_to_string(self.find_input()).strip()
        if not text:
            return
        if text == self.match_text and self.matches:
            self.match_pos = (self.match_pos + 1) % len(self.matches)     # same word again: next match
        else:
            self.matches, names = a.find(self._grid(), text)
            self.match_text, self.match_pos = text, 0
            if not self.matches:
                self.status_label.set('no class matches "%s"' % text)
                return
        self._select(self.matches[self.match_pos])

    def next_pressed(self):
        a = self._load()
        if a is None:
            return
        g = self._grid()
        cells = a.cells[g]
        n = len(cells['xy'])
        walk = any_to_string(self.walk_input())
        if self.k is None:
            self._select(a.nearest(g, g // 2, g // 2))
            return
        if walk == 'matches' and self.matches:
            self.match_pos = (self.match_pos + 1) % len(self.matches)
            k = self.matches[self.match_pos]
        elif walk == 'random':
            k = int(self.rng.integers(n))
        elif walk == 'snake':
            # rows left to right, then right to left, so each step is to a neighbour where it can be
            x, y = self._cell_xy()
            order = sorted(range(n), key=lambda j: (int(cells['xy'][j][1]),
                                                     int(cells['xy'][j][0]) * (1 if cells['xy'][j][1] % 2 == 0 else -1)))
            k = order[(order.index(self.k) + 1) % n]
        else:
            nb = a.neighbours(g, self.k)
            fresh = [j for j in nb if j not in self.visited]
            pool = fresh or nb
            if pool:
                k = int(self.rng.choice(pool))
            else:
                # an isolated cell: jump to the nearest other one
                x, y = self._cell_xy()
                d = ((cells['xy'] - np.array([x, y])) ** 2).sum(1).astype(np.float64)
                d[self.k] = np.inf
                k = int(np.argmin(d))
        self._select(k)

    def layer_changed(self):
        if self.in_loading_process:
            return
        xy = self._cell_xy()
        self.atlas = None
        self._reselect_at(xy)

    def grid_changed(self):
        if self.in_loading_process:
            return
        xy = self._cell_xy_on(getattr(self, '_last_grid', None))
        g = self._grid()
        if xy is not None:
            # the same place on the map, scaled to the new grid
            prev = getattr(self, '_last_grid', g)
            xy = (int(xy[0] * g / prev), int(xy[1] * g / prev))
        self._last_grid = g
        self.k = None
        self._reselect_at(xy)

    def whiten_changed(self):
        if self.in_loading_process or self.k is None or self.atlas is None:
            return
        self._select(self.k)

    def current_edited(self):
        if self.in_loading_process:
            return
        a = self._load()
        vals = any_to_string(self.current_option()).replace(',', ' ').split()
        if a is None or len(vals) < 2:
            return
        try:
            x, y = int(float(vals[0])), int(float(vals[1]))
        except ValueError:
            return
        k = a.nearest(self._grid(), x, y)
        if k != self.k:
            self._select(k)

    def custom_create(self, from_file):
        self._last_grid = self._grid()
        if not from_file and self._load() is not None:
            g = self._grid()
            self._select(self.atlas.nearest(g, g // 2, g // 2), send=False)

    def post_load_callback(self):
        self._last_grid = self._grid()
        a = self._load()
        if a is None:
            return
        vals = any_to_string(self.current_option()).split()
        g = self._grid()
        try:
            x, y = int(vals[0]), int(vals[1])
        except (ValueError, IndexError):
            x = y = g // 2
        self._select(a.nearest(g, x, y))

    def execute(self):
        pass
