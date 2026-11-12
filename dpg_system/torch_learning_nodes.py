import os
import math
import weakref

import numpy as np
import torch

from dpg_system.node import Node, SaveDialog
from dpg_system.conversion_utils import *
from dpg_system.torch_base_nodes import *


def register_torch_learning_nodes():
    Node.app.register_node('t.param', TorchParamNode.factory)
    Node.app.register_node('t.optimize', TorchOptimizeNode.factory)


def find_params(loss):
    """The t.param tensors a loss depends on, found by walking back through
    its gradient graph. Other tensors that carry gradients are left alone."""
    found = []
    seen = set()
    stack = [loss.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        variable = getattr(fn, 'variable', None)        # a leaf tensor's gradient collector
        if variable is not None:
            owner = getattr(variable, '_dpg_param', None)
            if owner is not None and owner() is not None:
                found.append(variable)
        stack.extend(next_fn for next_fn, _ in fn.next_functions)
    return found


class TorchParamNode(TorchDeviceDtypeNode):
    # A set of numbers for t.optimize to tune. Sends them as a tensor that carries
    # gradients, so a loss built from them by ordinary torch nodes can be traced
    # back here. With a range, the numbers ease toward its ends instead of being
    # clipped. With a path, they are kept in a .pt file and read back when the
    # patch loads.
    @staticmethod
    def factory(name, data, args=None):
        node = TorchParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        shape = []
        default_path = ''
        for i in range(len(args)):
            value, kind = decode_arg(args, i)
            if kind is int:
                shape.append(max(1, value))
            elif kind is str and value not in self.device_list and value not in self.dtype_dict:
                default_path = value
        self.shape = tuple(shape) if len(shape) > 0 else (1,)
        self.setup_dtype_device_grad(args)
        self.training = True
        self.raw = None

        self.input = self.add_input('in', triggers_execution=True)
        self.learn_input = self.add_input('learn', widget_type='checkbox', default_value=True,
                                          callback=self.update_tracking)
        self.save_button = self.add_input('save', widget_type='button', callback=self.save_to_file)
        self.reset_button = self.add_input('reset', widget_type='button', callback=self.reset)
        self.output = self.add_output('values')
        self.initial_option = self.add_option('initial value', widget_type='drag_float', default_value=0.0)
        self.low_option = self.add_option('low', widget_type='drag_float', default_value=0.0,
                                          callback=self.range_changed)
        self.high_option = self.add_option('high', widget_type='drag_float', default_value=0.0,
                                           callback=self.range_changed)
        self.create_device_property(option=True)
        self.path_option = self.add_option('path', widget_type='text_input', default_value=default_path,
                                           callback=self.path_changed)
        self.range = None
        self.make_raw(self.initial_values())

    def custom_create(self, from_file=False):
        self.path_changed()

    # ---- the values ----
    def bounded(self):
        low = self.low_option()
        high = self.high_option()
        if low is None or high is None or high <= low:
            return None
        return float(low), float(high)

    def initial_values(self):
        initial = self.initial_option()
        initial = 0.0 if initial is None else float(initial)
        return torch.full(self.shape, initial, dtype=torch.float32)

    def to_raw(self, values):
        # the stored number behind a value: inside a range, the value is
        # low + (high - low) * sigmoid(raw), so raw is its logit
        bounds = self.bounded()
        if bounds is None:
            return values
        low, high = bounds
        fraction = ((values - low) / (high - low)).clamp(1e-4, 1.0 - 1e-4)
        return torch.log(fraction / (1.0 - fraction))

    def from_raw(self, raw):
        bounds = self.bounded()
        if bounds is None:
            return raw
        low, high = bounds
        return low + (high - low) * torch.sigmoid(raw)

    def make_raw(self, values):
        values = values.detach().to(device=self.device, dtype=torch.float32)
        self.shape = tuple(values.shape)
        self.range = self.bounded()
        with torch.no_grad():
            raw = self.to_raw(values).clone()
        self.raw = raw
        self.raw._dpg_param = weakref.ref(self)
        self.update_tracking()

    def set_values(self, values):
        values = values.detach().to(device=self.device, dtype=torch.float32)
        if tuple(values.shape) == tuple(self.raw.shape):
            # in place, so t.optimize keeps its memory of this tensor
            with torch.no_grad():
                self.raw.copy_(self.to_raw(values))
        else:
            self.make_raw(values)

    def values(self):
        return self.from_raw(self.raw).detach()

    def update_tracking(self, input=None):
        learn = self.learn_input()
        track = self.training and (learn is None or any_to_bool(learn))
        if self.raw is not None and self.raw.requires_grad != track:
            self.raw.requires_grad_(track)

    def set_training(self, training):
        self.training = training
        self.update_tracking()

    def range_changed(self):
        # keep the values where they are; only the way they are stored changes
        if self.raw is None:
            return
        with torch.no_grad():
            values = self.from_raw_with(self.raw, self.range)
        self.make_raw(values)

    def from_raw_with(self, raw, bounds):
        if bounds is None:
            return raw
        low, high = bounds
        return low + (high - low) * torch.sigmoid(raw)

    def device_changed(self):
        super().device_changed()
        if self.raw is not None and self.raw.device != self.device:
            self.make_raw(self.values())

    def reset(self, input=None):
        self.make_raw(self.initial_values())
        self.send_values()

    # ---- running ----
    def execute(self):
        data = self.input()
        if not isinstance(data, str) and not (isinstance(data, (list, tuple)) and len(data) > 0
                                              and isinstance(data[0], str)):
            values = any_to_tensor(data, validate=True)
            if values is not None and values.numel() > 0:
                self.set_values(values.float())
        self.send_values()

    def send_values(self):
        if self.raw.requires_grad:
            out = self.from_raw(self.raw)
            if out is self.raw:
                out = self.raw.clone()      # never hand the stored tensor itself downstream
        else:
            out = self.values()
        self.output.send(out)

    # ---- the file ----
    def path_changed(self):
        path = self.path_option()
        if path is not None and path != '' and os.path.exists(path):
            self.load_file(path)

    def load_file(self, path):
        try:
            saved = torch.load(path, map_location='cpu', weights_only=True)
        except Exception as e:
            print(f'{self.label} error loading {path}: {type(e).__name__}: {e}')
            return
        values = saved.get('values') if isinstance(saved, dict) else saved
        if not isinstance(values, torch.Tensor):
            print(f'{self.label}: {path} holds no values')
            return
        self.make_raw(values.float())
        if self.app.verbose:
            print(f'{self.label}: read {tuple(values.shape)} values from {path}')

    def save_to_file(self, input=None):
        path = self.path_option()
        if path is None or path == '':
            SaveDialog(self, callback=self.save_dialog_callback, extensions=['.pt'],
                       default_filename='params.pt')
            return
        self.write_file(path)

    def save_dialog_callback(self, save_path):
        if save_path != '':
            if os.path.splitext(save_path)[1] == '':
                save_path += '.pt'
            self.path_option.set(save_path)
            self.write_file(save_path)

    def write_file(self, path):
        bounds = self.bounded()
        try:
            torch.save({'values': self.values().cpu(),
                        'low': bounds[0] if bounds else None,
                        'high': bounds[1] if bounds else None}, path)
        except Exception as e:
            print(f'{self.label} error writing {path}: {type(e).__name__}: {e}')

    def has_path(self):
        path = self.path_option()
        return path is not None and path != ''


class TorchOptimizeNode(TorchNode):
    # Tunes the t.param nodes that a loss depends on, by gradient descent. A loss
    # must stay a torch tensor all the way from the t.param nodes; t.optimize
    # finds them by tracing it back, so nothing needs to be wired to them except
    # 'again'. A step changes the numbers, and the loss has to be worked out
    # afresh before the next step: 'again' asks for that, so wire it back to the
    # t.param nodes (or to whatever starts the patch's calculation).
    optimizers = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW,
                  'sgd': torch.optim.SGD, 'rmsprop': torch.optim.RMSprop}

    @staticmethod
    def factory(name, data, args=None):
        node = TorchOptimizeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        rate = 0.01
        for i in range(len(args)):
            value, kind = decode_arg(args, i)
            if kind in (float, int):
                rate = float(value)

        self.optimizer = None
        self.params = []
        self.param_nodes = []
        self.step_count = 0
        self.in_loop = False
        self.pending = None
        self.misses = 0
        self.warned = set()

        self.input = self.add_input('loss', triggers_execution=True)
        self.train_input = self.add_input('train', widget_type='checkbox', default_value=True,
                                          callback=self.train_changed)
        self.step_button = self.add_input('step', widget_type='button', callback=self.step_requested)
        self.reset_button = self.add_input('reset', widget_type='button', callback=self.reset)
        self.mode_option = self.add_option('step on', widget_type='combo', default_value='each loss',
                                           callback=self.mode_changed)
        self.mode_option.widget.combo_items = ['each loss', 'bang', 'every frame']
        self.steps_option = self.add_option('steps', widget_type='drag_int', default_value=1, min=1)
        self.optimizer_option = self.add_option('optimizer', widget_type='combo', default_value='adam',
                                                callback=self.reset)
        self.optimizer_option.widget.combo_items = list(self.optimizers.keys())
        self.rate_option = self.add_option('learning rate', widget_type='drag_float', default_value=rate,
                                           callback=self.rate_changed)
        self.clip_option = self.add_option('clip gradient', widget_type='drag_float', default_value=0.0)
        self.pull_option = self.add_option('pull to zero', widget_type='drag_float', default_value=0.0)
        self.autosave_option = self.add_option('autosave every', widget_type='drag_int', default_value=0, min=0)

        self.loss_output = self.add_output('loss')
        self.count_output = self.add_output('steps')
        self.gradient_output = self.add_output('gradient size')
        self.again_output = self.add_output('again')

    def custom_create(self, from_file=False):
        self.mode_changed()

    def custom_cleanup(self):
        self.remove_frame_tasks()

    def warn_once(self, key, message):
        if key not in self.warned:
            self.warned.add(key)
            print(f'{self.label}: {message}')

    def training(self):
        train = self.train_input()
        return train is None or any_to_bool(train)

    def train_changed(self, input=None):
        training = self.training()
        for node_ref in self.param_nodes:
            node = node_ref()
            if node is not None:
                node.set_training(training)

    def mode_changed(self):
        if self.mode_option() == 'every frame':
            if not self.has_frame_task:
                self.add_frame_task()
        else:
            self.remove_frame_tasks()

    def rate_changed(self):
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                group['lr'] = float(self.rate_option())

    def reset(self, input=None):
        # forget the optimizer's momentum and the step count; the t.param
        # values are kept (their own reset puts them back)
        self.optimizer = None
        self.params = []
        self.step_count = 0
        self.warned = set()
        self.count_output.send(0)

    # ---- receiving a loss ----
    def execute(self):
        loss = self.input()
        if self.in_loop:
            self.pending = loss             # asked for by 'again': the loop steps on it
            return
        self.show_loss(loss)
        if self.training() and self.mode_option() == 'each loss':
            self.run(loss)

    def step_requested(self, input=None):
        if not self.in_loop and self.training():
            self.run(None)

    def frame_task(self):
        if not self.in_loop and self.training() and self.mode_option() == 'every frame':
            self.run(None)

    def show_loss(self, loss):
        if isinstance(loss, torch.Tensor):
            loss = loss.detach()
            loss = loss.sum().item() if loss.numel() > 0 else None
        if loss is not None:
            self.loss_output.send(loss)

    def ask_again(self):
        self.pending = None
        self.again_output.send('bang')
        loss = self.pending
        self.pending = None
        if loss is None:
            # a patch still loading its data misses a few; only a steady run of
            # misses means 'again' is not wired
            self.misses += 1
            if self.misses >= 60:
                self.warn_once('again', "no loss comes back from 'again' - wire 'again' to the t.param "
                                        "nodes (or to what starts the calculation)")
        else:
            self.misses = 0
            self.show_loss(loss)
        return loss

    def run(self, loss):
        # loss is the one just received, or None to ask for a fresh one. Each
        # further step asks again, and the loss that comes back while this loop
        # runs is held in pending rather than stepped there, so many steps never
        # nest one call inside another.
        self.in_loop = True
        try:
            steps = self.steps_option()
            steps = 1 if steps is None else max(1, int(steps))
            for i in range(steps):
                if loss is None:
                    loss = self.ask_again()
                    if loss is None:
                        break
                if not self.step(loss):
                    break
                loss = None
        finally:
            self.in_loop = False

    # ---- one step ----
    def step(self, loss):
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            self.warn_once('no grad', 'the loss carries no gradient. It must be worked out from t.param '
                                      'by torch nodes all the way; a node that turns it into a plain '
                                      'number or a numpy array cuts the trail')
            return False
        if loss.numel() != 1:
            loss = loss.sum()
        params = sorted(find_params(loss), key=id)
        if len(params) == 0:
            self.warn_once('no params', 'the loss does not depend on any t.param that is learning')
            return False
        # sorted the same way every time, so a new optimizer (which forgets its
        # momentum) is only built when the set of t.param nodes really changes
        if self.optimizer is None or len(params) != len(self.params) \
                or any(a is not b for a, b in zip(params, self.params)):
            self.build_optimizer(params)
        pull = self.pull_option()
        if pull is not None and pull > 0:
            # a gentle cost on the size of the numbers, so they only grow where
            # the loss really asks for it
            loss = loss + float(pull) * sum(p.pow(2).sum() for p in self.params)
        self.optimizer.zero_grad(set_to_none=True)
        try:
            loss.backward()
        except RuntimeError as e:
            # most often: this loss was worked out before the last step changed
            # the numbers, so its trail no longer matches them
            self.warn_once('backward', f'could not trace the loss back: {e}')
            return False
        grads = [p.grad for p in self.params if p.grad is not None]
        size = torch.sqrt(sum((g.detach().float() ** 2).sum() for g in grads)).item() if grads else 0.0
        clip = self.clip_option()
        if clip is not None and clip > 0:
            torch.nn.utils.clip_grad_norm_(self.params, float(clip))
        self.optimizer.step()
        self.step_count += 1
        autosave = self.autosave_option()
        if autosave is not None and autosave > 0 and self.step_count % int(autosave) == 0:
            self.save_params()
        self.gradient_output.send(size)
        self.count_output.send(self.step_count)
        return True

    def build_optimizer(self, params):
        self.params = params
        self.param_nodes = [p._dpg_param for p in params]
        kind = self.optimizers.get(self.optimizer_option(), torch.optim.Adam)
        self.optimizer = kind(params, lr=float(self.rate_option()))

    def save_params(self):
        for node_ref in self.param_nodes:
            node = node_ref()
            if node is not None and node.has_path():
                node.write_file(node.path_option())
