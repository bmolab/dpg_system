import os
import warnings

import numpy as np
import torch

from dpg_system.node import Node, SaveDialog, LoadDialog
from dpg_system.conversion_utils import *
from dpg_system.torch_base_nodes import *


def register_torch_model_nodes():
    Node.app.register_node('t.model', TorchModelNode.factory)
    Node.app.register_node('t.save', TorchSaveNode.factory)
    Node.app.register_node('t.load', TorchLoadNode.factory)


def to_cpu(data):
    # what t.save writes: tensors detached and on the cpu, so the file opens on
    # any machine; dicts and lists are walked so a whole set saves as one file
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {key: to_cpu(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        try:
            numeric = np.asarray(data).dtype.kind in 'biuf'     # a list of numbers, or of rows of them
        except ValueError:
            numeric = False                                     # ragged rows
        if numeric and len(data) > 0:
            return any_to_tensor(data, validate=True)
        return type(data)(to_cpu(d) for d in data)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data.copy())
    return data


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(d, device) for d in data)
    return data


class TorchModelNode(TorchDeviceDtypeNode):
    # Runs a saved, runnable model: TorchScript (.pt / .pth, from torch.jit.script
    # or torch.jit.trace) or an exported program (.pt2, from torch.export). A file
    # of weights alone has no code to run, so it is refused with a hint.
    @staticmethod
    def factory(name, data, args=None):
        node = TorchModelNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        default_path = ''
        counts = []
        for i in range(len(args)):
            value, kind = decode_arg(args, i)
            if kind is int:
                counts.append(value)
            elif kind is str and value not in self.device_list:
                default_path = value
        input_count = max(1, counts[0]) if len(counts) > 0 else 1
        output_count = max(1, counts[1]) if len(counts) > 1 else 1
        self.setup_dtype_device_grad(args)

        self.model = None
        self.loaded = None      # (path, device) of the model now held
        self.extra_warned = False

        self.input = self.add_input('in', triggers_execution=True)
        self.extra_inputs = [self.add_input(f'in {i + 2}') for i in range(input_count - 1)]
        self.load_button = self.add_input('load', widget_type='button', callback=self.open_load_dialog)
        self.create_device_property()
        self.path_option = self.add_option('path', widget_type='text_input', default_value=default_path,
                                           callback=self.path_changed)
        self.batch_option = self.add_option('add batch dim', widget_type='checkbox', default_value=False)
        self.output = self.add_output('out')
        self.outputs = [self.output] + [self.add_output(f'out {i + 2}') for i in range(output_count - 1)]

    def custom_create(self, from_file=False):
        self.path_changed()

    def path_changed(self):
        path = self.path_option()
        if path is not None and path != '':
            self.load_model(path)

    def device_changed(self):
        super().device_changed()
        if self.model is not None:
            path = self.loaded[0]
            try:
                self.model.to(self.device)
                self.loaded = (path, self.device_string)
            except Exception:
                # a model whose graph fixes its device cannot be moved; load it
                # afresh straight onto the new device
                self.load_model(path)

    def open_load_dialog(self, input=None):
        LoadDialog(self, callback=self.load_dialog_callback, extensions=['.pt', '.pth', '.pt2'])

    def load_dialog_callback(self, load_path):
        if load_path != '':
            self.path_option.set(load_path)
            self.load_model(load_path)

    def load_model(self, path):
        if self.loaded == (path, self.device_string) and self.model is not None:
            return
        if not os.path.exists(path):
            print(f'{self.label}: no file at {path}')
            return
        try:
            if path.endswith('.pt2'):
                model = torch.export.load(path).module()
            else:
                model = torch.jit.load(path, map_location=self.device)
        except Exception as e:
            print(f'{self.label}: could not load {path} as a runnable model: {type(e).__name__}: {e}')
            print(f'{self.label}: a file of weights (a state_dict) has no code to run - save the model '
                  f'with torch.jit.script(model).save(path), or torch.export, or read the weights with t.load')
            return
        try:
            model.eval()
        except Exception:
            pass        # an exported program refuses eval(); it is already in inference form
        try:
            model.to(self.device)
        except Exception as e:
            print(f'{self.label}: could not move the model to {self.device_string}: {e}')
        self.model = model
        self.loaded = (path, self.device_string)
        if self.app.verbose:
            print(f'{self.label}: loaded {path} on {self.device_string}')

    def prepare(self, data):
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            tensor = data.to(self.device)
        else:
            tensor = any_to_tensor(data, self.device, validate=True)
            if tensor is None:
                return None
            if not tensor.is_floating_point() and not tensor.is_complex():
                tensor = tensor.to(torch.float32)   # models want floats; a typed list arrives as ints
        if self.batch_option():
            tensor = tensor.unsqueeze(0)
        return tensor

    def execute(self):
        if self.model is None:
            if self.app.verbose:
                print(f'{self.label}: no model loaded')
            return
        inputs = [self.prepare(self.input())] + [self.prepare(inlet()) for inlet in self.extra_inputs]
        if any(t is None for t in inputs):
            return
        try:
            with torch.no_grad():
                result = self.model(*inputs)
        except Exception as e:
            print(f'{self.label}: the model failed on inputs of shape '
                  f'{[tuple(t.shape) for t in inputs]}: {type(e).__name__}: {e}')
            return
        if isinstance(result, dict):
            self.output.send(result)
            return
        results = list(result) if isinstance(result, (list, tuple)) else [result]
        if self.batch_option():
            results = [r.squeeze(0) if isinstance(r, torch.Tensor) and r.ndim > 0 and r.shape[0] == 1 else r
                       for r in results]
        if len(results) > len(self.outputs) and not self.extra_warned:
            print(f'{self.label}: the model gives {len(results)} results, but the node has '
                  f'{len(self.outputs)} outlets: add an output count to see them all')
            self.extra_warned = True
        for outlet, r in reversed(list(zip(self.outputs, results))):
            outlet.send(r)


class TorchSaveNode(Node):
    # Holds what arrives - a tensor, or a dict or list of them - and writes it with
    # torch.save only when asked, so a stream never rewrites the file every frame.
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSaveNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        default_path = args[0] if len(args) > 0 else ''
        self.data = None
        self.input = self.add_input('in', triggers_execution=True)
        self.write_button = self.add_input('write', widget_type='button', callback=self.write)
        self.save_as_button = self.add_input('save as', widget_type='button', callback=self.open_save_dialog)
        self.path_option = self.add_option('path', widget_type='text_input', default_value=default_path)
        self.output = self.add_output('saved path')

    def execute(self):
        self.data = self.input()

    def write(self, input=None):
        path = self.path_option()
        if path is None or path == '':
            self.open_save_dialog()
            return
        self.write_file(path)

    def open_save_dialog(self, input=None):
        SaveDialog(self, callback=self.save_dialog_callback, extensions=['.pt', '.pth'],
                   default_filename='untitled.pt')

    def save_dialog_callback(self, save_path):
        if save_path != '':
            self.path_option.set(save_path)
            self.write_file(save_path)

    def write_file(self, path):
        if self.data is None:
            if self.app.verbose:
                print(self.label, 'has nothing to write yet')
            return
        if os.path.splitext(path)[1] == '':
            path += '.pt'
        try:
            torch.save(to_cpu(self.data), path)
        except Exception as e:
            print(f'{self.label} error writing {path}: {type(e).__name__}: {e}')
            return
        self.output.send(path)


class TorchLoadNode(TorchDeviceDtypeNode):
    # Reads a file written by t.save or torch.save: a tensor, or a dict of tensors
    # such as a model's weights. It reads with weights_only, which refuses files
    # that would run code while loading.
    @staticmethod
    def factory(name, data, args=None):
        node = TorchLoadNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        default_path = ''
        for arg in args:
            if isinstance(arg, str) and arg not in self.device_list:
                default_path = arg
                break
        self.setup_dtype_device_grad(args)
        self.data = None
        self.load_button = self.add_input('load', widget_type='button', callback=self.open_load_dialog)
        self.add_input('send', widget_type='button', callback=self.send_data)
        self.create_device_property()
        self.path_option = self.add_option('path', widget_type='text_input', default_value=default_path,
                                           callback=self.path_changed)
        self.output = self.add_output('data out')

    def custom_create(self, from_file=False):
        self.path_changed()

    def path_changed(self):
        path = self.path_option()
        if path is not None and path != '':
            self.load_file(path)

    def device_changed(self):
        super().device_changed()
        if self.data is not None:
            self.data = to_device(self.data, self.device)

    def send_data(self, input=None):
        if self.data is not None:
            self.output.send(self.data)

    def open_load_dialog(self, input=None):
        LoadDialog(self, callback=self.load_dialog_callback, extensions=['.pt', '.pth'])

    def load_dialog_callback(self, load_path):
        if load_path != '':
            self.path_option.set(load_path)
            self.load_file(load_path)

    def load_file(self, path):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')     # torch warns before refusing a TorchScript file
                data = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as e:
            if 'TorchScript' in str(e):
                print(f'{self.label}: {path} is a runnable model - load it with t.model')
                return
            print(f'{self.label} error loading {path}: {type(e).__name__}: {e}')
            print(f'{self.label}: it reads tensors and dicts of them; a runnable model loads with t.model')
            return
        self.data = data
        if self.app.verbose:
            if isinstance(data, dict):
                print(f'{self.label}: loaded {path} with keys {list(data.keys())}')
            else:
                print(f'{self.label}: loaded {path}')
        self.output.send(self.data)
