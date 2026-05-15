from dpg_system.torch_base_nodes import *
import torch.fft

def register_torch_signal_processing_nodes():
    Node.app.register_node('t.window.blackman', TorchWindowNode_.factory)
    Node.app.register_node('t.window.bartlett', TorchWindowNode_.factory)
    Node.app.register_node('t.window.cosine', TorchWindowNode_.factory)
    Node.app.register_node('t.window.hamming', TorchWindowNode_.factory)
    Node.app.register_node('t.window.hann', TorchWindowNode_.factory)
    Node.app.register_node('t.window.nuttall', TorchWindowNode_.factory)
    Node.app.register_node('t.window.gaussian', TorchWindowNode_.factory)
    Node.app.register_node('t.window.general_hamming', TorchWindowNode_.factory)
    Node.app.register_node('t.window.kaiser', TorchWindowNode_.factory)
    Node.app.register_node('t.window.exponential', TorchWindowNode_.factory)

    Node.app.register_node('t.rfft', TorchFFTNode.factory)
    Node.app.register_node('t.irfft', TorchFFTNode.factory)
    Node.app.register_node('t.fft', TorchFFTNode.factory)
    Node.app.register_node('t.ifft', TorchFFTNode.factory)

class TorchWindowNode_(TorchDeviceDtypeNode):
    op_dict = {
        't.window.blackman': torch.signal.windows.blackman,
        't.window.bartlett': torch.signal.windows.bartlett,
        't.window.cosine': torch.signal.windows.cosine,
        't.window.hamming': torch.signal.windows.hamming,
        't.window.hann': torch.signal.windows.hann,
        't.window.nuttall': torch.signal.windows.nuttall,
        't.window.gaussian': torch.signal.windows.gaussian,
        't.window.general_hamming': torch.signal.windows.general_hamming,
        't.window.kaiser': torch.signal.windows.kaiser,
        't.window.exponential': torch.signal.windows.exponential
    }

    param_dict = {
        'blackman': [],
        'bartlett': [],
        'cosine': [],
        'hamming': [],
        'hann': [],
        'nuttall': [],
        'gaussian': [('std', 1.0)],
        'general_hamming': [('alpha', 0.54)],
        'kaiser': [('beta', 12.0)],
        'exponential': [('center', 1.0), ('tau', 1.0)]
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchWindowNode_(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # Fall back to hann so an unknown label still produces a usable window
        # instead of an AttributeError in execute().
        self.op = self.op_dict.get(self.label, torch.signal.windows.hann)
        window_type = self.label.split('.')[-1]

        self.param_1_name = ''
        self.param_2_name = ''
        self.default_1 = 1.0
        self.default_2 = 1.0

        param_list = []
        if window_type in self.param_dict:
            param_list = self.param_dict[window_type]
        if len(param_list) > 0:
            self.param_1_name, self.default_1 = param_list[0]
        if len(param_list) > 1:
            self.param_2_name, self.default_2 = param_list[1]

        m = 256
        sym = True
        self.help_file_name = 't.window_help'

        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.window_type = self.add_input('type', widget_type='combo', widget_width=120, default_value=window_type, callback=self.window_type_changed)
        self.window_type.widget.combo_items = list(self.param_dict.keys())
        # m is the window length — torch rejects values < 0 with a cryptic error.
        self.m = self.add_input('m', widget_type='drag_int', default_value=m, min=1)
        self.param_1 = self.add_input(self.param_1_name, widget_type='drag_float', default_value=self.default_1)
        self.param_2 = self.add_input(self.param_2_name, widget_type='drag_float', default_value=self.default_2)
        self.sym = self.add_input('sym', widget_type='checkbox', default_value=sym)
        self.setup_dtype_device_grad(args)
        self.output = self.add_output('window tensor out')
        self.create_dtype_device_grad_properties()

    def custom_create(self, from_file):
        if self.param_1_name == '':
            self.param_1.hide()
        if self.param_2_name == '':
            self.param_2.hide()

    def window_type_changed(self):
        self.label = 't.window.' + self.window_type()
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
        else:
            # Keep going with a sensible default if the combo picked up an
            # unregistered name (e.g. a typo'd preset).
            print(f'{self.label}: unknown window, falling back to hann')
            self.op = torch.signal.windows.hann
        self.set_title(self.label)
        param_list = []
        if self.window_type() in self.param_dict:
            param_list = self.param_dict[self.window_type()]
        if len(param_list) > 0:
            self.param_1_name = param_list[0][0]
            self.param_1.set_label(self.param_1_name)
            self.default_1 = param_list[0][1]
            self.param_1.set(self.default_1)
            self.param_1.show()
        else:
            self.param_1_name = ''
            self.param_1.set_label(self.param_1_name)
            self.param_1.hide()
        if len(param_list) > 1:
            self.param_2_name = param_list[1][0]
            self.param_2.set_label(self.param_2_name)
            self.default_2 = param_list[1][1]
            self.param_2.set(self.default_2)
            self.param_2.show()
        else:
            self.param_2_name = ''
            self.param_2.set_label(self.param_2_name)
            self.param_2.hide()

    def execute(self):
        m = self.m()
        sym = self.sym()
        common = {'sym': sym, 'dtype': self.dtype, 'device': self.device,
                  'requires_grad': self.requires_grad}

        try:
            if self.param_1_name == '':
                window_tensor = self.op(m, **common)
            elif self.label == 't.window.gaussian':
                std = self.param_1()
                if std <= 0:
                    print(f'{self.label}: std must be > 0, got {std}')
                    return
                window_tensor = self.op(m, std=std, **common)
            elif self.label == 't.window.general_hamming':
                window_tensor = self.op(m, alpha=self.param_1(), **common)
            elif self.label == 't.window.kaiser':
                beta = self.param_1()
                if beta < 0:
                    print(f'{self.label}: beta must be >= 0, got {beta}')
                    return
                window_tensor = self.op(m, beta=beta, **common)
            elif self.label == 't.window.exponential':
                tau = self.param_2()
                if tau <= 0:
                    print(f'{self.label}: tau must be > 0, got {tau}')
                    return
                # torch.signal.windows.exponential rejects center when sym=True
                # (center then defaults to M/2). Only pass center on the
                # asymmetric path. Previously this was gated on `if self.sym:`
                # (truthy NodeInput, never False), so the asym branch was dead.
                if sym:
                    window_tensor = self.op(m, tau=tau, **common)
                else:
                    window_tensor = self.op(m, center=self.param_1(), tau=tau, **common)
            else:
                if self.app.verbose:
                    print(f'{self.label}: no execute branch matched, skipping')
                return
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()
            return

        self.output.send(window_tensor)


class TorchFFTNode(TorchWithDimNode):
    op_dict = {
        't.rfft': torch.fft.rfft,
        't.fft': torch.fft.fft,
        't.irfft': torch.fft.irfft,
        't.ifft': torch.fft.ifft
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchFFTNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = self.op_dict.get(self.label, torch.fft.fft)
        self.input = self.add_input('tensor in', triggers_execution=True)
        norm = 'backward'
        if self.dim_specified:
            self.add_dim_input()
        self.norm = self.add_input('norm', widget_type='combo', default_value=norm)
        self.norm.widget.combo_items = ['forward', 'backward', 'ortho']
        self.output = self.add_output('fft tensor out')
        self.help_file_name = 't.fft_help'

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        try:
            # The dim widget previously did nothing — dim= was never forwarded
            # to the op. Pass it through when the user has specified a dim;
            # otherwise let torch use its default (-1).
            if self.dim_specified:
                fft_tensor = self.op(input_tensor, dim=self.dim, norm=self.norm())
            else:
                fft_tensor = self.op(input_tensor, norm=self.norm())
            self.output.send(fft_tensor)
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()
