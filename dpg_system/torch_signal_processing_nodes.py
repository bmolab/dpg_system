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

        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
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

        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.window_type = self.add_input('type', widget_type='combo', widget_width=120, default_value=window_type, callback=self.window_type_changed)
        self.window_type.widget.combo_items = list(self.param_dict.keys())
        self.m = self.add_input('m', widget_type='drag_int', default_value=m)
        self.param_1 = self.add_input(self.param_1_name, widget_type='drag_float', default_value=self.default_1)
        self.param_2 = self.add_input(self.param_2_name, widget_type='drag_float', default_value=self.default_2)
        self.sym = self.add_input('sym', widget_type='checkbox', default_value=sym)
        self.setup_dtype_device_grad(args)
        self.output = self.add_output('window tensor out')

    def custom_create(self, from_file):
        if self.param_1_name == '':
            self.param_1.hide()
        if self.param_2_name == '':
            self.param_2.hide()

    def window_type_changed(self):
        self.label = 't.window.' + self.window_type()
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
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
        window_tensor = None
        if self.param_1_name == '':
            window_tensor = self.op(self.m(), sym=self.sym(), dtype=self.dtype, device=self.device)
        elif self.label == 't.window.gaussian':
            window_tensor = self.op(self.m(), std=self.param_1(), sym=self.sym(), dtype=self.dtype, device=self.device)
        elif self.label == 't.window.general_hamming':
            window_tensor = self.op(self.m(), alpha=self.param_1(), sym=self.sym(), dtype=self.dtype, device=self.device)
        elif self.label == 't.window.kaiser':
            window_tensor = self.op(self.m(), beta=self.param_1(), sym=self.sym(), dtype=self.dtype, device=self.device)
        elif self.label == 't.window.exponential':
            if self.sym:
                window_tensor = self.op(self.m(), tau=self.param_2(), sym=self.sym(), dtype=self.dtype,
                                        device=self.device)
            else:
                window_tensor = self.op(self.m(), center=self.param_1(), tau=self.param_2(), sym=self.sym(), dtype=self.dtype,
                                        device=self.device)
        if window_tensor is not None:
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
        self.op = torch.fft.fft
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
        self.input = self.add_input("tensor in", triggers_execution=True)
        norm = 'backward'
        if self.dim_specified:
            self.add_dim_input()
        self.norm = self.add_input('norm', widget_type='combo', default_value=norm)
        self.norm.widget.combo_items = ['forward', 'backward', 'ortho']
        self.output = self.add_output('histogram tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            fft_tensor = self.op(input_tensor, norm=self.norm())
            self.output.send(fft_tensor)
