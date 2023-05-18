from dpg_system.torch_base_nodes import *

def register_torch_generator_nodes():
    Node.app.register_node('t.rand', TorchGeneratorNode.factory)
    Node.app.register_node('t.ones', TorchGeneratorNode.factory)
    Node.app.register_node('t.zeros', TorchGeneratorNode.factory)
    Node.app.register_node('t.full', TorchFullNode.factory)
    Node.app.register_node('t.linspace', TorchLinSpaceNode.factory)
    Node.app.register_node('t.logspace', TorchLinSpaceNode.factory)
    Node.app.register_node('t.range', TorchRangeNode.factory)
    Node.app.register_node('t.arange', TorchRangeNode.factory)
    Node.app.register_node('t.eye', TorchEyeNode.factory)

    Node.app.register_node('t.rand_like', TorchGeneratorLikeNode.factory)
    Node.app.register_node('t.ones_like', TorchGeneratorLikeNode.factory)
    Node.app.register_node('t.zeros_like', TorchGeneratorLikeNode.factory)


class TorchGeneratorNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchGeneratorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.shape = []
        for i in range(len(args)):
            d, t = decode_arg(args, i)
            if t == int:
                self.shape += (d,)
        if len(self.shape) == 0:
            self.shape = [1]

        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)

        self.shape_input = self.add_input('shape', widget_type='text_input', default_value=str(self.shape), callback=self.shape_changed)
        # self.shape_properties = []
        # for i in range(len(self.shape)):
        #     self.shape_properties.append(self.add_property('dim ' + str(i), widget_type='input_int', default_value=self.shape[i]))

        self.setup_dtype_device_grad(args)

        if self.label == 't.rand':
            self.min = 0
            self.max = 1
            self.min_input = self.add_input('min', widget_type='drag_float', default_value=self.min,
                                            callback=self.range_changed)
            self.max_input = self.add_input('max', widget_type='drag_float', default_value=self.max,
                                            callback=self.range_changed)

        out_label = 'random tensor'
        if self.label == 't.ones':
            out_label = 'tensor of ones'
        elif self.label == 't.zeros':
            out_label = 'tensor of zeros'
        self.output = self.add_output(out_label)

    def range_changed(self, val=None):
        self.min = self.min_input.get_widget_value()
        self.max = self.max_input.get_widget_value()

    def shape_changed(self, val=0):
        shape_text = self.shape_input.get_widget_value()
        shape_list = re.findall(r'[-+]?\d+', shape_text)
        shape = []
        for dim_text in shape_list:
            shape.append(any_to_int(dim_text))
        self.shape = shape

    def dtype_changed(self, val='torch.float'):
        super().dtype_changed(val)
        if self.label == 't.rand':
            if self.dtype == torch.uint8:
                if self.min < 0:
                    self.min_input.set(0.0)
                if self.max == 1.0 or self.max < 255:
                    self.max_input.set(255.0)
            elif self.dtype == torch.int64:
                if self.min < -32768:
                    self.min_input.set(-32768)
                if self.max == 1.0:
                    self.max_input.set(32767)
            elif self.dtype in [torch.float, torch.double, torch.float32, torch.float16, torch.bfloat16]:
                if self.min == -32768:
                    self.min_input.set(0.0)
                if self.max == 255:
                    self.max_input.set(1.0)
                elif self.max == 32767:
                    self.max_input.set(1.0)
            self.range_changed()

    def execute(self):
        size = tuple(self.shape)
        if self.label == 't.rand':
            if self.dtype in [torch.float, torch.float32, torch.double, torch.float16, torch.bfloat16, torch.complex32, torch.complex64, torch.complex128]:
                range_ = self.max - self.min
                out_array = torch.rand(size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad) * range_ + self.min
            elif self.dtype in [torch.int64, torch.uint8]:
                out_array = torch.randint(low=int(self.min), high=int(self.max), size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
            elif self.dtype == torch.bool:
                out_array = torch.randint(low=0, high=1, size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        elif self.label == 't.ones':
            out_array = torch.ones(size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        elif self.label == 't.zeros':
            out_array = torch.zeros(size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        self.output.send(out_array)

class TorchFullNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchFullNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.shape = []
        for i in range(len(args)):
            d, t = decode_arg(args, i)
            if t == int:
                self.shape += (d,)
        if len(self.shape) == 0:
            self.shape = [1]

        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.value = 1.0
        self.value_input = self.add_input('value', widget_type='drag_float', default_value=self.value, callback=self.val_changed)
        self.shape_properties = []
        for i in range(len(self.shape)):
            self.shape_properties.append(self.add_property('dim ' + str(i), widget_type='input_int', default_value=self.shape[i]))

        self.setup_dtype_device_grad(args)

        out_label = 'filled tensor'
        self.output = self.add_output(out_label)

    def val_changed(self, val=None):
        self.value = self.value_input.get_widget_value()

    def execute(self):
        for i in range(len(self.shape)):
            self.shape[i] = self.shape_properties[i].get_widget_value()
        size = tuple(self.shape)
        out_array = torch.full(size=size, fill_value=self.value, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        self.output.send(out_array)


class TorchGeneratorLikeNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchGeneratorLikeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('tensor in', triggers_execution=True)

        self.setup_dtype_device_grad(args)

        if self.label == 't.rand_like':
            self.min = 0
            self.max = 1
            self.min_input = self.add_input('min', widget_type='drag_float', default_value=self.min,
                                            callback=self.range_changed)
            self.max_input = self.add_input('max', widget_type='drag_float', default_value=self.max,
                                            callback=self.range_changed)

        out_label = 'random tensor'
        if self.label == 't.ones_like':
            out_label = 'tensor of ones'
        elif self.label == 't.zeros_like':
            out_label = 'tensor of zeros'
        self.output = self.add_output(out_label)

    def range_changed(self, val=None):
        self.min = self.min_input.get_widget_value()
        self.max = self.max_input.get_widget_value()

    def dtype_changed(self, val='torch.float'):
        super().dtype_changed(val)
        if self.label == 't.rand':
            if self.dtype == torch.uint8:
                if self.min < 0:
                    self.min_input.set(0.0)
                if self.max == 1.0 or self.max < 255:
                    self.max_input.set(255.0)
            elif self.dtype == torch.int64:
                if self.min < -32768:
                    self.min_input.set(-32768)
                if self.max == 1.0:
                    self.max_input.set(32767)
            elif self.dtype in [torch.float, torch.double, torch.float32, torch.float16, torch.bfloat16]:
                if self.min == -32768:
                    self.min_input.set(0.0)
                if self.max == 255:
                    self.max_input.set(1.0)
                elif self.max == 32767:
                    self.max_input.set(1.0)
            self.range_changed()

    def execute(self):
        if self.input.fresh_input:
            data = self.input.get_received_data()
            if type(data) == torch.Tensor:
                shape = data.shape
                size = tuple(shape)
                if self.label == 't.rand_like':
                    if self.dtype in [torch.float, torch.float32, torch.double, torch.float16, torch.bfloat16, torch.complex32, torch.complex64, torch.complex128]:
                        range_ = self.max - self.min
                        out_array = torch.rand(size=size, device=self.device, dtype=self.dtype) * range_ + self.min
                    elif self.dtype in [torch.int64, torch.uint8]:
                        out_array = torch.randint(low=int(self.min), high=int(self.max), size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
                    elif self.dtype == torch.bool:
                        out_array = torch.randint(low=0, high=1, size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
                elif self.label == 't.ones_like':
                    out_array = torch.ones(size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
                elif self.label == 't.zeros_like':
                    out_array = torch.zeros(size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
                self.output.send(out_array)


class TorchLinSpaceNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchLinSpaceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.shape = []
        self.start = 0.0
        self.stop = 1.0
        self.steps = 50

        self.op = torch.linspace
        out_label = 'linspace out'
        if self.label == 't.logspace':
            self.op = torch.logspace
            out_label = 'logspace out'

        if len(args) > 0:
            d, t = decode_arg(args, 0)
            if t in [float, int]:
                self.start = any_to_float(d)
        if len(args) > 1:
            d, t = decode_arg(args, 1)
            if t in [float, int]:
                self.stop = any_to_float(d)
        if len(args) > 2:
            d, t = decode_arg(args, 2)
            if t in [float, int]:
                self.steps = any_to_int(d)

        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.start_property = self.add_property('start', widget_type='drag_float', default_value=self.start)
        self.stop_property = self.add_property('stop', widget_type='drag_float', default_value=self.stop)
        self.steps_property = self.add_property('steps', widget_type='drag_int', default_value=self.steps)

        self.setup_dtype_device_grad(args)

        self.output = self.add_output(out_label)

    def execute(self):
        self.start = self.start_property.get_widget_value()
        self.stop = self.stop_property.get_widget_value()
        self.steps = self.steps_property.get_widget_value()
        out_array = self.op(self.start, self.stop, self.steps, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        self.output.send(out_array)


class TorchRangeNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchRangeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.shape = []
        self.start = 0.0
        self.stop = 1.0
        self.step = .01

        self.op = torch.arange
        out_label = 'arange out'
        if self.label == 't.range':
            self.op = torch.range
            out_label = 'range out'

        if len(args) > 0:
            d, t = decode_arg(args, 0)
            if t in [float, int]:
                self.start = any_to_float(d)
        if len(args) > 1:
            d, t = decode_arg(args, 1)
            if t in [float, int]:
                self.stop = any_to_float(d)
        if len(args) > 2:
            d, t = decode_arg(args, 2)
            if t in [float, int]:
                self.step = any_to_float(d)

        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.start_property = self.add_property('start', widget_type='drag_float', default_value=self.start)
        self.stop_property = self.add_property('stop', widget_type='drag_float', default_value=self.stop)
        self.step_property = self.add_property('step', widget_type='drag_float', default_value=self.step)

        self.setup_dtype_device_grad(args)

        self.output = self.add_output(out_label)

    def execute(self):
        self.start = self.start_property.get_widget_value()
        self.stop = self.stop_property.get_widget_value()
        self.step = self.step_property.get_widget_value()
        out_array = self.op(self.start, self.stop, self.step, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        self.output.send(out_array)


class TorchEyeNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchEyeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.shape = []
        if len(args) > 0:
            self.n = any_to_int(args[0])
        else:
            self.n = 4
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.n_input = self.add_input('n', widget_type='input_int', default_value=self.n, callback=self.n_changed)

        self.setup_dtype_device_grad(args)

        out_label = 'eye tensor'
        self.output = self.add_output(out_label)

    def n_changed(self, val=None):
        self.n = self.n_input.get_widget_value()

    def execute(self):
        out_array = torch.eye(n=self.n, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        self.output.send(out_array)
