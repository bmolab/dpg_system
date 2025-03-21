from dpg_system.torch_base_nodes import *

def register_torch_analyze_nodes():
    Node.app.register_node('t.any', TorchAnyAllNode.factory)
    Node.app.register_node('t.all', TorchAnyAllNode.factory)
    Node.app.register_node('t.count_nonzero', TorchCountNonZeroNode.factory)
    Node.app.register_node('t.bincount', TorchBinCountNode.factory)
    Node.app.register_node('t.bucketize', TorchBucketizeNode.factory)
    Node.app.register_node('t.histc', TorchHistogramNode.factory)
    Node.app.register_node('t.max', TorchMinMaxNode.factory)
    Node.app.register_node('t.min', TorchMinMaxNode.factory)
    Node.app.register_node('t.argmax', TorchArgMaxNode.factory)
    Node.app.register_node('t.argwhere', TorchArgWhereNode.factory)
    Node.app.register_node('t.non_zero', TorchArgWhereNode.factory)
    Node.app.register_node('t.argsort', TorchArgSortNode.factory)

class TorchCountNonZeroNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCountNonZeroNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('tensor in', triggers_execution=True)
        if self.dim_specified:
            self.add_dim_input()
        self.output = self.add_output('tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if self.dim_specified:
                result = torch.count_nonzero(input_tensor, dim=self.dim)
            else:
                result = torch.count_nonzero(input_tensor)
            self.output.send(result)


class TorchBinCountNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchBinCountNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('int tensor in', triggers_execution=True)
        self.output = self.add_output('bin count tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if input_tensor.dtype not in [torch.int, torch.uint8, torch.int8, torch.int64, torch.int32, torch.int16]:
                input_tensor = input_tensor.to(dtype=torch.int)
            if len(input_tensor.shape) > 1:
                input_tensor = torch.flatten(input_tensor)
            output_tensor = torch.bincount(input_tensor)
            self.output.send(output_tensor)


class TorchBucketizeNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchBucketizeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        output_int32 = False
        right = False
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.boundaries = self.add_input('boundaries tensor in')
        self.output_int32 = self.add_input('int32 indices', widget_type='checkbox', default_value=output_int32)
        self.right = self.add_input('right', widget_type='checkbox', default_value=right)
        self.output = self.add_output('bin count tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.boundaries()
            boundaries_tensor = self.data_to_tensor(data, match_tensor=input_tensor)
            output_tensor = torch.bucketize(input_tensor, boundaries_tensor, out_int32=self.output_int32(), right=self.right())
            self.output.send(output_tensor)


class TorchAnyAllNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchAnyAllNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = torch.all
        output_name = 'all'
        if self.label == 't.any':
            self.op = torch.any
            output_name == 'any'

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            result = self.op(input_tensor)
            self.output.send(result)


class TorchHistogramNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchHistogramNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        bin_count = 100
        min = 0
        max = 0

        self.input = self.add_input('', triggers_execution=True)
        self.bin_count = self.add_input('bin count', widget_type='drag_int', default_value=bin_count)
        self.min = self.add_input('min', widget_type='drag_float', default_value=min)
        self.max = self.add_input('max', widget_type='drag_float', default_value=max)
        self.output = self.add_output('histogram tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            histogram_tensor = torch.histc(input_tensor, bins=self.bin_count(), min=self.min(), max=self.max())
            self.output.send(histogram_tensor)


class TorchMinMaxNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchMinMaxNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        output_label = 'max tensor'
        index_label = 'max indices'
        self.op = torch.max
        if self.label == 't.min':
            output_label = 'min tensor'
            index_label = 'min indices'
            self.op = torch.min
        self.input = self.add_input('tensor in', triggers_execution=True)
        if self.dim_specified:
            self.add_dim_input()
        self.output = self.add_output(output_label)
        if self.dim_specified:
            self.index_output = self.add_output(index_label)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if self.dim_specified:
                output_tensor, index_tensor = self.op(input_tensor, dim=self.dim)
                self.index_output.send(index_tensor)
                self.output.send(output_tensor)
            else:
                output_tensor = self.op(input_tensor)
                self.output.send(output_tensor)


class TorchArgMaxNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchArgMaxNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        if self.dim_specified:
            self.add_dim_input()
        self.output = self.add_output('max index')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if self.dim_specified:
                output_tensor = torch.argmax(input_tensor, dim=self.dim)
            else:
                output_tensor = torch.argmax(input_tensor)
            self.output.send(output_tensor)


class TorchArgWhereNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchArgWhereNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('index tensor where non-zero')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(torch.argwhere(input_tensor))


class TorchArgSortNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchArgSortNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.dim = -1
        descending = False
        stable = False
        if self.dim_specified:
            self.add_dim_input()
        self.descending = self.add_input('descending', widget_type='checkbox', default_value=descending)
        self.stable = self.add_input('stable', widget_type='checkbox', default_value=stable)
        self.output = self.add_output('output')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if self.dim_specified:
                self.output.send(torch.argsort(input_tensor, dim=self.dim, descending=self.descending(), stable=self.stable()))
            else:
                self.output.send(torch.argsort(input_tensor, descending=self.descending(), stable=self.stable()))


