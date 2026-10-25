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
    Node.app.register_node('t.argmin', TorchArgMaxNode.factory)
    Node.app.register_node('t.argwhere', TorchArgWhereNode.factory)
    Node.app.register_node('t.non_zero', TorchArgWhereNode.factory)
    Node.app.register_node('t.argsort', TorchArgSortNode.factory)
    Node.app.register_node('t.sort', TorchSortNode.factory)
    Node.app.register_node('t.topk', TorchTopKNode.factory)
    Node.app.register_node('t.where', TorchWhereNode.factory)
    Node.app.register_node('t.unique', TorchUniqueNode.factory)
    Node.app.register_node('t.isnan', TorchFiniteTestNode.factory)
    Node.app.register_node('t.isfinite', TorchFiniteTestNode.factory)
    Node.app.register_node('t.isinf', TorchFiniteTestNode.factory)
    Node.app.register_node('t.nan_to_num', TorchNanToNumNode.factory)

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
            # torch.bincount requires non-negative ints; clamp at zero so we
            # produce a meaningful histogram instead of raising at the boundary.
            if input_tensor.numel() > 0 and bool((input_tensor < 0).any()):
                if self.app.verbose:
                    print(f'{self.label}: input contains negative values; clamping to 0')
                input_tensor = torch.clamp_min(input_tensor, 0)
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
            if data is None:
                return
            boundaries_tensor = self.data_to_tensor(data, match_tensor=input_tensor)
            if boundaries_tensor is None:
                return
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
            output_name = 'any'

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output(output_name)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            result = self.op(input_tensor)
            self.output.send(result)


class TorchHistogramNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchHistogramNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        bin_count = 100
        # Histogram bounds; min == max == 0 tells torch.histc to use the data's range.
        min_value = 0
        max_value = 0

        self.input = self.add_input('', triggers_execution=True)
        self.bin_count = self.add_input('bin count', widget_type='drag_int', default_value=bin_count, min=1)
        self.min = self.add_input('min', widget_type='drag_float', default_value=min_value)
        self.max = self.add_input('max', widget_type='drag_float', default_value=max_value)
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
        self.op = torch.argmin if label == 't.argmin' else torch.argmax
        self.input = self.add_input('tensor in', triggers_execution=True)
        if self.dim_specified:
            self.add_dim_input()
        self.output = self.add_output('min index' if label == 't.argmin' else 'max index')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if self.dim_specified:
                output_tensor = self.op(input_tensor, dim=self.dim)
            else:
                output_tensor = self.op(input_tensor)
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
        # Only override the default when no dim was passed in args; otherwise
        # TorchWithDimNode has already parsed the user-supplied value.
        if not self.dim_specified:
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




class TorchSortNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSortNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if not self.dim_specified:
            self.dim = -1
        self.input = self.add_input('tensor in', triggers_execution=True)
        if self.dim_specified:
            self.add_dim_input()
        self.descending = self.add_input('descending', widget_type='checkbox', default_value=False)
        self.stable = self.add_input('stable', widget_type='checkbox', default_value=False)
        self.output = self.add_output('sorted')
        self.index_output = self.add_output('indices')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        if not -input_tensor.ndim <= self.dim < max(input_tensor.ndim, 1):
            if self.app.verbose:
                print(self.label, 'dim', self.dim, 'out of range for shape', list(input_tensor.shape))
            return
        values, indices = torch.sort(input_tensor, dim=self.dim, descending=self.descending(),
                                     stable=self.stable())
        self.index_output.send(indices)
        self.output.send(values)


class TorchTopKNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchTopKNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        k = 3
        self.dim = -1
        dim_given = False
        if len(args) > 0:
            k = max(any_to_int(args[0]), 1)
        if len(args) > 1:
            self.dim = any_to_int(args[1])
            dim_given = True
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.k_input = self.add_input('k', widget_type='input_int', default_value=k, min=1)
        self.dim_input = None
        if dim_given:
            self.dim_input = self.add_input('dim', widget_type='input_int', default_value=self.dim)
        self.largest = self.add_input('largest', widget_type='checkbox', default_value=True)
        self.output = self.add_output('values')
        self.index_output = self.add_output('indices')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        dim = self.dim_input() if self.dim_input is not None else self.dim
        if not -input_tensor.ndim <= dim < max(input_tensor.ndim, 1):
            if self.app.verbose:
                print(self.label, 'dim', dim, 'out of range for shape', list(input_tensor.shape))
            return
        available = input_tensor.shape[dim] if input_tensor.ndim > 0 else 1
        # asking for more than there are gives all of them, best first
        k = min(max(self.k_input(), 1), available)
        values, indices = torch.topk(input_tensor, k, dim=dim, largest=self.largest())
        self.index_output.send(indices)
        self.output.send(values)


class TorchWhereNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchWhereNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        # t.where 1 0: numbers given here stand until a tensor or number arrives
        self.held = [None, None]
        for i in range(min(len(args), 2)):
            v, t = decode_arg(args, i)
            if t in [int, float]:
                self.held[i] = float(v)
        self.input = self.add_input('condition', triggers_execution=True)
        self.if_true_input = self.add_input('if true')
        self.if_false_input = self.add_input('if false')
        self.output = self.add_output('selected')

    def hold(self, data):
        # a single number stays a number, so it fills every place it is chosen
        tensor = held_tensor(data)
        if tensor is not None and tensor.numel() == 1:
            return float(tensor.item())
        return tensor

    def execute(self):
        for i, inlet in enumerate([self.if_true_input, self.if_false_input]):
            if inlet.fresh_input:
                self.held[i] = self.hold(inlet())
        condition = self.input_to_tensor()
        if condition is None or self.held[0] is None or self.held[1] is None:
            return
        if condition.dtype != torch.bool:
            condition = condition != 0
        choices = []
        like = next((h for h in self.held if isinstance(h, torch.Tensor)), None)
        for h in self.held:
            if isinstance(h, torch.Tensor):
                choices.append(h.to(device=condition.device, dtype=float_like(like).dtype))
            else:
                choices.append(h)
        if like is None:
            # two numbers: a float tensor on the condition's device
            choices[0] = torch.tensor(choices[0], device=condition.device)
        try:
            self.output.send(torch.where(condition, choices[0], choices[1]))
        except RuntimeError:
            if self.app.verbose:
                shapes = [list(c.shape) if isinstance(c, torch.Tensor) else 'number' for c in choices]
                print(self.label, 'shapes do not broadcast:', list(condition.shape), shapes)


class TorchUniqueNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchUniqueNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.values_output = self.add_output('values')
        self.counts_output = self.add_output('counts')
        self.inverse_output = self.add_output('inverse')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        values, inverse, counts = torch.unique(input_tensor, sorted=True, return_inverse=True,
                                               return_counts=True)
        self.inverse_output.send(inverse.reshape(input_tensor.shape))
        self.counts_output.send(counts)
        self.values_output.send(values)


class TorchFiniteTestNode(TorchNode):
    operations = {'t.isnan': torch.isnan, 't.isfinite': torch.isfinite, 't.isinf': torch.isinf}

    @staticmethod
    def factory(name, data, args=None):
        node = TorchFiniteTestNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = self.operations.get(label, torch.isnan)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('mask')
        self.count_output = self.add_output('count')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        mask = self.op(input_tensor)
        self.count_output.send(int(torch.count_nonzero(mask).item()))
        self.output.send(mask)


class TorchNanToNumNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchNanToNumNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        nan_value = 0.0
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                nan_value = float(v)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.nan_input = self.add_input('nan', widget_type='drag_float', default_value=nan_value)
        self.posinf_input = self.add_input('posinf', widget_type='drag_float', default_value=0.0)
        self.neginf_input = self.add_input('neginf', widget_type='drag_float', default_value=0.0)
        self.output = self.add_output('out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        if not input_tensor.is_floating_point():
            # whole numbers cannot hold nan or inf: nothing to replace
            self.output.send(input_tensor)
            return
        self.output.send(torch.nan_to_num(input_tensor, nan=self.nan_input(), posinf=self.posinf_input(),
                                          neginf=self.neginf_input()))
