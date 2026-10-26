from dpg_system.torch_base_nodes import *

def register_torch_calculation_nodes():
    Node.app.register_node('t.cdist', TorchCDistanceNode.factory)
    Node.app.register_node('t.dist', TorchDistanceNode.factory)
    Node.app.register_node('t.length', TorchDistanceNode.factory)
    Node.app.register_node('t.cosine_similarity', CosineSimilarityNode.factory)

    Node.app.register_node('t.corrcoef', TorchCovarianceCoefficientNode.factory)

    Node.app.register_node('t.diff', TorchDiffNode.factory)
    Node.app.register_node('t.energy', TorchEnergyNode.factory)

    Node.app.register_node('t.eq', TorchComparisonNode.factory)
    Node.app.register_node('t.gt', TorchComparisonNode.factory)
    Node.app.register_node('t.lt', TorchComparisonNode.factory)
    Node.app.register_node('t.ge', TorchComparisonNode.factory)
    Node.app.register_node('t.le', TorchComparisonNode.factory)
    Node.app.register_node('t.ne', TorchComparisonNode.factory)
    Node.app.register_node('t.minimum', TorchMinimumMaximumNode.factory)
    Node.app.register_node('t.maximum', TorchMinimumMaximumNode.factory)

    Node.app.register_node('t.clamp', TorchClampNode.factory)

    Node.app.register_node('t.real', TorchRealImaginaryNode.factory)
    Node.app.register_node('t.imag', TorchRealImaginaryNode.factory)
    Node.app.register_node('t.complex', TorchComplexNode.factory)

    Node.app.register_node('t.round', TorchRoundNode.factory)
    Node.app.register_node('t.floor', TorchFloorCeilingTruncNode.factory)
    Node.app.register_node('t.ceil', TorchFloorCeilingTruncNode.factory)
    Node.app.register_node('t.trunc', TorchFloorCeilingTruncNode.factory)
    Node.app.register_node('t.frac', TorchFloorCeilingTruncNode.factory)

    Node.app.register_node('t.cumsum', TorchCumSumNode.factory)
    Node.app.register_node('t.cumprod', TorchCumSumNode.factory)
    Node.app.register_node('t.cummax', TorchCumSumNode.factory)
    Node.app.register_node('t.cummin', TorchCumSumNode.factory)
    Node.app.register_node('t.logcumsumexp', TorchCumSumNode.factory)

    Node.app.register_node('t.normalize', TorchNormalizeNode.factory)

    Node.app.register_node('t.copysign', TorchCopySignNode.factory)

    Node.app.register_node('t.linalg.qr', TorchLinalgRQNode.factory)
    Node.app.register_node('t.linalg.svd', TorchLinalgSVDNode.factory)
    Node.app.register_node('t.linalg.pca_low_rank', TorchPCALowRankNode.factory)
    Node.app.register_node('t.linalg.eig', TorchLinalgEigenNode.factory)
    Node.app.register_node('t.linalg.norm', TorchLinalgNormNode.factory)
    Node.app.register_node('t.linalg.inv', TorchUnaryLinalgNode.factory)
    Node.app.register_node('t.linalg.det', TorchUnaryLinalgNode.factory)
    Node.app.register_node('t.linalg.pinv', TorchUnaryLinalgNode.factory)
    Node.app.register_node('t.linalg.solve', TorchLinalgSolveNode.factory)
    Node.app.register_node('t.linalg.lstsq', TorchLinalgSolveNode.factory)
    Node.app.register_node('t.matmul', TorchMatMulNode.factory)
    Node.app.register_node('t.einsum', TorchEinsumNode.factory)
    Node.app.register_node('t.cross', TorchVectorProductNode.factory)
    Node.app.register_node('t.dot', TorchVectorProductNode.factory)
    Node.app.register_node('t.outer', TorchVectorProductNode.factory)
    Node.app.register_node('t.cov', TorchCovNode.factory)
    Node.app.register_node('t.linalg.eigh', TorchEighNode.factory)
    Node.app.register_node('t.gradient', TorchGradientNode.factory)
    Node.app.register_node('t.trapezoid', TorchTrapezoidNode.factory)
    Node.app.register_node('t.cumulative_trapezoid', TorchTrapezoidNode.factory)
    Node.app.register_node('t.logical_and', TorchLogicalNode.factory)
    Node.app.register_node('t.logical_or', TorchLogicalNode.factory)
    Node.app.register_node('t.logical_xor', TorchLogicalNode.factory)
    Node.app.register_node('t.logical_not', TorchLogicalNode.factory)

    Node.app.register_node('t.gcd', TorchLCMGCDNode.factory)
    Node.app.register_node('t.lcm', TorchLCMGCDNode.factory)
    Node.app.register_node('t.mean', TorchMeanMedianNode.factory)
    Node.app.register_node('t.median', TorchMeanMedianNode.factory)
    Node.app.register_node('t.nanmean', TorchMeanMedianNode.factory)
    Node.app.register_node('t.nanmedian', TorchMeanMedianNode.factory)
    Node.app.register_node('t.sum', TorchMeanMedianNode.factory)
    Node.app.register_node('t.nansum', TorchMeanMedianNode.factory)
    Node.app.register_node('t.prod', TorchMeanMedianNode.factory)
    Node.app.register_node('t.std', TorchStdVarNode.factory)
    Node.app.register_node('t.var', TorchStdVarNode.factory)
    Node.app.register_node('t.quantile', TorchQuantileNode.factory)
    Node.app.register_node('t.mode', TorchModeNode.factory)

    Node.app.register_node('t.bernoulli', TorchDistributionTensorNode.factory)
    Node.app.register_node('t.poisson', TorchDistributionTensorNode.factory)
    Node.app.register_node('t.exponential', TorchDistributionTensorOneParamNode.factory)
    Node.app.register_node('t.geometric', TorchDistributionTensorOneParamNode.factory)
    Node.app.register_node('t.cauchy', TorchDistributionTensorTwoParamNode.factory)
    Node.app.register_node('t.log_normal', TorchDistributionTensorTwoParamNode.factory)
    Node.app.register_node('t.normal', TorchDistributionTensorTwoParamNode.factory)
    Node.app.register_node('t.uniform', TorchDistributionTensorTwoParamNode.factory)


class TorchCovarianceCoefficientNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCovarianceCoefficientNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            result = torch.corrcoef(input_tensor)
            self.output.send(result)


class TorchDistributionNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.distribution = torch.distributions.bernoulli.Bernoulli(probs=0.5)
        self.input = self.add_input('trigger', triggers_execution=True)
        self.probability = self.add_input('probability', widget_type='drag_float', min=0.0, max=1.0, default_value=0.5, callback=self.prob_changed)
        self.output = self.add_output('tensor out')

    def prob_changed(self):
        self.distribution = torch.distributions.bernoulli.Bernoulli(probs=self.probability())

    def execute(self):
        out_tensor = self.distribution.sample()
        self.output.send(out_tensor)


class TorchDistributionTensorNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionTensorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if self.label == 't.bernoulli':
            self.op = torch.bernoulli
        elif self.label == 't.poisson':
            self.op = torch.poisson

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        # torch.bernoulli requires input in [0, 1]; torch.poisson requires >= 0.
        # Validate before calling so the user gets a clear message instead of
        # the cryptic underlying RuntimeError.
        if self.label == 't.bernoulli':
            if input_tensor.min() < 0 or input_tensor.max() > 1:
                if self.app.verbose:
                    print(f'{self.label}: input must be in [0, 1], got '
                          f'[{input_tensor.min().item():.4g}, {input_tensor.max().item():.4g}]')
                return
        elif self.label == 't.poisson':
            if input_tensor.min() < 0:
                if self.app.verbose:
                    print(f'{self.label}: input must be non-negative, got '
                          f'min {input_tensor.min().item():.4g}')
                return
        try:
            out_tensor = self.op(input_tensor)
            self.output.send(out_tensor)
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()


class TorchDistributionTensorOneParamNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionTensorOneParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        param_1_name = ''
        param_1 = 1.0
        param_min = None
        param_max = None
        if self.label == 't.exponential':
            param_1_name = 'lambda'
            # exponential_ requires lambda > 0
            param_1 = 1.0
            param_min = 1e-6
        elif self.label == 't.geometric':
            param_1_name = 'p'
            # geometric_ requires p in (0, 1) — exclusive at both ends.
            param_1 = 0.5
            param_min = 1e-6
            param_max = 1.0 - 1e-6

        self.input = self.add_input('tensor in', triggers_execution=True)
        kwargs = {'widget_type': 'drag_float', 'default_value': param_1}
        if param_min is not None:
            kwargs['min'] = param_min
        if param_max is not None:
            kwargs['max'] = param_max
        self.param_1 = self.add_input(param_1_name, **kwargs)
        self.output = self.add_output('tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            out_tensor = input_tensor.clone()
            # get_data: the widget's value, clamped to its limits (a raw 0 sent
            # to the inlet would otherwise reach torch once)
            try:
                if self.label == 't.exponential':
                    out_tensor.exponential_(self.param_1.get_data())
                elif self.label == 't.geometric':
                    out_tensor.geometric_(self.param_1.get_data())
            except Exception as e:
                print(f'{self.label}: {type(e).__name__}: {e}')
                return
            self.output.send(out_tensor)


class TorchDistributionTensorTwoParamNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionTensorTwoParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        param_1_name = ''
        param_2_name = ''
        param_1 = 0
        param_2 = 1
        if self.label == 't.cauchy':
            param_1_name = 'median'
            param_2_name = 'sigma'
        elif self.label in ['t.log_normal', 't.normal']:
            param_1_name = 'mean'
            param_2_name = 'std'
        elif self.label == 't.uniform':
            param_1_name = 'from'
            param_2_name = 'to'

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.param_1 = self.add_input(param_1_name, widget_type='drag_float', default_value=param_1)
        self.param_2 = self.add_input(param_2_name, widget_type='drag_float', default_value=param_2)
        self.output = self.add_output('tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            out_tensor = input_tensor.clone()
            try:
                if self.label == 't.cauchy':
                    out_tensor.cauchy_(self.param_1(), self.param_2())
                elif self.label == 't.log_normal':
                    out_tensor.log_normal_(self.param_1(), self.param_2())
                elif self.label == 't.normal':
                    out_tensor.normal_(self.param_1(), self.param_2())
                elif self.label == 't.uniform':
                    out_tensor.uniform_(self.param_1(), self.param_2())
            except Exception as e:
                print(f'{self.label}: {type(e).__name__}: {e}')
                return
            self.output.send(out_tensor)


class TorchCDistanceNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCDistanceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('input', triggers_execution=True)
        self.input2 = self.add_input('points 2')
        self.output = self.add_output('output')
        self.points_2 = None

    def execute(self):
        # rows are points. With nothing in 'points 2' the answer is every row's
        # distance to every other row; with points there, from each input row to
        # each of those. A single vector still gives its length, as it always did
        # (this node used to do only that, and raised on more than one row).
        if self.input2.fresh_input:
            data = self.input2()
            # 'clear' (or an empty list) lets go of the held points
            if isinstance(data, str) or (isinstance(data, (list, tuple))
                                         and (len(data) == 0 or isinstance(data[0], str))):
                self.points_2 = None
            else:
                self.points_2 = self.data_to_tensor(data)
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        try:
            if input_tensor.ndim < 2 and self.points_2 is None:
                self.output.send(torch.linalg.vector_norm(input_tensor.float()).item())
                return
            a = input_tensor.float()
            a = a.reshape(1, -1) if a.ndim < 2 else a
            b = a if self.points_2 is None else self.points_2.to(a.device).float()
            b = b.reshape(1, -1) if b.ndim < 2 else b
            self.output.send(torch.cdist(a, b))
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')


class TorchDistanceNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistanceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        out_label = 'length'
        self.input2 = None
        self.input = self.add_input('tensor in', triggers_execution=True)
        if self.label == 't.dist':
            self.input2 = self.add_input("tensor 2 in")
            out_label = 'distance'

        self.output = self.add_output(out_label)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if self.input2 is not None:
                # t.dist: distance between two tensors. Skip if the second
                # tensor isn't connected/convertible yet.
                input2 = self.data_to_tensor(self.input2(), match_tensor=input_tensor)
                if input2 is None:
                    return
                euclidean_length = torch.dist(input_tensor, input2)
            else:
                # t.length: distance from the origin.
                euclidean_length = torch.dist(input_tensor, torch.zeros_like(input_tensor))

            self.output.send(euclidean_length.item())


class TorchDiffNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDiffNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('tensor in', triggers_execution=True)
        # Only override the default when no dim was passed in args; otherwise
        # TorchWithDimNode has already parsed the user-supplied value.
        if not self.dim_specified:
            self.dim = -1
        n = 1
        self.n = self.add_input('n', widget_type='input_int', default_value=n)
        if self.dim_specified:
            self.add_dim_input()
        self.output = self.add_output('tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            try:
                if self.dim_specified:
                    result = torch.diff(input_tensor, n=self.n(), dim=self.dim)
                else:
                    result = torch.diff(input_tensor, n=self.n())
                self.output.send(result)
            except Exception as e:
                traceback.print_exc()
                if self.app.verbose:
                    print(self.label, e)


class TorchEnergyNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchEnergyNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('tensor in', triggers_execution=True)
        if not self.dim_specified:
            self.dim = -1
        n = 1
        self.n = self.add_input('n', widget_type='input_int', default_value=n)
        if self.dim_specified:
            self.add_dim_input()
        self.output = self.add_output('tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            try:
                if self.dim_specified:
                    result = torch.diff(input_tensor, n=self.n(), dim=self.dim)
                else:
                    result = torch.diff(input_tensor, n=self.n())
                result = result.abs().sum()
                self.output.send(result)
            except Exception as e:
                traceback.print_exc()
                if self.app.verbose:
                    print(self.label, e)


class TorchMinimumMaximumNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchMinimumMaximumNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        output_label = 'maximum tensor'
        self.op = torch.maximum
        if self.label == 't.minimum':
            output_label = 'minimum tensor'
            self.op = torch.minimum
        self.input = self.add_input('tensor a in', triggers_execution=True)
        self.input_2 = self.add_input('tensor b in')

        self.output = self.add_output(output_label)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.input_2()
            if data is not None:
                input_tensor_2 = self.data_to_tensor(data, match_tensor=input_tensor)
                if input_tensor_2 is not None:
                    output_tensor = self.op(input_tensor, input_tensor_2)
                    self.output.send(output_tensor)


class TorchComparisonNode(TorchNode):
    op_dict = {
        't.eq': torch.eq,
        't.gt': torch.gt,
        't.lt': torch.lt,
        't.ge': torch.ge,
        't.le': torch.le,
        't.ne': torch.ne
    }
    @staticmethod
    def factory(name, data, args=None):
        node = TorchComparisonNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        output_label = 'tensor result'
        self.op = torch.eq
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
        self.input = self.add_input('tensor a in', triggers_execution=True)
        self.input_2 = self.add_input('tensor b in')

        self.output = self.add_output(output_label)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.input_2()
            if data is not None:
                input_tensor_2 = self.data_to_tensor(data, match_tensor=input_tensor)
                if input_tensor_2 is not None:
                    output_tensor = self.op(input_tensor, input_tensor_2)
                    self.output.send(output_tensor)


class TorchLCMGCDNode(TorchNode):
    op_dict = {
        't.gcd': torch.gcd,
        't.lcm': torch.lcm
    }
    @staticmethod
    def factory(name, data, args=None):
        node = TorchLCMGCDNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.op = torch.gcd
        # Default label matches the default op; override only when the lookup
        # actually selects a different op. Previously this branch unconditionally
        # set output_name = 'lcm' even for t.gcd.
        output_name = 'gcd'
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
            if self.label == 't.lcm':
                output_name = 'lcm'
        self.input = self.add_input('tensor a in', triggers_execution=True)
        self.input_2 = self.add_input('tensor b in')

        self.output = self.add_output(output_name + ' tensor out')

    @staticmethod
    def to_whole(tensor_):
        # gcd / lcm are integer-only: round float values to the nearest whole
        # number (so 12.0, or a list arriving as a float tensor, just works)
        if tensor_.is_floating_point():
            tensor_ = torch.round(tensor_).to(torch.int64)
        elif tensor_.dtype == torch.bool:
            tensor_ = tensor_.to(torch.int64)
        return tensor_

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.input_2()
            if data is not None:
                input_tensor_2 = any_to_tensor(data, device=input_tensor.device)
                if input_tensor_2 is not None:
                    try:
                        output_tensor = self.op(self.to_whole(input_tensor), self.to_whole(input_tensor_2))
                        self.output.send(output_tensor)
                    except Exception as e:
                        traceback.print_exc()
                        if self.app.verbose:
                            print(self.label, e)


# class TorchCropNode(Node):
#     @staticmethod
#     def factory(name, data, args=None):
#         node = TorchCropNode(name, data, args)
#         return node
#
#     def __init__(self, label: str, data, args):
#         super().__init__(label, data, args)
#         self.dim_count = 2
#         if len(args) > 0:
#             self.dim_count = string_to_int(args[0])
#
#         self.input = self.add_input("tensor in", triggers_execution=True)
#         self.croppers = []
#         for i in range(self.dims_count):
#             crop_min = self.add_input('dim ' + str(i) + ' min', widget_type='dragint', )
#             crop_max = self.add_input('dim ' + str(i) + ' max', widget_type='dragint', )
#         self.indices_property = self.add_property('', widget_type='text_input', width=200, default_value=index_string,
#                                                   callback=self.dim_changed)
#         self.output = self.add_output("output")


class TorchMeanMedianNode(TorchWithDimNode):
    op_dict = {
        't.sum': torch.sum,
        't.mean': torch.mean,
        't.median': torch.median,
        't.nansum': torch.nansum,
        't.nanmean': torch.nanmean,
        't.nanmedian': torch.nanmedian,
        't.prod': torch.prod
        }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchMeanMedianNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.op = torch.mean
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
        if not self.dim_specified:
            self.dim = -1
        keep_dims = False
        if self.dim_specified:
            self.add_dim_input()
            self.keep_dims = self.add_input('keep_dims', widget_type='checkbox', default_value=keep_dims)
        self.output = self.add_output('output')
        if self.label in ['t.median', 't.nanmedian'] and self.dim_specified:
            self.index_out = self.add_output("index output")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if self.dim_specified:
                if self.label in ['t.median', 't.nanmedian']:
                    out_tensor, index_tensor = self.op(input_tensor, dim=self.dim, keepdim=self.keep_dims())
                    self.index_out.send(index_tensor)
                    self.output.send(out_tensor)
                else:
                    self.output.send(self.op(input_tensor, dim=self.dim, keepdim=self.keep_dims()))
            else:
                self.output.send(self.op(input_tensor))


class TorchCumSumNode(TorchWithDimNode):
    op_dict = {
        't.cumsum': torch.cumsum,
        't.cumprod': torch.cumprod,
        't.cummax': torch.cummax,
        't.cummin': torch.cummin,
        't.logcumsumexp': torch.logcumsumexp
    }
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCumSumNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = torch.cumsum
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
        self.input = self.add_input('tensor in', triggers_execution=True)
        if self.dim_specified:
            self.add_dim_input()
        self.output = self.add_output('output')
        if self.label in ['t.cummax', 't.cummin']:
            self.index_output = self.add_output('indices')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            ndim = max(input_tensor.ndim, 1)
            if -ndim <= self.dim < ndim:
                if self.label in ['t.cummax', 't.cummin']:
                    output_tensor, indices_tensor = self.op(input_tensor, self.dim)
                    self.index_output.send(indices_tensor)
                    self.output.send(output_tensor)

                else:
                    self.output.send(self.op(input_tensor, self.dim))

class TorchRealImaginaryNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchRealImaginaryNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.real = True
        output_name = 'real tensor'
        if self.label == 't.imag':
            self.real = False
            output_name = 'imaginary tensor'

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output(output_name)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if input_tensor.dtype in [torch.complex32, torch.complex64, torch.complex128]:
                if self.real:
                    self.output.send(torch.real(input_tensor))
                else:
                    self.output.send(torch.imag(input_tensor))


class TorchComplexNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchComplexNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.real = True
        self.input = self.add_input('real tensor in', triggers_execution=True)
        self.imag_input = self.add_input('imag tensor in', triggers_execution=True)
        self.output = self.add_output('complex tensor out')

    def execute(self):
        real_tensor = self.input_to_tensor()
        if real_tensor is None:
            if self.app.verbose:
                print(self.label, 'real tensor is None')
            return
        data = self.imag_input()
        if data is None:
            if self.app.verbose:
                print(self.label, 'no input for imaginary tensor')
            return
        imag_tensor = self.data_to_tensor(data, match_tensor=real_tensor)
        if imag_tensor is None:
            if self.app.verbose:
                print(self.label, 'could not convert imaginary input to tensor')
            return
        if real_tensor.shape != imag_tensor.shape:
            if self.app.verbose:
                print(self.label, 'real and imaginary tensors have different shapes',
                      real_tensor.shape, imag_tensor.shape)
            return
        if real_tensor.dtype not in [torch.float16, torch.float32, torch.float64]:
            if self.app.verbose:
                print(self.label, 'real tensor wrong dtype', real_tensor.dtype)
            return
        if real_tensor.dtype != imag_tensor.dtype:
            if self.app.verbose:
                print(self.label, "real and imaginary tensor dtypes don't match",
                      real_tensor.dtype, imag_tensor.dtype)
            return
        complex_tensor = torch.complex(real_tensor, imag_tensor)
        self.output.send(complex_tensor)


class TorchClampNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchClampNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        min_value = 0.0
        max_value = 1.0
        if len(args) > 1:
            min_value = any_to_float(args[0])
            max_value = any_to_float(args[1])
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.min = self.add_input('min', widget_type='drag_float', default_value=min_value)
        self.max = self.add_input('max', widget_type='drag_float', default_value=max_value)
        self.output = self.add_output('output')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(torch.clamp(input_tensor, self.min(), self.max()))


class TorchRoundNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchRoundNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        decimals = 0
        self.decimals = self.add_input('decimals', widget_type='input_int', default_value=decimals)
        self.output = self.add_output('output')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(torch.round(input_tensor, decimals=self.decimals()))


class TorchFloorCeilingTruncNode(TorchNode):
    op_dict = {
        't.floor': torch.floor,
        't.ceil': torch.ceil,
        't.trunc': torch.trunc,
        't.frac': torch.frac
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchFloorCeilingTruncNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = torch.ceil
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('output')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor))


class TorchCopySignNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCopySignNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.sign_input = self.add_input('sign tensor')
        self.output = self.add_output('tensor with copied sign')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.sign_input()
            if data is not None:
                sign_tensor = self.data_to_tensor(data, match_tensor=input_tensor)
                if sign_tensor is not None:
                    try:
                        self.output.send(torch.copysign(input_tensor, sign_tensor))
                    except Exception as error:
                        print(f't.copysign: {type(error).__name__}: {error}')
                        traceback.print_exc()


class CosineSimilarityNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = CosineSimilarityNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.vector_2 = None
        # CosineSimilarity is stateless; per-instance construction is fine and
        # avoids the confused class-level/instance-level "inited" pattern that
        # was here before (which never actually short-circuited).
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        self.input = self.add_input('input 1', triggers_execution=True)
        self.input2 = self.add_input('input 2')
        self.output = self.add_output('output')

    def execute(self):
        vector_1 = self.input_to_tensor()
        if vector_1 is not None:
            # input 2 is held: it used to be needed fresh before every input 1
            if self.input2.fresh_input:
                self.vector_2 = self.data_to_tensor(self.input2(), match_tensor=vector_1)
            if self.vector_2 is not None:
                try:
                    similarity = self.cos(vector_1, self.vector_2)
                    self.output.send(similarity.item() if similarity.numel() == 1 else similarity)
                except Exception as e:
                    print(f'{self.label}: {type(e).__name__}: {e}')
                    traceback.print_exc()


class TorchNormalizeNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchNormalizeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.dim_input = self.add_input('dim', widget_type='input_int', default_value=-1)
        self.normalized_output = self.add_output('normalized tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            n = torch.nn.functional.normalize(input_tensor, dim=self.dim_input())
            self.normalized_output.send(n)


class TorchLinalgRQNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchLinalgRQNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.mode = 'reduced'
        self.mode_property = self.add_property('mode', widget_type='combo', default_value=self.mode, callback=self.mode_changed)
        self.mode_property.widget.combo_items = ['reduced', 'complete', 'r']
        self.q_output = self.add_output('Q tensor out')
        self.r_output = self.add_output('R tensor out')

    def mode_changed(self):
        self.mode = self.mode_property()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            q, r = torch.linalg.qr(input_tensor, self.mode)
            self.r_output.send(r)
            self.q_output.send(q)


class TorchLinalgSVDNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchLinalgSVDNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.full = True
        self.full_property = self.add_property('full', widget_type='checkbox', default_value=self.full, callback=self.full_changed)
        # the outlets always carried U, S and Vh in this order, but were
        # labelled S, V and D. The new names must not reuse the old ones: links
        # are matched by name first, and 'S tensor out' would move a saved link
        # from outlet 0 (U) to outlet 1. The archive keeps old patches wired to
        # the same data.
        self.u_output = self.add_output('U out')
        self.u_output.name_archive.append('S tensor out')
        self.s_output = self.add_output('S out')
        self.s_output.name_archive.append('V tensor out')
        self.vh_output = self.add_output('Vh out')
        self.vh_output.name_archive.append('D tensor out')

    def full_changed(self):
        self.full = self.full_property()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            u, s, vh = torch.linalg.svd(input_tensor, full_matrices=self.full)
            self.vh_output.send(vh)
            self.s_output.send(s)
            self.u_output.send(u)


class TorchPCALowRankNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchPCALowRankNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        center = True
        self.center = self.add_property('center', widget_type='checkbox', default_value=center)
        niter = 2
        self.niter = self.add_property('iterations', widget_type='input_int', default_value=niter, min=0)
        self.u_output = self.add_output('U tensor out')
        self.s_output = self.add_output('S tensor out')
        self.v_output = self.add_output('V tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            u, s, v = torch.pca_lowrank(input_tensor, center=self.center(), niter=self.niter())
            self.v_output.send(v)
            self.s_output.send(s)
            self.u_output.send(u)


class TorchLinalgEigenNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchLinalgEigenNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.l_output = self.add_output('L tensor out')
        self.v_output = self.add_output('V tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if len(input_tensor.shape) > 1:
                if input_tensor.shape[-1] == input_tensor.shape[-2]:
                    l, v = torch.linalg.eig(input_tensor)
                    self.v_output.send(v)
                    self.l_output.send(l)
                else:
                    if self.app.verbose:
                        print(self.label, 'tensor is not square')
            else:
                if self.app.verbose:
                    print(self.label, 'tensor has less than 2 dimensions')




class TorchMatMulNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchMatMulNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.operand = None
        self.input = self.add_input('in 1', triggers_execution=True)
        self.operand_input = self.add_input('in 2')
        self.output = self.add_output('result')

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = held_tensor(self.operand_input())
        input_tensor = self.input_to_tensor()
        if input_tensor is None or self.operand is None:
            return
        a = float_like(input_tensor)
        b = self.data_to_tensor(self.operand, match_tensor=a)
        try:
            # batched: a stack of matrices times a matrix (or a matching stack)
            self.output.send(torch.matmul(a, b))
        except RuntimeError as e:
            if self.app.verbose:
                print(self.label, 'shapes do not fit:', list(a.shape), 'and', list(b.shape), e)


class TorchEinsumNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchEinsumNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        # 'ij, jk -> ik' arrives as several args; spaces mean nothing to einsum
        subscripts = ''.join(str(a) for a in args) if len(args) > 0 else 'ij,jk->ik'
        self.operand_count = subscripts.split('->')[0].count(',') + 1
        self.held = [None] * self.operand_count
        self.input = self.add_input('in', triggers_execution=True)
        self.operand_inputs = [self.input]
        for i in range(1, self.operand_count):
            self.operand_inputs.append(self.add_input('in ' + str(i + 1)))
        self.output = self.add_output('result')
        self.subscripts_option = self.add_option('subscripts', widget_type='text_input', default_value=subscripts)

    def execute(self):
        for i in range(1, self.operand_count):
            if self.operand_inputs[i].fresh_input:
                self.held[i] = held_tensor(self.operand_inputs[i]())
        input_tensor = self.input_to_tensor()
        if input_tensor is None or any(h is None for h in self.held[1:]):
            return
        first = float_like(input_tensor)
        operands = [first] + [self.data_to_tensor(h, match_tensor=first) for h in self.held[1:]]
        subscripts = self.subscripts_option().replace(' ', '')
        if subscripts.split('->')[0].count(',') + 1 != self.operand_count:
            if self.app.verbose:
                print(self.label, subscripts, 'names a different number of operands than the',
                      self.operand_count, 'inlets this node was made with')
            return
        try:
            self.output.send(torch.einsum(subscripts, *operands))
        except RuntimeError as e:
            if self.app.verbose:
                print(self.label, e)


class TorchLinalgNormNode(TorchWithDimNode):
    # the name on the menu -> the ord torch.linalg.vector_norm takes
    orders = {'2': 2, '1': 1, 'inf': float('inf'), '0': 0}

    @staticmethod
    def factory(name, data, args=None):
        node = TorchLinalgNormNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        if self.dim_specified:
            self.add_dim_input()
            self.keep_dims = self.add_input('keep_dims', widget_type='checkbox', default_value=False)
        self.output = self.add_output('norm')
        self.ord_option = self.add_option('ord', widget_type='combo', default_value='2')
        self.ord_option.widget.combo_items = list(self.orders.keys())

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        input_tensor = float_like(input_tensor)
        order = self.orders.get(self.ord_option(), 2)
        if self.dim_specified:
            if not -input_tensor.ndim <= self.dim < max(input_tensor.ndim, 1):
                if self.app.verbose:
                    print(self.label, 'dim', self.dim, 'out of range for shape', list(input_tensor.shape))
                return
            self.output.send(torch.linalg.vector_norm(input_tensor, ord=order, dim=self.dim,
                                                      keepdim=self.keep_dims()))
        else:
            self.output.send(torch.linalg.vector_norm(input_tensor, ord=order))


class TorchUnaryLinalgNode(TorchNode):
    operations = {'t.linalg.inv': (torch.linalg.inv, 'inverse', True),
                  't.linalg.det': (torch.linalg.det, 'determinant', True),
                  't.linalg.pinv': (torch.linalg.pinv, 'pseudo-inverse', False)}

    @staticmethod
    def factory(name, data, args=None):
        node = TorchUnaryLinalgNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op, output_name, self.square_only = self.operations.get(label, self.operations['t.linalg.inv'])
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output(output_name)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        input_tensor = float_like(input_tensor)
        if input_tensor.ndim < 2:
            if self.app.verbose:
                print(self.label, 'needs a matrix, got shape', list(input_tensor.shape))
            return
        if self.square_only and input_tensor.shape[-1] != input_tensor.shape[-2]:
            if self.app.verbose:
                print(self.label, 'needs a square matrix, got shape', list(input_tensor.shape))
            return
        try:
            self.output.send(run_with_cpu_fallback(self.op, input_tensor))
        except RuntimeError as e:
            # a singular matrix has no inverse
            if self.app.verbose:
                print(self.label, e)


class TorchLinalgSolveNode(TorchNode):
    # t.linalg.solve: the x for which A x = b exactly (A square);
    # t.linalg.lstsq: the x that comes closest when there are more equations than unknowns
    @staticmethod
    def factory(name, data, args=None):
        node = TorchLinalgSolveNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.least_squares = label == 't.linalg.lstsq'
        self.b_value = None
        self.input = self.add_input('A matrix', triggers_execution=True)
        self.b_input = self.add_input('b')
        self.output = self.add_output('x')
        self.residuals_output = None
        if self.least_squares:
            self.residuals_output = self.add_output('residuals')

    def execute(self):
        if self.b_input.fresh_input:
            self.b_value = held_tensor(self.b_input())
        input_tensor = self.input_to_tensor()
        if input_tensor is None or self.b_value is None:
            return
        a = float_like(input_tensor)
        b = self.data_to_tensor(self.b_value, match_tensor=a)
        if a.ndim != 2:
            if self.app.verbose:
                print(self.label, 'A must be a 2-D matrix, got shape', list(a.shape))
            return
        if b.shape[0] != a.shape[0]:
            if self.app.verbose:
                print(self.label, 'b needs', a.shape[0], 'rows to match A, got shape', list(b.shape))
            return
        if not self.least_squares and a.shape[0] != a.shape[1]:
            if self.app.verbose:
                print(self.label, 'A must be square (t.linalg.lstsq takes any shape), got', list(a.shape))
            return
        vector = b.ndim == 1
        column_b = b.unsqueeze(-1) if vector else b
        try:
            if self.least_squares:
                x = run_with_cpu_fallback(torch.linalg.lstsq, a, column_b)[0]
            else:
                x = torch.linalg.solve(a, column_b)
        except RuntimeError as e:
            if self.app.verbose:
                print(self.label, e)
            return
        if self.least_squares:
            # the squared miss left over in each column of b
            residuals = ((a @ x - column_b) ** 2).sum(dim=0)
            self.residuals_output.send(residuals[0] if vector else residuals)
        self.output.send(x.squeeze(-1) if vector else x)


class TorchStdVarNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchStdVarNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = torch.var if label == 't.var' else torch.std
        self.input = self.add_input('tensor in', triggers_execution=True)
        if self.dim_specified:
            self.add_dim_input()
            self.keep_dims = self.add_input('keep_dims', widget_type='checkbox', default_value=False)
        self.output = self.add_output('output')
        # 1 (torch's default) divides by n - 1, the estimate from a sample;
        # 0 divides by n, as np.std and np.var do
        self.correction_option = self.add_option('correction', widget_type='input_int', default_value=1, min=0)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        input_tensor = float_like(input_tensor)
        correction = max(self.correction_option(), 0)
        if self.dim_specified:
            if not -input_tensor.ndim <= self.dim < max(input_tensor.ndim, 1):
                if self.app.verbose:
                    print(self.label, 'dim', self.dim, 'out of range for shape', list(input_tensor.shape))
                return
            self.output.send(self.op(input_tensor, dim=self.dim, correction=correction,
                                     keepdim=self.keep_dims()))
        else:
            self.output.send(self.op(input_tensor, correction=correction))


class TorchQuantileNode(TorchNode):
    interpolations = ['linear', 'lower', 'higher', 'nearest', 'midpoint']

    @staticmethod
    def factory(name, data, args=None):
        node = TorchQuantileNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.q = 0.5
        self.dim = None
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                self.q = float(v)
        if len(args) > 1:
            v, t = decode_arg(args, 1)
            if t == int:
                self.dim = v
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.q_input = self.add_input('q', widget_type='drag_float', default_value=self.q, callback=self.q_changed)
        self.dim_input = None
        if self.dim is not None:
            self.dim_input = self.add_input('dim', widget_type='input_int', default_value=self.dim)
            self.keep_dims = self.add_input('keep_dims', widget_type='checkbox', default_value=False)
        self.output = self.add_output('output')
        self.interpolation_option = self.add_option('interpolation', widget_type='combo', default_value='linear')
        self.interpolation_option.widget.combo_items = self.interpolations
        self.ignore_nan_option = self.add_option('ignore nan', widget_type='checkbox', default_value=False)

    def q_changed(self):
        # a list asks for several quantiles at once
        value = any_to_numerical(self.q_input(), validate=True)
        if value is not None:
            self.q = value

    def execute(self):
        if self.q_input.fresh_input:
            self.q_changed()
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        input_tensor = float_like(input_tensor)
        if input_tensor.is_complex():
            return
        q = torch.as_tensor(np.clip(np.asarray(self.q, dtype=float), 0.0, 1.0),
                            dtype=input_tensor.dtype, device=input_tensor.device)
        op = torch.nanquantile if self.ignore_nan_option() else torch.quantile
        interpolation = self.interpolation_option()
        if interpolation not in self.interpolations:
            interpolation = 'linear'
        try:
            if self.dim_input is not None:
                dim = self.dim_input()
                if not -input_tensor.ndim <= dim < max(input_tensor.ndim, 1):
                    if self.app.verbose:
                        print(self.label, 'dim', dim, 'out of range for shape', list(input_tensor.shape))
                    return
                self.output.send(op(input_tensor, q, dim=dim, keepdim=self.keep_dims(),
                                    interpolation=interpolation))
            else:
                self.output.send(op(input_tensor, q, interpolation=interpolation))
        except RuntimeError as e:
            # torch.quantile refuses inputs of more than 16 million values
            if self.app.verbose:
                print(self.label, e)


class TorchModeNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchModeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if not self.dim_specified:
            self.dim = -1
        self.input = self.add_input('tensor in', triggers_execution=True)
        if self.dim_specified:
            self.add_dim_input()
            self.keep_dims = self.add_input('keep_dims', widget_type='checkbox', default_value=False)
        self.output = self.add_output('mode')
        self.index_output = self.add_output('indices')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        if not -input_tensor.ndim <= self.dim < max(input_tensor.ndim, 1):
            if self.app.verbose:
                print(self.label, 'dim', self.dim, 'out of range for shape', list(input_tensor.shape))
            return
        keep = self.keep_dims() if self.dim_specified else False
        values, indices = run_with_cpu_fallback(torch.mode, input_tensor, self.dim, keep)
        self.index_output.send(indices)
        self.output.send(values)


class TorchVectorProductNode(TorchNode):
    # t.cross, t.dot and t.outer work on the vectors along the last dim (or the
    # dim chosen), so a stack of 24 vectors against one vector gives 24 answers
    @staticmethod
    def factory(name, data, args=None):
        node = TorchVectorProductNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.operand = None
        self.input = self.add_input('in 1', triggers_execution=True)
        self.operand_input = self.add_input('in 2')
        self.dim_input = None
        if label != 't.outer':
            dim = any_to_int(args[0]) if len(args) > 0 else -1
            self.dim_input = self.add_input('dim', widget_type='input_int', default_value=dim)
        names = {'t.cross': 'cross product', 't.dot': 'dot product', 't.outer': 'outer product'}
        self.output = self.add_output(names.get(label, 'result'))

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = held_tensor(self.operand_input())
        input_tensor = self.input_to_tensor()
        if input_tensor is None or self.operand is None:
            return
        a = float_like(input_tensor)
        b = self.data_to_tensor(self.operand, match_tensor=a)
        try:
            if self.label == 't.outer':
                # every pair of entries of the last dims multiplied: n x m
                result = a.unsqueeze(-1) * b.unsqueeze(-2)
            else:
                dim = self.dim_input()
                a, b = torch.broadcast_tensors(a, b)
                if self.label == 't.cross':
                    if a.shape[dim] != 3:
                        if self.app.verbose:
                            print(self.label, 'needs 3-vectors along dim', dim, 'got shape', list(a.shape))
                        return
                    result = torch.linalg.cross(a, b, dim=dim)
                else:
                    result = torch.linalg.vecdot(a, b, dim=dim)
        except (RuntimeError, IndexError) as e:
            if self.app.verbose:
                print(self.label, 'shapes do not fit:', list(a.shape), 'and', list(b.shape), e)
            return
        self.output.send(result)


class TorchCovNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCovNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        variables = 'columns'
        if len(args) > 0 and args[0] in ['rows', 'columns']:
            variables = args[0]
        self.input = self.add_input('in', triggers_execution=True)
        self.output = self.add_output('covariance')
        self.variables_option = self.add_option('variables', widget_type='combo', default_value=variables)
        self.variables_option.widget.combo_items = ['columns', 'rows']
        self.correction_option = self.add_option('correction', widget_type='input_int', default_value=1)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        data = float_like(input_tensor)
        if data.ndim != 2:
            if self.app.verbose:
                print(self.label, 'needs a 2-D tensor of observations, got shape', list(data.shape))
            return
        # torch.cov takes each row as a variable; the default here is columns (frames x channels)
        if self.variables_option() != 'rows':
            data = data.t()
        if data.shape[1] < 2:
            if self.app.verbose:
                print(self.label, 'needs at least 2 observations of each variable')
            return
        self.output.send(torch.cov(data, correction=self.correction_option()))


class TorchEighNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchEighNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('matrix', triggers_execution=True)
        self.output = self.add_output('eigenvalues')
        self.vector_output = self.add_output('eigenvectors')
        self.order_option = self.add_option('order', widget_type='combo', default_value='largest first')
        self.order_option.widget.combo_items = ['largest first', 'smallest first']

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        matrix = float_like(input_tensor)
        if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
            if self.app.verbose:
                print(self.label, 'needs a square matrix (or a stack of them), got shape', list(matrix.shape))
            return
        # eigh reads only the lower triangle: it assumes the matrix is symmetric
        values, vectors = run_with_cpu_fallback(torch.linalg.eigh, matrix)
        if self.order_option() == 'largest first':     # the principal axis first
            values = values.flip(-1)
            vectors = vectors.flip(-1)
        self.vector_output.send(vectors)
        self.output.send(values)


class TorchGradientNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchGradientNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        spacing = 1.0
        dim = -1
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                spacing = float(v)
        if len(args) > 1:
            v, t = decode_arg(args, 1)
            if t in [int, float]:
                dim = int(v)
        self.input = self.add_input('in', triggers_execution=True)
        self.spacing_input = self.add_input('spacing', widget_type='drag_float', default_value=spacing)
        self.dim_input = self.add_input('dim', widget_type='input_int', default_value=dim)
        self.output = self.add_output('gradient')
        self.edge_order_option = self.add_option('edge order', widget_type='combo', default_value='1')
        self.edge_order_option.widget.combo_items = ['1', '2']

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        data = float_like(input_tensor)
        dim = self.dim_input()
        if not -data.ndim <= dim < data.ndim:
            if self.app.verbose:
                print(self.label, 'dim', dim, 'out of range for shape', list(data.shape))
            return
        edge_order = 2 if self.edge_order_option() == '2' else 1
        if data.shape[dim] < edge_order + 1:
            if self.app.verbose:
                print(self.label, 'needs at least', edge_order + 1, 'values along the dim')
            return
        spacing = self.spacing_input()
        if spacing == 0:
            return
        self.output.send(torch.gradient(data, spacing=float(spacing), dim=dim, edge_order=edge_order)[0])


class TorchTrapezoidNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchTrapezoidNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.cumulative = label == 't.cumulative_trapezoid'
        spacing = 1.0
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                spacing = float(v)
        self.input = self.add_input('in', triggers_execution=True)
        self.spacing_input = self.add_input('spacing', widget_type='drag_float', default_value=spacing)
        self.dim_input = self.add_input('dim', widget_type='input_int', default_value=-1)
        self.output = self.add_output('running total' if self.cumulative else 'area')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        data = float_like(input_tensor)
        dim = self.dim_input()
        if not -data.ndim <= dim < data.ndim:
            if self.app.verbose:
                print(self.label, 'dim', dim, 'out of range for shape', list(data.shape))
            return
        spacing = float(self.spacing_input())
        if not self.cumulative:
            self.output.send(torch.trapezoid(data, dx=spacing, dim=dim))
            return
        # the area so far at every sample, starting from 0: the same length as the input
        totals = torch.cumulative_trapezoid(data, dx=spacing, dim=dim)
        shape = list(data.shape)
        shape[dim] = 1
        self.output.send(torch.cat([torch.zeros(shape, dtype=totals.dtype, device=totals.device), totals], dim=dim))


class TorchLogicalNode(TorchNode):
    operations = {'t.logical_and': torch.logical_and, 't.logical_or': torch.logical_or,
                  't.logical_xor': torch.logical_xor, 't.logical_not': torch.logical_not}
    output_types = {'bool': torch.bool, 'int': torch.int64, 'float': torch.float32}

    @staticmethod
    def factory(name, data, args=None):
        node = TorchLogicalNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = self.operations.get(label, torch.logical_and)
        self.unary = label == 't.logical_not'
        self.operand = None
        self.input = self.add_input('in', triggers_execution=True)
        if not self.unary:
            self.operand_input = self.add_input('in 2')
        self.output = self.add_output('result')
        self.output_type_option = self.add_option('output_type', widget_type='combo', default_value='bool')
        self.output_type_option.widget.combo_items = list(self.output_types.keys())

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        if self.unary:
            result = self.op(input_tensor)
        else:
            if self.operand_input.fresh_input:
                self.operand = held_tensor(self.operand_input())
            if self.operand is None:
                return
            # anything not zero counts as true, whatever its type
            b = self.operand.to(input_tensor.device)
            try:
                result = self.op(input_tensor, b)
            except RuntimeError:
                if self.app.verbose:
                    print(self.label, 'shapes do not broadcast:', list(input_tensor.shape), list(b.shape))
                return
        self.output.send(result.to(self.output_types.get(self.output_type_option(), torch.bool)))
