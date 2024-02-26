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

    Node.app.register_node('t.gcd', TorchLCMGCDNode.factory)
    Node.app.register_node('t.lcm', TorchLCMGCDNode.factory)
    Node.app.register_node('t.mean', TorchMeanMedianNode.factory)
    Node.app.register_node('t.median', TorchMeanMedianNode.factory)
    Node.app.register_node('t.nanmean', TorchMeanMedianNode.factory)
    Node.app.register_node('t.nanmedian', TorchMeanMedianNode.factory)
    Node.app.register_node('t.sum', TorchMeanMedianNode.factory)
    Node.app.register_node('t.nansum', TorchMeanMedianNode.factory)
    Node.app.register_node('t.prod', TorchMeanMedianNode.factory)

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
        if input_tensor is not None:
            out_tensor = self.op(input_tensor)
            self.output.send(out_tensor)


class TorchDistributionTensorOneParamNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionTensorOneParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        param_1_name = ''
        param_1 = 1
        if self.label == 't.exponential':
            param_1_name = 'lambda'
        elif self.label == 't.geometric':
            param_1_name = 'p'

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.param_1 = self.add_input(param_1_name, widget_type='drag_float', default_value=param_1)
        self.output = self.add_output('tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            out_tensor = input_tensor.clone()
            if self.label == 't.exponential':
                out_tensor.exponential_(self.param_1())
            elif self.label == 't.geometric':
                out_tensor.log_normal_(self.param_1())
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
            if self.label == 't.cauchy':
                out_tensor.cauchy_(self.param_1(), self.param_2())
            elif self.label == 't.log_normal':
                out_tensor.log_normal_(self.param_1(), self.param_2())
            elif self.label == 't.normal':
                out_tensor.normal_(self.param_1(), self.param_2())
            elif self.label == 't.uniform':
                out_tensor.uniform_(self.param_1(), self.param_2())
            self.output.send(out_tensor)


class TorchCDistanceNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCDistanceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('input', triggers_execution=True)
        self.output = self.add_output('output')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            input_tensor = input_tensor.unsqueeze(dim=0)
            euclidean_length = torch.cdist(input_tensor, torch.zeros_like(input_tensor))

            self.output.send(euclidean_length.item())


class TorchDistanceNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCDistanceNode(name, data, args)
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
                input2 = self.data_to_tensor(self.input2(), match_tensor=input_tensor)
                if input2 is not None:
                    euclidean_length = torch.dist(input_tensor, torch.zeros_like(input_tensor))
                else:
                    euclidean_length = torch.dist(input_tensor, input_tensor)
            else:
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
        output_name = 'gcd'
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
            output_name = 'lcm'
        self.input = self.add_input('tensor a in', triggers_execution=True)
        self.input_2 = self.add_input('tensor b in')

        self.output = self.add_output(output_name + ' tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.input_2()
            if data is not None:
                input_tensor_2 = any_to_tensor(data, dtype=torch.int64)
                if input_tensor_2 is not None:
                    output_tensor = self.op(input_tensor, input_tensor_2)
                    self.output.send(output_tensor)


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
            if -1 - len(input_tensor.shape) < self.dim <= len(input_tensor.shape):
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
        if real_tensor is not None:
            data = self.imag_input()
            if data is not None:
                imag_tensor = self.data_to_tensor(data, match_tensor=real_tensor)
                if imag_tensor is not None:
                    if real_tensor.shape == imag_tensor.shape:
                        if real_tensor.dtype in [torch.float16, torch.float32, torch.float64]:
                            if real_tensor.dtype == imag_tensor.dtype:
                                complex_tensor = torch.complex(real_tensor, imag_tensor)
                                self.output.send(complex_tensor)
                            else:
                                if self.app.verbose:
                                    print(self.label, 'real and imaginary tensor dtypes don\'t match', real_tensor.dtype, imag_tensor.dtype)
                        else:
                            if self.app.verbose:
                                print(self.label, 'real tensor wrong dtype', real_tensor.dtype)
                    else:
                        if self.app.verbose:
                            print(self.label, 'imaginary tensor is None')
                else:
                    if self.app.verbose:
                        print(self.label, 'no input for imaginary tensor')
        else:
            if self.app.verbose:
                print(self.label, 'real tensor is None')


class TorchClampNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchClampNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if len(args) > 1:
            min = any_to_float(args[0])
            max = any_to_float(args[1])
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.min = self.add_input('min', widget_type='drag_float', default_value=min)
        self.max = self.add_input('max', widget_type='drag_float', default_value=max)
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
                        print('t.copysign:', error)


class CosineSimilarityNode(TorchNode):
    cos = None
    inited = False

    @staticmethod
    def factory(name, data, args=None):
        node = CosineSimilarityNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.vector_2 = None
        if not self.inited:
            self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            self.inited = True
        self.input = self.add_input('input 1', triggers_execution=True)
        self.input2 = self.add_input('input 2')
        self.output = self.add_output('output')

    def execute(self):
        vector_1 = self.input_to_tensor()
        if vector_1 is not None:
            if self.input2.fresh_input:
                self.vector_2 = self.data_to_tensor(self.input2(), match_tensor=vector_1)
                if self.vector_2 is not None:
                    try:
                        similarity = self.cos(vector_1, self.vector_2)
                        self.output.send(similarity.item())
                    except Exception as e:
                        print(self.label, e)


class TorchNormalizeNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchNormalizeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.dim_input = self.add_input('dim', widget_type='input_int', default_value=1)
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
        self.s_output = self.add_output('S tensor out')
        self.v_output = self.add_output('V tensor out')
        self.d_output = self.add_output('D tensor out')

    def full_changed(self):
        self.full = self.full_property()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            s, v, d = torch.linalg.svd(input_tensor, self.full)
            self.d_output.send(d)
            self.v_output.send(v)
            self.s_output.send(s)


class TorchPCALowRankNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchPCALowRankNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        center = False
        self.center = self.add_property('center', widget_type='checkbox', default_value=center)
        niter = 2
        self.niter = self.add_property('full', widget_type='input_int', default_value=niter)
        self.u_output = self.add_output('U tensor out')
        self.s_output = self.add_output('S tensor out')
        self.v_output = self.add_output('V tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            u, s, v = torch.pca.low_rank(input_tensor, center=self.center(), niter=self.niter())
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
                if input_tensor.shape[-1] == input_tensor.shape[-1]:
                    l, v = torch.linalg.eig(input_tensor)
                    self.v_output.send(v)
                    self.l_output.send(l)
                else:
                    if self.app.verbose:
                        print(self.label, 'tensor is not square')
            else:
                if self.app.verbose:
                    print(self.label, 'tensor has less than 2 dimensions')


