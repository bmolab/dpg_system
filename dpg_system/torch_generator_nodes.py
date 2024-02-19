import torch.distributions.poisson

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

    Node.app.register_node('t.dist.bernoulli', TorchDistributionBernoulliNode.factory)
    Node.app.register_node('t.dist.continuous_bernoulli', TorchDistributionBernoulliNode.factory)
    Node.app.register_node('t.dist.chi2', TorchDistributionChi2Node.factory)
    Node.app.register_node('t.dist.exponential', TorchDistributionExponentialNode.factory)
    Node.app.register_node('t.dist.half_cauchy', TorchDistributionWithRateNode.factory)
    Node.app.register_node('t.dist.poisson', TorchDistributionWithRateNode.factory)
    Node.app.register_node('t.dist.half_normal', TorchDistributionHalfNormalNode.factory)
    # Node.app.register_node('t.dist.dirichlet', TorchDistributionDirichletNode.factory)

    Node.app.register_node('t.dist.cauchy', TorchDistributionLocScaleNode.factory)
    Node.app.register_node('t.dist.beta', TorchDistributionAlphaBetaNode.factory)
    Node.app.register_node('t.dist.fishersnedecor', TorchDistributionFisherSnedecorNode.factory)
    Node.app.register_node('t.dist.gamma', TorchDistributionGammaNode.factory)
    Node.app.register_node('t.dist.gumble', TorchDistributionLocScaleNode.factory)
    Node.app.register_node('t.dist.laplace', TorchDistributionLocScaleNode.factory)
    Node.app.register_node('t.dist.kumaraswamy', TorchDistributionAlphaBetaNode.factory)
    Node.app.register_node('t.dist.normal', TorchDistributionLocScaleNode.factory)
    Node.app.register_node('t.dist.lognormal', TorchDistributionLocScaleNode.factory)
    Node.app.register_node('t.dist.pareto', TorchDistributionParetoNode.factory)
    Node.app.register_node('t.dist.uniform', TorchDistributionUniformNode.factory)
    Node.app.register_node('t.dist.von_mises', TorchDistributionVonMisesNode.factory)
    Node.app.register_node('t.dist.weibull', TorchDistributionWeibullNode.factory)

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
        self.create_dtype_device_grad_properties()


    def range_changed(self):
        self.min = self.min_input()
        self.max = self.max_input()

    def shape_changed(self):
        shape_text = self.shape_input()
        shape_list = re.findall(r'[-+]?\d+', shape_text)
        shape = []
        for dim_text in shape_list:
            shape.append(any_to_int(dim_text))
        self.shape = shape

    def dtype_changed(self):
        super().dtype_changed()
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


class TorchDistributionNode(TorchNode):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.shape = []
        self.shape_input = None
        self.param_1 = None
        self.param_2 = None
        self.param_1_name = ''
        self.param_2_name = ''
        for i in range(len(args)):
            d, t = decode_arg(args, i)
            if t == int:
                self.shape += (d,)
        if len(self.shape) == 0:
            self.shape = [1]

        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.distribution = None

    def add_shape_input(self):
        self.shape_input = self.add_input('shape', widget_type='text_input', default_value=str(self.shape),
                                          callback=self.shape_changed)

    def add_param_1(self, name='loc', default_value=0.0, min=None, max=None):
        self.param_1_name = name
        self.param_1 = self.add_input(name, widget_type='drag_float', default_value=default_value, callback=self.params_changed)

    def add_param_2(self, name='scale', default_value=1.0, min=None, max=None):
        self.param_2_name = name
        self.param_2 = self.add_input(name, widget_type='drag_float', default_value=default_value, callback=self.params_changed)

    def shape_changed(self):
        shape_text = self.shape_input()
        shape_list = re.findall(r'[-+]?\d+', shape_text)
        shape = []
        for dim_text in shape_list:
            shape.append(any_to_int(dim_text))
        self.shape = shape

    def params_changed(self):
        pass

    def execute(self):
        size = tuple(self.shape)
        out_array = self.distribution.sample(sample_shape=size)
        self.output.send(out_array)


class TorchDistributionBernoulliNode(TorchDistributionNode):
    dist = torch.distributions.bernoulli.Bernoulli
    continuous_dist = torch.distributions.continuous_bernoulli.ContinuousBernoulli
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionBernoulliNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.mode = 'probs'
        for i in range(len(args)):
            d, t = decode_arg(args, i)
            if t == str:
                if d == 'probs':
                    self.mode = 'probs'
                else:
                    self.mode = 'logits'

        if self.label == 't.dist.bernoulli':
            if self.mode == 'probs':
                self.add_param_1('probability', default_value=0.5, min=0.0, max=1.0)
                self.distribution = self.dist(probs=0.5)
            else:
                self.add_param_1('logits', default_value=1.0)
                self.distribution = self.dist(logits=1.0)
        elif self.label == 't.dist.continuous_bernoulli':
            if self.mode == 'probs':
                self.add_param_1('probability', default_value=0.5, min=0.0, max=1.0)
                self.distribution = self.continuous_dist(probs=0.5)
            else:
                self.add_param_1('logits', default_value=1.0)
                self.distribution = self.continuous_dist(logits=1.0)
        self.output = self.add_output('random tensor')

    def params_changed(self):
        if self.label == 't.dist.bernoulli':
            if self.mode == 'probs':
                self.distribution = self.dist(probs=self.param_1())
            else:
                self.distribution = self.dist(logits=self.param_1())
        else:
            if self.mode == 'probs':
                self.distribution = self.continuous_dist(probs=self.param_1())
            else:
                self.distribution = self.continuous_dist(logits=self.param_1())


class TorchDistributionChi2Node(TorchDistributionNode):
    dist = torch.distributions.chi2.Chi2
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionChi2Node(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.add_param_1('df', default_value=1.0, min=0.000001)
        self.distribution = self.dist(df=1.0)

        self.output = self.add_output('random tensor')

    def params_changed(self):
        self.distribution = self.dist(df=self.param_1())


class TorchDistributionExponentialNode(TorchDistributionNode):
    dist = torch.distributions.exponential.Exponential

    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionExponentialNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.add_param_1('rate', default_value=1.0, min=0.000001)
        self.distribution = self.dist(rate=1.0)

        self.output = self.add_output('random tensor')

    def params_changed(self):
        self.distribution = self.dist(rate=self.param_1())


class TorchDistributionWithRateNode(TorchDistributionNode):
    dist_dict = {
        't.dist.half_cauchy': torch.distributions.half_cauchy.HalfCauchy,
        't.dist.poisson': torch.distributions.poisson.Poisson
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionWithRateNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.add_param_1('rate', default_value=1.0, min=0.000001)
        self.dist = torch.distributions.half_cauchy.HalfCauchy
        if self.label in self.dist_dict:
            self.dist = self.dist_dict[self.label]
        self.distribution = self.dist(scale=1.0)
        self.output = self.add_output('random tensor')

    def params_changed(self):
        self.distribution = self.dist(scale=self.param_1())


class TorchDistributionHalfNormalNode(TorchDistributionNode):
    dist = torch.distributions.half_normal.HalfNormal
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionHalfNormalNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.add_param_1('scale', default_value=1.0, min=0.000001)
        self.distribution = self.dist(scale=1.0)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def params_changed(self):
        self.distribution = self.dist(scale=self.param_1())


class TorchDistributionLocScaleNode(TorchDistributionNode):
    dist_dict = {
        't.dist.cauchy': torch.distributions.cauchy.Cauchy,
        't.dist.gumbel': torch.distributions.gumbel.Gumbel,
        't.dist.laplace': torch.distributions.laplace.Laplace,
        't.dist.log_normal': torch.distributions.log_normal.LogNormal,
        't.dist.normal': torch.distributions.normal.Normal
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionLocScaleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.add_param_1('loc', default_value=0.0)
        self.add_param_2('scale', default_value=1.0, min=0.000001)
        if self.label in self.dist_dict:
            self.dist = self.dist_dict[self.label]
        else:
            self.dist = self.dist_dict['t.dist.cauchy']
        self.distribution = self.dist(loc=0.0, scale=1.0)

        self.output = self.add_output('random tensor')

    def params_changed(self):
        self.distribution = self.dist(loc=self.param_1(), scale=self.param_2())


class TorchDistributionParetoNode(TorchDistributionNode):
    dist = torch.distributions.pareto.Pareto
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionParetoNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.scale = self.add_input('scale', widget_type='drag_float', default_value=1.0, min=0.000001, callback=self.params_changed)
        self.alpha = self.add_input('alpha', widget_type='drag_float', default_value=1.0, min=0.000001, callback=self.params_changed)
        self.distribution = self.dist(scale=1.0, alpha=1.0)

        self.output = self.add_output('random tensor')

    def params_changed(self):
        self.distribution = self.dist(scale=self.scale(), alpha=self.alpha())


class TorchDistributionAlphaBetaNode(TorchDistributionNode):
    dist_dict = {
        't.dist.beta': torch.distributions.beta.Beta,
        't.dist.kumaraswamy': torch.distributions.kumaraswamy.Kumaraswamy
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionAlphaBetaNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        default_alpha = 1.0
        default_beta = 1.0
        if self.label == 't.dist.beta':
            default_alpha = 0.5
            default_beta = 0.5
        if self.label in self.dist_dict:
            self.dist = self.dist_dict[self.label]
        else:
            self.dist = torch.distributions.beta.Beta

        self.add_param_1('alpha', default_value=default_alpha, min=0.000001)
        self.add_param_2('beta', default_value=default_beta, min=0.000001)
        self.distribution = self.dist(concentration1=default_alpha, concentration0=default_beta)

        self.output = self.add_output('random tensor')

    def params_changed(self):
        self.distribution = self.dist(concentration1=self.param_1(), concentration0=self.param_2())


class TorchDistributionUniformNode(TorchDistributionNode):
    dist = torch.distributions.uniform.Uniform
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionUniformNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.add_param_1('low', default_value=0.0)
        self.add_param_2('high', default_value=1.0)
        self.distribution = self.dist(low=0.0, high=1.0)
        self.output = self.add_output('random tensor')

    def params_changed(self):
        self.distribution = self.dist(low=self.param_1(), high=self.param_2())


class TorchDistributionVonMisesNode(TorchDistributionNode):
    dist = torch.distributions.von_mises.VonMises
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionVonMisesNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.add_param_1('concentration', default_value=1.0, min=0.000001)
        self.add_param_2('loc', default_value=1.0)
        self.distribution = self.dist(concentration=1.0, loc=1.0)
        self.output = self.add_output('random tensor')

    def params_changed(self):
        self.distribution = self.dist(concentration=self.param_1(), loc=self.param_2())


class TorchDistributionWeibullNode(TorchDistributionNode):
    dist = torch.distributions.weibull.Weibull

    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionWeibullNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.add_param_1('scale', default_value=1.0, min=0.000001)
        self.add_param_2('concentration', default_value=1.0, min=0.000001)
        self.distribution = self.dist(scale=1.0, concentration=1.0)
        self.output = self.add_output('random tensor')

    def params_changed(self):
        self.distribution = self.dist(scale=self.param_1(), concentration=self.param_2())


class TorchDistributionFisherSnedecorNode(TorchDistributionNode):
    dist = torch.distributions.fishersnedecor.FisherSnedecor

    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionFisherSnedecorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.add_param_1('df1', default_value=0.5, min=0.000001)
        self.add_param_2('df2', default_value=0.5, min=0.000001)
        self.distribution = self.dist(df1=0.5, df2=0.5)
        self.output = self.add_output('random tensor')

    def params_changed(self):
        self.distribution = self.dist(df1=self.param_1(), df2=self.param_2())


class TorchDistributionGammaNode(TorchDistributionNode):
    dist = torch.distributions.gamma.Gamma
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionGammaNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.add_param_1('concentration', default_value=0.5, min=0.000001)
        self.add_param_2('rate', default_value=0.5, min=0.000001)
        self.distribution = self.dist(concentration=0.5, rate=0.5)
        self.output = self.add_output('random tensor')

    def params_changed(self):
        self.distribution = self.dist(concentration=self.param_1(), rate=self.param_2())


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
        self.create_dtype_device_grad_properties()


    def val_changed(self):
        self.value = self.value_input()

    def execute(self):
        for i in range(len(self.shape)):
            self.shape[i] = self.shape_properties[i]()
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
        self.create_dtype_device_grad_properties()


    def range_changed(self):
        self.min = self.min_input()
        self.max = self.max_input()

    def dtype_changed(self):
        super().dtype_changed()
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
            data = self.input()
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
        self.create_dtype_device_grad_properties()


    def execute(self):
        self.start = self.start_property()
        self.stop = self.stop_property()
        self.steps = self.steps_property()
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
        self.create_dtype_device_grad_properties()


    def execute(self):
        self.start = self.start_property()
        self.stop = self.stop_property()
        self.step = self.step_property()
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
        self.create_dtype_device_grad_properties()


    def n_changed(self):
        self.n = self.n_input()

    def execute(self):
        out_array = torch.eye(n=self.n, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        self.output.send(out_array)
