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
    Node.app.register_node('t.dist.half_cauchy', TorchDistributionHalfCauchyNode.factory)
    Node.app.register_node('t.dist.half_normal', TorchDistributionHalfNormalNode.factory)
    Node.app.register_node('t.dist.poisson', TorchDistributionPoissonNode.factory)

    Node.app.register_node('t.dist.cauchy', TorchDistributionCauchyNode.factory)
    Node.app.register_node('t.dist.beta', TorchDistributionBetaNode.factory)
    Node.app.register_node('t.dist.fishersnedecor', TorchDistributionFisherSnedecorNode.factory)
    Node.app.register_node('t.dist.gamma', TorchDistributionGammaNode.factory)
    Node.app.register_node('t.dist.gumble', TorchDistributionGumbelNode.factory)
    Node.app.register_node('t.dist.laplace', TorchDistributionLaplaceNode.factory)
    Node.app.register_node('t.dist.kumaraswamy', TorchDistributionKumaraswamyNode.factory)
    Node.app.register_node('t.dist.normal', TorchDistributionNormalNode.factory)
    Node.app.register_node('t.dist.lognormal', TorchDistributionLogNormalNode.factory)
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
        self.mode = 'probs'
        for i in range(len(args)):
            d, t = decode_arg(args, i)
            if t == int:
                self.shape += (d,)
        if len(self.shape) == 0:
            self.shape = [1]

    def add_shape_input(self):
        self.shape_input = self.add_input('shape', widget_type='text_input', default_value=str(self.shape),
                                          callback=self.shape_changed)

    def shape_changed(self):
        shape_text = self.shape_input()
        shape_list = re.findall(r'[-+]?\d+', shape_text)
        shape = []
        for dim_text in shape_list:
            shape.append(any_to_int(dim_text))
        self.shape = shape

    def execute(self):
        size = tuple(self.shape)
        out_array = self.distribution.sample(sample_shape=size)
        self.output.send(out_array)



class TorchDistributionBernoulliNode(TorchDistributionNode):
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

        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)

        self.add_shape_input()

        if self.label == 't.dist.bernoulli':
            if self.mode == 'probs':
                self.probability = self.add_input('probability', widget_type='drag_float', default_value=0.5, min=0.0, max=1.0, callback=self.probability_changed)
                self.distribution = torch.distributions.bernoulli.Bernoulli(probs=0.5)
            else:
                self.probability = self.add_input('logits', widget_type='drag_float', default_value=1.0, callback=self.probability_changed)
                self.distribution = torch.distributions.bernoulli.Bernoulli(logits=1.0)
        elif self.label == 't.dist.continuous_bernoulli':
            if self.mode == 'probs':
                self.probability = self.add_input('probability', widget_type='drag_float', default_value=0.5, min=0.0,
                                                  max=1.0, callback=self.probability_changed)
                self.distribution = torch.distributions.continuous_bernoulli.ContinuousBernoulli(probs=0.5)
            else:
                self.probability = self.add_input('logits', widget_type='drag_float', default_value=1.0,
                                                  callback=self.probability_changed)
                self.distribution = torch.distributions.continuous_bernoulli.ContinuousBernoulli(logits=1.0)
        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def probability_changed(self):
        if self.mode == 'probs':
            self.distribution = torch.distributions.continuous_bernoulli.ContinuousBernoulli(probs=self.probability())
        else:
            self.distribution = torch.distributions.continuous_bernoulli.ContinuousBernoulli(logits=self.probability())


class TorchDistributionChi2Node(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionChi2Node(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.df = self.add_input('df', widget_type='drag_float', default_value=1.0, min=0.0, callback=self.df_changed)
        self.distribution = torch.distributions.chi2.Chi2(df=1.0)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def df_changed(self):
        self.distribution = torch.distributions.chi2.Chi2(df=self.df())


class TorchDistributionExponentialNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionExponentialNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.rate = self.add_input('rate', widget_type='drag_float', default_value=1.0, min=0.0, callback=self.rate_changed)
        self.distribution = torch.distributions.exponential.Exponential(rate=1.0)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def rate_changed(self):
        self.distribution = torch.distributions.exponential.Exponential(rate=self.rate())


class TorchDistributionHalfCauchyNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionHalfCauchyNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.scale = self.add_input('scale', widget_type='drag_float', default_value=1.0, min=0.0, callback=self.scale_changed)
        self.distribution = torch.distributions.half_cauchy.HalfCauchy(scale=1.0)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def scale_changed(self):
        self.distribution = torch.distributions.half_cauchy.HalfCauchy(scale=self.scale())


class TorchDistributionHalfNormalNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionHalfNormalNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.scale = self.add_input('scale', widget_type='drag_float', default_value=1.0, min=0.0, callback=self.scale_changed)
        self.distribution = torch.distributions.half_normal.HalfNormal(scale=1.0)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def scale_changed(self):
        self.distribution = torch.distributions.half_normal.HalfNormal(scale=self.scale())


class TorchDistributionPoissonNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionPoissonNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.rate = self.add_input('rate', widget_type='drag_float', default_value=1.0, min=0.0, callback=self.rate_changed)
        self.distribution = torch.distributions.poisson.Poisson(rate=1.0)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def rate_changed(self):
        self.distribution = torch.distributions.poisson.Poisson(rate=self.rate())


class TorchDistributionCauchyNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionCauchyNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.loc = self.add_input('loc', widget_type='drag_float', default_value=0.0, callback=self.params_changed)
        self.scale = self.add_input('scale', widget_type='drag_float', default_value=1.0, callback=self.params_changed)
        self.distribution = torch.distributions.cauchy.Cauchy(loc=0.0, scale=1.0)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def params_changed(self):
        self.distribution = torch.distributions.cauchy.Cauchy(loc=self.loc(), scale=self.scale())


class TorchDistributionGumbelNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionGumbelNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.loc = self.add_input('loc', widget_type='drag_float', default_value=0.0, callback=self.params_changed)
        self.scale = self.add_input('scale', widget_type='drag_float', default_value=1.0, callback=self.params_changed)
        self.distribution = torch.distributions.gumbel.Gumbel(loc=0.0, scale=1.0)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def params_changed(self):
        self.distribution = torch.distributions.gumbel.Gumbel(loc=self.loc(), scale=self.scale())


class TorchDistributionLaplaceNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionLaplaceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.loc = self.add_input('loc', widget_type='drag_float', default_value=0.0, callback=self.params_changed)
        self.scale = self.add_input('scale', widget_type='drag_float', default_value=1.0, callback=self.params_changed)
        self.distribution = torch.distributions.laplace.Laplace(loc=0.0, scale=1.0)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def params_changed(self):
        self.distribution = torch.distributions.laplace.Laplace(loc=self.loc(), scale=self.scale())


class TorchDistributionLogNormalNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionLogNormalNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.loc = self.add_input('loc', widget_type='drag_float', default_value=0.0, callback=self.params_changed)
        self.scale = self.add_input('scale', widget_type='drag_float', default_value=1.0, callback=self.params_changed)
        self.distribution = torch.distributions.log_normal.LogNormal(loc=0.0, scale=1.0)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def params_changed(self):
        self.distribution = torch.distributions.log_normal.LogNormal(loc=self.loc(), scale=self.scale())


class TorchDistributionNormalNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionNormalNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.loc = self.add_input('loc', widget_type='drag_float', default_value=0.0, callback=self.params_changed)
        self.scale = self.add_input('scale', widget_type='drag_float', default_value=1.0, callback=self.params_changed)
        self.distribution = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def params_changed(self):
        self.distribution = torch.distributions.normal.Normal(loc=self.loc(), scale=self.scale())


class TorchDistributionParetoNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionParetoNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.scale = self.add_input('scale', widget_type='drag_float', default_value=1.0, callback=self.params_changed)
        self.alpha = self.add_input('alpha', widget_type='drag_float', default_value=1.0, callback=self.params_changed)
        self.distribution = torch.distributions.pareto.Pareto(scale=1.0, alpha=1.0)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def params_changed(self):
        self.distribution = torch.distributions.pareto.Pareto(scale=self.scale(), alpha=self.alpha())


class TorchDistributionBetaNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionBetaNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.alpha = self.add_input('alpha', widget_type='drag_float', default_value=0.5, callback=self.params_changed)
        self.beta = self.add_input('beta', widget_type='drag_float', default_value=0.5, callback=self.params_changed)
        self.distribution = torch.distributions.beta.Beta(concentration1=0.5, concentration0=0.5)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def params_changed(self):
        self.distribution = torch.distributions.beta.Beta(concentration1=self.alpha(), concentration0=self.beta())


class TorchDistributionUniformNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionUniformNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.low = self.add_input('low', widget_type='drag_float', default_value=0.0, callback=self.params_changed)
        self.high = self.add_input('high', widget_type='drag_float', default_value=1.0, callback=self.params_changed)
        self.distribution = torch.distributions.uniform.Uniform(low=0.0, high=1.0)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def params_changed(self):
        self.distribution = torch.distributions.uniform.Uniform(low=self.low(), high=self.high())


class TorchDistributionVonMisesNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionVonMisesNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.concentration = self.add_input('concentration', widget_type='drag_float', default_value=1.0, callback=self.params_changed)
        self.loc = self.add_input('loc', widget_type='drag_float', default_value=1.0, callback=self.params_changed)
        self.distribution = torch.distributions.von_mises.VonMises(concentration=1.0, loc=1.0)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def params_changed(self):
        self.distribution = torch.distributions.von_mises.VonMises(concentration=self.concentration(), loc=self.loc())


class TorchDistributionWeibullNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionWeibullNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.scale = self.add_input('scale', widget_type='drag_float', default_value=1.0, callback=self.params_changed)
        self.concentration = self.add_input('concentration', widget_type='drag_float', default_value=1.0, callback=self.params_changed)
        self.distribution = torch.distributions.weibull.Weibull(scale=1.0, concentration=1.0)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def params_changed(self):
        self.distribution = torch.distributions.weibull.Weibull(scale=self.scale(), concentration=self.concentration())


class TorchDistributionFisherSnedecorNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionFisherSnedecorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.df1 = self.add_input('df1', widget_type='drag_float', default_value=0.5, min=0.0, callback=self.params_changed)
        self.df2 = self.add_input('df2', widget_type='drag_float', default_value=0.5, min=0.0, callback=self.params_changed)
        self.distribution = torch.distributions.fishersnedecor.FisherSnedecor(df1=0.5, df2=0.5)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def params_changed(self):
        self.distribution = torch.distributions.fishersnedecor.FisherSnedecor(df1=self.df1(), df2=self.df2())


class TorchDistributionGammaNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionGammaNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.concentration = self.add_input('concentration', widget_type='drag_float', default_value=0.5, min=0.0, callback=self.params_changed)
        self.rate = self.add_input('rate', widget_type='drag_float', default_value=0.5, min=0.0, callback=self.params_changed)
        self.distribution = torch.distributions.gamma.Gamma(concentration=0.5, rate=0.5)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def params_changed(self):
        self.distribution = torch.distributions.gamma.Gamma(concentration=self.concentration(), rate=self.rate())


class TorchDistributionKumaraswamyNode(TorchDistributionNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionKumaraswamyNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.add_shape_input()
        self.alpha = self.add_input('alpha', widget_type='drag_float', default_value=1.0, min=0.0, callback=self.params_changed)
        self.beta = self.add_input('beta', widget_type='drag_float', default_value=1.0, min=0.0, callback=self.params_changed)
        self.distribution = torch.distributions.kumaraswamy.Kumaraswamy(concentration1=1.0, concentration0=1.0)

        out_label = 'random tensor'
        self.output = self.add_output(out_label)

    def params_changed(self):
        self.distribution = torch.distributions.kumaraswamy.Kumaraswamy(concentration1=self.alpha(), concentration0=self.beta())


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

    def n_changed(self):
        self.n = self.n_input()

    def execute(self):
        out_array = torch.eye(n=self.n, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        self.output.send(out_array)
