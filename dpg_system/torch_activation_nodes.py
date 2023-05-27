from dpg_system.torch_base_nodes import *

def register_torch_activation_nodes():
    Node.app.register_node('t.nn.Threshold', TorchNNThresholdNode.factory)
    Node.app.register_node('t.nn.relu', TorchActivationNode.factory)
    Node.app.register_node('t.nn.hardswish', TorchActivationNode.factory)
    Node.app.register_node('t.nn.relu6', TorchActivationNode.factory)
    Node.app.register_node('t.nn.selu', TorchActivationNode.factory)
    Node.app.register_node('t.nn.glu', TorchActivationNode.factory)
    Node.app.register_node('t.nn.gelu', TorchActivationNode.factory)
    Node.app.register_node('t.nn.logsigmoid', TorchActivationNode.factory)
    Node.app.register_node('t.nn.tanhshrink', TorchActivationNode.factory)
    Node.app.register_node('t.nn.softsign', TorchActivationNode.factory)
    Node.app.register_node('t.nn.tanh', TorchActivationNode.factory)
    Node.app.register_node('t.nn.sigmoid', TorchActivationNode.factory)
    Node.app.register_node('t.nn.hardsigmoid', TorchActivationNode.factory)
    Node.app.register_node('t.nn.silu', TorchActivationNode.factory)
    Node.app.register_node('t.nn.mish', TorchActivationNode.factory)
    Node.app.register_node('t.nn.elu', TorchActivationOneParamNode.factory)
    Node.app.register_node('t.nn.celu', TorchActivationOneParamNode.factory)
    Node.app.register_node('t.nn.leaky_relu', TorchActivationOneParamNode.factory)
    # Node.app.register_node('t.nn.prelu', TorchActivationOneParamNode.factory)
    Node.app.register_node('t.nn.hardshrink', TorchActivationOneParamNode.factory)
    Node.app.register_node('t.nn.softshrink', TorchActivationOneParamNode.factory)
    Node.app.register_node('t.nn.hardtanh', TorchActivationTwoParamNode.factory)
    Node.app.register_node('t.nn.rrelu', TorchActivationTwoParamNode.factory)
    Node.app.register_node('t.nn.softplus', TorchActivationTwoParamNode.factory)
    Node.app.register_node('t.nn.softmax', TorchSoftmaxNode.factory)
    Node.app.register_node('t.nn.softmin', TorchSoftmaxNode.factory)
    Node.app.register_node('t.nn.log_softmax', TorchSoftmaxNode.factory)
    Node.app.register_node('t.nn.gumbel_softmax', TorchActivationThreeParamNode.factory)

    Node.app.register_node('t.special.airy_ai', TorchSpecialNode.factory)
    Node.app.register_node('t.special.bessel_j0', TorchSpecialNode.factory)
    Node.app.register_node('t.special.bessel_j1', TorchSpecialNode.factory)
    Node.app.register_node('t.special.digamma', TorchSpecialNode.factory)
    Node.app.register_node('t.special.entr', TorchSpecialNode.factory)
    Node.app.register_node('t.special.erf', TorchSpecialNode.factory)
    Node.app.register_node('t.special.erfc', TorchSpecialNode.factory)
    Node.app.register_node('t.special.erfcx', TorchSpecialNode.factory)
    Node.app.register_node('t.special.erfinv', TorchSpecialNode.factory)
    Node.app.register_node('t.special.exp2', TorchSpecialNode.factory)
    Node.app.register_node('t.special.expit', TorchSpecialNode.factory)
    Node.app.register_node('t.special.expm1', TorchSpecialNode.factory)
    Node.app.register_node('t.special.gammaln', TorchSpecialNode.factory)
    Node.app.register_node('t.special.i0', TorchSpecialNode.factory)
    Node.app.register_node('t.special.i0e', TorchSpecialNode.factory)
    Node.app.register_node('t.special.i1', TorchSpecialNode.factory)
    Node.app.register_node('t.special.i1e', TorchSpecialNode.factory)
    Node.app.register_node('t.special.log1p', TorchSpecialNode.factory)
    Node.app.register_node('t.special.logndtr', TorchSpecialNode.factory)
    Node.app.register_node('t.special.ndtr', TorchSpecialNode.factory)
    Node.app.register_node('t.special.ndtri', TorchSpecialNode.factory)
    Node.app.register_node('t.special.scaled_modified_bessel_k0', TorchSpecialNode.factory)
    Node.app.register_node('t.special.sinc', TorchSpecialNode.factory)
    Node.app.register_node('t.special.spherical_bessel_j0', TorchSpecialNode.factory)
    Node.app.register_node('t.special.softmax', TorchSpecialDimNode.factory)
    Node.app.register_node('t.special.log_softmax', TorchSpecialDimNode.factory)
    Node.app.register_node('t.special.polygamma', TorchSpecialPolygammaNode.factory)
    Node.app.register_node('t.special.logits', TorchSpecialLogitNode.factory)
    Node.app.register_node('t.special.multigammaln', TorchSpecialMultiGammaLnNode.factory)
    Node.app.register_node('t.special.zeta', TorchSpecialTwoTensorOrNumberNode.factory)
    Node.app.register_node('t.special.xlogy', TorchSpecialTwoTensorOrNumberNode.factory)
    Node.app.register_node('t.special.xlog1py', TorchSpecialTwoTensorOrNumberNode.factory)
    Node.app.register_node('t.special.xlog1py', TorchSpecialTwoTensorOrNumberNode.factory)
    Node.app.register_node('t.special.gammainc', TorchSpecialTwoTensorNode.factory)
    Node.app.register_node('t.special.gammaincc', TorchSpecialTwoTensorNode.factory)


class TorchNNThresholdNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchNNThresholdNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        threshold = 0
        replace = 0
        if len(args) > 0:
            threshold = any_to_int(args[0])
        if len(args) > 1:
            replace = any_to_int(args[1])

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.threshold = self.add_input('threshold', widget_type='drag_float', default_value=threshold, callback=self.params_changed)
        self.replace = self.add_input('replacenent', widget_type='drag_float', default_value=replace, callback=self.params_changed)
        self.op = torch.nn.Threshold(self.threshold, self.replace)
        self.output = self.add_output("output")

    def params_changed(self):
        self.op = torch.nn.Threshold(self.threshold(), self.replace())

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor))


class TorchActivationNode(TorchNode):
    op_dict = {
    't.nn.relu': torch.nn.functional.relu,
    't.nn.hardswish': torch.nn.functional.hardswish,
    't.nn.hardtanh': torch.nn.functional.hardtanh,
    't.nn.relu6': torch.nn.functional.relu6,
    't.nn.selu': torch.nn.functional.selu,
    't.nn.glu': torch.nn.functional.glu,
    't.nn.gelu': torch.nn.functional.gelu,
    't.nn.logsigmoid': torch.nn.functional.logsigmoid,
    't.nn.tanhshrink': torch.nn.functional.tanhshrink,
    't.nn.softsign': torch.nn.functional.softsign,
    't.nn.tanh': torch.nn.functional.tanh,
    't.nn.sigmoid': torch.nn.functional.sigmoid,
    't.nn.hardsigmoid': torch.nn.functional.hardsigmoid,
    't.nn.silu': torch.nn.functional.silu,
    't.nn.mish': torch.nn.functional.mish
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchActivationNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.op = torch.nn.functional.relu
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.output = self.add_output("output")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor))


class TorchActivationTwoParamNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchActivationTwoParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = torch.nn.functional.hardtanh
        param_1_name = 'minimum'
        param_2_name = 'maximum'
        parameter_1 = -1
        parameter_2 = 1

        if self.label == 't.nn.hardtanh':
            self.op = torch.nn.functional.hardtanh
        if self.label == 't.nn.rrelu':
            self.op = torch.nn.functional.rrelu
            param_1_name = 'lower'
            param_2_name = 'upper'
            parameter_1 = 0.125
            parameter_2 = 0.3333333333333
        if self.label == 't.nn.softplus':
            self.op = torch.nn.functional.softplus
            param_1_name = 'beta'
            param_2_name = 'threshold'
            parameter_1 = 1.0
            parameter_2 = 20

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.parameter_1 = self.add_input(param_1_name, widget_type='drag_float', default_value=parameter_1)
        self.parameter_2 = self.add_input(param_2_name, widget_type='drag_float', default_value=parameter_2)
        self.output = self.add_output("output")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor, self.parameter_1(), self.parameter_2()))


class TorchActivationThreeParamNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchActivationThreeParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = torch.nn.functional.gumbel_softmax
        param_1_name = 'tau'
        param_2_name = 'hard'
        param_3_name = 'dim'
        parameter_1 = 1
        parameter_2 = False
        parameter_3 = -1

        if self.label == 't.nn.gumbel_softmax':
            self.op = torch.nn.functional.gumbel_softmax

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.parameter_1 = self.add_input(param_1_name, widget_type='drag_float', default_value=parameter_1)
        self.parameter_2 = self.add_input(param_2_name, widget_type='checkbox', default_value=parameter_2)
        self.parameter_3 = self.add_input(param_3_name, widget_type='input_int', default_value=parameter_3)
        self.output = self.add_output("output")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor, self.parameter_1(), self.parameter_2(), self.parameter_3()))


class TorchSoftmaxNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSoftmaxNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = torch.nn.functional.softmax
        dim = 0

        if self.label == 't.nn.softmax':
            self.op = torch.nn.functional.softmax
        if self.label == 't.nn.softmin':
            self.op = torch.nn.functional.softmin
        if self.label == 't.nn.log_softmax':
            self.op = torch.nn.functional.log_softmax

        if len(args) > 0:
            dim = any_to_int(args[0])

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.dim = self.add_input('dim', widget_type='input_int', default_value=dim)
        self.output = self.add_output("output")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor, self.dim()))


class TorchActivationOneParamNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchActivationOneParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = torch.nn.functional.elu
        param_name = ''
        parameter = 1
        if self.label == 't.nn.elu':
            self.op = torch.nn.functional.elu
            param_name = 'alpha'
            parameter = 1.0
        elif self.label == 't.nn.celu':
            self.op = torch.nn.functional.celu
            param_name = 'alpha'
            parameter = 1.0
        elif self.label == 't.nn.leaky_relu':
            self.op = torch.nn.functional.leaky_relu
            param_name = 'negative slope'
            parameter = 0.01
        # elif self.label == 't.nn.prelu':
        #     self.op = torch.nn.functional.prelu
        #     param_name = 'weight'
        #     parameter = 1.0
        elif self.label == 't.nn.hardshrink':
            self.op = torch.nn.functional.hardshrink
            param_name = 'lambda'
            parameter = 0.5
        elif self.label == 't.nn.softshrink':
            self.op = torch.nn.functional.softshrink
            param_name = 'lambda'
            parameter = 0.5

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.parameter = self.add_input(param_name, widget_type='drag_float', default_value=parameter)
        self.output = self.add_output("output")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor, self.parameter()))

class TorchSpecialNode(TorchNode):
    op_dict = {
        't.special.airy_ai': torch.special.airy_ai,
        't.special.bessel_j0': torch.special.bessel_j0,
        't.special.bessel_j1': torch.special.bessel_j1,
        't.special.digamma': torch.special.digamma,
        't.special.entr': torch.special.entr,
        't.special.erf': torch.special.erf,
        't.special.erfc': torch.special.erfc,
        't.special.erfcx': torch.special.erfcx,
        't.special.erfinv': torch.special.erfinv,
        't.special.exp2': torch.special.exp2,
        't.special.expit': torch.special.expit,
        't.special.expm1': torch.special.expm1,
        't.special.gammaln': torch.special.gammaln,
        't.special.i0': torch.special.i0,
        't.special.i0e': torch.special.i0e,
        't.special.i1': torch.special.i1,
        't.special.i1e': torch.special.i1e,
        't.special.log1p': torch.special.log1p,
        't.special.logndtr': torch.special.log_ndtr,
        't.special.ndtr': torch.special.ndtr,
        't.special.ndtri': torch.special.ndtri,
        't.special.scaled_modified_bessel_k0': torch.special.scaled_modified_bessel_k0,
        't.special.sinc': torch.special.sinc,
        't.special.spherical_bessel_j0': torch.special.spherical_bessel_j0
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchSpecialNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = torch.special.exp2
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.output = self.add_output("tensor out")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor))


class TorchSpecialDimNode(TorchWithDimNode):
    op_dict = {
        't.special.log_softmax': torch.special.log_softmax,
        't.special.softmax': torch.special.softmax
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchSpecialDimNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = torch.special.log_softmax
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
        self.input = self.add_input("tensor in", triggers_execution=True)
        if self.dim_specified:
            self.add_dim_input()
        self.output = self.add_output("tensor out")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor, dim=self.dim))


class TorchSpecialPolygammaNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSpecialPolygammaNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        n = 0
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.n = self.add_input('n', widget_type='input_int', default_value=n, min=0)
        self.output = self.add_output("tensor out")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(torch.special.polygamma(self.n(), input_tensor))


class TorchSpecialLogitNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSpecialLogitNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        eps = 1e-8
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.eps = self.add_input('eps', widget_type='drag_float', default_value=eps)

        self.output = self.add_output("tensor out")

    def custom_create(self, from_file):
        self.eps.widget.set_format('%.8f')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(torch.special.logit(input_tensor, self.eps()))


class TorchSpecialTwoTensorNode(TorchNode):
    op_dict = {
        't.special.gammainc': torch.special.gammainc,
        't.special.gammaincc': torch.special.gammaincc
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchSpecialTwoTensorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.op = torch.special.gammainc
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
        self.input = self.add_input("tensor 1 in", triggers_execution=True)
        self.second_input = self.add_input('tensor 2 in')
        self.output = self.add_output("tensor out")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.second_input()
            if data is not None:
                second_tensor = self.data_to_tensor(data, match_tensor=input_tensor)
                if second_tensor is not None:
                    self.output.send(self.op(input_tensor, second_tensor))


class TorchSpecialTwoTensorOrNumberNode(TorchNode):
    op_dict = {
        't.special.zeta': torch.special.zeta,
        't.special.xlogy': torch.special.xlogy,
        't.special.xlog1py': torch.special.xlog1py
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchSpecialTwoTensorOrNumberNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.op = torch.special.gammainc
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
        self.input = self.add_input("tensor 1 or number in", triggers_execution=True)
        self.second_input = self.add_input('tensor 2 or number in')
        self.output = self.add_output("tensor out")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.second_input()
            if data is not None:
                second_tensor = self.data_to_tensor(data, match_tensor=input_tensor)
                if second_tensor is not None:
                    self.output.send(self.op(input_tensor, second_tensor))


class TorchSpecialMultiGammaLnNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSpecialMultiGammaLnNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        p = 1e-8
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.p = self.add_input('p', widget_type='input_int', default_value=p)
        self.output = self.add_output("tensor out")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(torch.special.multigammaln(input_tensor, self.p()))

