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
        self.threshold = 0
        self.replace = 0
        if len(args) > 0:
            self.threshold = any_to_int(args[0])
        if len(args) > 1:
            self.replace = any_to_int(args[1])

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.threshold_input = self.add_input('threshold', widget_type='drag_float', default_value=self.threshold, callback=self.threshold_changed)
        self.replace_input = self.add_input('replacenent', widget_type='drag_float', default_value=self.replace, callback=self.replacement_changed)
        self.op = torch.nn.Threshold(self.threshold, self.replace)
        self.output = self.add_output("output")

    def threshold_changed(self, val=None):
        self.threshold = self.threshold_input.get_widget_value()
        self.op = torch.nn.Threshold(self.threshold, self.replace)

    def replacement_changed(self, val=None):
        self.replace = self.replace_input.get_widget_value()
        self.op = torch.nn.Threshold(self.threshold, self.replace)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor))


class TorchActivationNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchActivationNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = None
        if self.label == 't.nn.relu':
            self.op = torch.nn.functional.relu
        elif self.label == 't.nn.hardswish':
            self.op = torch.nn.functional.hardswish
        elif self.label == 't.nn.hardtanh':
            self.op = torch.nn.functional.hardtanh
        elif self.label == 't.nn.relu6':
            self.op = torch.nn.functional.relu6
        elif self.label == 't.nn.selu':
            self.op = torch.nn.functional.selu
        elif self.label == 't.nn.glu':
            self.op = torch.nn.functional.glu
        elif self.label == 't.nn.gelu':
            self.op = torch.nn.functional.gelu
        elif self.label == 't.nn.logsigmoid':
            self.op = torch.nn.functional.logsigmoid
        elif self.label == 't.nn.tanhshrink':
            self.op = torch.nn.functional.tanhshrink
        elif self.label == 't.nn.softsign':
            self.op = torch.nn.functional.softsign
        elif self.label == 't.nn.tanh':
            self.op = torch.nn.functional.tanh
        elif self.label == 't.nn.sigmoid':
            self.op = torch.nn.functional.sigmoid
        elif self.label == 't.nn.hardsigmoid':
            self.op = torch.nn.functional.hardsigmoid
        elif self.label == 't.nn.silu':
            self.op = torch.nn.functional.silu
        elif self.label == 't.nn.mish':
            self.op = torch.nn.functional.mish

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
        self.op = None
        param_1_name = 'minimum'
        param_2_name = 'maximum'
        self.parameter_1 = -1
        self.parameter_2 = 1

        if self.label == 't.nn.hardtanh':
            self.op = torch.nn.functional.hardtanh
        if self.label == 't.nn.rrelu':
            self.op = torch.nn.functional.rrelu
            param_1_name = 'lower'
            param_2_name = 'upper'
            self.parameter_1 = 0.125
            self.parameter_2 = 0.3333333333333
        if self.label == 't.nn.softplus':
            self.op = torch.nn.functional.softplus
            param_1_name = 'beta'
            param_2_name = 'threshold'
            self.parameter_1 = 1.0
            self.parameter_2 = 20

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.parameter_1_input = self.add_input(param_1_name, widget_type='drag_float', default_value=self.parameter_1, callback=self.parameter_changed)
        self.parameter_2_input = self.add_input(param_2_name, widget_type='drag_float', default_value=self.parameter_2, callback=self.parameter_changed)
        self.output = self.add_output("output")

    def parameter_changed(self, val=None):
        self.parameter_1 = self.parameter_1_input.get_widget_value()
        self.parameter_2 = self.parameter_2_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor, self.parameter_1, self.parameter_2))


class TorchActivationThreeParamNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchActivationThreeParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = None
        param_1_name = 'tau'
        param_2_name = 'hard'
        param_3_name = 'dim'
        self.parameter_1 = 1
        self.parameter_2 = False
        self.parameter_3 = -1

        if self.label == 't.nn.gumbel_softmax':
            self.op = torch.nn.functional.gumbel_softmax

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.parameter_1_input = self.add_input(param_1_name, widget_type='drag_float', default_value=self.parameter_1, callback=self.parameter_changed)
        self.parameter_2_input = self.add_input(param_2_name, widget_type='checkbox', default_value=self.parameter_2, callback=self.parameter_changed)
        self.parameter_3_input = self.add_input(param_3_name, widget_type='input_int', default_value=self.parameter_3, callback=self.parameter_changed)
        self.output = self.add_output("output")

    def parameter_changed(self, val=None):
        self.parameter_1 = self.parameter_1_input.get_widget_value()
        self.parameter_2 = self.parameter_2_input.get_widget_value()
        self.parameter_3 = self.parameter_3_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor, self.parameter_1, self.parameter_2, self.parameter_3))


class TorchSoftmaxNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSoftmaxNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = None
        self.dim = 0

        if self.label == 't.nn.softmax':
            self.op = torch.nn.functional.softmax
        if self.label == 't.nn.softmin':
            self.op = torch.nn.functional.softmin
        if self.label == 't.nn.log_softmax':
            self.op = torch.nn.functional.log_softmax

        if len(args) > 0:
            self.dim = any_to_int(args[0])

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.dim_input = self.add_input('dim', widget_type='input_int', default_value=self.dim, callback=self.dim_changed)
        self.output = self.add_output("output")

    def dim_changed(self, val=None):
        self.dim = self.dim_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor, self.dim))


class TorchActivationOneParamNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchActivationOneParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = None
        param_name = ''
        self.parameter = 1
        if self.label == 't.nn.elu':
            self.op = torch.nn.functional.elu
            param_name = 'alpha'
            self.parameter = 1.0
        elif self.label == 't.nn.celu':
            self.op = torch.nn.functional.celu
            param_name = 'alpha'
            self.parameter = 1.0
        elif self.label == 't.nn.leaky_relu':
            self.op = torch.nn.functional.leaky_relu
            param_name = 'negative slope'
            self.parameter = 0.01
        # elif self.label == 't.nn.prelu':
        #     self.op = torch.nn.functional.prelu
        #     param_name = 'weight'
        #     self.parameter = 1.0
        elif self.label == 't.nn.hardshrink':
            self.op = torch.nn.functional.hardshrink
            param_name = 'lambda'
            self.parameter = 0.5
        elif self.label == 't.nn.softshrink':
            self.op = torch.nn.functional.softshrink
            param_name = 'lambda'
            self.parameter = 0.5

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.parameter_input = self.add_input(param_name, widget_type='drag_float', default_value=self.parameter, callback=self.parameter_changed)
        self.output = self.add_output("output")

    def parameter_changed(self, val=None):
        self.parameter = self.parameter_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor, self.parameter))

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
        self.n = 0
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.n_input = self.add_input('n', widget_type='input_int', default_value=self.n, min=0, callback=self.n_changed)
        self.output = self.add_output("tensor out")

    def n_changed(self, val=0):
        self.n = self.n_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(torch.special.polygamma(self.n, input_tensor))



class TorchSpecialLogitNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSpecialLogitNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.eps = 1e-8
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.eps_input = self.add_input('eps', widget_type='drag_float', default_value=self.eps, callback=self.eps_changed)

        self.output = self.add_output("tensor out")

    def custom_setup(self, from_file):
        self.eps_input.widget.set_format('%.8f')

    def eps_changed(self, val=0):
        self.eps = self.eps_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(torch.special.logit(input_tensor, self.eps))


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
            data = self.second_input.get_received_data()
            if data is not None:
                second_tensor = self.data_to_tensor(data)
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
            data = self.second_input.get_received_data()
            if data is not None:
                second_tensor = self.data_to_tensor(data)
                if second_tensor is not None:
                    self.output.send(self.op(input_tensor, second_tensor))

class TorchSpecialMultiGammaLnNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSpecialMultiGammaLnNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.p = 1e-8
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.p_input = self.add_input('p', widget_type='input_int', default_value=self.p, callback=self.p_changed)
        self.output = self.add_output("tensor out")

    def p_changed(self, val=0):
        self.p = self.p_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(torch.special.multigammaln(input_tensor, self.p))

