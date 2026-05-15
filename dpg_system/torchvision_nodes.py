from dpg_system.torch_base_nodes import *
import torchvision

def register_torchvision_nodes():
    Node.app.register_node('tv.Grayscale', TorchvisionGrayscaleNode.factory)
    Node.app.register_node('tv.gaussian_blur', TorchvisionGaussianBlurNode.factory)
    Node.app.register_node('tv.adjust_hue', TorchvisionAdjustOneParamNode.factory)
    Node.app.register_node('tv.adjust_saturation', TorchvisionAdjustOneParamNode.factory)
    Node.app.register_node('tv.adjust_contrast', TorchvisionAdjustOneParamNode.factory)
    Node.app.register_node('tv.adjust_sharpness', TorchvisionAdjustOneParamNode.factory)
    Node.app.register_node('tv.adjust_brightness', TorchvisionAdjustOneParamNode.factory)


class TorchvisionGrayscaleNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchvisionGrayscaleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('output')
        self.op = torchvision.transforms.Grayscale()

    def execute(self):
        input_tensor = self.input_to_torchvision_tensor()
        if input_tensor is None:
            return
        try:
            output_tensor = self.op(input_tensor)
            self.output.send(output_tensor)
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()


class TorchvisionGaussianBlurNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchvisionGaussianBlurNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.kernel_size = 9
        self.sigma = 0.5

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.kernel_size_property = self.add_property('kernel size', widget_type='combo', default_value=self.kernel_size, callback=self.params_changed)
        self.kernel_size_property.widget.combo_items = [3, 5, 7, 9, 11, 13, 15, 17, 19]
        self.sigma_property = self.add_property('sigma', widget_type='drag_float', default_value=self.sigma, callback=self.params_changed)
        self.output = self.add_output('output')
        self.op = torchvision.transforms.functional.gaussian_blur

    def params_changed(self):
        self.sigma = self.sigma_property()
        if self.sigma <= 0:
            # Push the clamped value back to the widget so the UI matches
            # what we'll actually hand to torchvision. Previously we kept
            # the user's invalid number on screen and quietly filtered
            # with sigma=0.1.
            self.sigma = 0.1
            self.sigma_property.set(self.sigma)
        self.kernel_size = int(self.kernel_size_property())

    def execute(self):
        input_tensor = self.input_to_torchvision_tensor()
        if input_tensor is None:
            return
        try:
            output_tensor = self.op(input_tensor, self.kernel_size, self.sigma)
            self.output.send(output_tensor)
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()


class TorchvisionAdjustOneParamNode(TorchNode):
    # The if/elif chain that previously initialised self.op compared
    # self.label against the wrong strings (the 'torchvision.' prefix
    # instead of the registered 'tv.' prefix), so none of the branches
    # ever matched, self.op was never assigned, and the subsequent
    # print(self.op) crashed every construction with AttributeError.
    # Switched to a single source-of-truth dispatch dict keyed on the
    # actual registered names.
    _DISPATCH = {
        'tv.adjust_hue':        (torchvision.transforms.functional.adjust_hue,        'hue offset', 0.0, -0.5, 0.5),
        'tv.adjust_saturation': (torchvision.transforms.functional.adjust_saturation, 'saturation', 1.0, 0.0, 10.0),
        'tv.adjust_sharpness':  (torchvision.transforms.functional.adjust_sharpness,  'sharpness',  1.0, 0.0, 10.0),
        'tv.adjust_contrast':   (torchvision.transforms.functional.adjust_contrast,   'contrast',   1.0, 0.0, 10.0),
        'tv.adjust_brightness': (torchvision.transforms.functional.adjust_brightness, 'brightness', 1.0, 0.0, 10.0),
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchvisionAdjustOneParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # Fall back to adjust_hue's settings if we somehow get an unregistered
        # label so the node still constructs rather than crashing on a missing
        # self.op in execute().
        op, param_name, default_param, param_min, param_max = self._DISPATCH.get(
            self.label, self._DISPATCH['tv.adjust_hue']
        )
        self.op = op
        self.param = default_param

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.param_input = self.add_input(param_name, widget_type='drag_float',
                                          default_value=default_param,
                                          min=param_min, max=param_max,
                                          callback=self.params_changed)
        self.output = self.add_output('output')

    def params_changed(self):
        self.param = self.param_input()

    def execute(self):
        input_tensor = self.input_to_torchvision_tensor()
        if input_tensor is None:
            return
        try:
            output_tensor = self.op(input_tensor, self.param)
            self.output.send(output_tensor)
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()

