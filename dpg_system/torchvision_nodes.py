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
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.output = self.add_output("output")
        self.op = torchvision.transforms.Grayscale()

    def execute(self):
        input_tensor = self.input_to_torchvision_tensor()
        if input_tensor is not None:
            output_tensor = self.op(input_tensor)
            self.output.send(output_tensor)


class TorchvisionGaussianBlurNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchvisionGaussianBlurNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.kernel_size = 9
        self.sigma = 0.5

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.kernel_size_property = self.add_property('kernel size', widget_type='combo', default_value=self.kernel_size, callback=self.params_changed)
        self.kernel_size_property.widget.combo_items = [3, 5, 7, 9, 11, 13, 15, 17, 19]
        self.sigma_property = self.add_property('sigma', widget_type='drag_float', default_value=self.sigma, callback=self.params_changed)
        self.output = self.add_output("output")
        self.op = torchvision.transforms.functional.gaussian_blur

    def params_changed(self):
        self.sigma = self.sigma_property()
        if self.sigma <= 0:
            self.sigma = 0.1
        self.kernel_size = int(self.kernel_size_property())
        self.op = torchvision.transforms.functional.gaussian_blur

    def execute(self):
        input_tensor = self.input_to_torchvision_tensor()
        if input_tensor is not None:
            output_tensor = self.op(input_tensor, self.kernel_size, self.sigma)
            self.output.send(output_tensor)


class TorchvisionAdjustOneParamNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchvisionAdjustOneParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.param = 0.0
        param_name = 'hue offset'
        min = -.5
        max = .5

        if self.label == 'torchvision.adjust_hue':
            self.param = 0.0
            param_name = 'hue offset'
            min = -.5
            max = .5
            self.op = torchvision.transforms.functional.adjust_hue
        elif self.label == 'torchvision.adjust_saturation':
            self.param = 1.0
            param_name = 'saturation'
            min = 0
            max = 10
            self.op = torchvision.transforms.functional.adjust_saturation
        elif self.label == 'torchvision.adjust_sharpness':
            self.param = 1.0
            param_name = 'sharpness'
            min = 0
            max = 10
            self.op = torchvision.transforms.functional.adjust_sharpness
        elif self.label == 'torchvision.adjust_contrast':
            self.param = 1.0
            param_name = 'contrast'
            min = 0
            max = 10
            self.op = torchvision.transforms.functional.adjust_contrast
        elif self.label == 'torchvision.adjust_brightness':
            self.param = 1.0
            param_name = 'brightness'
            min = 0
            max = 10
            self.op = torchvision.transforms.functional.adjust_brightness

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.param_input = self.add_input(param_name, widget_type='drag_float', default_value=self.param, min=min, max=max, callback=self.params_changed)
        self.output = self.add_output("output")
        print(self.op)

    def params_changed(self):
        self.param = self.param_input()

    def execute(self):
        input_tensor = self.input_to_torchvision_tensor()
        if input_tensor is not None:
            output_tensor = self.op(input_tensor, self.param)
            self.output.send(output_tensor)

