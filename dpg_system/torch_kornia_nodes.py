from dpg_system.torch_base_nodes import *
import kornia as K

def register_kornia_nodes():
    Node.app.register_node('k.rgb_to_grayscale', KorniaGrayscaleNode.factory)
    Node.app.register_node('k.rgb_to_hls', KorniaRGBToHLSNode.factory)
    Node.app.register_node('k.apply_colormap', KorniaColorMapNode.factory)
    Node.app.register_node('k.canny', KorniaCannyNode.factory)
    Node.app.register_node('k.sobel', KorniaSobelNode.factory)
    Node.app.register_node('k.gaussian_blur', KorniaGaussianBlurNode.factory)
    Node.app.register_node('k.dog_response_single', KorniaDOGResponseSingleNode.factory)

class KorniaGrayscaleNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = KorniaGrayscaleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('output')

    def execute(self):
        input_tensor = self.input_to_torchvision_tensor()
        if input_tensor is not None:
            output_tensor = K.color.rgb_to_grayscale(input_tensor)
            self.output.send(output_tensor)


class KorniaRGBToHLSNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = KorniaRGBToHLSNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('output')

    def execute(self):
        input_tensor = self.input_to_torchvision_tensor().float()
        if input_tensor is not None:
            output_tensor = K.color.rgb_to_hls(input_tensor)
            self.output.send(output_tensor)


class KorniaColorMapNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = KorniaColorMapNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('output')

    def execute(self):
        input_tensor = self.input_to_torchvision_tensor().float()
        if input_tensor is not None:
            if input_tensor.shape[-3] == 3:
                input_tensor = K.color.rgb_to_grayscale(input_tensor)
            output_tensor = K.color.apply_colormap(input_tensor, K.color.AUTUMN())
            self.output.send(output_tensor)


class KorniaCannyNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = KorniaCannyNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('output')

    def execute(self):
        input_tensor = self.input_to_torchvision_tensor().float()
        if input_tensor is not None:
            if len(input_tensor.shape) == 3:
                input_tensor = input_tensor.unsqueeze(dim=0)
            magnitude, edges = K.filters.canny(input_tensor)
            self.output.send(edges[0])

class KorniaSobelNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = KorniaSobelNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('output')

    def execute(self):
        input_tensor = self.input_to_torchvision_tensor().float()
        if input_tensor is not None:
            if len(input_tensor.shape) == 3:
                input_tensor = input_tensor.unsqueeze(dim=0)
            edges = K.filters.sobel(input_tensor)
            self.output.send(edges[0])


class KorniaGaussianBlurNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = KorniaGaussianBlurNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.kernel_size = 9
        self.sigma = 0.5

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.kernel_size_property = self.add_property('kernel size', widget_type='combo', default_value=self.kernel_size, callback=self.params_changed)
        self.kernel_size_property.widget.combo_items = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        self.sigma_property = self.add_property('sigma', widget_type='drag_float', default_value=self.sigma, callback=self.params_changed)
        self.output = self.add_output('output')

    def params_changed(self):
        self.sigma = self.sigma_property()
        if self.sigma <= 0:
            self.sigma = 0.1
        self.kernel_size = int(self.kernel_size_property())

    def execute(self):
        input_tensor = self.input_to_torchvision_tensor().float()
        if input_tensor is not None:
            if len(input_tensor.shape) == 3:
                input_tensor = input_tensor.unsqueeze(dim=0)
            blur_tensor = K.filters.gaussian_blur2d(input_tensor, kernel_size=(self.kernel_size, self.kernel_size), sigma=(self.sigma, self.sigma))
            self.output.send(blur_tensor[0])


class KorniaDOGResponseSingleNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = KorniaDOGResponseSingleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.sigma_1 = 1.0
        self.sigma_2 = 1.6

        if len(args) > 1:
            self.sigma_1 = any_to_float(arg[0])
            self.sigma_2 = any_to_float(arg[1])

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.sigma_1_property = self.add_property('sigma_1', widget_type='drag_float', default_value=self.sigma_1, min=.01, callback=self.params_changed)
        self.sigma_2_property = self.add_property('sigma_2', widget_type='drag_float', default_value=self.sigma_2, min=.01, callback=self.params_changed)
        self.output = self.add_output('output')

    def params_changed(self):
        self.sigma_1 = self.sigma_1_property()
        self.sigma_2 = self.sigma_2_property()


    def execute(self):
        input_tensor = self.input_to_torchvision_tensor().float()
        if input_tensor is not None:
            if len(input_tensor.shape) == 3:
                input_tensor = input_tensor.unsqueeze(dim=0)
            dog_tensor = K.feature.dog_response_single(input_tensor, self.sigma_1, self.sigma_2)
            self.output.send(dog_tensor[0])