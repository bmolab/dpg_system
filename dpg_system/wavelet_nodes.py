from dpg_system.torch_base_nodes import *
import numpy as np
#from wavelets_pytorch.transform import WaveletTransform        # SciPy version

from dpg_system.wavelets_pytorch.transform import WaveletTransformTorch   # PyTorch version
from dpg_system.wavelets_pytorch.wavelets import Morlet

def register_wavelet_nodes():
    Node.app.register_node('t.cwt', TorchCWTNode.factory)
    pass


class TorchCWTNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCWTNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.sample_scaling = self.add_input('sample_scaling', widget_type='drag_float', default_value=.01, callback=self.transform_changed)
        self.scale_distribution = self.add_input('scale_distribution', widget_type='drag_float', default_value=0.125, callback=self.transform_changed)
        self.unbias = self.add_input('unbias', widget_type='checkbox', default_value=False, callback=self.transform_changed)
        self.wavelet_constant = self.add_input('wavelet_constant', widget_type='input_int', default_value=6, callback=self.transform_changed)
        self.output = self.add_output('wavelets out')
        self.wavelet = Morlet(w0=6)
        self.transform = WaveletTransformTorch(dt=.01, dj=0.125, wavelet=self.wavelet, unbias=False, cuda=False)

    def transform_changed(self):
        self.wavelet = Morlet(w0=self.wavelet_constant())
        self.transform = WaveletTransformTorch(dt=self.sample_scaling(), dj=self.scale_distribution(), wavelet=self.wavelet, unbias=self.unbias(), cuda=False)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            power_torch = self.transform.power(input_tensor)
            self.output.send(power_torch)






