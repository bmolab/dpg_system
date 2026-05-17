from dpg_system.torch_base_nodes import *
#from wavelets_pytorch.transform import WaveletTransform        # SciPy version

from dpg_system.wavelets_pytorch.transform import WaveletTransformTorch   # PyTorch version
from dpg_system.wavelets_pytorch.wavelets import Morlet


def register_wavelet_nodes():
    Node.app.register_node('t.cwt', TorchCWTNode.factory)


# Floors that keep WaveletTransformTorch construction well-defined. dt and dj
# feed log/division paths inside the transform; w0 below ~5 makes the Morlet
# wavelet non-admissible, but we accept down to 1 to leave room for
# experimentation while still rejecting <=0.
_MIN_DT = 1e-9
_MIN_DJ = 1e-6
_MIN_W0 = 1


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

        self.wavelet = None
        self.transform = None
        try:
            self.wavelet, self.transform = self._build_transform(0.01, 0.125, 6, False)
        except Exception as e:
            print('t.cwt: failed to build initial transform:', e)

    def _build_transform(self, dt, dj, w0, unbias):
        try:
            dt = float(dt)
        except (TypeError, ValueError):
            dt = _MIN_DT
        try:
            dj = float(dj)
        except (TypeError, ValueError):
            dj = _MIN_DJ
        try:
            w0 = int(w0)
        except (TypeError, ValueError):
            w0 = _MIN_W0

        if dt < _MIN_DT:
            dt = _MIN_DT
        if dj < _MIN_DJ:
            dj = _MIN_DJ
        if w0 < _MIN_W0:
            w0 = _MIN_W0

        wavelet = Morlet(w0=w0)
        transform = WaveletTransformTorch(dt=dt, dj=dj, wavelet=wavelet, unbias=bool(unbias), cuda=False)
        return wavelet, transform

    def transform_changed(self):
        try:
            wavelet, transform = self._build_transform(
                self.sample_scaling(),
                self.scale_distribution(),
                self.wavelet_constant(),
                self.unbias(),
            )
        except Exception as e:
            # Keep the previous valid transform rather than leaving the node
            # in a half-rebuilt state.
            print('t.cwt: rebuild failed, keeping previous transform:', e)
            return
        self.wavelet = wavelet
        self.transform = transform

    def execute(self):
        if self.transform is None:
            return
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        # WaveletTransformTorch.cwt only handles 1D or 2D [n_batch, signal_length]
        # input; anything else falls through to a misshapen conv and yields
        # confusing failures further down.
        if input_tensor.ndim not in (1, 2):
            print('t.cwt: expected 1D or 2D tensor, got shape', tuple(input_tensor.shape))
            return
        try:
            power_torch = self.transform.power(input_tensor)
        except Exception as e:
            print('t.cwt: power() failed:', e)
            return
        if power_torch is None:
            return
        self.output.send(power_torch)
