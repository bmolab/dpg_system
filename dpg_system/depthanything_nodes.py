import os

from dpg_system.torch_base_nodes import *


from dpg_system.depth_anything_v2.dpt import DepthAnythingV2


def register_depthanything_nodes():
    Node.app.register_node('depth_anything', DepthAnythingNode.factory)


model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

max_depth = 20

_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'depth_anything_v2', 'checkpoints')


def _select_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class DepthAnythingNode(Node):
    # Keyed by model_mode so different encoders can coexist. Previously a
    # single class-level model was reused even when callers asked for a
    # different encoder, which silently loaded mismatched weights.
    _models = {}

    @staticmethod
    def factory(name, data, args=None):
        node = DepthAnythingNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.model_name = 'depth_anything_v2_metric_hypersim_vitl.pth'
        self.model_mode = 'vitl'

        if args:
            if any_to_string(args[0]) == 'small':
                self.model_name = 'depth_anything_v2_metric_hypersim_vits.pth'
                self.model_mode = 'vits'

        self.input_size = 518
        self.DEVICE = _select_device()

        self.model = self._get_or_load_model(self.model_mode, self.model_name)

        self.image_input = self.add_input('input_image', triggers_execution=True)
        self.depth_out = self.add_output('depth_image')

    @classmethod
    def _get_or_load_model(cls, mode, checkpoint_name):
        cached = cls._models.get(mode)
        if cached is not None:
            return cached

        if mode not in model_configs:
            print('depth_anything: unknown model_mode', mode)
            return None

        try:
            model = DepthAnythingV2(**{**model_configs[mode], 'max_depth': max_depth})
        except Exception as e:
            print('depth_anything: failed to construct DepthAnythingV2:', e)
            return None

        checkpoint_path = os.path.join(_CHECKPOINT_DIR, checkpoint_name)
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
        except FileNotFoundError:
            print('depth_anything: checkpoint not found at', checkpoint_path)
            return None
        except Exception as e:
            print('depth_anything: torch.load failed for', checkpoint_path, ':', e)
            return None

        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print('depth_anything: load_state_dict failed:', e)
            return None

        try:
            model = model.to(_select_device()).eval()
        except Exception as e:
            print('depth_anything: model.to(device).eval() failed:', e)
            return None

        cls._models[mode] = model
        return model

    def infer_depth(self, raw_image):
        if self.model is None or raw_image is None:
            return None
        try:
            return self.model.infer_image(raw_image, self.input_size)
        except Exception as e:
            print('depth_anything: infer_image failed:', e)
            return None

    def execute(self):
        image = self.image_input()
        if image is None:
            return
        depth_image = self.infer_depth(image)
        if depth_image is None:
            return
        self.depth_out.send(depth_image)