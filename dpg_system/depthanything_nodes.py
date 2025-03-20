from dpg_system.torch_base_nodes import *


from dpg_system.depth_anything_v2.dpt import DepthAnythingV2


def register_depth_anything_nodes():
    Node.app.register_node('depth_anything', DepthAnythingNode.factory)

model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

max_depth = 20

class DepthAnythingNode(Node):
    model = None
    tokenizer = None
    inited = False

    @staticmethod
    def factory(name, data, args=None):
        node = DepthAnythingNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.model_name = 'depth_anything_v2_metric_hypersim_vitl.pth'
        self.model_mode = 'vitl'

        if len(args) > 0:
            if args[0] == 'small':
                self.model_name = 'depth_anything_v2_metric_hypersim_vits.pth'
                self.model_mode = 'vits'

        if not self.__class__.inited:
            self.__class__.model = DepthAnythingV2(**{**model_configs[self.model_mode], 'max_depth': max_depth})
            self.__class__.inited = True
        self.input_size = 518

        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.__class__.model.load_state_dict(torch.load('dpg_system/depth_anything_v2/checkpoints/' + self.model_name, map_location='cpu'))
        self.__class__.model = self.__class__.model.to(self.DEVICE).eval()
        self.image_input = self.add_input('input_image', triggers_execution=True)
        self.depth_out = self.add_output('depth_image')

    def infer_depth(self, raw_image):
        depth = self.__class__.model.infer_image(raw_image, self.input_size)
        return depth

    def execute(self):
        image = self.image_input()
        depth_image = self.infer_depth(image)
        self.depth_out.send(depth_image)

