import torch

from dpg_system.torch_base_nodes import *

def register_torch_voxel_nodes():
    Node.app.register_node('t.point_cloud_crop', TorchPointCloudCropNode.factory)
    Node.app.register_node('t.point_cloud_voxels', TorchPointCloudVoxelsNode.factory)

class TorchPointCloudCropNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchPointCloudCropNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('point cloud in', triggers_execution=True)
        self.left_input = self.add_input('left (m)', widget_type='drag_float', default_value=-6.0)
        self.right_input = self.add_input('right (m)', widget_type='drag_float', default_value=6.0)
        self.top_input = self.add_input('top (m)', widget_type='drag_float', default_value=-6.0)
        self.bottom_input = self.add_input('bottom (m)', widget_type='drag_float', default_value=6.0)
        self.front_input = self.add_input('front (m)', widget_type='drag_float', default_value=.100)
        self.back_input = self.add_input('back (m)', widget_type='drag_float', default_value=10.0)

        self.output = self.add_output('cropped point cloud out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            left = self.left_input()
            right = self.right_input()
            top = self.top_input()
            bottom = self.bottom_input()
            front = self.front_input()
            back = self.back_input()

            valid_x = torch.logical_and(input_tensor[:, 0].ge(left), input_tensor[:, 0].le(right))
            valid_y = torch.logical_and(input_tensor[:, 1].le(bottom), input_tensor[:, 1].ge(top))
            valid_z = torch.logical_and(input_tensor[:, 2].ge(front), input_tensor[:, 2].le(back))
            valid = torch.logical_and(valid_x, valid_y)
            valid = torch.logical_and(valid, valid_z)

            valid_indices = torch.nonzero(valid).flatten()
            output_tensor = torch.index_select(input_tensor, 0, valid_indices)
            self.output.send(output_tensor)


class TorchPointCloudVoxelsNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchPointCloudVoxelsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.left = -6.0
        self.right = 6.0
        self.top = -6.0
        self.bottom = 6.0
        self.front = 0.1
        self.back = 10.0

        self.voxel_size = 0.1

        self.input = self.add_input('point cloud in', triggers_execution=True)
        self.left_input = self.add_input('left (m)', widget_type='drag_float', default_value=self.left, callback=self.adjust_params)
        self.right_input = self.add_input('right (m)', widget_type='drag_float', default_value=self.right, callback=self.adjust_params)
        self.top_input = self.add_input('top (m)', widget_type='drag_float', default_value=self.top, callback=self.adjust_params)
        self.bottom_input = self.add_input('bottom (m)', widget_type='drag_float', default_value=self.bottom, callback=self.adjust_params)
        self.front_input = self.add_input('front (m)', widget_type='drag_float', default_value=self.front, callback=self.adjust_params)
        self.back_input = self.add_input('back (m)', widget_type='drag_float', default_value=self.back, callback=self.adjust_params)
        self.voxel_size_input = self.add_input('voxel size (mm)', widget_type='drag_float', default_value=self.voxel_size, callback=self.adjust_params)
        self.output_point_cloud = self.add_input('output point cloud', widget_type='checkbox', default_value=False)
        self.output_voxels = self.add_input('output voxels', widget_type='checkbox', default_value=False)
        self.output_voxel_cloud = self.add_input('output voxels cloud', widget_type='checkbox', default_value=True)

        self.point_cloud_output = self.add_output('point cloud out')
        self.voxel_output = self.add_output('voxels out')
        self.voxel_cloud_output = self.add_output('voxels cloud out')

        # self.voxel_size_reciprocal = 1.0 / self.voxel_size
        self.device = torch.device('cpu')
        # self.lower_bounds = torch.Tensor([self.left, self.top, self.front]).to(device=self.device)
        # self.upper_bounds = torch.Tensor([self.right, self.bottom, self.back]).to(device=self.device)

        # self.bin_count_x = int((self.right - self.left) / self.voxel_size + 0.5)
        # self.bin_count_y = int((self.bottom - self.top) / self.voxel_size + 0.5)
        # self.bin_count_z = int((self.back - self.front) / self.voxel_size + 0.5)
        # self.bin_count = int(self.bin_count_x * self.bin_count_y * self.bin_count_z)
        # self.x_stride = 1
        # self.y_stride = self.bin_count_x
        # self.z_stride = self.bin_count_x * self.bin_count_y
        # self.strides = torch.tensor([self.x_stride, self.y_stride, self.z_stride]).to(device=self.device)

        self.valid_cache = None
        self.bin_indices_cache = None
        # self.bins = torch.zeros((self.bin_count, ), device=self.device)

        self.vals_cache = None
        self.voxel_centres = None
        self.make_adjustments()

    def adjust_params(self):
        self.left = self.left_input()
        self.right = self.right_input()
        self.top = self.top_input()
        self.bottom = self.bottom_input()
        self.front = self.front_input()
        self.back = self.back_input()
        self.voxel_size = self.voxel_size_input()
        self.make_adjustments()

    def make_adjustments(self):
        self.voxel_size_reciprocal = 1.0 / self.voxel_size
        self.lower_bounds = torch.Tensor([self.left, self.top, self.front]).to(device=self.device)
        self.upper_bounds = torch.Tensor([self.right, self.bottom, self.back]).to(device=self.device)

        self.bin_count_x = int((self.right - self.left) / self.voxel_size + 0.5)
        self.bin_count_y = int((self.bottom - self.top) / self.voxel_size + 0.5)
        self.bin_count_z = int((self.back - self.front) / self.voxel_size + 0.5)
        self.bin_count = int(self.bin_count_x * self.bin_count_y * self.bin_count_z)
        self.bins = torch.zeros((self.bin_count, ), device=self.device)

        self.x_stride = 1
        self.y_stride = self.bin_count_x
        self.z_stride = self.bin_count_x * self.bin_count_y
        self.strides = torch.tensor([self.x_stride, self.y_stride, self.z_stride]).to(device=self.device)

        self.voxel_centres = torch.zeros([self.bin_count_z, self.bin_count_y, self.bin_count_x, 3], device=self.device)
        x_base = torch.linspace(0, self.bin_count_x - 1, self.bin_count_x) * self.voxel_size + self.voxel_size / 2.0 + self.left
        y_base = torch.linspace(0, self.bin_count_y - 1, self.bin_count_y) * self.voxel_size + self.voxel_size / 2.0 + self.top
        z_base = torch.linspace(0, self.bin_count_z - 1, self.bin_count_z) * self.voxel_size + self.voxel_size / 2.0 + self.front

        for y in range(self.bin_count_y):
            self.voxel_centres[:, y, :, 0] = x_base

        y_base.unsqueeze_(1)
        for z in range(self.bin_count_z):
            self.voxel_centres[z, :, :, 1] = y_base

        z_base.unsqueeze_(1)
        for x in range(self.bin_count_x):
            self.voxel_centres[:, :, x, 2] = z_base

        self.voxel_centres = self.voxel_centres.to(device=self.device)
        self.voxel_centres = self.voxel_centres.reshape((-1, 3))


    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return

        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        if self.valid_cache is None or self.valid_cache.shape != input_tensor.shape:
            self.valid_cache = torch.empty_like(input_tensor, dtype=torch.bool)

        if self.device != input_tensor.device:
            self.device = input_tensor.device
            self.upper_bounds = self.upper_bounds.to(self.device)
            self.lower_bounds = self.lower_bounds.to(self.device)
            self.bins = self.bins.to(self.device)
            self.strides = self.strides.to(self.device)
            self.voxel_centres = self.voxel_centres.to(device=self.device)
            self.valid_cache = self.valid_cache.to(device=self.device)

        torch.logical_and(
            torch.ge(input_tensor, self.lower_bounds, out=self.valid_cache),
            torch.le(input_tensor, self.upper_bounds),
            out=self.valid_cache
        )
        # if self.valid_lower_cache is None or self.valid_lower_cache.shape != input_tensor.shape:
        #     self.valid_lower_cache = torch.empty_like(input_tensor, dtype=torch.bool)
        #     self.valid_upper_cache = torch.empty_like(input_tensor, dtype=torch.bool)

        # self.valid_lower_cache = torch.ge(input_tensor, self.lower_bounds)
        # self.valid_upper_cache = torch.le(input_tensor, self.upper_bounds)
        # self.valid_upper_cache = torch.logical_and(self.valid_lower_cache, self.valid_upper_cache)

        valid_indices = self.valid_cache.all(dim=1).nonzero().flatten()

        cropped_points = torch.index_select(input_tensor, 0, valid_indices)
        if self.output_point_cloud():
            self.point_cloud_output.send(cropped_points)

        if self.output_voxels() or self.output_voxel_cloud():
            cropped_points_offset = torch.sub(cropped_points, self.lower_bounds)
            bin_indices = cropped_points_offset.mul_(self.voxel_size_reciprocal).to(dtype=torch.int64)

            bin_linear_indices = torch.sum(bin_indices * self.strides, dim=1 )
            # bin_linear_indices = (bin_indices[:, 0].mul_(self.x_stride).add_(bin_indices[:, 1].mul_(self.y_stride)).add_(bin_indices[:, 2].mul_(self.z_stride)))

            self.bins.zero_()

            # Sort indices for more efficient scatter_add
            unique_indices, counts = torch.unique(bin_linear_indices, return_counts=True)

            # Use accumulated counts instead of ones
            self.bins.scatter_add_(0, unique_indices, counts.to(self.bins.dtype))

            if self.output_voxels():
                self.voxel_output.send(self.bins.view((self.bin_count_z, self.bin_count_y, self.bin_count_x)))

            if self.output_voxel_cloud():
                voxel_linear_indices = self.bins.nonzero().flatten()
                self.voxel_cloud = torch.index_select(self.voxel_centres, 0, voxel_linear_indices)
                self.voxel_cloud_output.send(self.voxel_cloud)




