from dpg_system.torch_base_nodes import *

def register_torch_manipulation_nodes():
    Node.app.register_node('t[]', TorchSubtensorNode.factory)
    Node.app.register_node('t.permute', TorchPermuteNode.factory)
    Node.app.register_node('t.transpose', TorchTransposeNode.factory)
    Node.app.register_node('t.flip', TorchFlipNode.factory)
    Node.app.register_node('t.select', TorchSelectNode.factory)
    Node.app.register_node('t.squeeze', TorchSqueezeNode.factory)
    Node.app.register_node('t.unsqueeze', TorchUnsqueezeNode.factory)
    Node.app.register_node('t.cat', TorchCatNode.factory)
    Node.app.register_node('t.stack', TorchStackNode.factory)
    Node.app.register_node('t.hstack', TorchHStackNode.factory)
    Node.app.register_node('t.row_stack', TorchHStackNode.factory)
    Node.app.register_node('t.vstack', TorchHStackNode.factory)
    Node.app.register_node('t.column_stack', TorchHStackNode.factory)
    Node.app.register_node('t.dstack', TorchHStackNode.factory)
    Node.app.register_node('t.repeat', TorchRepeatNode.factory)
    Node.app.register_node('t.tile', TorchTileNode.factory)
    Node.app.register_node('t.chunk', TorchChunkNode.factory)
    Node.app.register_node('t.tensor_split', TorchChunkNode.factory)
    Node.app.register_node('t.view', TorchViewNode.factory)
    Node.app.register_node('t.reshape', TorchViewNode.factory)
    Node.app.register_node('t.narrow', TorchNarrowNode.factory)
    Node.app.register_node('t.roll', TorchRollNode.factory)
    Node.app.register_node('t.flatten', TorchViewVariousNode.factory)
    Node.app.register_node('t.adjoint', TorchViewVariousNode.factory)
    Node.app.register_node('t.t', TorchViewVariousNode.factory)
    Node.app.register_node('t.ravel', TorchRavelNode.factory)
    Node.app.register_node('t.masked_select', TorchMaskedSelectNode.factory)
    Node.app.register_node('t.index_select', TorchIndexSelectNode.factory)
    Node.app.register_node('t.take', TorchTakeNode.factory)
    Node.app.register_node('t.take_along_dim', TorchTakeAlongDimNode.factory)
    Node.app.register_node('t.diag', TorchDiagNode.factory)
    Node.app.register_node('t.tril', TorchTriangleNode.factory)
    Node.app.register_node('t.triu', TorchTriangleNode.factory)


class TorchSqueezeNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSqueezeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        if self.dim_specified:
            self.add_dim_input()
        self.output = self.add_output('output')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if -1 - len(input_tensor.shape) < self.dim < len(input_tensor.shape):
                if self.dim_specified:
                    self.output.send(torch.squeeze(input_tensor, self.dim))
                else:
                    self.output.send(torch.squeeze(input_tensor))
                return


class TorchUnsqueezeNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchUnsqueezeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        if self.dim_specified:
            self.dim_input = self.add_input('dim', widget_type='input_int', default_value=self.dim, callback=self.dim_changed)
        self.output = self.add_output('output')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if -1 - len(input_tensor.shape) < self.dim <= len(input_tensor.shape):
                self.output.send(torch.unsqueeze(input_tensor, self.dim))
                return


class TorchViewNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchViewNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.shape = [-1]
        if len(args) > 0:
            self.shape = []
            for arg in args:
                self.shape.append(any_to_int(arg))
        shape_string = str(self.shape)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.shape_input = self.add_input('', widget_type='text_input', widget_width=200, default_value=shape_string,
                                                  callback=self.shape_changed)
        self.output = self.add_output('output')

    def shape_changed(self):
        shape_text = self.shape_input()
        shape_list = re.findall(r'[-+]?\d+', shape_text)
        shape = []
        for dim_text in shape_list:
            shape.append(any_to_int(dim_text))
        self.shape = shape

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            try:
                if self.label == 't.view':
                    view_tensor = input_tensor.view(self.shape)
                else:
                    view_tensor = input_tensor.reshape(self.shape)
                self.output.send(view_tensor)
            except Exception as e:
                traceback.print_exception(e)
                print(self.label, e)


class TorchViewVariousNode(TorchNode):
    op_dict = {
        't.adjoint': torch.adjoint,
        't.t': torch.t,
        't.flatten': torch.flatten
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchViewVariousNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = torch.t
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            out_tensor = self.op(input_tensor)
            self.output.send(out_tensor)


class TorchRavelNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchRavelNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('ravelled tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            out_tensor = torch.ravel(input_tensor)
            self.output.send(out_tensor)


class TorchTransposeNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchTransposeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.transpose1 = 0
        self.transpose2 = 1
        if len(args) > 0:
            self.transpose1 = any_to_int(args[0])
        if len(args) > 1:
            self.transpose2 = any_to_int(args[1])
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.transpose1_property = self.add_input('dim 1', widget_type='input_int', default_value=self.transpose1, callback=self.transpose_changed)
        self.transpose2_property = self.add_input('dim 2', widget_type='input_int', default_value=self.transpose2, callback=self.transpose_changed)
        self.output = self.add_output('permuted tensor out')

    def transpose_changed(self):
        self.transpose1 = self.transpose1_property()
        self.transpose2 = self.transpose2_property()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if len(input_tensor.shape) <= 1:
                self.output.send(input_tensor)
                if self.app.verbose:
                    print('WARNING: torch.transpose - too few dims to transpose')
            else:
                transposed = torch.transpose(input_tensor, self.transpose1, self.transpose2)
                self.output.send(transposed)


class TorchFlipNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchFlipNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.flip_list = [0]
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.flip_property = self.add_property('flip dims', widget_type='text_input', default_value='0',
                                                  callback=self.flip_changed)
        self.output = self.add_output('output')

    def flip_changed(self):
        flip_text = self.flip_property()
        flip_split = re.findall(r'[-+]?\d+', flip_text)
        flip_list, _, _ = list_to_hybrid_list(flip_split)
        self.flip_list = flip_list

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if len(input_tensor.shape) < len(self.flip_list) or len(self.flip_list) == 0:
                self.output.send(input_tensor)
            else:
                permuted = torch.flip(input_tensor, self.flip_list)
                self.output.send(permuted)


class TorchStackCatNode(TorchNode):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input_count = 2
        self.dim = 0
        if len(args) > 0:
            self.input_count = string_to_int(args[0])
        self.other_inputs = []
        self.input = self.add_input('tensor 1', triggers_execution=True)
        for i in range(self.input_count - 1):
            self.other_inputs.append(self.add_input('tensor ' + str(i + 2)))
        self.dim_input = self.add_input('dim', widget_type='input_int', default_value=self.dim, callback=self.dim_changed)
        self.output = self.add_output('output')

    def dim_changed(self):
        self.dim = self.dim_input()


class TorchStackNode(TorchStackCatNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchStackNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            stack_list = [input_tensor]
            for i in range(self.input_count - 1):
                an_in = self.other_inputs[i]
                a_tensor = self.data_to_tensor(an_in(), match_tensor=input_tensor)
                if a_tensor is not None:
                    if len(a_tensor.shape) == len(input_tensor.shape):
                        ok_shape = True
                        for j in range(len(a_tensor.shape)):
                            if a_tensor.shape[j] != input_tensor.shape[j]:
                                ok_shape = False
                                if self.app.verbose:
                                    print('t.stack input tensor ' + str(i + 1) + ' has wrong shape')
                                break
                        if ok_shape:
                            stack_list.append(a_tensor)
                    else:
                        if self.app.verbose:
                            print('t.stack input tensor ' + str(i + 1) + ' has wrong number of dimensions')
            if -len(input_tensor.shape) <= self.dim <= len(input_tensor.shape):
                output_tensor = torch.stack(stack_list, self.dim)
                self.output.send(output_tensor)
            else:
                if self.app.verbose:
                    print('t.stack dim is out of range', self.dim)


class TorchCatNode(TorchStackCatNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCatNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            cat_list = [input_tensor]
            for i in range(self.input_count - 1):
                an_in = self.other_inputs[i]
                a_tensor = self.data_to_tensor(an_in(), match_tensor=input_tensor)
                if a_tensor is not None:
                    if len(a_tensor.shape) == len(input_tensor.shape):
                        ok_shape = True
                        for j in range(len(a_tensor.shape)):
                            if j != self.dim:
                                if a_tensor.shape[j] != input_tensor.shape[j]:
                                    ok_shape = False
                                    if self.app.verbose:
                                        print('t.cat input tensor ' + str(i) + ' has wrong shape')
                                    break
                        if ok_shape:
                            cat_list.append(a_tensor)
                    else:
                        if self.app.verbose:
                            print('t.cat input tensor ' + str(i) + ' has wrong number of dimensions')
            if self.dim < len(input_tensor.shape):
                output_tensor = torch.cat(cat_list, self.dim)
                self.output.send(output_tensor)
            else:
                if self.app.verbose:
                    print('t.cat dim is out of range', self.dim)


class TorchHStackNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchHStackNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input_count = 2
        if len(args) > 0:
            self.input_count = string_to_int(args[0])
        self.op = torch.hstack
        if self.label in ['t.vstack', 't.row_stack']:
            self.op = torch.vstack
        elif self.label == 't.dstack':
            self.op = torch.dstack

        self.other_inputs = []
        self.input = self.add_input('tensor 1', triggers_execution=True)
        for i in range(self.input_count - 1):
            self.other_inputs.append(self.add_input('tensor ' + str(i + 2)))
        self.output = self.add_output('stacked tensors')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            stack_list = [input_tensor]
            for i in range(self.input_count - 1):
                an_in = self.other_inputs[i]
                a_tensor = self.data_to_tensor(an_in(), match_tensor=input_tensor)
                if a_tensor is not None:
                    if a_tensor.shape != input_tensor.shape:
                        if self.app.verbose:
                            print(self.label + ' input tensors must have the same shape')
                        return
                    else:
                        stack_list.append(a_tensor)
            output_tensor = self.op(tuple(stack_list))
            self.output.send(output_tensor)


class TorchChunkNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchChunkNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.dim = 0
        self.splits = 2
        self.op = torch.tensor_split
        if self.label == 't.chunk':
            self.op = torch.chunk
        if len(args) > 0:
            self.splits = any_to_int(args[0])
        if len(args) > 1:
            self.dim = any_to_int(args[1])

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.dim_input = self.add_input('dim', widget_type='input_int', default_value=self.dim, callback=self.dim_changed)
        self.split_count_option = self.add_option('split', widget_type='input_int', default_value=self.splits)
        self.tensor_outputs = []

        for i in range(self.splits):
            self.tensor_outputs.append(self.add_output('tensor ' + str(i)))

    def dim_changed(self):
        self.dim = self.dim_input()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if -1 - len(input_tensor.shape) < self.dim < len(input_tensor.shape):
                if self.splits < input_tensor.shape[self.dim]:
                    tensors = self.op(input_tensor, self.splits, self.dim)
                    for idx, tensor_ in enumerate(tensors):
                        if idx < len(self.outputs):
                            self.tensor_outputs[idx].send(tensor_)


class TorchPermuteNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchPermuteNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.permute = []
        if len(args) > 0:
            for i in range(len(args)):
                self.permute.append(any_to_int(args[i]))
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.permute_property = self.add_input('permute', widget_type='text_input', default_value=self.permute, callback=self.permute_changed)
        self.output = self.add_output('permuted tensor out')

    def permute_changed(self):
        permute_text = self.permute_property()
        permute_split = re.findall(r'[-+]?\d+', permute_text)
        permute_list, _, _ = list_to_hybrid_list(permute_split)
        self.permute = permute_list

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if len(input_tensor.shape) != len(self.permute):
                self.output.send(input_tensor)
                if self.app.verbose:
                    print('WARNING: torch.permute - permute list and channel count mismatch')
            else:
                permuted = torch.permute(input_tensor, self.permute)
                self.output.send(permuted)


class TorchRepeatNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchRepeatNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.repeat = []
        if len(args) > 0:
            for i in range(len(args)):
                self.repeat.append(any_to_int(args[i]))
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.repeat_property = self.add_input('repeats', widget_type='text_input', default_value=self.repeat, callback=self.repeat_changed)
        self.output = self.add_output('repeated tensor out')

    def repeat_changed(self):
        repeat_text = self.repeat_property()
        repeat_split = re.findall(r'\d+', repeat_text)
        repeat_list, _, _ = list_to_hybrid_list(repeat_split)
        self.repeat = repeat_list

    def execute(self):
        input_tensor = self.input_to_tensor()
        repeat = self.repeat
        if input_tensor is not None:
            repeated = input_tensor.repeat(repeat)
            self.output.send(repeated)


class TorchTileNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchTileNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.tile = []
        if len(args) > 0:
            for i in range(len(args)):
                self.tile.append(any_to_int(args[i]))
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.tile_property = self.add_input('tiling', widget_type='text_input', default_value=self.tile, callback=self.tile_changed)
        self.output = self.add_output('repeated tensor out')

    def tile_changed(self):
        tile_text = self.tile_property()
        tile_split = re.findall(r'\d+', tile_text)
        tile_list, _, _ = list_to_hybrid_list(tile_split)
        self.tile = tile_list

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            tiled = input_tensor.repeat(self.tile)
            self.output.send(tiled)


class TorchRollNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchRollNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.dims_tuple = (-1,)
        self.shifts_tuple = (1,)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.dims_input = self.add_input('roll dims', widget_type='text_input', default_value='-1',
                                                  callback=self.dims_changed)
        self.shifts_input = self.add_input('roll shifts', widget_type='text_input', default_value='1',
                                                  callback=self.shifts_changed)
        self.output = self.add_output('rolled tensor')

    def dims_changed(self):
        dims_text = self.dims_input()
        dims_split = re.findall(r'[-+]?\d+', dims_text)
        dims_list, _, _ = list_to_hybrid_list(dims_split)
        self.dims_tuple = tuple(dims_list)

    def shifts_changed(self):
        shifts_text = self.shifts_input()
        shifts_split = re.findall(r'[-+]?\d+', shifts_text)
        shifts_list, _, _ = list_to_hybrid_list(shifts_split)
        self.shifts_tuple = tuple(shifts_list)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            try:
                rolled = torch.roll(input_tensor, self.shifts_tuple, self.dims_tuple)
                self.output.send(rolled)
            except Exception as e:
                traceback.print_exception(e)
                if self.app.verbose:
                    print(self.label, e)


class TorchSubtensorNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSubtensorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.dim_list = None
        index_string = ''
        for i in range(len(args)):
            index_string += args[i]
        if index_string == '':
            index_string = ':'
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.indices_input = self.add_input('', widget_type='text_input', widget_width=200, default_value=index_string,
                                                  callback=self.dim_changed)
        self.output = self.add_output('output')

    def dim_changed(self):
        dim_text = any_to_string(self.indices_input())

        dim_split = dim_text.split(',')
        dimmers = []

        for i in range(len(dim_split)):
            dimmer = dim_split[i]
            dimmer = dimmer.split(':')
            dim_nums = []
            if len(dimmer) == 1:
                dim_num = re.findall(r'[-+]?\d+', dimmer[0])
                if len(dim_num) > 0:
                    dim_num = string_to_int(dim_num[0])
                    dim_nums.append([dim_num])
                    dim_nums.append([dim_num + 1])
            else:
                for j in range(len(dimmer)):
                    dim_num = re.findall(r'[-+]?\d+', dimmer[j])
                    if len(dim_num) == 0:
                        if j == 0:
                            dim_nums.append([0])
                        else:
                            dim_nums.append([1000000])
                    else:
                        dim_num = string_to_int(dim_num[0])
                        dim_nums.append([dim_num])
            dimmers.append(dim_nums)

        self.dim_list = dimmers

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if self.dim_list is None:
                self.dim_changed()
            if len(input_tensor.shape) < len(self.dim_list) or len(self.dim_list) == 0:
                self.output.send(input_tensor)
            else:
                dim_list_now = []
                for i in range(len(self.dim_list)):
                    dim_dim = self.dim_list[i]
                    dim_list_now.append(dim_dim[0][0])
                    if dim_dim[1][0] == 1000000:
                        dim_list_now.append(input_tensor.shape[i])
                    else:
                        dim_list_now.append(dim_dim[1][0])
                sub_tensor = input_tensor
                if len(dim_list_now) == 2:
                    sub_tensor = input_tensor[dim_list_now[0]:dim_list_now[1]]
                elif len(dim_list_now) == 4:
                    sub_tensor = input_tensor[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3]]
                elif len(dim_list_now) == 6:
                    sub_tensor = input_tensor[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3], dim_list_now[4]:dim_list_now[5]]
                elif len(dim_list_now) == 8:
                    sub_tensor = input_tensor[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3],
                             dim_list_now[4]:dim_list_now[5], dim_list_now[6]:dim_list_now[7]]
                self.output.send(sub_tensor)


class TorchSelectNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSelectNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.dim = 0
        self.index = 0
        if len(args) > 0:
            self.dim = any_to_int(args[0])
        if len(args) > 1:
            self.index = any_to_int(args[1])

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.dim_input = self.add_input('dim', widget_type='input_int', default_value=self.dim, callback=self.dim_changed)
        self.index_input = self.add_input('index', widget_type='input_int', default_value=self.index, callback=self.index_changed)
        self.output = self.add_output('output')

    def dim_changed(self):
        self.dim = self.dim_input()

    def index_changed(self):
        self.index = self.index_input()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if -1 - len(input_tensor.shape) < self.dim < len(input_tensor.shape):
                if -1 - input_tensor.shape[self.dim] < self.index < input_tensor.shape[self.dim]:
                    self.output.send(torch.select(input_tensor, self.dim, self.index))
                    return


class TorchMaskedSelectNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchMaskedSelectNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('source tensor', triggers_execution=True)
        self.mask_input = self.add_input('mask')
        self.out = self.add_output('selection tensor')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.mask_input()
            if data is not None:
                mask_tensor = self.data_to_tensor(data, device=input_tensor.device, dtype=torch.bool, requires_grad=input_tensor.requires_grad)
                if mask_tensor is not None:
                    try:
                        out_tensor = torch.masked_select(input_tensor, mask_tensor)
                        self.out.send(out_tensor)
                    except Exception as e:
                        traceback.print_exception(e)
                        print(self.label, e)


class TorchTakeNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchTakeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.index_input = self.add_input('indices in')
        self.output = self.add_output('output')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.index_input()
            if data is not None:
                index_tensor = self.data_to_tensor(data, dtype=torch.long, device=input_tensor.device, requires_grad=input_tensor.requires_grad)
                if index_tensor is not None:
                    try:
                        taken = torch.take(input_tensor, index_tensor)
                        self.output.send(taken)
                    except Exception as e:
                        traceback.print_exception(e)
                        print(self.label, e)
                else:
                    if self.app.verbose:
                        print(self.label, ' no index tensor')
            else:
                if self.app.verbose:
                    print(self.label, ' invalid input tensor')


class TorchTakeAlongDimNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchTakeAlongDimNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.dim_specified = True
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.index_input = self.add_input('indices in')
        self.add_dim_input()
        self.output = self.add_output('output')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.index_input()
            if data is not None:
                index_tensor = self.data_to_tensor(data, device=input_tensor.device, requires_grad=input_tensor.requires_grad, dtype=torch.long)
                if index_tensor is not None:
                    if -1 - len(input_tensor.shape) < self.dim <= len(input_tensor.shape):
                        try:
                            taken = torch.take_along_dim(input_tensor, indices=index_tensor, dim=self.dim)
                            self.output.send(taken)
                        except Exception as e:
                            traceback.print_exception(e)
                            print(self.label, 'failed')
                else:
                    if self.app.verbose:
                        print(self.label, ' no index tensor')
            else:
                if self.app.verbose:
                    print(self.label, ' invalid input tensor')


class TorchIndexSelectNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchIndexSelectNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.index_input = self.add_input('indices in')
        if self.dim_specified:
            self.add_dim_input()

        self.output = self.add_output('output')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.index_input()
            if data is not None:
                index_tensor = self.data_to_tensor(data, device=input_tensor.device, requires_grad=input_tensor.requires_grad, dtype=torch.long)
                if index_tensor is not None:
                    if -1 - len(input_tensor.shape) < self.dim < len(input_tensor.shape):
                        self.output.send(torch.index_select(input_tensor, self.dim, index_tensor))
                    else:
                        if self.app.verbose:
                            print('t.index_select dim is invalid', self.dim)
                else:
                    if self.app.verbose:
                        print('t.index_select no index tensor')
            else:
                if self.app.verbose:
                    print('t.index_select invalid input tensor')


class TorchNarrowNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchNarrowNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.dim_specified = True
        self.start = 0
        self.length = 1
        self.add_dim_input()
        self.start_input = self.add_input('start', widget_type='drag_int', default_value=self.start, callback=self.params_changed)
        self.length_input = self.add_input('length', widget_type='drag_int', default_value=self.length, callback=self.params_changed)
        self.output = self.add_output('tensor out')

    def params_changed(self):
        self.start = self.start_input()
        self.length = self.length_input()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if self.dim < len(input_tensor.shape):
                critical_dim = input_tensor.shape[self.dim]
                if -critical_dim <= self.start < critical_dim and self.start + self.length <= critical_dim:
                    output_tensor = torch.narrow(input_tensor, dim=self.dim, start=self.start, length=self.length)
                    self.output.send(output_tensor)


class TorchDiagNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDiagNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.flip_list = [0]
        self.diag = 0
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.which_property = self.add_property('which diag', widget_type='input_int', default_value=self.diag,
                                                  callback=self.diag_changed)
        self.output = self.add_output('output')

    def diag_changed(self):
        self.diag = self.which_property()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            output_tensor = torch.diag(input_tensor, self.diag)
            self.output.send(output_tensor)


class TorchTriangleNode(TorchNode):
    op_dict = {
        't.tril': torch.tril,
        't.triu': torch.triu
    }
    @staticmethod
    def factory(name, data, args=None):
        node = TorchTriangleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.op = torch.tril
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]

        self.diag = 0
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.which_property = self.add_property('which diag', widget_type='input_int', default_value=self.diag,
                                                  callback=self.diag_changed)
        self.output = self.add_output('output')

    def diag_changed(self):
        self.diag = self.which_property()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            output_tensor = self.op(input_tensor, self.diag)
            self.output.send(output_tensor)


