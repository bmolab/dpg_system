import torch

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
    Node.app.register_node('t.ravel', TorchViewVariousNode.factory)
    Node.app.register_node('t.masked_select', TorchMaskedSelectNode.factory)
    Node.app.register_node('t.index_select', TorchIndexSelectNode.factory)
    Node.app.register_node('t.take', TorchTakeNode.factory)
    Node.app.register_node('t.take_along_dim', TorchTakeAlongDimNode.factory)
    Node.app.register_node('t.diag', TorchDiagNode.factory)
    Node.app.register_node('t.tril', TorchTriangleNode.factory)
    Node.app.register_node('t.triu', TorchTriangleNode.factory)
    Node.app.register_node('t.scatter', TorchScatterNode.factory)
    Node.app.register_node('t.scatter_hold', TorchScatterHoldNode.factory)


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
        return TorchViewNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # 1. Initialize State
        # Default to [-1] (flatten) if no args provided
        self.target_shape = [-1]
        if args and len(args) > 0:
            self.target_shape = [any_to_int(arg) for arg in args]

        # 2. Format default string
        default_str = str(self.target_shape)[1:-1]

        self.input = self.add_input('tensor in', triggers_execution=True)

        # 3. Hybrid Input
        # Changed label from '' to 'shape' for clarity in the UI
        self.shape_input = self.add_input('shape', widget_type='text_input', widget_width=200, default_value=default_str, callback=self.shape_changed)

        self.output = self.add_output('output')

    def shape_changed(self):
        self.target_shape = self.shape_input.conform_to_int_list()

    def execute(self):
        input_tensor = self.input_to_tensor()

        if input_tensor is None:
            return

        # If shape is empty/invalid, we can't view/reshape. Pass through or return.
        if not self.target_shape:
            self.output.send(input_tensor)
            return

        try:
            # Determine operation based on label (flexible check)
            # If label contains 'view' (e.g. "t.view"), use view. Otherwise reshape.
            if 'view' in self.label.lower():
                # .view() requires the tensor to be contiguous
                if not input_tensor.is_contiguous():
                    # Helpful warning before the inevitable crash
                    if self.app.verbose:
                        print(
                            f"Node {self.label} WARNING: Input is non-contiguous. .view() will fail. Use .reshape() or .contiguous().view()")

                # Unpack shape list
                view_tensor = input_tensor.view(*self.target_shape)
            else:
                # .reshape() handles non-contiguous tensors by copying if necessary
                view_tensor = input_tensor.reshape(*self.target_shape)

            self.output.send(view_tensor)

        except RuntimeError as e:
            # Captures shape mismatch errors (e.g. numel doesn't match)
            print(f"Node {self.label}: Shape Mismatch - {e}")
        except Exception as e:
            print(f"Node {self.label}: Error - {e}")
            traceback.print_exc()


class TorchViewVariousNode(TorchNode):
    op_dict = {
        't.adjoint': torch.adjoint,
        't.t': torch.t,
        't.flatten': torch.flatten,
        't.ravel': torch.ravel
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
        return TorchPermuteNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # 1. Initialize internal state
        self.permute_dims = []
        if args and len(args) > 0:
            self.permute_dims = [any_to_int(a) for a in args]

        # 2. Prepare default string (e.g., "0, 2, 1")
        default_str = str(self.permute_dims)[1:-1]

        self.input = self.add_input('tensor in', triggers_execution=True)

        # 3. Hybrid Input for Permutation indices
        self.permute_input = self.add_input('permute', widget_type='text_input', default_value=default_str, callback=self.permute_changed)

        self.output = self.add_output('permuted tensor out')

    def permute_changed(self):
        self.permute_dims = self.permute_input.conform_to_int_list()

    def execute(self):
        input_tensor = self.input_to_tensor()

        if input_tensor is None:
            return

        # If no permutation list is defined, pass through or return
        if not self.permute_dims:
            self.output.send(input_tensor)
            return

        try:
            # VALIDATION: Check if dimensions match
            # Permute requires the number of indices to match the tensor rank exactly.
            if len(input_tensor.shape) != len(self.permute_dims):
                if self.app.verbose:
                    print(
                        f"Node {self.label} WARNING: Rank Mismatch. Tensor is {len(input_tensor.shape)}D, Permute list is {len(self.permute_dims)}D.")

                # Option A: Fail Gracefully (Pass original) - Matches your previous code
                self.output.send(input_tensor)
                return

            # Execute Permute
            # torch.permute(input, dims) expects a tuple/list of ints
            permuted = torch.permute(input_tensor, self.permute_dims)
            self.output.send(permuted)

        except RuntimeError as e:
            print(f"Node {self.label}: Permute Error - {e}")
        except Exception as e:
            print(f"Node {self.label}: Error - {e}")


class TorchRepeatNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        return TorchRepeatNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # 1. Initialize internal state from args
        self.current_repeats = []
        if args and len(args) > 0:
            self.current_repeats = [any_to_int(a) for a in args]

        # 2. Format default string for widget (e.g., "2, 2")
        default_str = str(self.current_repeats)[1:-1]

        self.input = self.add_input('tensor in', triggers_execution=True)

        # 3. Hybrid Input: Acts as both a Data Port and a Text Widget
        self.repeats_input = self.add_input('repeats', widget_type='text_input', default_value=default_str, callback=self.repeat_changed)

        self.output = self.add_output('repeated tensor out')

    def repeat_changed(self):
        self.current_repeats = self.repeats_input.conform_to_int_list()

    def execute(self):
        input_tensor = self.input_to_tensor()

        if input_tensor is None:
            return

        try:
            if not self.current_repeats:
                # No repeats defined? Pass original tensor
                self.output.send(input_tensor)
                return

            # Note: PyTorch .repeat() expects *args (varargs)
            # We unpack the list using *
            repeated = input_tensor.repeat(*self.current_repeats)
            self.output.send(repeated)

        except RuntimeError as e:
            # Handles dimension mismatches (e.g. repeating a 3D tensor with 2 args)
            print(f"Node {self.label}: Repeat Error (Dimension mismatch?) - {e}")
        except Exception as e:
            print(f"Node {self.label}: Error - {e}")


class TorchTileNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        return TorchTileNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # Internal state for tiling dimensions
        self.current_tile_dims = []
        if args and len(args) > 0:
            self.current_tile_dims = [any_to_int(a) for a in args]

        # Create a string representation for the widget default
        default_str = str(self.current_tile_dims)[1:-1]

        self.input = self.add_input('tensor in', triggers_execution=True)

        # Single input acting as both data port and UI widget
        self.tile_input = self.add_input('tiling', widget_type='text_input', default_value=default_str, callback=self.tile_changed)

        self.output = self.add_output('tiled tensor out')

    def tile_changed(self):
        self.current_tile_dims = self.tile_input.conform_to_int_list()

    def execute(self):
        input_tensor = self.input_to_tensor()

        if input_tensor is None:
            return

        try:
            if not self.current_tile_dims:
                # Pass through input if no tiling specified
                self.output.send(input_tensor)
                return

            # We use * unpacking to handle torch.Size objects or lists correctly
            tiled = input_tensor.repeat(*self.current_tile_dims)
            self.output.send(tiled)

        except RuntimeError as e:
            print(f"Node {self.label}: Tiling Error (Shape mismatch?) - {e}")
        except Exception as e:
            print(f"Node {self.label}: Error - {e}")


class TorchRollNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        return TorchRollNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # 1. Initialize Internal State
        # Expecting args to be [[shifts], [dims]] if provided
        self.roll_shifts = [1]
        self.roll_dims = [0]  # Default to rolling dim 0

        if args and len(args) >= 1:
            self.roll_shifts = [any_to_int(x) for x in args[0]] if isinstance(args[0], list) else [any_to_int(args[0])]
        if args and len(args) >= 2:
            self.roll_dims = [any_to_int(x) for x in args[1]] if isinstance(args[1], list) else [any_to_int(args[1])]

        # 2. Defaults for Widgets
        shifts_str = str(self.roll_shifts)[1:-1]
        dims_str = str(self.roll_dims)[1:-1]

        self.input = self.add_input('tensor in', triggers_execution=True)

        self.shifts_input = self.add_input('shifts', widget_type='text_input', default_value=shifts_str, callback=self.shifts_changed)
        self.dims_input = self.add_input('dims', widget_type='text_input', default_value=dims_str, callback=self.dims_changed)

        self.output = self.add_output('rolled tensor')

    def shifts_changed(self):
        self.roll_shifts = self.shifts_input.conform_to_int_list()

    def dims_changed(self):
        self.roll_dims = self.dims_input.conform_to_int_list()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return

        try:
            # Basic validation: Lists must be populated
            if not self.roll_shifts or not self.roll_dims:
                self.output.send(input_tensor)
                return

            # Advanced validation: Length mismatch
            # torch.roll((x, y), shifts=(1, 2), dims=(0,)) -> Crash
            # The lengths must match if they are lists/tuples
            if len(self.roll_shifts) != len(self.roll_dims):
                if self.app.verbose:
                    print(
                        f"Node {self.label} WARNING: Mismatch in 'shifts' ({len(self.roll_shifts)}) and 'dims' ({len(self.roll_dims)}) counts.")
                # We return here to prevent a hard crash, passing original data
                self.output.send(input_tensor)
                return

            # Execute Roll
            # PyTorch expects tuples or ints
            rolled = torch.roll(input_tensor, shifts=tuple(self.roll_shifts), dims=tuple(self.roll_dims))
            self.output.send(rolled)

        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            print(f"Node {self.label} Error: {e}")


class TorchSubtensorNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSubtensorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.slice_obj = (slice(None),)

        # Determine default string
        index_string = ''.join(args) if args else ':'

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.indices_input = self.add_input(
            'indices',
            widget_type='text_input',
            widget_width=200,
            default_value=index_string,
            callback=self.dim_changed
        )
        self.output = self.add_output('output')

        # Parse the default string immediately
        self.dim_changed()

    def dim_changed(self):
        """
        Parses the input string (e.g., "0:5, -1, ::2") into a tuple
        of slice objects and integers that PyTorch can consume directly.
        """
        dim_text = any_to_string(self.indices_input())

        # Clean up brackets/whitespace
        dim_text = dim_text.strip().strip("[]")

        if not dim_text:
            self.slice_obj = (slice(None),)
            self.execute()
            return

        parts = dim_text.split(',')
        slices = []

        for part in parts:
            part = part.strip()

            if part == '...':
                slices.append(Ellipsis)
            elif ':' in part:
                # Handle slice notation
                sub_parts = part.split(':')
                slice_args = []
                for sp in sub_parts:
                    if sp.strip() == '':
                        slice_args.append(None)  # None indicates start/end/step default
                    else:
                        try:
                            slice_args.append(int(sp))
                        except ValueError:
                            slice_args.append(None)
                slices.append(slice(*slice_args))
            else:
                # Handle single integer index
                try:
                    slices.append(int(part))
                except ValueError:
                    # Fallback for garbage input
                    slices.append(slice(None))

        self.slice_obj = tuple(slices)
        self.execute()

    def execute(self):
        # 1. Get Tensor (using parent class helper)
        # Note: self.input_to_tensor() usually returns None if not connected
        input_tensor = self.input_to_tensor()

        # 2. SILENT CHECK: No Input
        if input_tensor is None:
            return

        # 3. SILENT CHECK: Dimension Mismatch (Waiting for real data)
        # Calculate required dimensions (excluding Ellipsis which is wild)
        required_dims = sum(1 for s in self.slice_obj if s is not Ellipsis)

        # If the input tensor (e.g. 1D placeholder) is smaller than the requested
        # slice depth (e.g. 3D), we assume the real data hasn't arrived yet.
        if input_tensor.ndim < required_dims:
            return

        # 4. Apply Slice
        try:
            sub_tensor = input_tensor[self.slice_obj]
            self.output.send(sub_tensor)
        except Exception as e:
            # If dimensions matched but something else was wrong (e.g. index out of bounds),
            # we print the error to help debug.
            print(f"TorchSubtensorNode Error: {e}")


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


class TorchScatterNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchScatterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.dim_specified = True
        self.input = self.add_input('tensor to scatter into', triggers_execution=True)
        self.index_input = self.add_input('indices in')
        self.source_input = self.add_input('source in')

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
                        data_2 = self.source_input()
                        source_tensor = self.data_to_tensor(data_2, device=input_tensor.device, requires_grad=input_tensor.requires_grad)
                        if source_tensor is not None:
                            try:
                                taken = torch.scatter(input_tensor, self.dim, index_tensor, source_tensor)
                                self.output.send(taken)
                            except Exception as e:
                                traceback.print_exception(e)
                                print(self.label, 'failed')
                        else:
                            if self.app.verbose:
                                print(self.label, ' no index tensor')
                else:
                    if self.app.verbose:
                        print(self.label, ' no index tensor')
            else:
                if self.app.verbose:
                    print(self.label, ' invalid input tensor')


class TorchScatterHoldNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchScatterHoldNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        length = 16
        if len(args) > 0:
            length = any_to_int(args[0])
        self.dim_specified = True
        self.input = self.add_input('list of index value pairs', triggers_execution=True)
        self.length_input = self.add_input('length of target tensor', widget_type='drag_int', default_value=length, min=0, callback=self.resize)
        self.clear_in = self.add_input('clear', widget_type='button', callback=self.clear)
        self.output = self.add_output('output')
        self.accum = torch.zeros(length)

    def clear(self):
        self.accum[:] = 0

    def resize(self):
        self.accum = torch.zeros(self.length_input())

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            pairs_tensor = input_tensor.view(-1, 2).transpose(0, 1)
            indices = pairs_tensor[0].type(torch.int64)
            indices = indices.clamp(0, self.length_input() - 1)
            values = pairs_tensor[1]
            try:
                self.accum.scatter_(0, indices, values)
                self.output.send(self.accum)
            except Exception as e:
                print(e)


class TorchIndexSelectNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        return TorchIndexSelectNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.index_input = self.add_input('indices')
        if self.dim_specified:
            self.add_dim_input()

        self.output = self.add_output('output')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return

        index_data = self.index_input()
        if index_data is None:
            return

        try:
            # --- 1. Robust Conversion to Index Tensor ---
            index_tensor = self.data_to_tensor(index_data, device=input_tensor.device, dtype=torch.long)

            # --- 2. Shape Validation ---
            if index_tensor.dim() != 1:
                # Optional: Auto-flatten if user passed [[1, 2]], or error out.
                # Here we strictly enforce 1D but provide a helpful error.
                if self.app.verbose:
                    print(
                        f"Node {self.label} Warning: Indices must be 1D. Got shape {index_tensor.shape}. Flattening...")
                index_tensor = index_tensor.reshape(-1)

            # --- 3. Dimension Validation ---
            ndim = input_tensor.ndim
            # self.dim comes from the TorchWithDimNode parent
            if not (-ndim <= self.dim < ndim):
                if self.app.verbose:
                    print(f"Node {self.label} Error: Dim {self.dim} is out of bounds for {ndim}D tensor.")
                return

            # --- 4. Execution ---
            selected = torch.index_select(input_tensor, self.dim, index_tensor)
            self.output.send(selected)

        except RuntimeError as e:
            # Handles index out of bounds errors
            print(f"Node {self.label}: Runtime Error (Index out of bounds?) - {e}")
        except Exception as e:
            traceback.print_exc()
            print(f"Node {self.label}: Error - {e}")


class TorchNarrowNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchNarrowNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)

        # Initialize state
        self.start = 0
        self.length = 1

        # Setup specific to parent class logic
        self.dim_specified = True
        self.add_dim_input()

        # Widgets with callbacks
        self.start_input = self.add_int_input('start', widget_type='drag_int', default_value=self.start,
                                          callback=self.params_changed)
        self.length_input = self.add_int_input('length', widget_type='drag_int', default_value=self.length,
                                           callback=self.params_changed)

        self.output = self.add_output('tensor out')

    def params_changed(self):
        # Update internal state
        self.start = self.start_input()
        self.length = self.length_input()
        # IMPROVEMENT: Trigger execution immediately when widgets change
        # self.execute()

    def execute(self):
        input_tensor = self.input_to_tensor()

        if input_tensor is None:
            return

        # Validate Dimension
        if self.dim >= len(input_tensor.shape) or self.dim < -len(input_tensor.shape):
            print(f"Error: Dim {self.dim} out of bounds for shape {input_tensor.shape}")
            return

        dim_size = input_tensor.shape[self.dim]

        # Handle negative indexing (Pythonic style)
        actual_start = self.start
        if actual_start < 0:
            actual_start += dim_size

        # Validate Start
        if actual_start < 0 or actual_start >= dim_size:
            print(
                f"Error: Start index {self.start} results in {actual_start}, which is out of bounds for dim size {dim_size}")
            return

        # Validate Length (and safeguard against 0 or negative length if desired)
        if self.length <= 0:
            print("Error: Length must be positive")
            return

        if actual_start + self.length > dim_size:
            print(f"Error: Slice ends at {actual_start + self.length}, but dim size is only {dim_size}")
            return

        try:
            output_tensor = torch.narrow(input_tensor, dim=self.dim, start=actual_start, length=self.length)
            self.output.send(output_tensor)
        except Exception as e:
            print(f"Torch Error: {e}")


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


