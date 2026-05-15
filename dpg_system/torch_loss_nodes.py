import torch.nn.functional

from dpg_system.torch_base_nodes import *

def register_torch_loss_nodes():
    Node.app.register_node('t.mse_loss', TorchMSELossNode.factory)
    Node.app.register_node('t.l1_loss', TorchL1LossNode.factory)
    Node.app.register_node('t.cross_entropy_loss', TorchCrossEntropyLossNode.factory)


class TorchMSELossNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchMSELossNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.target_input = self.add_input('target')
        self.loss_output = self.add_output('loss')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        target_tensor = self.data_to_tensor(self.target_input(), match_tensor=input_tensor)
        if target_tensor is None:
            return
        try:
            loss = torch.nn.functional.mse_loss(input_tensor, target_tensor, reduction='sum')
            self.loss_output.send(loss.item())
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()


class TorchL1LossNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchL1LossNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.target_input = self.add_input('target')
        self.loss_output = self.add_output('loss')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        target_tensor = self.data_to_tensor(self.target_input(), match_tensor=input_tensor)
        if target_tensor is None:
            return
        try:
            loss = torch.nn.functional.l1_loss(input_tensor, target_tensor, reduction='sum')
            self.loss_output.send(loss.item())
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()


class TorchCrossEntropyLossNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCrossEntropyLossNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.target_input = self.add_input('target')
        self.loss_output = self.add_output('loss')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        target_tensor = self.data_to_tensor(self.target_input(), match_tensor=input_tensor)
        if target_tensor is None:
            return
        try:
            loss = torch.nn.functional.cross_entropy(input_tensor, target_tensor)
            self.loss_output.send(loss.item())
        except Exception as e:
            # cross_entropy is picky about target dtype/shape (class indices
            # must be long; soft targets must match input shape). The error
            # message from torch is the most informative thing to surface.
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()

