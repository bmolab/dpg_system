from dataclasses import dataclass
import torch
import os
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dpg_system.node import Node
from dpg_system.conversion_utils import *


# NOT SURE OF THE UTILITY OF THIS...

def register_torch_dataloader_nodes():
    Node.app.register_node("t.data_set", TorchDataSetNode.factory)


class TorchDataSet(Dataset):
    def __init__(self, dataset_dir):
        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt','')
            self.ds[k] = torch.load(data_fname)

    def __len__(self):
       return 0

    def __getitem__(self, idx):
        data =  {k: self.ds[k][idx] for k in self.ds.keys()}
        return data


class TorchDataSetNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDataSetNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.dataset = None
        self.dataset_directory = ''
        if len(args) > 0:
            self.dataset_directory = args[0]
            self.dataset = TorchDataSet(self.dataset_directory)


