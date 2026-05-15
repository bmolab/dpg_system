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
        if not os.path.isdir(dataset_dir):
            print(f'TorchDataSet: dataset dir not found: {dataset_dir!r}')
            return
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt', '')
            # Per-file try/except: one corrupt .pt file shouldn't kill the
            # whole dataset load — skip it and keep the rest available.
            try:
                self.ds[k] = torch.load(data_fname)
            except Exception as e:
                print(f'TorchDataSet: failed to load {data_fname}: '
                      f'{type(e).__name__}: {e}')

    def __len__(self):
        # Was: return 0 (so DataLoader produced no iterations). Take the
        # min leading-dim across loaded tensors so __getitem__ is always
        # in-bounds even when files have different lengths.
        if not self.ds:
            return 0
        lengths = []
        for v in self.ds.values():
            try:
                lengths.append(len(v))
            except TypeError:
                # A scalar / 0-dim tensor doesn't define len; treat as 1.
                lengths.append(1)
        return min(lengths)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self)):
            raise IndexError(f'TorchDataSet index {idx} out of range [0, {len(self)})')
        return {k: self.ds[k][idx] for k in self.ds.keys()}


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
            # Wrap construction so a missing dir or unreadable .pt file
            # doesn't tear down the node — leave self.dataset = None and
            # surface the error instead of failing during creation.
            try:
                self.dataset = TorchDataSet(self.dataset_directory)
            except Exception as e:
                print(f'{self.label}: could not load dataset from '
                      f'{self.dataset_directory!r}: {type(e).__name__}: {e}')
                traceback.print_exc()


