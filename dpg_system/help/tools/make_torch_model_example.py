"""Make the example files the t.model and t.load help pages read (2026-10-25).

    python3 dpg_system/help/tools/make_torch_model_example.py

Trains a tiny network (1 -> 16 -> 16 -> 1) to draw one cycle of a sine wave over
inputs 0 to 1, then writes it twice into help/data:

    t_model_example.pt    the runnable model, as TorchScript (t.model runs it)
    t_load_example.pt     its weights alone, a dict of tensors (t.load reads it)

The seed is fixed, so a rerun writes the same network.
"""
import os
import math
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


class SineNet(torch.nn.Module):
    # takes any shape of values; each value is one input to the network
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(),
                                       torch.nn.Linear(16, 16), torch.nn.Tanh(),
                                       torch.nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x.unsqueeze(-1)).squeeze(-1)


def main():
    torch.manual_seed(1)
    model = SineNet()
    x = torch.linspace(0, 1, 256)
    y = torch.sin(2 * math.pi * x)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    for step in range(3000):
        optimiser.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        optimiser.step()
    print(f'final mean squared error {loss.item():.6f}')
    model.eval()
    torch.jit.script(model).save(os.path.join(DATA_DIR, 't_model_example.pt'))
    torch.save(model.state_dict(), os.path.join(DATA_DIR, 't_load_example.pt'))


if __name__ == '__main__':
    main()
