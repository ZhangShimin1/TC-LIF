import torch.nn as nn
import torch
from spiking_neuron import base, neuron
from surrogate import Triangle as SG


class RecurrentContainer(base.MemoryModule):
    def __init__(self, sub_module: nn.Module, element_wise_function=lambda x, y: x + y, step_mode='s', hid_dim=64):
        super().__init__()
        self.hid_weight = nn.Linear(hid_dim, hid_dim)
        # nn.init.orthogonal_(self.hid_weight.weight)
        self.step_mode = step_mode
        assert not hasattr(sub_module, 'step_mode') or sub_module.step_mode == 's'
        self.sub_module = sub_module
        self.element_wise_function = element_wise_function
        self.register_memory('y', None)

    def forward(self, x: torch.Tensor):
        if self.y is None:
            self.y = torch.zeros_like(x.data)
        self.y = self.sub_module(self.element_wise_function(self.hid_weight(self.y), x))
        return self.y

    def extra_repr(self) -> str:
        return f'element-wise function={self.element_wise_function}, step_mode={self.step_mode}'


class fbMnist(nn.Module):
    def __init__(self, in_dim=8, spiking_neuron=None):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, 64),
                   RecurrentContainer(spiking_neuron(v_threshold=1.0), hid_dim=64)]
        layers += [nn.Linear(64, 256),
                   RecurrentContainer(spiking_neuron(v_threshold=1.0), hid_dim=256)]
        layers += [nn.Linear(256, 256),
                   spiking_neuron(v_threshold=1.0)]
        layers += [nn.Linear(256, 10)]
        self.features = nn.Sequential(*layers)
        self.in_dim = in_dim

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"  # x: [bs, 784, 1]
        output_current = []
        for time in range(x.size(1)):  # T loop
            start_idx = time
            if start_idx < (x.size(1) - self.in_dim):
                x_t = x[:, start_idx:start_idx+self.in_dim, :].reshape(-1, self.in_dim)
            else:
                x_t = x[:, 784-self.in_dim:784, :].reshape(-1, self.in_dim)
            output_current.append(self.features(x_t))
        res = torch.stack(output_current, 0)
        return res.sum(0)


class ffMnist(nn.Module):
    def __init__(self, in_dim=8, spiking_neuron=None):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, 64),
                   spiking_neuron()]
        layers += [nn.Linear(64, 256),
                   spiking_neuron()]
        layers += [nn.Linear(256, 256),
                   spiking_neuron()]
        layers += [nn.Linear(256, 10)]
        self.features = nn.Sequential(*layers)
        self.in_dim = in_dim

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"  # x: [bs, 784, 1]
        output_current = []
        for time in range(x.size(1)):  # T loop
            start_idx = time
            if start_idx < (x.size(1) - self.in_dim):
                x_t = x[:, start_idx:start_idx+self.in_dim, :].reshape(-1, self.in_dim)
            else:
                x_t = x[:, 784-self.in_dim:784, :].reshape(-1, self.in_dim)
            output_current.append(self.features(x_t))
        res = torch.stack(output_current, 0)
        return res.sum(0)


