import torch.nn as nn
import torch
from utils import temporal_loop_stack
from spiking_neuron import base


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


class fbGSC(nn.Module):
    def __init__(self, in_dim=40, spiking_neuron=None, drop=0.3):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, 300), nn.Dropout(drop),
                   RecurrentContainer(spiking_neuron(), hid_dim=300)]
        layers += [nn.Linear(300, 300), nn.Dropout(drop),
                   spiking_neuron()]
        layers += [nn.Linear(300, 12)]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"  # x: [T, bs, 80]
        res = temporal_loop_stack(x, self.features)
        return res.sum(0)


class ffGSC(nn.Module):
    def __init__(self, in_dim=40, spiking_neuron=None, drop=0.3):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, 300), nn.Dropout(drop),
                   spiking_neuron()]
        layers += [nn.Linear(300, 300), nn.Dropout(drop),
                   spiking_neuron()]
        layers += [nn.Linear(300, 12)]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"  # x: [T, bs, 80]
        res = temporal_loop_stack(x, self.features)
        return res.sum(0)