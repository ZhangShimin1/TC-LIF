import torch.nn as nn
import torch
from utils import temporal_loop_stack
import torch.nn.functional as F
from spiking_neuron import base
import surrogate


class LinearRecurrentContainer(base.MemoryModule):
    def __init__(self, sub_module: nn.Module, in_features: int, out_features: int, bias: bool = True,
                 step_mode='s') -> None:
        super().__init__()
        self.step_mode = step_mode
        assert not hasattr(sub_module, 'step_mode') or sub_module.step_mode == 's'
        self.sub_module_out_features = out_features
        self.rc = nn.Linear(in_features, in_features, bias)
        self.sub_module = sub_module
        self.register_memory('y', None)

    def forward(self, x: torch.Tensor):
        if self.y is None:
            if x.ndim == 2:
                self.y = torch.zeros([x.shape[0], self.sub_module_out_features]).to(x)
            else:
                out_shape = [x.shape[0]]
                out_shape.extend(x.shape[1:-1])
                out_shape.append(self.sub_module_out_features)
                self.y = torch.zeros(out_shape).to(x)
        #x = torch.cat((x, self.y), dim=-1)
        self.y = self.sub_module(self.rc(self.y) + x)
        return self.y

    def extra_repr(self) -> str:
        return f', step_mode={self.step_mode}'

class ElementWiseRecurrentContainer(base.MemoryModule):
    def __init__(self, sub_module: nn.Module, element_wise_function=lambda x, y: x + y, step_mode='s', hid_dim=64):
        super().__init__()
        self.hid_weight = nn.Linear(hid_dim, hid_dim)
        self.step_mode = step_mode
        assert not hasattr(sub_module, 'step_mode') or sub_module.step_mode == 's'
        self.sub_module = sub_module
        self.element_wise_function = element_wise_function
        self.register_memory('y', None)

        #nn.init.orthogonal_(self.hid_weight.weight)

    def forward(self, x: torch.Tensor):
        if self.y is None:
            self.y = torch.zeros_like(x.data)
        self.y = self.sub_module(self.element_wise_function(self.hid_weight(self.y), x))
        return self.y

    def extra_repr(self) -> str:
        return f'element-wise function={self.element_wise_function}, step_mode={self.step_mode}'

class BatchNorm1d(nn.BatchNorm1d, base.StepModule):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x):
        if self.step_mode == 's':
            return super().forward(x)


class fbMnist(nn.Module):
    def __init__(self, in_dim=8, spiking_neuron=None):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, 64),
                   LinearRecurrentContainer(spiking_neuron(), in_features=64, out_features=64)]
        layers += [nn.Linear(64, 256),
                   LinearRecurrentContainer(spiking_neuron(), in_features=256, out_features=256)]
        layers += [nn.Linear(256, 10)]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"  # x: [bs, 784, 1]
        output_current = []
        for time in range(x.size(1)):  # T loop
            start_idx = time
            if start_idx < (x.size(1) - 8):
                x_t = x[:, start_idx:start_idx+8, :].reshape(-1, 8)
            else:
                x_t = x[:, 776:784, :].reshape(-1, 8)
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
        layers += [nn.Linear(256, 10)]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"  # x: [bs, 784, 1]
        output_current = []
        for time in range(x.size(1)):  # T loop
            start_idx = time
            if start_idx < (x.size(1) - 8):
                x_t = x[:, start_idx:start_idx+8, :].reshape(-1, 8)
            else:
                x_t = x[:, 776:784, :].reshape(-1, 8)
            output_current.append(self.features(x_t))
        res = torch.stack(output_current, 0)
        return res.sum(0)

class ff_SHD(nn.Module):
    def __init__(self, in_dim=8, hidden =128, out_dim=20, spiking_neuron=None,drop=0.0):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, hidden),
                   nn.Dropout(drop),
                   spiking_neuron()]
        layers += [nn.Linear(hidden, hidden),
                   nn.Dropout(drop),
                   spiking_neuron()]
        layers += [nn.Linear(hidden, out_dim)]
        self.features = nn.Sequential(*layers)

    def forward(self, x): # x [N, T, F]
        assert x.dim() == 3, "dimension of x is not correct!"
        output_current = []
        for time in range(x.size(1)):  # T loop
            x_t = x[:, time, :]
            output_current.append(self.features(x_t))
        res = torch.stack(output_current, 0)
        return res.sum(0)

'''
class fb_SHD(nn.Module):
    def __init__(self, in_dim=8, hidden =128, out_dim=20, spiking_neuron=None, drop=0.0):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, hidden),
                   nn.Dropout(drop),
                   ElementWiseRecurrentContainer(spiking_neuron())]
        layers += [nn.Linear(hidden, hidden),
                   nn.Dropout(drop),
                   ElementWiseRecurrentContainer(spiking_neuron())]
        layers += [nn.Linear(hidden, out_dim)]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"  # x: [bs, 784, 1]
        output_current = []
        for time in range(x.size(1)):  # T loop
            x_t = x[:, time, :]
            output_current.append(self.features(x_t))
        res = torch.stack(output_current, 0)
        return res.sum(0)
'''

class SRNN_multi(nn.Module):
    def __init__(self, in_dim=8, hidden =128, out_dim=20, spiking_neuron=None, drop=0.0):
        super(SRNN_multi, self).__init__()
        self.input_size = in_dim
        self.output_size = out_dim
        self.hidden_size = hidden
        self.i2h_1 = nn.Linear(self.input_size, self.hidden_size)
        self.h2h_1 = nn.Linear(self.hidden_size, hidden)

        self.i2h_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.h2h_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.h2o_2 = nn.Linear(self.hidden_size, self.output_size)
        self.spiking_neuron = spiking_neuron()

    def forward(self, input):
        N = input.size(0)
        T = input.size(1)
        device = input.device
        h2h1_spike = torch.zeros(N, self.hidden_size, device=device)
        h2h2_spike = torch.zeros(N, self.hidden_size, device=device)
        output_sum = torch.zeros(N, self.output_size, device=device)

        for step in range(T):
            input_x = input[:, step, :]
            h1_input = self.i2h_1(input_x.float()) + self.h2h_1(h2h1_spike)
            h2h1_spike = self.spiking_neuron(h1_input)

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_spike = self.spiking_neuron(h2_input)
            h2o3_mem = self.h2o_2(h2h2_spike)
            output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.

        outputs = output_sum / T
        return outputs

class fb_SHD_BN(nn.Module):
    def __init__(self, in_dim=8, hidden =128, out_dim=20, spiking_neuron=None, drop=0.0):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, hidden),
                   BatchNorm1d(hidden),
                   nn.Dropout(drop),
                   ElementWiseRecurrentContainer(spiking_neuron(), hid_dim=hidden)]
        layers += [nn.Linear(hidden, hidden),
                   BatchNorm1d(hidden),
                   nn.Dropout(drop),
                   ElementWiseRecurrentContainer(spiking_neuron(), hid_dim=hidden)]
        layers += [nn.Linear(hidden, out_dim)]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"
        output_current = []
        for time in range(x.size(1)):  # T loop
            x_t = x[:, time, :]
            output_current.append(self.features(x_t))
        res = torch.stack(output_current, 0)
        return res.sum(0)

class fb_SHD_1RNN(nn.Module):
    def __init__(self, in_dim=8, hidden =128, out_dim=20, spiking_neuron=None, drop=0.0):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, hidden),
                   nn.Dropout(drop),
                   ElementWiseRecurrentContainer(spiking_neuron(), hid_dim=hidden)]
        layers += [nn.Linear(hidden, out_dim)]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"
        output_current = []
        for time in range(x.size(1)):  # T loop
            x_t = x[:, time, :]
            output_current.append(self.features(x_t))
        res = torch.stack(output_current, 0)
        return res.sum(0)


class fb_SHD_v1(nn.Module):
    def __init__(self, in_dim=8, hidden =128, out_dim=20, spiking_neuron=None, drop=0.0):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, hidden),
                   nn.Dropout(drop),
                   ElementWiseRecurrentContainer(spiking_neuron(), hid_dim=hidden)]
        layers += [nn.Linear(hidden, hidden),
                   nn.Dropout(drop),
                   ElementWiseRecurrentContainer(spiking_neuron(), hid_dim=hidden)]
        layers += [nn.Linear(hidden, out_dim)]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"
        output_current = []
        for time in range(x.size(1)):  # T loop
            x_t = x[:, time, :]
            output_current.append(self.features(x_t))
        res = torch.stack(output_current, 0)
        return res.sum(0)

class fb_SHD(nn.Module):
    def __init__(self, in_dim=8, hidden =128, out_dim=20, spiking_neuron=None, drop=0.0):
        super().__init__()
        self.out_dim= out_dim
        layers = []
        layers += [nn.Linear(in_dim, hidden),
                   nn.Dropout(drop),
                   ElementWiseRecurrentContainer(spiking_neuron(), hid_dim=hidden)]
        layers += [nn.Linear(hidden, hidden),
                   nn.Dropout(drop),
                   ElementWiseRecurrentContainer(spiking_neuron(), hid_dim=hidden)]
        layers += [nn.Linear(hidden, out_dim)]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"
        outputs = 0
        for time in range(x.size(1)):  # T loop
            x_t = x[:, time, :]
            out_t = self.features(x_t)
            outputs = outputs + out_t #F.softmax(out_t, dim=1)
        outputs = F.log_softmax(outputs/x.size(1), dim=1)
        return outputs
#'''

class fb_SHD_v2(nn.Module):
    def __init__(self, in_dim=8, hidden =128, out_dim=20, spiking_neuron=None, drop=0.0):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, hidden),
                   nn.Dropout(drop),
                   ElementWiseRecurrentContainer(spiking_neuron(), hid_dim=hidden)]
        layers += [nn.Linear(hidden, hidden),
                   nn.Dropout(drop),
                   spiking_neuron()]
        layers += [nn.Linear(hidden, out_dim)]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"
        output_current = []
        for time in range(x.size(1)):  # T loop
            x_t = x[:, time, :]
            output_current.append(self.features(x_t))
        res = torch.stack(output_current, 0)
        return res.sum(0)

class fb_SHD_Linear(nn.Module):
    def __init__(self, in_dim=8, hidden =128, out_dim=20, spiking_neuron=None, drop=0.0):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, hidden),
                   nn.Dropout(drop),
                   LinearRecurrentContainer(spiking_neuron(), in_features=hidden, out_features=hidden)]
        layers += [nn.Linear(hidden, hidden),
                   nn.Dropout(drop),
                   LinearRecurrentContainer(spiking_neuron(), in_features=hidden, out_features=hidden)]
        layers += [nn.Linear(hidden, out_dim)]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"
        x = x.permute(1, 0, 2).contiguous()  # x shape [T, N, F]
        res = temporal_loop_stack(x, self.features)
        return res.sum(0)

class ff_SHD_dp(nn.Module):
    def __init__(self, in_dim=8, hidden =128, out_dim=20, spiking_neuron=None, dp=0.0):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, hidden),
                   nn.Dropout(dp),
                   spiking_neuron()]
        layers += [nn.Linear(hidden, hidden),
                   nn.Dropout(dp),
                   spiking_neuron()]
        layers += [nn.Linear(hidden, out_dim)]
        self.features = nn.Sequential(*layers)

    def forward(self, x): # x [N, T, F]
        assert x.dim() == 3, "dimension of x is not correct!"
        output_current = []
        for time in range(x.size(1)):  # T loop
            x_t = x[:, time, :]
            output_current.append(self.features(x_t))
        res = torch.stack(output_current, 0)
        return res.sum(0)
