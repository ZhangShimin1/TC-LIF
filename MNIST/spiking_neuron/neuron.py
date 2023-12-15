from abc import abstractmethod
from typing import Callable
import torch
from spiking_neuron import base
from spikingjelly.activation_based import neuron


class BaseNode(base.MemoryModule):
    def __init__(self,
                 v_threshold: float = 1.,
                 surrogate_function=None,
                 hard_reset: bool = False,
                 detach_reset: bool = False):

        assert isinstance(v_threshold, float)
        assert isinstance(hard_reset, bool)
        assert isinstance(detach_reset, bool)
        super().__init__()

        self.register_memory('v', 0.)

        self.v_threshold = v_threshold

        self.hard_reset = hard_reset
        self.detach_reset = detach_reset

        self.surrogate_function = surrogate_function

    def forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.hard_reset:
            self.v = self.v * (1. - spike_d)
        else:
            self.v = self.v - spike_d * self.v_threshold

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, detach_reset={self.detach_reset}, hard_reset={self.hard_reset}'

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class LIFNode(BaseNode):
    def __init__(self,
                 decay_factor: torch.Tensor = None,
                 v_threshold: float = 1.,
                 surrogate_function: Callable = None,
                 hard_reset: bool = False,
                 detach_reset: bool = False,
                 gamma: float = 0.):
        self.decay_factor = torch.tensor(0.8).float()

        super().__init__(v_threshold, surrogate_function, hard_reset, detach_reset)

    def extra_repr(self):
        return super().extra_repr() + f', decay_factor={self.decay_factor}'

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v * self.decay_factor + x
