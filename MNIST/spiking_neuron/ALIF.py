from abc import abstractmethod
from typing import Callable
import torch
from spiking_neuron import base
from spikingjelly.activation_based import neuron


class BaseNode(base.MemoryModule):
    def __init__(self,
                 v_threshold: float = 1.,
                 v_reset: float = 0.,
                 surrogate_function: Callable = None,
                 detach_reset: bool = False,
                 step_mode='s', backend='torch',
                 store_v_seq: bool = False):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold

        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq


    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset

        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: torch.Tensor):
        v = v - spike * v_threshold
        return v

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        # return self.surrogate_function(self.v - self.names[''])
        raise NotImplementedError

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class ALIF(BaseNode):
    def __init__(self,
                 v_threshold=1.,
                 v_reset=0.,
                 surrogate_function: Callable = None,
                 detach_reset=False,
                 hard_reset=False,
                 vn_reset=True,
                 step_mode='s',
                 backend='torch',
                 store_v_seq: bool = False,
                 k=1,
                 alpha=1,
                 decay_factor: torch.Tensor = torch.full([1, 2], 0, dtype=torch.float),
                 gamma = 0.):
        # a = torch.zeros([1, 1], dtype=torch.float)
        super(ALIF, self).__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.k = k
        for i in range(1, self.k + 1):
            self.register_memory('v' + str(i), 0.)
        self.register_memory('yita', 0.)
        self.register_memory('threshold', 1.)

        self.names = self._memories
        self.hard_reset = hard_reset
        self.vn_reset = vn_reset
        assert 0 < alpha <= 1
        self.decay_factor = torch.nn.Parameter(decay_factor)

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        # print(self.decay_factor)
        spike_lastT = self.neuronal_fire()
        alpha = torch.exp(-1 / torch.sigmoid(self.decay_factor[0][0]))
        r0 = torch.exp(-1 / torch.sigmoid(self.decay_factor[0][1]))
        self.names['yita'] = r0 * self.names['yita'] + (1 - r0) * spike_lastT
        self.names['threshold'] = 0.01 + 1.8 * self.names['yita']
        self.names['v1'] = alpha * self.names['v1'] + (1 - alpha) * x
        self.v = self.names['v1']

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.names['threshold'])

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if not self.hard_reset:
            # soft reset
            self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_d, self.names['threshold'])
            self.v = self.jit_soft_reset(self.v, spike_d, self.names['threshold'])

        else:
            # hard reset
            self.v1 = self.jit_hard_reset(self.v1, spike_d, self.v_reset)
            if self.vn_reset:
                for i in range(2, self.k + 1):
                    self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_d,  self.v_reset)

    def multi_step_forward(self, x_seq: torch.Tensor):
        # if self.training:
        if self.backend == 'torch':
            return super().multi_step_forward(x_seq)
        elif self.backend == 'cupy':
            raise NotImplementedError
        else:
            raise ValueError(self.backend)

    def forward(self, x: torch.Tensor):
        # if self.training:
        return super().single_step_forward(x)

    def extra_repr(self):
        return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, hard_reset={self.hard_reset}, " \
               f"vn_reset={self.vn_reset}, k={self.k}, step_mode={self.step_mode}, backend={self.backend}"

