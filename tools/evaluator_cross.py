import sys
import torch
import torch.optim as optim

import numpy as np
import copy

from utils import eval, eval_ensemble_actor, eval_actor, get_optimizer, get_policy

import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal, TanhTransform, TransformedDistribution
from typing import Any, Dict, List, Optional, Tuple, Union

def _fanin_init(tensor, alpha=0):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    # bound = 1. / np.sqrt(fan_in)
    bound = np.sqrt(1. / ((1 + alpha * alpha) * fan_in))
    return tensor.data.uniform_(-bound, bound)


def _uniform_init(tensor, param=3e-3):
    return tensor.data.uniform_(-param, param)


def _constant_bias_init(tensor, constant=0.1):
    tensor.data.fill_(constant)


def layer_init(layer, weight_init=_fanin_init, bias_init=_constant_bias_init):
    weight_init(layer.weight)
    bias_init(layer.bias)


def basic_init(layer):
    layer_init(layer, weight_init=_fanin_init, bias_init=_constant_bias_init)


def uniform_init(layer):
    layer_init(layer, weight_init=_uniform_init, bias_init=_uniform_init)


def _orthogonal_init(tensor, gain=np.sqrt(2)):
    nn.init.orthogonal_(tensor, gain=gain)


def orthogonal_init(layer, scale=np.sqrt(2), constant=0):
    layer_init(
        layer,
        weight_init=lambda x: _orthogonal_init(x, gain=scale),
        bias_init=lambda x: _constant_bias_init(x, 0))


def basic_init(layer):
    layer_init(layer, weight_init=_fanin_init, bias_init=_constant_bias_init)
    
class MLPBase(nn.Module):
    def __init__(
            self, input_shape, hidden_shapes,
            activation_func=nn.ReLU,
            init_func=basic_init,
            last_activation_func=None):
        super().__init__()

        self.activation_func = activation_func
        self.fcs = []
        if last_activation_func is not None:
            self.last_activation_func = last_activation_func
        else:
            self.last_activation_func = activation_func
        input_shape = np.prod(input_shape)

        self.output_shape = input_shape
        for next_shape in hidden_shapes:
            fc = nn.Linear(input_shape, next_shape)
            init_func(fc)
            self.fcs.append(fc)
            self.fcs.append(activation_func())
            input_shape = next_shape
            self.output_shape = next_shape
        self.fcs.pop(-1)
        self.fcs.append(self.last_activation_func())
        self.seq_fcs = nn.Sequential(*self.fcs)

    def forward(self, x):
        return self.seq_fcs(x)


class Net(nn.Module):
    def __init__(
            self,
            output_shape,
            base_type,
            append_hidden_shapes=[],
            append_hidden_init_func=basic_init,
            net_last_init_func=uniform_init,
            activation_func=nn.ReLU,
            **kwargs):
        super().__init__()
        self.base = base_type(activation_func=activation_func, **kwargs)
        self.activation_func = activation_func
        append_input_shape = self.base.output_shape
        self.append_fcs = []
        for next_shape in append_hidden_shapes:
            fc = nn.Linear(append_input_shape, next_shape)
            append_hidden_init_func(fc)
            self.append_fcs.append(fc)
            self.append_fcs.append(self.activation_func())
            append_input_shape = next_shape

        self.last = nn.Linear(append_input_shape, output_shape)
        net_last_init_func(self.last)

        self.append_fcs.append(self.last)
        self.seq_append_fcs = nn.Sequential(*self.append_fcs)

    def forward(self, x):
        out = self.base(x)
        out = self.seq_append_fcs(out)
        return out


class DetContPolicy(Net):
    def __init__(self, tanh_action=False, **kwargs):
        self.tanh_action = tanh_action
        super(DetContPolicy, self).__init__(**kwargs)

    def forward(self, x):
        if self.tanh_action:
            return torch.tanh(super().forward(x))
        return super().forward(x)

    def eval_act(self, x):
        with torch.no_grad():
            return self.forward(x).squeeze(0).detach().cpu().numpy()

    def explore(self, x):
        return {
            "action": self.forward(x).squeeze(0)
        }


class ReparameterizedTanhGaussian(nn.Module):
    # log_prob: https://pytorch.org/docs/stable/distributions.html#transformeddistribution
    def __init__(
        self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
            # output shape = 2 * action_dim 是因为要学mean, std
        )

        if orthogonal_init:
            self.base_network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.base_network[-1], False)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,  # default False
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        #_, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        #actions = torch.tanh(mean)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 2,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

        layers = [
            nn.Linear(observation_dim + action_dim, 256),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))

        self.network = nn.Sequential(*layers)
        if orthogonal_init:
            self.network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.network[-1], False)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)



class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class TwoLayerTanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
            # output shape = 2 * action_dim 是因为要学mean, std
        )

        if orthogonal_init:
            self.base_network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.base_network[-1], False)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,  # default False
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        #_, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        #actions = torch.tanh(mean)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class ThreeLayerTanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
            # output shape = 2 * action_dim 是因为要学mean, std
        )

        if orthogonal_init:
            self.base_network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.base_network[-1], False)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,  # default False
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        #_, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        #actions = torch.tanh(mean)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class FiveLayerTanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
            # output shape = 2 * action_dim 是因为要学mean, std
        )

        if orthogonal_init:
            self.base_network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.base_network[-1], False)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,  # default False
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        #_, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        #actions = torch.tanh(mean)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class SixLayerTanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
            # output shape = 2 * action_dim 是因为要学mean, std
        )

        if orthogonal_init:
            self.base_network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.base_network[-1], False)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,  # default False
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        #_, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        #actions = torch.tanh(mean)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class ResMLPTanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh
        
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 2 * action_dim)
        
    
        if orthogonal_init:
            #self.base_network.apply(lambda m: init_module_weights(m, True))
            self.layer1.apply(lambda m: init_module_weights(m, True))
            self.layer2.apply(lambda m: init_module_weights(m, True))
            self.layer3.apply(lambda m: init_module_weights(m, True))
            self.layer4.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.base_network[-1], False)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,  # default False
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        #base_network_output = self.base_network(observations)
        out_1 = F.relu(self.layer1(observations))
        out = F.relu(self.layer2(out_1))
        out = F.relu(self.layer3(out))
        out = out + out_1
        base_network_output = self.layer4(out)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        #_, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        #actions = torch.tanh(mean)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class EvaluatorCross(object):
    def __init__(
            self,
            env,
            net_arch,
            config
        ):

        # configs
        self.config = config
        self.env = env    

        self.net_arch = net_arch

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        self.policy_type = config.policy_type

        self.eval_ensemble = config.eval_ensemble
        self.ensemble_policy_num = config.ensemble_policy_num

        self.device = config.device
        self.loss_func = torch.nn.MSELoss()



    def _create_policy(self, net_arch):
        if net_arch == '2layer_mlp':
            policy = TwoLayerTanhGaussianPolicy(self.state_dim, self.action_dim, self.max_action, orthogonal_init=True)
        elif net_arch == '3layer_mlp':
            policy = ThreeLayerTanhGaussianPolicy(self.state_dim, self.action_dim, self.max_action, orthogonal_init=True)
        elif net_arch == '4layer_mlp':
            policy = TanhGaussianPolicy(self.state_dim, self.action_dim, self.max_action, orthogonal_init=True)
        elif net_arch == '5layer_mlp':
            policy = FiveLayerTanhGaussianPolicy(self.state_dim, self.action_dim, self.max_action, orthogonal_init=True)
        elif net_arch == '6layer_mlp':
            policy = SixLayerTanhGaussianPolicy(self.state_dim, self.action_dim, self.max_action, orthogonal_init=True)
        elif net_arch == 'res_mlp':
            policy = ResMLPTanhGaussianPolicy(self.state_dim, self.action_dim, self.max_action, orthogonal_init=True)
        else:
            raise Exception("Non-recognized net_arch.")
        
        return policy.to(self.device)

    
    def _get_pred_acts(self, policy_type, policy, obs):
        if policy_type == 'gaussian':
            pred_actions, _ = policy(obs)
        elif policy_type == 'deterministic':
            pred_actions = policy(obs)
        else:
            raise Exception("Non-recognized policy_type.") 

        return pred_actions

    
    def _eval_policy(self, policy_type, policy):
        if policy_type == 'gaussian':
            eval_info = eval_actor(self.env, policy, device=self.config.device, n_episodes=self.config.evaluation.eval_episodes, seed=self.config.seed)
        elif policy_type == 'deterministic':
            eval_info = eval(self.env, policy, device=self.config.device, eval_episodes=self.config.evaluation.eval_episodes, seed=self.config.seed)
        else:
            raise Exception("Non-recognized policy_type.") 

        return eval_info



    def _train(self, policy, synset):
        policy.train()

        for epoch in range(self.config.bptt.inner_steps):

            #obs, actions = synset.to(self.device)
            obs = synset.observations.weight.to(self.device)
            actions = synset.actions.weight.to(self.device)

            optimizer = get_optimizer(policy.parameters(), self.config.bptt_optim)#optim.Adam(policy.parameters(), lr=self.config.evaluation.policy_lr)

            pred_actions = self._get_pred_acts(self.policy_type, policy, obs)
            #new_actions, _ = policy(obs)
            policy_loss = self.loss_func(pred_actions, actions)
            
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

        policy.eval()

        return None 


    def _create_and_train_ensemble_policy(self, policy_type, ensemble_policy_num, synset):
        policy_list = []
        for i in range(ensemble_policy_num):
            policy = self._create_policy(policy_type)
            self._train(policy, synset)
            policy_list.append(policy)
        
        return policy_list


    def trajectory_return(self, synset):

        policy = self._create_policy(self.net_arch)
        self._train(policy, synset)

        eval_info = self._eval_policy(self.policy_type, policy)
        #eval_info = eval_actor(self.env, policy, device=self.config.device, n_episodes=self.config.evaluation.eval_episodes, seed=self.config.seed)

        return eval_info


    def offline_loss(self, synset, offline_testloader):
        policy = self._create_policy(self.policy_type)
        trained_policy = self._train(policy, synset)

        loss_list = []
        for i, batch in enumerate(offline_testloader):
            obs = batch['obs']
            actions = batch['acts']

            obs = torch.Tensor(obs).to(self.device)
            actions = torch.Tensor(actions).to(self.device)

            pred_actions = self._get_pred_acts(self.policy_type, policy, obs)
            #new_actions, _ = policy(obs)

            loss = self.loss_func(pred_actions, actions)
            loss_list.append(loss.item())
        
        return np.mean(loss_list)       