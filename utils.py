import torch
import torch.optim as optim
import numpy as np
import os
import uuid
import random
import time
import gym
import wandb
from network import MLPBase
from typing import Any, Dict, List, Optional, Tuple, Union




def set_seed(
    seed: int, env: None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: float = 0.0,
    state_std: float = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


# evaluate policy
def eval(env, pf, eval_episodes, device, seed=0):
    env.seed(seed)
    pf.eval()
    rewards = []
    lengths = []
    start_time = time.time()
    for _ in range(eval_episodes):
        ob = env.reset()
        done = False
        episode_reward = 0
        length = 0
        while not done:
            length += 1
            ob_tensor = torch.Tensor(ob).to(device)
            act = pf.eval_act(ob_tensor)
            ob, r, done, _ = env.step(act)
            episode_reward += r
        rewards.append(episode_reward)
        lengths.append(length)
    return {
        "episode_rewards": np.mean(rewards),
        "normalized_return": env.get_normalized_score(np.mean(episode_rewards)),
        "episode_lengths": np.mean(lengths),
        "eval_times": time.time() - start_time
    }


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: torch.nn.Module, device: str, n_episodes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    lengths = []
    start_time = time.time()
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        length = 0
        while not done:
            length += 1
            action = actor.act(state, device)
            state, reward, done, env_infos = env.step(action)
            episode_reward += reward

        # Valid only for environments with goal
        lengths.append(length)
        episode_rewards.append(episode_reward)
    
    actor.train()
    return {
        "episode_rewards": np.mean(episode_rewards),
        "normalized_return": env.get_normalized_score(np.mean(episode_rewards)),
        "episode_lengths": np.mean(lengths),
        "eval_times": time.time() - start_time
    }


@torch.no_grad()
def eval_ensemble_actor(
    env: gym.Env, actor_list: list, device: str, n_episodes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:

    def _get_ensemble_policy_preds(policy_list, obs):
        preds_mat = np.zeros((len(policy_list), env.action_space.shape[0]))
        for i, policy in enumerate(policy_list):
            preds = policy.act(obs, device)
            preds_mat[i] = preds
            
        return np.mean(preds_mat, axis=0)

    env.seed(seed)
    [actor.eval() for actor in actor_list]
    episode_rewards = []
    lengths = []
    start_time = time.time()
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        length = 0
        while not done:
            length += 1
            action = _get_ensemble_policy_preds(actor_list, state)
            state, reward, done, env_infos = env.step(action)
            episode_reward += reward

        # Valid only for environments with goal
        lengths.append(length)
        episode_rewards.append(episode_reward)
    
    [actor.train() for actor in actor_list]
    return {
        "episode_rewards": np.mean(episode_rewards),
        "normalized_return": env.get_normalized_score(np.mean(episode_rewards)),
        "episode_lengths": np.mean(lengths),
        "eval_times": time.time() - start_time
    }


def update(env, pf, plr, bc_loss, batch, device):

    obs = batch['obs']
    actions = batch['acts']

    obs = torch.Tensor(obs).to(device)
    actions = torch.Tensor(actions).to(device)
    
    plr = 3e-4
    
    pf_optimizer = optim.Adam(
            pf.parameters(),
            lr=plr,
        )

    """
    Policy Loss.
    """

    new_actions = pf(obs)
    if pf.tanh_action:
        lb = torch.Tensor(
            env.action_space.low).to(device)
        ub = torch.Tensor(
            env.action_space.high).to(device)
        new_actions = lb + (new_actions + 1) * 0.5 * (ub - lb)
    policy_loss = bc_loss(new_actions, actions)

    """
    Update Networks
    """

    pf_optimizer.zero_grad()
    policy_loss.backward()
    pf_optimizer.step()

    # Information For Logger
    info = {}
    info['Training/policy_loss'] = policy_loss.item()

    info['new_actions/mean'] = new_actions.mean().item()
    info['new_actions/std'] = new_actions.std().item()
    info['new_actions/max'] = new_actions.max().item()
    info['new_actions/min'] = new_actions.min().item()

    return info


def update_gaussian(env, pf, plr, batch, device):

    obs = batch['obs']
    actions = batch['acts']

    obs = torch.Tensor(obs).to(device)
    actions = torch.Tensor(actions).to(device)
    
    pf_optimizer = optim.Adam(
            pf.parameters(),
            lr=plr,
        )

    """
    Policy Loss.
    """
    bc_loss = torch.nn.MSELoss()
    new_actions, next_log_pi = pf(obs)
    policy_loss = bc_loss(new_actions, actions)

    """
    Update Networks
    """

    pf_optimizer.zero_grad()
    policy_loss.backward()
    pf_optimizer.step()

    # Information For Logger
    info = {}
    info['Training/policy_loss'] = policy_loss.item()

    return info


def get_policy(input_shape, output_shape, policy_cls, policy_params):

    return policy_cls(
        input_shape=input_shape,
        output_shape=output_shape,
        base_type=MLPBase,
        **policy_params)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


"""
  Get optimizer for a model
"""
def get_optimizer(parameters, optim_config):
    if optim_config.optimizer == 'AdamW':
        return optim.AdamW(parameters, lr=optim_config.lr, weight_decay=float(optim_config.weight_decay))
    elif optim_config.optimizer == 'Adam':
        return optim.Adam(parameters, lr=optim_config.lr, weight_decay=float(optim_config.weight_decay))
    elif optim_config.optimizer == 'SGD':
        return optim.SGD(parameters, lr=optim_config.lr, momentum=optim_config.momentum)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(optim_config.optimizer))