import importlib
import os, sys, time
import copy
import random
import numpy as np
import time
import yaml
import argparse
import d4rl
import gym
import wandb
import uuid
import torch
from torch import nn
import torch.utils.data as tdata

from dataclasses import asdict
from data_load import D4RLDataset
import d4rl.gym_mujoco

from tools.evaluator import Evaluator

from configs.command_parser import command_parser, merge_args

from utils import get_optimizer, set_seed, get_hms, wrap_env, compute_mean_std, normalize_states
import warnings
warnings.filterwarnings('ignore')

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_args_and_config():
    # parse arguments
    parser = command_parser()
    args   = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
        config = dict2namespace(config)

    config = merge_args(args, config)
    return args, config


# args and config
args, config = parse_args_and_config()
setattr(config, 'pid', str(os.getpid()))


# Init Wandb
def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

# Load Environment
env = gym.make(config.dataset.name)
dataset = env.get_dataset()

# Normalize ENV
if config.normalize:
    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
else:
    state_mean, state_std = 0, 1

dataset["observations"] = normalize_states(
    dataset["observations"], state_mean, state_std
)
dataset["next_observations"] = normalize_states(
    dataset["next_observations"], state_mean, state_std
)
env = wrap_env(env, state_mean=state_mean, state_std=state_std)

evaluator = Evaluator(env=env, config=config)
eval_intervals = np.arange(0, config.training.n_iters + 1, config.training.eval_every)

save_folder_path = os.path.join(config.save_dir, config.dataset.name[:-3])

wandb_init(config)

time_start = time.time()

over_time_avg_return_list = []
over_time_avg_normalize_return_list = []

for it in eval_intervals[:-2]:
    print("%s iterations"%(it))
    h, m, s = get_hms(time.time() - time_start)
    print("Execute time: %dh %dm %ds"%(h, m, s))

    # Load synset
    if config.q_weight:
        save_path_name = os.path.join(save_folder_path, config.dataset.name[:-3] + '_size_' + str(config.synset_size) + '_' + config.match_objective + '_q-weight_' + str(config.beta_weight) + '_init_' + str(config.synset_init) + '_iter_' + str(it) + '_seed_' + str(config.seed) + '.pt')
    else:
        save_path_name = os.path.join(save_folder_path, config.dataset.name[:-3] + '_size_' + str(config.synset_size) + '_' + config.match_objective + '_no-q-weight_init_' + str(config.synset_init) + '_iter_' + str(it) + '_seed_' + str(config.seed) + '.pt')

    synset = torch.load(save_path_name)

    # Evaluate return
    return_list = []
    normalize_return_list = []

    for i in range(10):
        trajectory_return_info = evaluator.trajectory_return(synset)
        episode_return = trajectory_return_info["episode_rewards"]
        return_list.append(episode_return)
        normalize_return_list.append(env.get_normalized_score(episode_return))
        
    print(np.mean(return_list))
    wandb.log({"avg_episode_rewards": np.mean(return_list)}, step=it)
    wandb.log({"avg_normalize_episode_rewards": np.mean(normalize_return_list)}, step=it)

    over_time_avg_return_list.append(np.mean(return_list))
    over_time_avg_normalize_return_list.append(np.mean(normalize_return_list))

    if config.q_weight:
        if len(over_time_avg_return_list) > 4:
            wandb.log({"sooth_avg_episode_rewards": np.mean(over_time_avg_return_list[-5:])}, step=it)
            wandb.log({"sooth_avg_normalize_episode_rewards": np.mean(over_time_avg_normalize_return_list[-5:])}, step=it)
    else:
        if len(over_time_avg_return_list) > 10:
            wandb.log({"sooth_avg_episode_rewards": np.mean(over_time_avg_return_list[-10:])}, step=it)
            wandb.log({"sooth_avg_normalize_episode_rewards": np.mean(over_time_avg_normalize_return_list[-10:])}, step=it)