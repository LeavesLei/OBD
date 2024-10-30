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
from data_lib.data_bptt import SynSetBPTT 
import d4rl.gym_mujoco

from data_load import D4RLDataset

from tools.evaluator import Evaluator

from configs.command_parser import command_parser, merge_args

from utils import get_optimizer, set_seed, get_hms, wrap_env, compute_mean_std, normalize_states
import warnings
warnings.filterwarnings('ignore')


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

"""
  Flatten tensors
"""
def flatten(data):
    return torch.cat([ele.flatten() for ele in data])


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


"""
  Train function for compressor
"""
def train(config):
    eval_intervals = np.arange(0, config.training.n_iters + 1, config.training.eval_every)

    # import synset libraries
    synset_lib = importlib.import_module('data_lib.syndset')

    # prepare dataset
    env = gym.make(config.dataset.name)
    observation_space = env.observation_space.shape
    action_space = env.action_space.shape
    max_action = float(env.action_space.high[0])

    dataset = d4rl.qlearning_dataset(env)

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

    dataloader = tdata.DataLoader(D4RLDataset(dataset), batch_size=config.evaluation.batch_size, shuffle=True, num_workers=config.evaluation.num_workers)

    seed = config.seed
    set_seed(seed, env)

    # wandb setting
    print(config.group)
    wandb_init(config)

    # offline policy path
    config.offline_policy_path = os.path.join(config.offline_policy_dir, 'Cal-QL-' + config.dataset.name, 'checkpoint.pt')

    """
      Define synset, BPTT model
    """
    synset = synset_lib.Net(
                      syn_dset_size=config.synset_size,
                      observation_space=observation_space,
                      action_space=action_space,
                      config=config,
                      device=config.device,
                  )
    # initialize synset
    synset.init_synset(config.synset_init, config.dataset.name, dataset)

    synset_bptt = SynSetBPTT(
                          synset=synset,
                          observation_space=observation_space,
                          action_space=action_space,
                          max_action=max_action,
                          config=config
                      ).to(config.device)

    synset_optimizer = get_optimizer(synset_bptt.synset.parameters(), config.synset_optim)

    print('Synset optimizer:', synset_optimizer)

    evaluator = Evaluator(env=env, config=config)

    time_start = time.time()

    # create folder for saving synset
    save_folder_path = os.path.join(config.save_dir, config.dataset.name[:-3])
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # Train for N iterations
    for it in range(config.training.n_iters):
        # evaluate at intervals
        if it in eval_intervals:
            print("%s iterations"%(it))
            h, m, s = get_hms(time.time() - time_start)
            print("Execute time: %dh %dm %ds"%(h, m, s))

            synset_copy = copy.deepcopy(synset_bptt.synset)

            # save synset
            if config.q_weight:
                save_path_name = os.path.join(save_folder_path, config.dataset.name[:-3] + '_size_' + str(config.synset_size) + '_' + config.match_objective + '_q-weight_' + str(config.beta_weight) + '_init_' + str(config.synset_init) + '_iter_' + str(it) + '_seed_' + str(config.seed) + '.pt')
            else:
                save_path_name = os.path.join(save_folder_path, config.dataset.name[:-3] + '_size_' + str(config.synset_size) + '_' + config.match_objective + '_no-q-weight_init_' + str(config.synset_init) + '_iter_' + str(it) + '_seed_' + str(config.seed) + '.pt')
            torch.save(synset_copy, save_path_name)

            trajectory_return_info = evaluator.trajectory_return(synset_copy)
            offline_loss = evaluator.offline_loss(synset_copy, dataloader)

            print("Normalized return: " + str(trajectory_return_info["normalized_return"]))

            wandb.log(trajectory_return_info, step=it)
            wandb.log({"offline_test_loss": offline_loss}, step=it)

        # Optimize compressor_bptt model with inner loops
        synset_optimizer.zero_grad()
        loss, dL_dc, dL_dw  = synset_bptt.forward(test_dataloader=dataloader)
        wandb.log({"train loss": loss}, step=it)
        torch.cuda.empty_cache()

        synset.assign_grads(grads=[flatten(ele) for ele in dL_dc])

        torch.nn.utils.clip_grad_norm_(synset_bptt.parameters(), max_norm=2)
        synset_optimizer.step()
        synset_optimizer.zero_grad()

    print('Training completed.')


def parse_args_and_config():
    # parse arguments
    parser = command_parser()
    args   = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
        config = dict2namespace(config)

    # merge arguments and config
    config = merge_args(args, config)

    return args, config
    

def main():
    # args and config
    args, config = parse_args_and_config()
    setattr(config, 'pid', str(os.getpid()))

    train(config)
    

if __name__ == '__main__':
    sys.exit(main())