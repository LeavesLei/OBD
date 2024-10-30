import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from network import FullyConnectedQFunction

from sklearn.cluster import KMeans, kmeans_plusplus

ENV_START_STATE = {"hopper": [1.25] + [0] * 10, "halfcheetah": [0] * 17, "walker2d": [1.25] + [0] *16}
ENV_RESET_SCALE = {"hopper": 5e-3, "halfcheetach": 0.1, "walker2d": 5e-3}


''' Synthetic data generator '''
class Net(nn.Module):
    def __init__(
            self,
            syn_dset_size,
            observation_space,
            action_space,
            config,
            device
        ):
        super(Net, self).__init__()
        self.name = 'syn data'
        self.syn_dset_size = syn_dset_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

        self.prev_imgs = None

        self.observations = nn.Embedding(self.syn_dset_size, np.prod(self.observation_space))
        self.actions = nn.Embedding(self.syn_dset_size, np.prod(self.action_space))

        self.config = config
        self.offline_critic = self.load_critic(self.config.offline_policy_path)


    def load_critic(self, critic_path):
        critic = FullyConnectedQFunction(self.observation_space[0], self.action_space[0])
        checkpoint = torch.load(critic_path)
        
        critic.load_state_dict(state_dict=checkpoint["critic1"])
        critic.eval()

        return critic.to(self.device)


    def init_synset(self, init_type, env, dataset):
        if init_type == 'real':
            perm = torch.randperm(dataset['observations'].shape[0])
            selected_idx = perm[:self.syn_dset_size]
            
            self.observations.weight = torch.nn.Parameter(torch.Tensor(dataset['observations'][selected_idx]))
            self.actions.weight = torch.nn.Parameter(torch.Tensor(dataset['actions'][selected_idx]))

        elif init_type == 'q-value-real':
            q_value = self.offline_critic(torch.Tensor(dataset['observations']).to('cuda'), torch.Tensor(dataset['actions']).to('cuda'))
            idx = torch.argsort(q_value.detach().cpu(), descending=True)
            selected_idx = idx[:self.syn_dset_size]

            self.observations.weight = torch.nn.Parameter(torch.Tensor(dataset['observations'][selected_idx]))
            self.actions.weight = torch.nn.Parameter(torch.Tensor(dataset['actions'][selected_idx]))

        elif init_type == 'q-value-kmeans-real':
            model = KMeans(n_clusters=self.syn_dset_size, random_state=0, max_iter=1)
            kmeans = model.fit(dataset['observations'])
            cluster_label = kmeans.labels_

            q_value = self.offline_critic(torch.Tensor(dataset['observations']).to('cuda'), torch.Tensor(dataset['actions']).to('cuda'))
            q_min = torch.min(q_value.detach().cpu())
            idx = torch.argsort(q_value.detach().cpu(), descending=True)

            selected_idx = []
            for i in range(self.syn_dset_size):
                q_value_copy = copy.deepcopy(q_value.detach().cpu())
                q_value_copy[cluster_label != i] = q_min - 1.
                selected_idx.append(torch.argsort(q_value_copy.detach().cpu(), descending=True)[0])

            self.observations.weight = torch.nn.Parameter(torch.Tensor(dataset['observations'][selected_idx]))
            self.actions.weight = torch.nn.Parameter(torch.Tensor(dataset['actions'][selected_idx]))

        elif init_type == 'kmeans++':
            centers, selected_idx = kmeans_plusplus(dataset['observations'], n_clusters=self.syn_dset_size)

            self.observations.weight = torch.nn.Parameter(torch.Tensor(dataset['observations'][selected_idx]))
            self.actions.weight = torch.nn.Parameter(torch.Tensor(dataset['actions'][selected_idx]))

        elif init_type == 'random':
            env_class = env.split("-")[0]
            state_mean, state_std = torch.Tensor(ENV_START_STATE[env_class]), ENV_RESET_SCALE[env_class]
            
            self.observations.weight = torch.nn.Parameter(state_mean + state_std * torch.randn(self.observation_space))
            self.actions.weight = torch.nn.Parameter(2 * torch.rand(self.action_space) - 1.) # actions in [-1, 1]
        
        elif init_type == 'xavier':
            torch.nn.init.xavier_uniform(self.observations.weight)
            torch.nn.init.xavier_uniform(self.actions.weight)
        else:
            raise Exception("Synset initialization type can note be recognized.")


    def assign_grads(self, grads):
        
        obs_grads = grads[0]
        actions_grads = grads[1]

        self.observations.weight.grad = obs_grads.to(self.observations.weight.data.device).view(self.observations.weight.shape)
        self.actions.weight.grad = actions_grads.to(self.actions.weight.data.device).view(self.actions.weight.shape)

    def forward(self, placeholder=None):

        observations = self.observations
        actions = self.actions

        return observations, actions