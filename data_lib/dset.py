import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# get sample batch_size data from the offline data
def get_data(batch_size, observations_all, actions_all):
    idx_shuffle = np.random.permutation(observations_all.shape[0])[:n]
    return observations_all[idx_shuffle], actions_all

''' Real dataset sampler '''
class Net(nn.Module):
    def __init__(self, dataset, device, batch_size):
        super(Net, self).__init__()
        self.name        = 'dset'
        self.device      = device
        self.batch_size  = batch_size
        self.observations_all, self.actions_all = dataset['observations'], dataset['actions']

    def forward(self, placeholder=None, task_indices=None, cls=None, new_batch_size=None, no_cuda=False):
        batch_size = self.batch_size if new_batch_size is None else new_batch_size
        
        batched_observations, batched_actions = get_data(batch_size, self.observations_all, self.actions_all)
        
        if no_cuda:
            return batched_observations, batched_actions 
        else:
            return batched_observations.to(self.device), batched_actions.to(self.device)