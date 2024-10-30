import numpy as np
from torch.utils.data import Dataset

class D4RLDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.obs = self.dataset["observations"]
        self.acts = self.dataset["actions"]
        self.next_obs = self.dataset["observations"]
        self.rews = self.dataset["rewards"]
        self.dones = self.dataset["terminals"]

        self.len = self.obs.shape[0]

    def __getitem__(self, index):
        return {
            "obs": self.obs[index],
            "acts": self.acts[index],
            "next_obs": self.next_obs[index],
            "rews": self.rews[index, np.newaxis],
            "dones": self.dones[index, np.newaxis]
        }

    def __len__(self):
        return self.len-1