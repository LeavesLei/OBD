import sys
import torch
import torch.optim as optim

import numpy as np
import copy

from utils import eval, eval_ensemble_actor, eval_actor, get_optimizer, get_policy
from network import TanhGaussianPolicy, DetContPolicy


policy_params = { 
            "hidden_shapes": [400,300],
            "append_hidden_shapes":[],
            "tanh_action": True
        }


"""
  An Evaluator class:
    first, training the actor with synset
    1. evaluate by sample in the env
    2. evaluate by MSE loss on the offline data
"""


class Evaluator(object):
    def __init__(
            self,
            env,
            config
        ):

        # configs
        self.config = config
        self.env = env    
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        self.policy_type = config.policy_type

        self.eval_ensemble = config.eval_ensemble
        self.ensemble_policy_num = config.ensemble_policy_num

        self.device = config.device
        self.loss_func = torch.nn.MSELoss()


    def _create_policy(self, policy_type):
        if policy_type == 'gaussian':
            policy = TanhGaussianPolicy(self.state_dim, self.action_dim, self.max_action, orthogonal_init=True)
        elif policy_type == 'deterministic':
            policy = get_policy(
            input_shape=self.state_dim,
            output_shape=self.action_dim,
            policy_cls=DetContPolicy,
            policy_params=policy_params
        )
        else:
            raise Exception("Non-recognized policy_type.")
        
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

            optimizer = get_optimizer(policy.parameters(), self.config.bptt_optim)

            pred_actions = self._get_pred_acts(self.policy_type, policy, obs)
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
        if self.eval_ensemble:
            policy_list = self._create_and_train_ensemble_policy(self.policy_type, self.ensemble_policy_num, synset)
            eval_info = eval_ensemble_actor(self.env, policy_list, device=self.config.device, n_episodes=self.config.evaluation.eval_episodes, seed=self.config.seed)

        else:
            policy = self._create_policy(self.policy_type)
            self._train(policy, synset)

            eval_info = self._eval_policy(self.policy_type, policy)

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

            loss = self.loss_func(pred_actions, actions)
            loss_list.append(loss.item())
        
        return np.mean(loss_list)