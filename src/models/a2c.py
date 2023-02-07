import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.mlp import MLP
from torch import Tensor
from torch.distributions import Categorical

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad) 

class ActorCritic(nn.Module):
    """
    implements both actor and critic in one model
    TODO:
        - make in/out layers have variable dimensions
    """

    def __init__(self, obs_dim=2048, action_dim=4, hidden_dim=256, layers=8):
        super().__init__()

        self.actor = MLP(obs_dim, action_dim, hidden_dim, layers)
        self.critic = MLP(obs_dim, 1, hidden_dim, layers)
        self.critic_target = MLP(obs_dim, 1, hidden_dim, layers)
        self.critic_target.requires_grad_(False)
        self.train_steps = 0

        print('actor has', count_parameters(self.actor), 'params and', layers, 'layers')
        print('critic has', count_parameters(self.critic), 'params and', layers, 'layers')

    def forward(self, x):

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_probs = F.softmax(self.actor(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.critic(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_probs, state_values

    def forward_t(self, x):

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_probs = F.softmax(self.actor(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.critic(x)
        target_values = self.critic_target(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_probs, state_values, target_values

    def predict(self, cur_state, deterministic=False, **kwargs):
        """get the action and value from the current state"""
        action_probs, state_values = self(cur_state)
        # need to make generic for continuous action spaces
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, state_values

    def forward_actor(self, x):
        logits = self.actor(x)
        action_probs = F.softmax(logits, dim=-1)
        return logits, action_probs

    def update_critic_target(self):
        self.critic_target.load_state_dict(
            self.critic.state_dict())  # type: ignore
        self.critic_target.requires_grad_(False)


class Discriminator(nn.Module):
    def __init__(self, obs_dim=2048, out_dim=1, hidden_dim=128, layers=4):
        super(Discriminator, self).__init__()

        self.discrim = MLP(obs_dim, out_dim, hidden_dim, layers)
        print('discrim has', count_parameters(self.discrim), 'params and', layers, 'layers')

    def forward(self, x):
        logits = self.discrim(x)
        probs = torch.sigmoid(logits)

        return logits, probs
