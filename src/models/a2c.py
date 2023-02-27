import torch
import torch.nn as nn
import torch.optim as optim
from common.mlp import MLP
from models.distributions import get_distribution_func
from torch import Tensor
import torch.nn as nn

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad) 

class ActorCritic(nn.Module):
    """
    implements both actor and critic in one model
    TODO:
        - make in/out layers have variable dimensions
    """

    def __init__(self, action_type="normal", obs_dim=2048, action_dim=4, hidden_dim=256, layers=8):
        super().__init__()

        assert action_type == "discrete" or action_type == "normal" \
            or action_type == "trunc_normal" or action_type == "normal_tanh"
        
        if action_type == "discrete":
            self.actor = MLP(obs_dim, action_dim, hidden_dim, layers)
            self.bc_loss = nn.CrossEntropyLoss()
        else:
            self.actor = MLP(obs_dim, 2*action_dim, hidden_dim, layers)
            self.bc_loss = nn.MSELoss()

        self.critic = MLP(obs_dim, 1, hidden_dim, layers)
        self.critic_target = MLP(obs_dim, 1, hidden_dim, layers)
        self.critic_target.requires_grad_(False)
        self.train_steps = 0
        self.action_type = action_type
        self.action_dist = get_distribution_func(action_type)

        print('actor has', count_parameters(self.actor), 'params and', layers, 'layers')
        print('critic has', count_parameters(self.critic), 'params and', layers, 'layers')


    def forward(self, x):

        # actor: choses action to take from state s_t
        # by returning probability of each action
        a_dist = self.action_dist(self.actor(x))
        
        # critic: evaluates being in the state s_t
        state_values = self.critic(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return a_dist, state_values

    def forward_t(self, x):

        # actor: choses action to take from state s_t
        # by returning probability of each action
        a_dist = self.action_dist(self.actor(x))

        # critic: evaluates being in the state s_t
        state_values = self.critic(x)
        target_values = self.critic_target(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return a_dist, state_values, target_values

    def predict(self, cur_state, deterministic=False, **kwargs):
        """get the action and value from the current state"""
        a_dist, state_values = self(cur_state)
        # need to make generic for continuous action spaces
        action = a_dist.sample()
        return action, state_values

    def get_bc_loss(self, states, expert_actions):
        if self.action_type == "discrete":
            logits = self.actor(states)
            BCLoss = self.bc_loss(logits, expert_actions).mean()
        else:
            a_dist, _ = self.forward(states)
            act_sample = a_dist.sample()
            BCLoss = self.bc_loss(act_sample, expert_actions).mean()
        return BCLoss

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
