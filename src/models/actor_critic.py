import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class ActorCriticOld(nn.Module):
    """
    implements both actor and critic in one model
    TODO:
        - make in/out layers have variable dimensions
    """

    def __init__(self, obs_dim=4, action_dim=2):
        super().__init__()
        self.affine1 = nn.Linear(obs_dim, 128)

        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x):

        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_probs = F.softmax(self.actor_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.critic_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_probs, state_values

    def predict(self, cur_state):
        """get the action and value from the current state"""
        action_probs, state_values = self(cur_state)
        # need to make generic for continuous action spaces
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, state_values
