import math
import os
import sys

import d4rl_atari
import gym
import numpy as np
import torch
import cv2
from torch.utils.data import Sampler
import matplotlib.pyplot as plt

# TODO: Combine these into a single sampler


class EquiSampler(Sampler):
    """Equidistant batch sampler.

    Yields n (where n==batch_size) equidistant indices, steps through
    the dataset by adding the sequence length to each index and yielding
    the new set of indices. 
    """

    def __init__(self,  data_size, seq_len, batch_size, init_idx=None):
        self.data_size = data_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.init_idx = init_idx
        self.chunk_size = math.ceil(self.data_size/self.batch_size)
        self.n_steps = math.ceil(self.chunk_size/self.seq_len)
        print("Chunk size:", self.chunk_size)
        print("n steps:", self.n_steps)

    def __iter__(self):
        if self.init_idx is None:
            init_idx = np.random.randint(self.data_size)
        else:
            init_idx = self.init_idx
        for i in range(self.n_steps):
            iters = []
            for j in range(self.batch_size):
                start_idx = (init_idx + i*self.seq_len +
                             j*self.chunk_size) % self.data_size
                iters.append(start_idx)
            yield iters

    def __len__(self):
        return self.n_steps


class ACSampler(Sampler):
    """Actor critic sampler

    For any given episode, this sampler will yield a start index between 0 
    and episode_length - seq_len 
    """

    def __init__(self,  n_transitions, episode_starts, seq_len, batch_size, init_idx=None):
        self.indices = self.build_indices(
            n_transitions, episode_starts, seq_len)
        self.data_size = len(self.indices)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.init_idx = init_idx

        self.chunk_size = math.ceil(self.data_size/self.batch_size)
        self.n_steps = math.ceil(self.chunk_size/self.seq_len)

        print("Chunk size:", self.chunk_size)
        print("n steps:", self.n_steps)

    @staticmethod
    def build_indices(n_transitions, episode_starts, seq_len):
        last_start_idx = episode_starts[0]
        indices = []
        for start_idx in episode_starts[1:]:
            indices.extend(np.arange(last_start_idx, start_idx-seq_len))
            last_start_idx = start_idx
        indices.extend(np.arange(last_start_idx, n_transitions-seq_len))
        return np.array(indices)

    def __iter__(self):
        if self.init_idx is None:
            init_idx = np.random.randint(self.data_size)
        else:
            init_idx = self.init_idx
        for i in range(self.n_steps):
            iters = []
            for j in range(self.batch_size):
                start_idx = (init_idx + i*self.seq_len +
                             j*self.chunk_size) % self.data_size
                idx = self.indices[start_idx]
                iters.append(idx)
            yield iters

    def __len__(self):
        return self.n_steps

# TODO: Implement inverse discounting to prioritize long-horizon rewards
def lambda_return(rewards, values, lambda_=0.95, gamma=0.99, inverse=False):
    """
    \lambda return recursive definition, according to Dreamerv2 paper:

    if t<H:
        V^\lambda_t = r_t + \gamma * ((1-\lambda)*v_hat_t+1 + \lambda * V^\lambda_t+1)
    elif t=H:
        V^\lambda_t = r_t + \gamma * v_hat[-1]

    according to Director paper, 
    if t=H: V^\lambda_t = v_hat[-1]
    """
    # returns[-1] += values[-1]
    R = values[-1]
    returns = [R]
    # ignores last reward and first value
    rewards_less_last = rewards[:-1]
    values_less_first = values[1:]
    for r_t, v_tplus1 in zip(rewards_less_last[::-1], values_less_first[::-1]):
        R = r_t + gamma * ( (1-lambda_)*v_tplus1 + lambda_ * R)
        returns.insert(0,R)
    returns = torch.stack(returns)
    return returns

def MC_return(latent_rewards, bootstrap, norm=False, gamma=0.99, eps=1e-8):
    latent_rewards[-1] += bootstrap
    R = torch.zeros((len(latent_rewards[0])), device=latent_rewards[0].device)
    returns = []
    for r in latent_rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.stack(returns)
    if norm:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns

def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def get_d4rl_data(dataset_name):
    sys.stdout = open(os.devnull, 'w')
    env = gym.make(dataset_name)
    data = env.get_dataset()
    sys.stdout = sys.__stdout__
    return data


def fix_terms(terms):
    resets = np.zeros(len(terms), dtype=np.float32).astype(bool)
    idxs = np.where(terms)[0]+1
    resets[idxs] = True
    return resets

def fix_obs(img, new_hw=64):
    """normalizes image"""
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    img = np.transpose(img, (1, 2, 0))
    img = np.array(cv2.resize(img, dsize=(
        new_hw, new_hw), interpolation=cv2.INTER_AREA))
    img = np.expand_dims(img, 0)
    img = img.astype(np.float32)
    img = img / 255.0 - 0.5
    img = np.expand_dims(img,0)
    return img

def fix_obs_sb3(img):
    s = np.expand_dims(img.squeeze(), axis=(0,1)).astype(np.float32)
    s = (s - 33)/55
    return s



def cat_actions(actions, cats=18):
    targets = actions.reshape(-1)
    one_hot = np.eye(cats)[targets]
    return one_hot


def fix_actions(actions, reset, cats=18):
    """takes array of scalar actions and converts to one-hot.
    also offsets actions b/c dreamer uses pre-actions instead of post-actions"""
    rolled_actions = np.roll(actions, 1)
    one_hot = cat_actions(rolled_actions)
    ridxs = np.where(reset)[0]
    one_hot[ridxs] = np.zeros_like(one_hot[0])
    return one_hot


def transpose_collate(batch):
    """transposes batch and time dimension
    (B, T, ...) -> (T, B, ...)"""
    from torch.utils.data._utils.collate import default_collate
    return {k: torch.transpose(v, 0, 1) for k, v in default_collate(batch).items()}
