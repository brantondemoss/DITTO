import itertools
import random
from collections import deque, namedtuple
from re import L
from typing import Tuple

import cv2
import gym
import numpy as np
import torch
from numba import njit
from torch.utils.data import Dataset
from tqdm import tqdm

Transition = namedtuple(
    "Transition",
    field_names=[
        "transition_id",
        "hidden_state",
        "action",
        "last_state",
        "current_state",
        "reward",
        "done",
        "info",
    ],
)


class Episode(object):
    def __init__(self, init_obs, action_dim=18, vecobs=True, info={}):
        self.vecobs = vecobs
        self.observations = []
        self.rewards = []
        self.terminals = []
        self.actions = []
        self.length = 0
        self.action_dim = action_dim
        self.infos = []
        # print(info)
        self.add_init_state(init_obs, info=info)
        self.action_dim = 4

    def add_init_state(self, init_obs, reward=0, action=np.array([0.1, 0.1, 0.1]), terminal=False, info={}):
        # print(info)
        self.push(init_obs, reward, action, terminal, info)
        # print(self.infos)

    def reshape_img(self, img, new_hw=64):
        img = img.astype(np.uint8)
        img = np.array(cv2.resize(img, dsize=(
            new_hw, new_hw), interpolation=cv2.INTER_AREA))
        img = np.expand_dims(img, 0)
        return img

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def push(self, obs, reward, action, terminal, info):
        obs = self.reshape_img(obs)
        # assert obs.shape == (84, 84, 1)
        self.observations.append(obs)
        self.rewards.append(reward)
        self.actions.append(action)
        self.terminals.append(terminal)
        if "img" in info.keys():
            info["img"] = self.reshape_img(self.rgb2gray(info["img"]))
        else:
            print("AAAAHHH")
        self.infos.append(info)
        self.length += 1


class ReplayBuffer(object):
    def __init__(self) -> None:
        self.episodes = deque()
        self.size = 0

    def push(self, episode):
        """Add episode to the buffer.

        Args:
            episode (Episode) : class with
                transition_id, hidden_state, action, last_state, current_state, reward, done, info)
        """
        self.episodes.append(episode)
        self.size += 1

    # def save(self, buffer_save_path):
    #     np.save(buffer_save_path, np.array(self.episodes))

    def save_one(self, episode, savepath):
        import pathlib

        # print(savepath)
        pathlib.Path(savepath).mkdir(parents=True, exist_ok=True)
        vecobs = None
        if episode.vecobs:
            # print("Here")
            vecobs = np.array(episode.observations)
            print(episode.infos[0]["img"].shape)
            images = np.array([info["img"] for info in episode.infos])
        else:
            #print(len(episode.infos))
            # print(episode.infos.keys())
            images = np.array([info for info in episode.infos])

        actions = np.array(episode.actions)
        rewards = np.array(episode.rewards)
        terminals = np.array(episode.terminals)
        resets = np.zeros((episode.length), dtype=bool)
        print("Actions shape: {}".format(actions.shape))
        print("Rewards shape: {}".format(rewards.shape))
        print("Terminals shape: {}".format(terminals.shape))
        resets[0] = True
        print(f"{savepath}/episode-{self.size}")
        np.savez(f"{savepath}/episode-{self.size}", image=images, vecobs=vecobs,
                 action=actions, reward=rewards, terminal=terminals, reset=resets)
        self.size += 1

    def save(self, buffer_save_path):
        import pathlib
        pathlib.Path(buffer_save_path).mkdir(parents=True, exist_ok=True)

        for i, episode in enumerate(self.episodes):
            vecobs = None
            if episode.vecobs:
                # print("Here")
                vecobs = np.array(episode.observations)
                images = np.array([info["img"] for info in episode.infos])
            else:
                images = np.array(episode.observations)

            actions = np.array(episode.actions)
            rewards = np.array(episode.rewards)
            terminals = np.array(episode.terminals)
            resets = np.zeros((episode.length), dtype=bool)
            resets[0] = True
            print(f"{buffer_save_path}/episode-{i}")
            np.savez(f"{buffer_save_path}/episode-{i}", image=images, vecobs=vecobs,
                     action=actions, reward=rewards, terminal=terminals, reset=resets)

    def load(self, buffer_load_path, allow_pickle=True):
        self.episodes = np.load(buffer_load_path, allow_pickle=allow_pickle)
        self.size = len(self.episodes)

    def sample(self):
        pass
