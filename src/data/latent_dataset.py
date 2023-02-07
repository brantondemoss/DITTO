import glob
import math

import cv2
import d4rl_atari
import gym
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class LatentDataset(Dataset):
    def __init__(self, data_dir="None"):
        self.data = self.load_data(data_dir)
        self.data_keys = ["actions", "features"]

    def build_data(self):
        pass

    def load_data(self, data_dir):
        data = {}
        data["features"] = np.load(data_dir+"expertv2-features.npy")
        # data_dict = np.load(data_dir + "breakout-expert-v2.npz")
        env = gym.make("breakout-expert-v2")
        data_dict = env.get_dataset()
        data["actions"] = np.eye(18)[data_dict["actions"]]
        return data

    def __len__(self):
        return len(self.data["actions"])

    def __getitem__(self, idx):
        action, feature = [self.data[key][idx] for key in self.data_keys]
        return action, feature
