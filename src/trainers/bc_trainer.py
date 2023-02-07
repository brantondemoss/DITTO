import os
import sys
from contextlib import redirect_stdout
from multiprocessing import Pool
from pathlib import Path
from time import time

import cv2
import d4rl_atari
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pyvirtualdisplay import Display
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecFrameStack, VecMonitor,
                                              VecNormalize, VecVideoRecorder,
                                              is_vecenv_wrapped)
from torch.cuda.amp import GradScaler, autocast
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm, trange

import wandb
from agent import Agent
from common.mlp import MLP
from data.ac_dataset import ACDataset
from data.common import (ACSampler, MC_return, fix_actions, fix_obs,
                         fix_obs_sb3, lambda_return)
from data.image_dataset import ImageDataset
from data.latent_dataset import LatentDataset
# from models.actor_critic import ActorCritic
from models.a2c import ActorCritic, Discriminator
from models.encoders import CnnEncoder
from models.world_model import WorldModelRSSM

# from ac_trainer import get_accuracy


class BCTrainer(object):
    def __init__(self, conf, encoder_type="cnn"):

        self.lr = float(conf["ac_trainer_config"]["lr"])
        # print(conf)
        # print(conf["ac_trainer_config"]["lr"])
        # print(conf["env_name"])
        # print(conf["action_dim"])
        # print(conf["dataset_path"])
        action_space = {"breakout": 4,
                        "atari_pong": 6,
                        "beamrider": 9,
                        "mspacman": 9,
                        "qbert": 6,
                        "spaceInvaders": 6}
        self.dataset_path = conf["dataset_config"]["dataset_path"]
        self.pixel_mean = conf["ac_trainer_config"]["pixel_mean"]
        self.pixel_std = conf["ac_trainer_config"]["pixel_std"]
        self.env_id = conf["ac_trainer_config"]["env_id"]
        self.out_dim = action_space[conf["ac_trainer_config"]["env_name"]]
        self.episode_limit = conf["ac_dataset_config"]["episode_limit"]
        self.batch_size = 8
        self.n_games = conf["ac_trainer_config"]["validation"]["n_games"]
        print(self.out_dim, "HERE")

        """ need to add these 
        self.out_dim (action_dim) - also pass to ImageDataset
        self.file_path (line 85, need path to npz files)
        """
        # self.env_id = "BeamRiderNoFrameskip-v4"
        # self.out_dim = 9
        # self.file_path = "/data/beamrider/ppo/episodes/episode-9.npz"
        print("LR:", self.lr)
        self.device = "cuda"
        # in_dim, out_dim, hidden_dim, hidden_layers
        in_dim = 2048 if encoder_type == "mlp" else 1024
        self.model = nn.Sequential(CnnEncoder(in_channels=1), nn.Flatten(), MLP(in_dim=in_dim, out_dim=self.out_dim,
                                                                                hidden_dim=2048, hidden_layers=3),
                                   ).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_dataloader = self.build_dataloader()
        self.steps = 0
        self.pbar = tqdm()

    def build_dataloader(self):
        dataset = ImageDataset(self.dataset_path, self.out_dim, self.pixel_mean,
                               self.pixel_std, self.episode_limit)  # get image dataset
        dataloader = DataLoader(dataset,
                                pin_memory=True,
                                shuffle=True,
                                batch_size=self.batch_size)
        return dataloader

    def train(self):
        while True:
            for batch in tqdm(self.train_dataloader, leave=False):
                action, image = [x.to(self.device) for x in batch]
                self.optimizer.zero_grad()

                # print(image.shape, image.dtype, "here")

                a_hat = self.model(image)

                # print(type(a_hat), type(action))
                # print(a_hat.dtype, action.dtype)
                # print(a_hat.shape, action.shape)
                loss = self.loss_fn(a_hat, action).mean()

                loss.backward()
                self.optimizer.step()

                self.log_stats(loss.item(), a_hat, action)

    def validate(self):
        mean_rwd, max_rwd, min_rwd, std_rwd, rwds, actions = self.parallel_test_agent(
            n_games=self.n_games)
        val_metrics = {"val/actions": wandb.Histogram(actions),
                       "val/returns": wandb.Histogram(rwds),
                       "val/mean_ep_return": mean_rwd,
                       "val/stdev_ep_return": std_rwd,
                       "val/max_ep_return": max_rwd,
                       "val/min_ep_return": min_rwd
                       }
        return val_metrics

    def get_action2(self, img, scalar=False, target=False, student=False, epsilon=0.0, policy=None):
        logits = self.model(img)

        action_probs = F.softmax(logits, dim=-1)
        a_dist = Categorical(action_probs)
        a_hat = a_dist.sample()

        if len(a_hat.shape) == 0:
            a_hat = a_hat.unsqueeze(0)

        return a_hat

    def make_env(self, n_envs=1, original_fn=True, seed=None):
        env = make_atari_env(self.env_id, n_envs=n_envs,
                             wrapper_kwargs={
                                 "clip_reward": False, "screen_size": 64},
                             vec_env_cls=SubprocVecEnv, seed=seed)

        return env

    @torch.inference_mode()
    def parallel_test_agent(self, n_envs=1, n_games=10, print_reward=False, policy=None):
        rewards = []
        actions = []
        # self.pixel_mean = 33
        # self.pixel_std = 55

        def prep_obs(img):
            img = img.astype(np.float32)
            img = (img - self.pixel_mean) / self.pixel_std
            img = np.transpose(img, (0, 3, 1, 2))
            # features = torch.tensor([fix_obs(x) for x in init_obs])
            img = torch.from_numpy(img)
            img = img.to(self.device)
            return img

        pbar = tqdm(total=n_games, leave=False, desc="test_agent2")

        env = self.make_env(n_envs=n_envs, original_fn=False)

        img = env.reset()
        # print(img.shape, img.dtype)
        img = prep_obs(img)
        # print(img.shape, img.dtype, "here")

        is_monitor_wrapped = is_vecenv_wrapped(
            env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
        n_complete = 0
        while n_complete < n_games:
            s_action = self.get_action2(img, policy=policy)
            actions.extend(s_action.tolist())
            img, r, done, info = env.step(s_action)
            img = prep_obs(img)

            for i in range(n_envs):
                if done[i] and is_monitor_wrapped and "episode" in info[i]:
                    n_complete += 1
                    if print_reward:
                        print(info[i]["episode"])
                    rewards.append(info[i]["episode"]["r"])
                    pbar.update(1)
        return np.mean(rewards), np.max(rewards), np.min(rewards), np.std(rewards), rewards, actions

    @staticmethod
    def get_accuracy(actions, ac_actions):
        # print(actions.shape, ac_actions.shape)
        # print(actions.dtype, ac_actions.dtype)

        assert actions.shape == ac_actions.shape, print(
            actions.shape, ac_actions.shape)
        # T, B, action_dim
        B = actions.shape[1]
        accuracy_per_step = []
        for i in range(len(actions)):
            matches = len(np.where((actions[i] == ac_actions[i]).all(1))[0])
            acc = matches/B
            accuracy_per_step.append(acc)
        return accuracy_per_step[0]

    def log_stats(self, loss, a_hat_logits, action):
        if self.steps % 25 == 0:
            self.pbar.update(25)
        if self.steps % 200 == 0:
            val_metrics = self.validate()
            wandb.log(val_metrics, step=self.steps)
        action = torch.unsqueeze(action, 0).cpu()
        a_hat_dist = Categorical(F.softmax(a_hat_logits, dim=-1))
        a_hat = torch.Tensor(np.eye(self.out_dim)[a_hat_dist.sample().cpu()])
        a_hat = torch.reshape(a_hat, action.shape)
        entropy = a_hat_dist.entropy().mean()
        # print(action,a_hat)
        wandb.log({"train/entropy": entropy.cpu(), "train/loss": loss,
                  "train/acc": self.get_accuracy(a_hat, action)}, step=self.steps)
        self.steps += 1
