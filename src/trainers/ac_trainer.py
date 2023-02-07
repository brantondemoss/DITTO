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
import wandb
from agent import Agent
from data.ac_dataset import ACDataset
from data.common import (ACSampler, MC_return, fix_obs, fix_obs_sb3,
                         lambda_return)
# from models.actor_critic import ActorCritic
from models.a2c import ActorCritic, Discriminator
from models.world_model import WorldModelRSSM
from pyvirtualdisplay import Display
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecFrameStack, VecMonitor,
                                              VecNormalize, VecVideoRecorder,
                                              is_vecenv_wrapped)
from torch.cuda.amp import autocast
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm, trange

#display = Display(visible=0, size=(400, 300))
#display.start()

class ACTrainer(object):
    def __init__(self, conf):
        self.conf = conf.ac_trainer_config
        self.do_checkpoint = self.conf.do_checkpoint
        if self.do_checkpoint:
            self.checkpoint_path = self.conf.checkpoint_path + wandb.run.name + "/"
            Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
            print('Got checkpoint path', self.checkpoint_path)
        self.sb3 = self.conf.use_sb3
        print("WEIGHTS:", self.conf.wm_path)
        print('self.sb3 =', self.sb3)

        self.lr = float(self.conf.lr)
        self.device = self.conf.train_device

        self.batch_size = self.conf.batch_size
        self.seq_length = self.conf.seq_length
        self.unrolls = self.seq_length

        self.env_name = self.conf.env_name
        self.env_id = self.conf.env_id

        # in_dim, out_dim
        self.obs_dim = 2048
        action_space ={"breakout": 4,
        "atari_pong": 6,
        "beamrider": 9,
        "mspacman": 9,
        "qbert": 6,
        "spaceInvaders": 6}
        self.action_dim = action_space[self.conf.env_name]
        print('got action dim', self.action_dim)
        self.policy = ActorCritic(
            obs_dim=self.obs_dim, action_dim=self.action_dim, hidden_dim=self.conf.hidden_dim, layers=self.conf.layers).to(self.device)
        self.policy_gail =  ActorCritic(
            obs_dim=self.obs_dim, action_dim=self.action_dim, hidden_dim=self.conf.hidden_dim, layers=self.conf.layers).to(self.device)
        self.policy_bc =  ActorCritic(
            obs_dim=self.obs_dim, action_dim=self.action_dim, hidden_dim=self.conf.hidden_dim, layers=self.conf.layers).to(self.device)
        # self.policy = self.load_policy()
        # self.student = ActorCritic(self.obs_dim//2, 4).to(self.device)
        self.discrim = Discriminator(obs_dim=self.obs_dim).to(self.device)
        self.discrim_buffer = []
        self.world_model = self.load_wm(conf.wm_config).to(self.device)
        # self.init_h = self.world_model.rssm_core.cell.init_h.detach().to(self.device)
        # self.init_z = self.world_model.rssm_core.cell.init_z.detach().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_gail = optim.Adam(self.policy_gail.parameters(), lr=self.lr)
        self.optimizer_bc = optim.Adam(self.policy_bc.parameters(), lr=self.lr)
        self.discrim_optimizer = optim.Adam(self.discrim.parameters(), lr=self.lr)
        self.discrim_loss = nn.BCEWithLogitsLoss()
        self.bc_loss = nn.CrossEntropyLoss()

        self.train_dataloader, self.val_dataloader = self.build_dataloaders(
            conf)
        self.steps = 0
        self.pbar = tqdm()

        self.gamma = self.conf.gamma
        self.lambda_gae = self.conf.lambda_gae
        self.eps = np.finfo(np.float32).eps.item()

        self.max_val_reward = 100
        self.max_val_reward_bc = 100
        self.max_val_reward_gail = 100

    def load_policy(self, policy_model_path="../checkpoints/breakout_old/policy-r339.625.pth"):
        policy_model = ActorCritic(
            obs_dim=self.obs_dim, action_dim=self.action_dim, hidden_dim=self.conf.hidden_dim, layers=self.conf.layers)
        policy_model.load_state_dict(torch.load(policy_model_path))
        policy_model.to(self.device)
        return policy_model

    def load_wm(self, wm_config):
        """load world model from checkpoint"""
        # this dict contains keys: steps, model_state_dict, optimizer_state_dict
        state_dict = torch.load(self.conf.wm_path)['model_state_dict']
        print('state_dict', self.conf.wm_path)
        wm = WorldModelRSSM(wm_config)
        wm.load_state_dict(state_dict)
        wm.to(self.device)
        wm.eval()
        wm.requires_grad_(False)

        return wm

    def val_train_split(self, dataset, episode_starts, split=0.9):
        print("Splitting data..")
        split_idx_ep = int(len(episode_starts)*split)
        split_idx = episode_starts[split_idx_ep]
        idxs = torch.arange(len(dataset))

        train_dataset = Subset(dataset, idxs[:split_idx])
        train_ep_starts = episode_starts[:split_idx_ep]

        val_dataset = Subset(dataset, idxs[split_idx:])
        val_ep_starts = episode_starts[split_idx_ep:] - \
            episode_starts[split_idx_ep]

        print("set lens:", len(train_dataset), len(val_dataset))
        return train_dataset, val_dataset, train_ep_starts, val_ep_starts

    def build_dataloader(self, dataset, ep_starts):
        batch_sampler = ACSampler(
            len(dataset), ep_starts, self.seq_length, self.batch_size)
        dataloader = DataLoader(dataset,
                                pin_memory=True,
                                batch_sampler=batch_sampler,
                                collate_fn=self.dataset.ac_collate)
        return dataloader

    def build_dataloaders(self, conf):
        # dataset_config = self.conf.dataset_config
        self.dataset = ACDataset(conf)
        episode_starts = np.where(self.dataset.data["resets"])[0]

        train_dataset, val_dataset, train_ep_starts, val_ep_starts = \
            self.val_train_split(self.dataset, episode_starts, split=0.9)

        train_dataloader = self.build_dataloader(
            train_dataset, train_ep_starts)
        val_dataloader = self.build_dataloader(val_dataset, val_ep_starts)

        return train_dataloader, val_dataloader

    @staticmethod
    def max_cos(input, target):
        n_i = torch.norm(input, dim=-1, keepdim=True)
        n_t = torch.norm(target, dim=-1, keepdim=True)
        norms = torch.cat((n_i, n_t), dim=-1)
        max_norm = torch.max(norms, dim=-1)[0]
        dot_prod = (input * target).sum(dim=-1)
        max_cos = dot_prod/torch.square(max_norm)

        return max_cos

    def get_action(self, latent, scalar=False, target=False, student=False, policy=None):
        if target:
            action_probs, state_values, target_values = \
                policy.forward_t(latent)
        else:
            action_probs, state_values = policy(latent)
        a_dist = Categorical(action_probs)
        mean_entropy = a_dist.entropy().mean()
        a_hat = a_dist.sample()
        a_onehot = torch.eye(18)[a_hat].to(self.device)
        log_prob = a_dist.log_prob(a_hat)

        action = (a_hat.item(), a_onehot) if scalar else a_onehot
        state_values = (
            state_values, target_values) if target else state_values

        return (action, log_prob, mean_entropy), state_values

    def wm_step(self, latent, a_onehot):
        with torch.no_grad():
            h, z = latent.split([1024, 1024], -1)
            with autocast(enabled=True):
                (h, z) = self.world_model.dream(a_onehot, (h, z))

            next_latent = torch.cat((h, z), dim=-1)
        return next_latent

    def unroll_policy(self, latents, resets, student=True, discrim=False):
        if discrim:
            policy = self.policy_gail
        else:
            policy = self.policy
        ac_buffer, rewards, actions, entropy, target_buffer = \
            [], [], [], [], []
        state = latents[0]
        assert not torch.any(resets[1:]).item(), print(resets.shape)
        for k in range(self.unrolls):
            stateplusgoal = torch.concat(
                (state, latents[-1]), dim=-1) if not student else state
            prev_state = state

            (a_onehot, log_prob, mean_entropy), (v_hat, v_t) = self.get_action(stateplusgoal,
                                                                               target=True, 
                                                                               student=student,
                                                                               policy=policy)

            state = self.wm_step(state, a_onehot)

            if k < self.unrolls-1:
                if discrim:
                    # Discriminator block
                    cat_state = torch.cat(
                        (prev_state[..., :1024], state[..., :1024]), dim=-1)
                    cat_latent = torch.cat(
                        (latents[k][..., :1024],latents[k+1][..., :1024]), dim=-1)
                    d_logits, d_probs = self.discrim(cat_state)
                    # print('d_logits.shape', d_logits.shape)
                    reward = d_probs.squeeze()
                    self.discrim_buffer.append((d_logits.squeeze(),
                        torch.zeros(state.shape[0], dtype=torch.float32, device=self.device))
                        )
                    t_logits, t_probs = self.discrim(cat_latent)
                    self.discrim_buffer.append((t_logits.squeeze(),
                        torch.ones(latents[k+1].shape[0], dtype=torch.float32, device=self.device))
                        )
                else:
                    reward = self.max_cos(
                        state[..., :1024], latents[k+1][..., :1024])
            else:
                reward = torch.zeros_like(rewards[-1]).to(self.device)
            ac_buffer.append([log_prob, v_hat.squeeze()])
            target_buffer.append(v_t.squeeze())
            rewards.append(reward)
            actions.append(a_onehot)
            entropy.append(mean_entropy)

        entropy = torch.mean(torch.stack(entropy))
        actions = torch.stack(actions)

        return ac_buffer, rewards, actions, entropy, target_buffer

    def BehaviorClone(self, latents, actions):
        # input shapes = (T, B, dim)

        # drop last action b/c we've rolled that action over and lost it
        latents = latents[:-1]
        TB_latents = torch.reshape(latents, (-1, 2048))

        # turn pre-actions into post-actions
        post_actions = torch.roll(actions, shifts=(-1, 0, 0), dims=(0, 1, 2))
        # as above, remove last action
        post_actions = post_actions[:-1]
        TB_actions = torch.reshape(
            post_actions, (-1, post_actions.shape[-1]))[..., :self.action_dim]

        logits, action_probs = self.policy_bc.forward_actor(TB_latents)
        a_dist = Categorical(action_probs)
        entropy = a_dist.entropy().mean()

        # Cross entropy loss
        BCLoss = self.bc_loss(logits, TB_actions).mean()

        return BCLoss, entropy

    @staticmethod
    def calculate_advantage(value_buffer, returns, norm=True):
        # values.shape == returns.shape == (T,B)
        values = torch.stack(value_buffer)

        advantage = returns - values
        if norm:
            advantage = (advantage - advantage.mean())/(advantage.std() + 1e-8)

        return advantage

    @staticmethod
    def calculate_losses(ac_buffer, returns, advantage):
        log_probs = torch.stack([log_prob for (log_prob, value) in ac_buffer])
        policy_loss = (-log_probs[:-1] * advantage.detach()[:-1]).mean()

        values = torch.stack([value for (log_prob, value) in ac_buffer])
        value_loss = 0.5*F.mse_loss(values[:-1], returns.detach()[:-1])

        return policy_loss, value_loss

    def train(self):
        epoch = tqdm(desc="epoch")
        while True:
            epoch.update(1)
            for batch in tqdm(self.train_dataloader, leave=False):
                # (T, B, 2048)
                latents, resets, actions = [x.to(self.device) for x in batch]

                # BC just for fun
                BCLoss, entropy_bc = self.BehaviorClone(latents, actions)

                # Main policy training
                ac_buffer, latent_rewards, ac_actions, entropy, target_buff = self.unroll_policy(
                    latents, resets, discrim=False)
                v_hats = [y for (x, y) in ac_buffer]

                unnorm_returns = lambda_return(latent_rewards, target_buff,
                                               lambda_=self.lambda_gae, gamma=self.gamma)

                advantage = self.calculate_advantage(
                    target_buff, unnorm_returns)

                policy_loss, value_loss = self.calculate_losses(
                    ac_buffer, unnorm_returns, advantage)

                accuracy = self.get_accuracy(actions.cpu(), ac_actions.cpu())

                # GAIL policy training
                ac_buffer_gail, latent_rewards_gail, ac_actions_gail, entropy_gail, target_buff_gail = self.unroll_policy(
                    latents, resets, discrim=True)
                v_hats_gail = [y for (x, y) in ac_buffer_gail]

                unnorm_returns_gail = lambda_return(latent_rewards_gail, target_buff_gail,
                                               lambda_=self.lambda_gae, gamma=self.gamma)

                advantage_gail = self.calculate_advantage(
                    target_buff_gail, unnorm_returns_gail)

                policy_loss_gail, value_loss_gail = self.calculate_losses(
                    ac_buffer_gail, unnorm_returns_gail, advantage_gail)

                accuracy_gail = self.get_accuracy(actions.cpu(), ac_actions_gail.cpu())

                
                discrim_logits = torch.stack(
                    [logits for (logits, targets) in self.discrim_buffer])
                discrim_targets = torch.stack(
                    [targets for (logits, targets) in self.discrim_buffer])
                # print('discrim_logits:', discrim_logits.shape, discrim_logits[0], discrim_logits[-1])
                # print('discrim_targets:', discrim_targets.shape, discrim_targets[0], discrim_targets[-1])
                discrim_loss = self.discrim_loss(
                    discrim_logits, discrim_targets)

                self.discrim_optimizer.zero_grad()
                discrim_loss.backward()
                self.discrim_optimizer.step()
                del self.discrim_buffer[:]

                loss = policy_loss + 0.01*value_loss - 0.05*entropy
                loss_bc = BCLoss - 0.1*entropy_bc
                loss_gail = policy_loss_gail + 0.01*value_loss_gail - 0.05*entropy_gail


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.optimizer_bc.zero_grad()
                loss_bc.backward()
                self.optimizer_bc.step()

                self.optimizer_gail.zero_grad()
                loss_gail.backward()
                self.optimizer_gail.step()

                with torch.no_grad():
                    metrics = {
                        'train/policy_loss': policy_loss.cpu(),
                        'train/value_loss': value_loss.cpu(),
                        'train/unnorm_return': unnorm_returns.mean().cpu(),
                        'train/reward': torch.stack(latent_rewards).mean().cpu(),
                        'train/acc': accuracy,
                        'train/entropy': entropy.cpu(),
                        # 'train/advantage': advantage.mean().cpu(),
                        'train/bc_loss': BCLoss.cpu(),
                        'train/discrim_loss': discrim_loss.cpu(),###
                        'train/policy_loss_gail': policy_loss_gail.cpu(),
                        'train/value_loss_gail': value_loss_gail.cpu(),
                        'train/unnorm_return_gail': unnorm_returns_gail.mean().cpu(),
                        'train/reward_gail': torch.stack(latent_rewards_gail).mean().cpu(),
                        'train/acc_gail': accuracy_gail,
                        'train/entropy_gail': entropy_gail.cpu(),
                    }
                self.log_stats(metrics)

    @staticmethod
    def get_accuracy(actions, ac_actions):
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

    def log_stats(self, metrics_dict):
        if self.steps % 200 == 0:
            val_metrics = self.validate()
            wandb.log(val_metrics, step=self.steps)
        wandb.log(metrics_dict, step=self.steps)
        self.steps += 1

    def validate(self):
        #TODO: fix this to a config value for target weight updte rate
        self.policy.update_critic_target()
        policy_losses = 0
        value_losses = 0
        run_accuracy = 0
        run_entropy = 0
        self.policy.eval()
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, leave=False):
                latents, resets, actions = [x.to(self.device) for x in batch]

                ac_buffer, latent_rewards, ac_actions, entropy, target_buff = self.unroll_policy(
                    latents, resets, discrim=False)
                accuracy = self.get_accuracy(actions.cpu(), ac_actions.cpu())

                unnorm_returns = MC_return(latent_rewards, target_buff[-1],
                                           norm=False, gamma=self.gamma, eps=self.eps)

                advantage = self.calculate_advantage(
                    target_buff, unnorm_returns)

                policy_loss, value_loss = self.calculate_losses(
                    ac_buffer, unnorm_returns, advantage)

                policy_losses += policy_loss
                value_losses += value_loss
                run_accuracy += accuracy
                run_entropy += entropy
            if self.sb3:
                # replay_buffer = self.parallel_test_agent(n_games=2, n_env=2)
                # mean_reward, max_reward, min_reward, stdev_reward, rewards = replay_buffer.calc_stats()
                # actions = replay_buffer.get_actions()
                mean_reward, max_reward, min_reward, stdev_reward, rewards, actions = self.parallel_test_agent(
                    n_games=8, n_envs=4, policy=self.policy)
                mean_reward_bc, max_reward_bc, min_reward_bc, stdev_reward_bc, rewards_bc, actions_bc = self.parallel_test_agent(
                    n_games=8, n_envs=4, policy=self.policy_bc)
                mean_reward_gail, max_reward_gail, min_reward_gail, stdev_reward_gail, rewards_gail, actions_gail = self.parallel_test_agent(
                    n_games=8, n_envs=4, policy=self.policy_gail)

            else:
                mean_reward, max_reward, min_reward, stdev_reward, rewards, actions = self.test_agent(
                    n_games=2)

        if self.do_checkpoint and mean_reward > self.max_val_reward:
            self.max_val_reward = mean_reward
            model_name = self.checkpoint_path + f"policy-r{mean_reward}.pth"
            # print('Saving model', model_name)
            torch.save(self.policy.state_dict(), model_name)
        if self.do_checkpoint and mean_reward_bc > self.max_val_reward_bc:
            self.max_val_reward_bc = mean_reward_bc
            model_name = self.checkpoint_path + f"policy-bc-r{mean_reward_bc}.pth"
            # print('Saving model', model_name)
            torch.save(self.policy_bc.state_dict(), model_name)
        if self.do_checkpoint and mean_reward_gail > self.max_val_reward_gail:
            self.max_val_reward_gail = mean_reward_gail
            model_name = self.checkpoint_path + f"policy-gail-r{mean_reward_gail}.pth"
            # print('Saving model', model_name)
            torch.save(self.policy_gail.state_dict(), model_name)

        self.policy.train()
        lvdl = len(self.val_dataloader)
        val_metrics = {"val/policy_loss": policy_losses/lvdl,
                       "val/value_loss": value_losses/lvdl,
                       "val/acc": run_accuracy/lvdl,
                       "val/entropy": run_entropy/lvdl,
                       "val/mean_ep_return": mean_reward,
                       "val/stdev_ep_return": stdev_reward,
                       "val/max_ep_return": max_reward,
                       "val/min_ep_return": min_reward,
                       "val/actions": wandb.Histogram(actions),
                       "val/returns": wandb.Histogram(rewards),
                       "val/mean_ep_return_gail": mean_reward_gail,
                       "val/stdev_ep_return_gail": stdev_reward_gail,
                       "val/max_ep_return_gail": max_reward_gail,
                       "val/min_ep_return_gail": min_reward_gail,
                       "val/actions_gail": wandb.Histogram(actions_gail),
                       "val/returns_gail": wandb.Histogram(rewards_gail),
                       "val/mean_ep_return_bc": mean_reward_bc,
                       "val/stdev_ep_return_bc": stdev_reward_bc,
                       "val/max_ep_return_bc": max_reward_bc,
                       "val/min_ep_return_bc": min_reward_bc,
                       "val/actions_bc": wandb.Histogram(actions_bc),
                       "val/returns_bc": wandb.Histogram(rewards_bc)}
        return val_metrics

    def make_env(self, n_envs=1, original_fn=True, seed=None):

        if self.env_name == "breakout" or True:
            if original_fn:
                print('original fn true')
                env = gym.make(self.env_id)
                env = AtariWrapper(env, clip_reward=False, screen_size=64)
            else:
                env = make_atari_env(self.env_id, n_envs=n_envs,
                                     wrapper_kwargs={
                                         "clip_reward": False, "screen_size": 64},
                                     vec_env_cls=SubprocVecEnv, seed=seed)
                env = WmEnv(env, self.world_model, n_envs, self.device, 18, 
                            pixel_mean = self.conf.pixel_mean, pixel_std = self.conf.pixel_std)
        elif self.env_name == "bipedalwalker":
            print('REALLY FUCKED')
            env = make_vec_env(self.env_id, n_envs=n_envs)
            env = VecFrameStack(env, n_stack=1)
            path = f"../third_party/rl-baselines3-zoo/rl-trained-agents/ppo/{self.env_id}_1/{self.env_id}/vecnormalize.pkl"
            env = VecNormalize.load(path, env)
            env = WmEnv(env, self.world_model, n_envs,
                        self.device, self.conf.action_dim, use_render=True)
        return env

    def get_action2(self, latent, scalar=False, target=False, student=False, epsilon=0.0, policy=None):
        action_probs, state_values = policy(latent)
        a_dist = Categorical(action_probs)
        a_hat = a_dist.sample()

        # if epsilon > np.random.rand: 
            # dist = Unifrom over action space
            # action = dist.sample()

        if len(a_hat.shape) == 0:
            a_hat = a_hat.unsqueeze(0)

        return a_hat

    @torch.inference_mode()
    def parallel_test_agent(self, n_envs=1, n_games=10, print_reward=False, policy=None):
        rewards = []
        actions = []

        pbar = tqdm(total=n_games, leave=False, desc="test_agent2")
        #with open(os.devnull, 'w') as f:
        #    with redirect_stdout(f):
        env = self.make_env(n_envs=n_envs, original_fn=False)

        init_obs = env.reset()
        features = env.prep_obs(init_obs)

        is_monitor_wrapped = is_vecenv_wrapped(
            env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
        n_complete = 0
        while n_complete < n_games:
            s_action = self.get_action2(features, policy=policy)
            actions.extend(s_action.tolist())
            features, r, done, info = env.step(s_action)

            for i in range(n_envs):
                if done[i] and is_monitor_wrapped and "episode" in info[i]:
                    n_complete += 1
                    if print_reward:
                        print(info[i]["episode"])
                    rewards.append(info[i]["episode"]["r"])
                    pbar.update(1)
        return np.mean(rewards), np.max(rewards), np.min(rewards), np.std(rewards), rewards, actions


class WmEnv(gym.Wrapper):

    def __init__(self, env, wm, n_env, device, action_dim, use_render=False, pixel_mean = 33, pixel_std = 55):
        self.wm = wm
        self.action_dim = action_dim
        self.n_env = n_env
        self.in_states = self.wm.init_state(n_env)
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.device = device
        self.observations = []
        self.dones = 0
        self.use_render = use_render
        super().__init__(env)

    def wm_step(self, latent, a_onehot):
        with torch.no_grad():
            h, z = latent.split([1024, 1024], -1)
            with autocast(enabled=True):
                (h, z) = self.world_model.dream(a_onehot, (h, z))

            next_latent = torch.cat((h, z), dim=-1)
        return next_latent

    def reshape_img(self, img, new_hw=64):
        img = img.astype(np.uint8)
        img = np.array(cv2.resize(img, dsize=(
            new_hw, new_hw), interpolation=cv2.INTER_AREA))
        img = np.expand_dims(img, 2)
        return img

    def prep_obs(self, img, action=None, done=None, info=None):
        img = img.astype(np.float32)
        img = (img - self.pixel_mean) / self.pixel_std
        # print("HERE", img.shape)
        img = np.transpose(img, (0, 3, 1, 2))
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)
        img = img.to(self.device)

        reset = torch.zeros((1, img.shape[1]), dtype=torch.bool)

        if done is None:
            action = torch.zeros(
                (1, img.shape[1], self.action_dim)).to(self.device)
            reset = torch.ones(
                (1, img.shape[1]), dtype=torch.bool)
        else:
            d = [True if info[i]["lives"] == 0
                 else False for i in range(len(done))]
            reset[0, np.where(d)[0]] = 1
            if len(action.shape) == 1:
                action = torch.eye(self.action_dim)[
                    action].unsqueeze(0).to(self.device)
            idxs = np.where(done)[0]
            if len(idxs) > 0:
                action[:, idxs] = torch.zeros(
                    (len(idxs), self.action_dim)).to(self.device)

        obs = {"obs": img, "action": action, "reset": reset.to(self.device)}
        features, out_states = self.wm(obs, self.in_states)
        self.in_states = out_states
        features = features.squeeze()
        return features

    def step(self, action):
        # print(action.item())
        #print(action)
        obs, reward, done, info = self.env.step(action)
        if self.use_render:
            obs = self.env.render(mode="rgb_array")
            obs = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140])
            obs = self.reshape_img(obs)
            obs = np.expand_dims(obs, 0)
        latent = self.prep_obs(obs, action, done, info)
        if isinstance(info, dict):
            info["obs"] = obs
        else:
            for i in range(len(info)):
                info[i]["obs"] = obs[i]

        return latent, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.use_render:
            obs = self.env.render(mode="rgb_array")
            # print("HERE:", obs.shape)
            obs = self.reshape_img(
                np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140]))
            obs = np.expand_dims(obs, 0)
        # latent = self.prep_obs(obs)
        # exit()
        return obs
