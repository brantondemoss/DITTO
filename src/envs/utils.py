from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecFrameStack, VecMonitor,
                                              VecNormalize, VecVideoRecorder,
                                              is_vecenv_wrapped)
import gym
import numpy as np
import torch

def make_wm_env(env_name, env_id, wm, conf, device="cpu", n_envs=1, original_fn=True, seed=None):
    atari_envs = ["breakout", "qbert", "spaceInvaders", "mspacman", "beamrider"]

    if env_name in atari_envs:
        if original_fn:
            print('original fn true')
            env = gym.make(env_id)
            env = AtariWrapper(env, clip_reward=False, screen_size=64)
        else:
            env = make_atari_env(env_id, n_envs=n_envs,
                                    wrapper_kwargs={
                                        "clip_reward": False, "screen_size": 64},
                                    vec_env_cls=SubprocVecEnv, seed=seed)
            env = WmEnv(env, wm, n_envs, device, 18, 
                        pixel_mean = conf.pixel_mean, pixel_std = conf.pixel_std)
    elif env_name == "bipedalwalker":
        print('REALLY FUCKED')
        env = make_vec_env(env_id, n_envs=n_envs)
        env = VecFrameStack(env, n_stack=1)
        path = f"../third_party/rl-baselines3-zoo/rl-trained-agents/ppo/{env_id}_1/{env_id}/vecnormalize.pkl"
        env = VecNormalize.load(path, env)
        env = WmEnv(env, wm, n_envs,
                    device, conf.action_dim, use_render=True)
    else:
        env = make_vec_env(env_id, n_envs=n_envs)
        env = VecFrameStack(env, n_stack=1)
        env = WmEnv(env, wm, n_envs,
                    device, conf.action_dim, use_render=True)
    return env


class WmEnv(gym.Wrapper):

    def __init__(self, env, wm, n_env, device, action_dim, use_render=False, pixel_mean=33, pixel_std=55):
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
