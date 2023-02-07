import gym
import gym.spaces
import numpy as np
import yaml
from gym.envs.classic_control import CartPoleEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, VecFrameStack,
                                              VecNormalize)

# class AbstractEnv(gym.Wrapper):
#     def __init__(self, config):
#         env_id = config["env_id"]
#         n_stack = config["n_stack"]
#         n_env = config["n_env"]
#         env = make_vec_env(env_id, n_envs=n_env, seed=0)
#         env = VecFrameStack(env, n_stack=n_stack)


class CartPoleEnv(gym.Wrapper):
    def __init__(self, config):
        env_id = config["env_id"]
        self.n_stack = config["n_stack"]
        self.n_env = config["n_env"]
        env = make_vec_env(env_id, n_envs=self.n_env, seed=0)
        env = VecFrameStack(env, n_stack=self.n_stack)
        # Automatically normalize the input features and reward
        # env = VecNormalize(env, norm_obs=True, norm_reward=True,clip_obs=10.)

        super().__init__(env)


class BipedalWalkerEnv(gym.Wrapper):
    def __init__(self, config):
        env_id = "BipedalWalker-v3"
        self.n_stack = 1
        self.n_env = 1
        env = make_vec_env(env_id, n_envs=self.n_env)
        env = VecFrameStack(env, n_stack=self.n_stack)
        path = f"../third_party/rl-baselines3-zoo/rl-trained-agents/ppo/{env_id}_1/{env_id}/vecnormalize.pkl"
        env = VecNormalize.load(path, env)
        super().__init__(env)


class PongEnv(gym.Wrapper):
    # last obs = obs[...,-1]
    #wrapper_kwargs = {"frame_skip":10}
    def __init__(self, config):
        env_id = config["env_id"]
        self.n_stack = config["n_stack"]
        self.n_env = config["n_env"]
        env = make_atari_env(env_id, n_envs=self.n_env)
        env = VecFrameStack(env, n_stack=self.n_stack)
        super().__init__(env)


class BreakoutEnv(gym.Wrapper):
    # last obs = obs[...,-1]
    #wrapper_kwargs = {"frame_skip":10}
    def __init__(self, config):
        env_id = config["env_id"]
        self.n_stack = config["n_stack"]
        self.n_env = config["n_env"]
        env = make_atari_env(env_id, n_envs=self.n_env)
        env = VecFrameStack(env, n_stack=self.n_stack)
        super().__init__(env)


class EnvFactory:
    def __init__(self, config_file="config.yaml"):
        with open(config_file, "r") as f:
            self.envs_conf = yaml.safe_load(f)

    def create_env(self, env_name, **config_args):
        # base_conf = self.envs_conf["base"]
        # env_conf = self.envs_conf[env_name]
        # env_conf = {**base_conf, **env_conf, **config_args}
        env_conf = {}
        if env_name == "cartpole":
            return CartPoleEnv(env_conf)
        elif env_name == "bipedalwalker" or env_name == "bipedalwalkerhardcore":
            return BipedalWalkerEnv(env_conf)
        elif env_name == "atari_pong":
            return PongEnv(env_conf)
        elif env_name == "atari_breakout":
            return BreakoutEnv(env_conf)
        else:
            print(f"Env {env_name} not implemented")
