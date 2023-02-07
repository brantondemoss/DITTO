import multiprocessing as mp
import os
from multiprocessing import Manager, Pool, Process

import numpy as np
from IPython import display
from IPython.display import Video
from PIL import Image
from pyvirtualdisplay import Display
from stable_baselines3 import DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, VecFrameStack,
                                              VecMonitor, VecNormalize,
                                              is_vecenv_wrapped)
from tqdm.notebook import tqdm, trange

from agent import Agent
from envs.cartpole import EnvFactory

# os.environ['DISPLAY'] = ':1'


display = Display(visible=0, size=(400, 300))
display.start()


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


n_cores = mp.cpu_count()


def build_expert(env_name, n_env, agent_type="ppo"):
    # env = env_factory.create_env(env_name,n_env=n_env)
    # env_id = env_factory.envs_conf[env_name]["env_id"]
    env_id = "BipedalWalker-v3"
    # env = make_vec_env(env_id, n_envs=n_env)
    # env = VecFrameStack(env, n_stack=1)
    env = build_normalized_vec()
    expert_path = f"../third_party/rl-baselines3-zoo/rl-trained-agents/{agent_type}/{env_id}_1/{env_id}.zip"
    # env = gym.make('breakout-mixed-v0')
    expert = Agent.from_sb3_file(expert_path, agent_type, env)
    return expert


def build_env(env_id="BreakoutNoFrameskip-v4", n_envs=4, n_stack=4):
    env = make_atari_env(env_id, n_envs=n_envs,
                         wrapper_kwargs={"clip_reward": False})
    env = VecFrameStack(env, n_stack=n_stack)
    return env


def build_normalized_vec(env_id="BipedalWalker-v3", n_stack=1, n_env=1):
    env = make_vec_env(env_id, n_envs=n_env)
    env = VecFrameStack(env, n_stack=n_stack)
    path = f"../third_party/rl-baselines3-zoo/rl-trained-agents/ppo/{env_id}_1/{env_id}/vecnormalize.pkl"
    env = VecNormalize.load(path, env)
    return env


def generate_data(env_name, agent_type, max_steps, num_episodes, outfile, n_env):
    out_path = "../data/bipedalwalker/episodes"
    expert = build_expert(env_name, n_env, agent_type)

    expert.run_n_episodes(num_episodes, out_path, max_steps)
    # expert.save_buffer(out_path)
    return out_path


if __name__ == "__main__":
    env_factory = EnvFactory("config/config.yaml")
    env_names = ["cartpole", "bipedalwalker",
                 "bipedalwalkerhardcore", "atari_pong", "atari_breakout"]
    env_name = env_names[1]
    agent_type = "ppo"
    max_steps = None

    n_env = 1  # n_cores
    num_episodes = 1000
    outfile = f"{env_name}_{agent_type}_{max_steps}-steps_{num_episodes}-eps.npy"

    data_file = generate_data(env_name, agent_type,
                              max_steps, num_episodes, outfile, n_env)
    data_file
