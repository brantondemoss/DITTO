import cv2
import gym
import gym.envs
import numpy as np
import torch
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, VecFrameStack,
                                              VecMonitor, VecVideoRecorder,
                                              is_vecenv_wrapped)
from tqdm.auto import tqdm, trange

from replay_buffer import Episode, ReplayBuffer

ALGOS = {"a2c": A2C, "ddpg": DDPG, "dqn": DQN,
         "ppo": PPO, "sac": SAC, "td3": TD3}

custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}


class Agent:
    def __init__(self, policy_model, env=None):
        """
        Args:
            model : policy model
            env : Gym environment

        """
        self.replay_buffer = ReplayBuffer()
        self.policy_model = policy_model
        self.deterministic = False  # whether to choose action with max prob or sample

        self.env = self.policy_model.get_env() if env is None else env

        self.state = self.env.reset()

    @classmethod
    def from_sb3_file(cls, policy_model_path, algo_name, env, device="cpu", **kwargs):
        """Instantiates a stablebaselines3 expert from file

        Args:
            policy_model_path (str): filepath to expert policy (using stablebaselines3)
            algo_name (str) : name of policy (in ALGOS macro)
            env (gym.Env) : a gym environment
            device (Optional[str]) : compute device, defaults to CPU
            kwargs (dict) : additonal arguments to send to stablebaselines policy loader
        Returns:
            agent (Agent) : instance of Agent class
        """
        kwargs = {"seed": 0}
        policy_model = ALGOS[algo_name].load(
            policy_model_path,
            env=env,
            custom_objects=custom_objects,
            device=device,
            **kwargs
        )
        agent = cls(policy_model)
        return agent

    @classmethod
    def from_state_dict(cls, policy_model_path, policy_class, env, device="cuda:0"):
        policy_model = policy_class()
        policy_model.load_state_dict(torch.load(policy_model_path))
        policy_model.eval()
        policy_model.to(device)

        agent = cls(policy_model, env)
        return agent

    @staticmethod
    def record_video(
        env_id, policy_model, video_length=500, prefix="", video_folder="videos/"
    ):
        """Unroll under agent's current policy

        Args:
            env_id (str) :
            policy_model (RL model) :
            video_length (int) :
            prefix (str) :
            video_folder (str) :
        """
        eval_env = DummyVecEnv([lambda: gym.make(env_id)])

        eval_env = VecVideoRecorder(
            eval_env,
            video_folder=video_folder,
            record_video_trigger=lambda step: step == 0,
            video_length=video_length,
            name_prefix=prefix,
        )

        obs = eval_env.reset()
        for _ in range(video_length):
            action, _ = policy_model.predict(obs)
            obs, _, _, _ = eval_env.step(action)

        eval_env.close()

    def get_action(self, obs, episode_start=None, hidden_state=None):
        action, hidden_state = self.policy_model.predict(
            obs, state=hidden_state, deterministic=self.deterministic
        )
        return action, hidden_state

    def take_action(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def action_step(self, obs):
        action, hidden_state = self.get_action(obs)
        obs, reward, done, info = self.take_action(action)
        return obs

    @staticmethod
    def add_transition(env, action, state, reward, done, episode: Episode, info):
        if env.n_stack > 1:
            state = state[-1]
        else:
            state = state
            # print(state.shape, "ajajaj")
            # state = torch.transpose()

        episode.push(state, reward, action, done, info)

    def run_n_episodes(self, n_episodes, savepath=None, max_steps_per_episode=None):
        n_envs = self.env.num_envs

        total_eps = 0
        is_monitor_wrapped = is_vecenv_wrapped(
            self.env, VecMonitor) or self.env.env_is_wrapped(Monitor)[0]

        # this has dims (n_envs, n_stack, *obs_shape)
        last_states = self.env.reset()
        # print(last_states.shape, "Hereeeeeee")
        hidden_states = None

        episode_step_counts = np.zeros(n_envs, dtype="int")
        episode_starts = np.ones((n_envs), dtype=bool)
        info = {"img": self.env.render(mode="rgb_array")}

        # inits n_env episodes with starting states. -1 to grab last frame (default 4 frames per obs in Atari)
        episodes = [Episode(last_states[i, -1], info=info) for i in range(n_envs)]

        pbar = tqdm(total=n_episodes)

        while True:

            actions, hidden_states = self.get_action(
                last_states, episode_start=episode_starts, hidden_state=hidden_states)
            # if len(actions.shape) == 0:
            #     actions = actions.unsqueeze(0)
            current_states, rewards, dones, infos = self.take_action(actions)
            info = {"img": self.env.render(mode="rgb_array")}
            for i in range(n_envs):
                if total_eps >= n_episodes:
                    return

                episode_starts[i] = dones[i]
                hit_max_steps = max_steps_per_episode and max_steps_per_episode == episode_step_counts[
                    i]

                if dones[i] or hit_max_steps:
                    finished_ep = is_monitor_wrapped and "episode" in infos[i].keys(  # this is atari specific
                    )
                    if finished_ep or hit_max_steps:
                        total_eps += 1
                        pbar.update(1)
                        if savepath:
                            print("Saving..")
                            self.replay_buffer.save_one(episodes[i], savepath)
                        else:
                            # you need to run Agent.replay_buffer.save(path) manually to save all episodes in this case
                            print("Saving to buffer...")
                            self.replay_buffer.push(episodes[i])
                        episodes[i] = Episode(current_states[i, -1], info=info)
                        episode_step_counts[i] = 0
                else:
                    # print("here:", current_states.shape)

                    self.add_transition(
                        self.env, actions[i], current_states[i], rewards[i], dones[i], episodes[i], info)
                    last_states[i] = current_states[i]
                    episode_step_counts[i] += 1
