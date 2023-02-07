import glob
import os
import sys

import cv2
import d4rl_atari
import gym
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class D4RLDataset(Dataset):
    def __init__(self, dataset_config):
        self.action_type = dataset_config.action_type
        self.seq_length = dataset_config.seq_length
        self.device = dataset_config.load_device
        self.num_transitions = 0
        self.data_keys = dataset_config.data_keys

        self.pixel_mean = dataset_config.pixel_mean
        self.pixel_std = dataset_config.pixel_std
        # self.pixel_mean = 33
        # self.pixel_std = 55

        if "dataset_path" in dataset_config:
            self.data = self.load_files(dataset_config.dataset_path)
        else:
            self.data = self.load_d4rl_data(dataset_config.dataset_names)

        print("total transitions:", self.num_transitions)
        print("num batch elements:", self.num_transitions//self.seq_length)

    def fix_obs(self, img, new_hw=64, resize=True, sb3=False):
        """normalizes image"""
        img = np.transpose(img, (1, 2, 0))
        if resize:
            img = np.array(cv2.resize(img, dsize=(
                new_hw, new_hw), interpolation=cv2.INTER_AREA))
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)
        img = (img - self.pixel_mean) / self.pixel_std

        return img

    def fix_terminals(self, terminals):
        resets = np.zeros(len(terminals), dtype=np.float32).astype(bool)
        idxs = (np.where(terminals)[0]+1)%len(terminals)
        resets[idxs] = True
        return resets

    def fix_actions(self, actions, reset, cats=18):
        """takes array of scalar actions and converts to one-hot.
        also offsets actions b/c dreamer uses pre-actions instead of post-actions"""
        ridxs = np.where(reset)[0]
        # actions = actions.reshape(-1)
        targets = np.roll(actions, 1, axis=0)
        if self.action_type == "discrete":
            one_hot = np.eye(cats)[targets]
            one_hot[ridxs] = np.zeros_like(one_hot[0])
            return one_hot
        else:
            return targets

    def get_data(self, dataset_name):
        sys.stdout = open(os.devnull, 'w')
        env = gym.make(dataset_name)
        data = env.get_dataset()
        sys.stdout = sys.__stdout__
        return data

    def load_files(self, path):
        data = {k: [] for k in self.data_keys}
        print("loading dataset from", path)

        filenames = glob.glob(path+'/*.npz')
        filenames = filenames[:1000]
        print('got', len(filenames), 'files...')

        rewards=[]

        for filename in tqdm(filenames, desc="loading files.."):
            npz_dict = np.load(filename,allow_pickle=True)

            obs_key = "images" # or "observations" or "vecobs"
            
            episode_length = len(npz_dict["actions"])
            self.num_transitions += episode_length
            resets = self.fix_terminals(npz_dict["terminals"])
            data["reset"].extend(resets)
            data["action"].extend(self.fix_actions(
                npz_dict["actions"], resets))
            data["obs"].extend([self.fix_obs(x, resize=True, sb3=True) #should resize be here?
                                for x in npz_dict[obs_key]])
            rewards.append(np.sum(npz_dict["rewards"]))
        print("finished loading")
        
        print("Reward min, max, mean, std", np.min(rewards), 
            np.max(rewards), np.mean(rewards), np.std(rewards))
        

        data = {k: torch.tensor(np.array(v, dtype=np.float32)).to(self.device)
                for k, v in data.items()}
        print(data["action"].shape, "here")
        data["reset"] = data["reset"].bool()
        return data

    def load_d4rl_data(self, dataset_names):
        data = {k: [] for k in self.data_keys}
        print("loading the following datasets:", dataset_names)

        for dataset_name in dataset_names:
            npz_dict = self.get_data(dataset_name)
            episode_length = len(npz_dict["actions"])
            self.num_transitions += episode_length
            data["reset"].extend(self.fix_terminals(npz_dict["terminals"]))
            data["action"].extend(self.fix_actions(
                npz_dict["actions"], data["reset"]))
            data["obs"].extend([self.fix_obs(x)
                                for x in npz_dict["observations"]])
        print("finished loading")

        data = {k: torch.tensor(np.array(v, dtype=np.float32)).to(self.device)
                for k, v in data.items()}
        data["reset"] = data["reset"].bool()
        return data

    def __len__(self):
        return self.num_transitions

    def __getitem__(self, idx):
        end_idx = idx+self.seq_length
        action, obs, reset = \
            [self.data[key][idx:end_idx] for key in self.data_keys]

        pad_size = end_idx-self.num_transitions
        if pad_size > 0:
            action = torch.cat((action, self.data['action'][:pad_size]), dim=0)
            obs = torch.cat(
                (obs, self.data['obs'][:pad_size]), dim=0)
            reset = torch.cat((reset, self.data['reset'][:pad_size]), dim=0)

        ret = {"action": action, "obs": obs, "reset": reset}
        return ret

    def get_trans(self, idx):
        action, obs, reset = \
            [self.data[key][idx].unsqueeze(0).unsqueeze(0)
             for key in self.data_keys]
        return action, obs, reset

# from pathlib import Path

# import numpy as np
# from torch.utils.data import get_worker_info

# from data.d4rl_dataset import D4RLDataset


# class D4RLIterableDataset(IterableDataset):
#     def __init__(self, config):
#         self.files = list(Path(config.filepath).rglob('*.npz'))
        
#         self.batch_size = config.batch_size
#         self.seq_len = config.seq_len
        
#         self.chunk_size = math.ceil(self.num_files/self.batch_size)
#         self.n_steps = math.ceil(self.chunk_size/self.seq_len)


#     def __iter__(self):
        
#         worker_id = get_worker_info().id if get_worker_info() else 0
#         chunk_size = math.ceil(len(self.files)/self.num_workers) # num episodes each worker is responsible for
#         worker_files = self.files[worker_id*chunk_size:(worker_id+1)*chunk_size]
  

#     def iter_files(self):
#         """
#         e.g. batch size 50, 2 workers
#         each worker has a list of 50 eqiudistance indices corresponding to episodes across the whole dataset
#             - the start index for each worker is offset by init_idx+chunk_size//worker_id
#                 - chunk_size is len btwn indices
#             - each index corresponds to a file (episode) in the file list
#             - the worker keeps these files open 
#                 - puts in a list along with the within-episode indices for each episode
#             - puts this data in a list and returns
#             - when one of the files is done, the next file is opened and placed in the list
#         """
#         obs = {"action":[],"image":[],"reset":[]}
#         obs_len = lambda obs: len(obs[list(obs.keys())[0]])
#         for file in self.files:
#             npz_dict = np.load(file)
#             episode_length = obs_len(npz_dict)
#             for i in range(0,episode_length,self.seq_len):
#                 remaining_to_fill = self.seq_len-obs_len(obs) 
#                 obs = {k:obs[k]+npz_dict[k][i:i+remaining_to_fill] for k,v in npz_dict.items()}
#                 if obs_len(obs) == self.seq_len:
#                     yield obs
#                     obs = {"action":[],"image":[],"reset":[]}
                    
#         if obs_len(obs) != 0:
#             file = files[0]
#             npz_dict = np.load(file)
#             remaining_to_fill = self.seq_len-obs_len(obs)
#             obs = {k:obs[k]+npz_dict[k][i:i+remaining_to_fill] for k,v in npz_dict.items()}
#             yield obs
                                       
        