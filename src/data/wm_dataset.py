import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.common import fix_actions, fix_terms, get_d4rl_data

class WorldModelDataset(Dataset):
    def __init__(self, dataset_config):
        self.seq_length = dataset_config["batch_length"]
        self.device = dataset_config["load_device"]
        self.limit = dataset_config["episode_limit"]
        self.partitions = dataset_config.get("partitions", None)
        self.rank = dataset_config.get("rank", None)
        self.num_transitions = 0
        self.data_keys = ["action", "state", "reset"]
        if "data_names" in dataset_config:
            print("Building WorldModelDataset for",dataset_config["data_names"])
            self.data = self.load_data_names(dataset_config["data_names"])
        else:
            print("Building WorldModelDataset from",dataset_config["data_path"])
            self.data = self.load_data(dataset_config["data_path"])

        print("total transitions:", self.num_transitions)
        print("num batch elements:", self.num_transitions//self.seq_length)

    def process_img(self, img, new_hw=64):
        """normalizes image"""
        if img.shape[0] != 64:
            img = np.array(cv2.resize(img, dsize=(
                new_hw, new_hw), interpolation=cv2.INTER_AREA))
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        img = img.astype(np.float32)
        img = img / 255.0 - 0.5
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return img

    def scalar_onehot(self, scalars, categories=18):
        """takes array of scalar actions and converts to one-hot"""
        if len(scalars.shape) == 2:
            return scalars
        targets = scalars.reshape(-1)
        one_hot = np.eye(categories)[targets]
        return one_hot

    def init_resets(self, resets, prob=0.8):
        """randomly adds resets as memory dropout"""
        reset_idxs = np.where(resets)[0]
        sample_bounds = [((j-i)//3 + i, j) for i, j in
                         zip(reset_idxs, np.append(reset_idxs[1:], len(resets)))]
        for bounds in sample_bounds:
            idx = np.random.randint(bounds[0], bounds[1])
            resets[idx] = \
                np.random.rand(
            ) > prob or resets[idx]
        assert resets[0] is not False
        return resets

    def load_data_names(self, names):
        data = {k: [] for k in self.data_keys}

        for name in names:
            dataset = get_d4rl_data(name)
            episode_length = len(dataset["actions"])
            self.num_transitions += episode_length
            reset = fix_terms(dataset["terminals"])
            data["reset"].extend(reset)
            data["action"].extend(fix_actions(dataset["actions"], reset))
            data["state"].extend([self.process_img(x)
                                for x in np.squeeze(dataset["observations"])])

        data = {k: torch.tensor(np.array(v, dtype=np.float32)).to(self.device)
                for k, v in data.items()}
        data["reset"] = data["reset"].bool()
        return data


    def load_data(self, data_dir):
        data = {k: [] for k in self.data_keys}
        path = data_dir+"/*.npz"
        files = glob.glob(path, recursive=False)
        if self.limit != -1:
            files = files[:self.limit]

        if self.partitions is not None:
            start = len(files)//self.partitions*self.rank
            end = len(files)//self.partitions*(self.rank+1)
            files = files[start:end]

        for file in files:
            npz_dict = np.load(file)
            episode_length = len(npz_dict["action"])
            self.num_transitions += episode_length
            # data["reset"].extend(self.init_resets(npz_dict["reset"]))
            data["reset"].extend(npz_dict["reset"])
            data["action"].extend(self.scalar_onehot(npz_dict["action"]))
            data["state"].extend([self.process_img(x)
                                 for x in npz_dict["image"]])

        data = {k: torch.tensor(np.array(v, dtype=np.float32)).to(self.device)
                for k, v in data.items()}
        data["reset"] = data["reset"].bool()
        return data

    def __len__(self):
        return self.num_transitions

    def __getitem__(self, idx):
        end_idx = idx+self.seq_length
        action, state, reset = \
            [self.data[key][idx:end_idx] for key in self.data_keys]

        pad_size = end_idx-self.num_transitions
        if pad_size > 0:
            action = torch.cat((action, self.data['action'][:pad_size]), dim=0)
            state = torch.cat(
                (state, self.data['state'][:pad_size]), dim=0)
            reset = torch.cat((reset, self.data['reset'][:pad_size]), dim=0)

        return action, state, reset

    def get_trans(self, idx):
        action, state, reset = \
            [self.data[key][idx].unsqueeze(0).unsqueeze(0)
             for key in self.data_keys]
        return action, state, reset
