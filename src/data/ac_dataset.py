from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.common import cat_actions, fix_terms, get_d4rl_data
from data.featurizer import Featurizer


class ACDataset(Dataset):
    def __init__(self, conf):
        #self.data_path = dataset_config["data_path"]

        # self.conf = conf

        self.conf = conf.ac_dataset_config
        self.seq_length = self.conf.seq_length
        self.device = self.conf.load_device

        self.dataset_config = conf.dataset_config
        self.featurizer_config = self.conf.featurizer_config

        self.data_keys = ["actions", "features", "resets"]
        self.data = self.build_data(conf)

        #self.data = self.load_data()
        # self.data_dir = "../../data/breakout/valf/"
        # self.data = self.load_data_old(self.data_dir)

        self.num_transitions = len(self.data["features"])
        print("ACDataset size:", self.num_transitions)

        # train_device
        # batch_size
        # batch_length
        # dataset_config
        # self.conf.wm_config

    def build_data(self, conf):
        data = {k: [] for k in self.data_keys}

        print("BUILDING DATAAAA")

        if self.conf.use_cached_features and Path(self.conf.cached_features_path).is_file():
            npz_dict = np.load(self.conf.cached_features_path)
            (features, actions, resets) = (
                npz_dict['features'], npz_dict['actions'], npz_dict['resets'])
            print('loaded features from path', self.conf.cached_features_path)
        else:
            print("making features..")
            print(self.conf)
            features, actions, resets = Featurizer(
                self.conf).get_features()
            np.savez(self.conf.cached_features_path, features=features, actions=actions, resets=resets)
            print('Built Features', self.conf.cached_features_path)
        if self.conf.episode_limit > 0:
            reset_idxs = np.where(resets)[0]
            if self.conf.episode_limit < len(reset_idxs):
                print('Cut dataset down to', self.conf.episode_limit, 'episodes')
                ep_idx = reset_idxs[self.conf.episode_limit]
                (features, actions, resets) = (
                    features[:ep_idx], actions[:ep_idx], resets[:ep_idx])

        data["features"].extend(features)
        data["actions"].extend(actions)
        data["resets"].extend(resets)

        data = {k: torch.tensor(np.array(v, dtype=np.float32)).to(self.device)
                for k, v in data.items()}
        data["resets"] = data["resets"].bool()

        assert len(data["features"]) == len(data["resets"])
        print("Built ACDataset from", self.conf.wm_path)
        return data

    def load_data(self):
        data = np.load(self.data_path)
        data = {k: torch.tensor(np.array(v, dtype=np.float32))
                for k, v in data.items()}
        data["resets"] = data["resets"].bool()
        return data

    def __len__(self):
        return self.num_transitions

    def __getitem__(self, idx):
        end_idx = idx+self.seq_length
        feature = self.data["features"][idx:end_idx]
        reset = self.data["resets"][idx:end_idx]
        action = self.data["actions"][idx:end_idx]
        return feature, reset, action

    @staticmethod
    def ac_collate(batch):
        from torch.utils.data._utils.collate import default_collate
        return [torch.transpose(x, 0, 1) for x in default_collate(batch)]
