import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data.common import fix_actions, fix_obs
import glob
from tqdm.auto import tqdm

class ImageDataset(Dataset):
    def __init__(self, data_dir, out_dim, pixel_mean, pixel_std, episode_limit, device="cuda"):
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.out_dim = out_dim
        self.device = device
        self.num_transitions = 0
        self.episode_limit = episode_limit
        self.data_keys = ["actions", "images"]
        print(data_dir)
        self.data = self.load_data(data_dir)

    def fix_obs(self, img, new_hw=64, resize=True, sb3=False):
        """normalizes image"""
        img = np.transpose(img, (1, 2, 0))
        if resize:
            img = np.array(cv2.resize(img, dsize=(
                new_hw, new_hw), interpolation=cv2.INTER_AREA))
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)
        img = (img - self.pixel_mean) / self.pixel_std
        img = np.expand_dims(img, 1)

        return img

    # def load_data(self, data_dir):
    #     data_dir = "/data/beamrider/ppo/episodes/episode-9.npz"
    #     data = {k: [] for k in self.data_keys}
    #     data_dict = np.load(data_dir)
    #     data["images"] = [self.fix_obs(x) for x in data_dict["images"]]
    #     data["actions"] = np.eye(self.out_dim)[data_dict["actions"]]
    #     data = {k: torch.tensor(np.array(v, dtype=np.float32)).to("cpu")
    #             for k, v in data.items()}
    #     return data



    def load_data(self, path):
        data = {k: [] for k in self.data_keys}
        print("loading dataset from", path)

        filenames = glob.glob(path+'/*.npz')
        filenames = filenames[:self.episode_limit]
        print('got', len(filenames), 'files...')

        for filename in tqdm(filenames, desc="loading files.."):
            npz_dict = np.load(filename,allow_pickle=True)

            obs_key = "images" # or "observations" or "vecobs"
            
            episode_length = len(npz_dict["actions"])
            self.num_transitions += episode_length
            data["actions"].extend(np.eye(self.out_dim)[npz_dict["actions"]])
            data["images"].extend([self.fix_obs(x, resize=True, sb3=True) #should resize be here?
                                for x in npz_dict[obs_key]])
        print("finished loading")
        

        data = {k: torch.tensor(np.array(v, dtype=np.float32)).to("cpu")
                for k, v in data.items()}
        print(data["actions"].shape, "here")
        return data

    def __len__(self):
        return self.num_transitions

    # def __len__(self):
    #     return len(self.data["actions"])

    def __getitem__(self, idx):
        action, feature = [self.data[key][idx] for key in self.data_keys]
        return action, feature
