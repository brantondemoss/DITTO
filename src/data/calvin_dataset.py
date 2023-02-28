import torch 
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm 
import glob
import numpy as np
import cv2
#from src.common import EquiSampler, transpose_collate

class CalvinDataset(Dataset):
    def __init__(self, dataset_config, train=True):
        self.action_type = dataset_config.action_type
        self.seq_length = dataset_config.seq_length #dataset_config.seq_length
        self.device = dataset_config.load_device
        self.num_transitions = 0
        self.data_keys = dataset_config.data_keys
        
        self.pixel_mean = dataset_config.pixel_mean
        self.pixel_std = dataset_config.pixel_std
        
        #self.pixel_mean = 138.9931
        #self.pixel_std = 65.1393
        
        if train: # TODO unhardcode paths
            self.data_path = "/home/alex/repos/calvin/dataset/calvin_debug_dataset/training" # NOTE THis is hardcoded
        else:
            self.data_path = "/home/alex/repos/calvin/dataset/calvin_debug_dataset/validation"
        
        self.data = self.load_files(self.data_path)
        
    def load_files(self, path):
        data = {k: [] for k in self.data_keys}
        print("loading dataset from", path)
        
        start_ids = self._ep_start_ids(path)
        #print('start ids', start_ids)
        filenames = glob.glob(path+'/*.npz')
        print(f'got {len(filenames)} files...')
        self.num_transitions = len(filenames)
        
        for filename in tqdm(filenames, desc="loading files.."):
            npz_dict = np.load(filename,allow_pickle=True)
            idx = int(filename[-11:-4])
            obs_key = "rgb_static"
            
            # TODO add normalisation (fix functions)
            data["action"].extend(self._fix_actions(npz_dict["actions"]))
            data["obs"].extend(self._fix_obs(npz_dict[obs_key]))
            if idx in start_ids:
                data["reset"].extend([True])
            else:
                data["reset"].extend([False])

            # files: ['actions', 'rel_actions', 'robot_obs', 'scene_obs', 'rgb_static', 'rgb_gripper', 'rgb_tactile', 'depth_static', 'depth_gripper', 'depth_tactile']
        
        data = {k: torch.tensor(np.array(v, dtype=np.float32))
                for k, v in data.items()} # TODO add device
        data["reset"] = data["reset"].bool()
        return data
            
    def _fix_obs(self, img, new_hw=64, resize=True):
        img = np.moveaxis(img, -1, 0)
        
        if resize:
            new_img = np.empty((3, new_hw, new_hw))
            img = np.squeeze(img)  # 3, 200, 200
            for i in range(img.shape[0]):
                new_img[i] = np.array(cv2.resize(img[i], dsize=(
                    new_hw, new_hw), interpolation=cv2.INTER_AREA)) 
            img = new_img
            
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)
        img = (img - self.pixel_mean) / self.pixel_std
        
        return img
        
            
    def _fix_actions(self, action):
        action = np.expand_dims(action, 0)
        return action
    
    def _ep_start_ids(self, path) -> np.ndarray :
        ep_start_end_ids = np.load(path + "/ep_start_end_ids.npy")
        return np.array([i[0] for i in ep_start_end_ids])
            
    def __len__(self):
        return self.num_transitions
    
    def __getitem__(self, idx):    
        end_idx = idx+self.seq_length
        action, obs, reset = \
            [self.data[key][idx:end_idx] for key in self.data_keys]
        
        pad_size = end_idx-self.num_transitions
        if pad_size>0:
            action = torch.cat((action, self.data['action'][:pad_size]), dim=0)
            obs = torch.cat(
                (obs, self.data['obs'][:pad_size]), dim=0)
            reset = torch.cat((reset, self.data['reset'][:pad_size]), dim=0)
        
        # TODO  ask Branton why this does not throw an error. seems like the above line will throw an error first tho
        
        ret = {"action": action, "obs": obs, "reset": reset}
        return ret

    def get_trans(self, idx):
        action, obs, reset = \
            [self.data[key][idx].unsqueeze(0).unsqueeze(0)
             for key in self.data_keys]
        return action, obs, reset

if __name__=="__main__":
    
    config = {
        "action_type": "continuous",
        "seq_length": 50,
        "device": "cuda",
        "data_keys": ["actions", "obs", "reset"],
    }
    #config = DotMap(config)
    
    data = CalvinDataset(config)
    
    path = '/home/alex/repos/calvin/dataset/calvin_debug_dataset/training'
    
    data.data = data.load_files(path)
    
    batch_sampler = EquiSampler(
        len(data), 5, 10
    )
    dataloader = DataLoader(
        data, 
        pin_memory=True,
        batch_sampler=batch_sampler,
        collate_fn=transpose_collate,
    )
    
    # statistics
    #mean = torch.mean(data.data["obs"])#, axis=0)
    #std = torch.std(data.data["obs"])#, axis=0)
    #print(mean, std)
        