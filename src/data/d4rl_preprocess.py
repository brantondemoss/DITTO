import gc
import itertools
import os
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import d4rl_atari
import gym
import numpy as np
from tqdm.auto import tqdm


class D4RLPreprocess(object):

    @staticmethod
    def fix_all(npz_dict):
        reset = D4RLPreprocess.fix_terminals(npz_dict["terminals"])
        del npz_dict["terminals"]
        action = D4RLPreprocess.fix_actions(npz_dict["actions"], reset)
        del npz_dict["actions"]
        image = [D4RLPreprocess.fix_obs(x)for x in npz_dict["observations"]]
        return action, image, reset

    @staticmethod
    def fix_obs(img, new_hw=64):
        """normalizes image"""
        img = np.transpose(img, (1, 2, 0))
        img = np.array(cv2.resize(img, dsize=(
            new_hw, new_hw), interpolation=cv2.INTER_AREA))
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)
        img = img / 255.0 - 0.5
        return img

    @staticmethod
    def fix_terminals(terminals):
        resets = np.zeros(len(terminals), dtype=np.float32).astype(bool)
        resets[0] = True
        resets[np.where(terminals)[0]+1] = True
        return resets

    @staticmethod
    def fix_actions(actions, reset, cats=18):
        """takes array of scalar actions and converts to one-hot.
        also offsets actions b/c dreamer uses pre-actions instead of post-actions"""
        one_hot = np.eye(cats)[np.roll(actions.reshape(-1), 1)]
        one_hot[np.where(reset)[0]] = np.zeros_like(one_hot[0])
        return one_hot

    @staticmethod
    def get_data(dataset_name):
        sys.stdout = open(os.devnull, 'w')
        env = gym.make(dataset_name)
        data = env.get_dataset()
        sys.stdout = sys.__stdout__
        return data

    @staticmethod
    def split_into_episodes(action, image, reset):
        idxs = np.where(reset)[0][1:]  # ignore fist idx, will cause empty list
        action = np.split(action, idxs)
        image = np.split(image, idxs)
        reset = np.split(reset, idxs)
        return action, image, reset

    @staticmethod
    def preprocess_dataset(args):
        dataset_name, temp_dir_name = args
        action, image, reset = D4RLPreprocess.fix_all(
            D4RLPreprocess.get_data(dataset_name))
        gc.collect()

        n_episodes = len(np.where(reset)[0])

        out_dir = temp_dir_name+f"/{dataset_name}/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        action, image, reset = D4RLPreprocess.split_into_episodes(
            action, image, reset)

        for i, episode in tqdm(enumerate(zip(action, image, reset)), leave=False, desc="writing episodes to disk", position=1, total=n_episodes):
            outfile = out_dir + f"episode-{i}.npz"
            np.savez(outfile, action=episode[0],
                     image=episode[1], reset=episode[2])

    @staticmethod
    def preprocess_dist():
        # load, transform, save to temp dir
        ids = {"mixed": [0, 1, 2, 3, 4],
               "medium": [0, 1, 2, 3, 4],
               "expert": [0, 1, 2]}
        # dataset_name = 'breakout-expert-v2'

        # pbar = tqdm(total=13, desc="d4rl preprocessing")
        temp_dir = tempfile.TemporaryDirectory()
        dataset_names = []
        for level, versions in ids.items():
            for version in versions:
                dataset_name = 'breakout-{}-v{}'.format(level, version)
                dataset_names.append(dataset_name)
        args = zip(dataset_names, itertools.repeat(temp_dir.name))
        with Pool(2) as p:
            data = list(
                tqdm(
                    p.imap_unordered(D4RLPreprocess.preprocess_dataset, args),
                    total=len(dataset_names),
                )
            )

        return temp_dir
