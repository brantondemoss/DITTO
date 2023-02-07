import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from ml_collections import config_dict
from models.world_model import WorldModelRSSM
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from trainers.metrics import MetricsHelper

from data.common import EquiSampler, transpose_collate
from data.d4rl_dataset import D4RLDataset
from data.wm_dataset import WorldModelDataset


class Featurizer(object):
    """ Featurizer reads in conf, loads world model from checkpoint, loads dataset in train dir
        calculates features for that dataset under that world model
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf.featurizer_config
        # self.dataset_config = self.conf.dataset_config

        self.device = torch.device(self.conf.train_device)
        self.wm_path = conf.wm_path
        self.batch_length = self.conf.seq_length
        self.batch_size = self.conf.batch_size

        #self.groundtruth = np.load("/home/bdemoss/research/div-rl/data/breakout/valf/expertv2-features.npy")

    def get_features(self):
        print("Building WM Dataset for featurizer...")
        dataloader, dataset = self.build_dataloader()

        chunk_size = math.ceil(len(dataset)/self.batch_size)

        print("Loading in trained world model...")
        # this is why you need the full conf all the way down here
        wm = WorldModelRSSM(self.conf.wm_config)
        wm.load_state_dict(torch.load(self.wm_path)[
                           'model_state_dict'])
        wm.to(self.device)
        wm.eval()
        wm.requires_grad_(False)

        print("Getting features...")
        features = self.infer_features(dataset, dataloader, wm)
        return features, dataset.data['action'].cpu().numpy(), dataset.data['reset'].cpu().numpy()

    def build_dataloader(self):
        dataset_config = config_dict.ConfigDict(self.conf)
        dataset = D4RLDataset(dataset_config)
        batch_sampler = EquiSampler(
            len(dataset),  self.batch_length, self.batch_size, init_idx=0)
        dataloader = DataLoader(dataset,
                                pin_memory=True,
                                batch_sampler=batch_sampler,
                                collate_fn=transpose_collate)
        return dataloader, dataset

    @torch.inference_mode()
    def infer_features(self, dataset, dataloader, wm):
        s = time.time()

        # TODO: don't hardcode dim
        feats = np.zeros((len(dataset), 2048), dtype=np.float32)

        for epoch in range(2):
            # potentially need to do more than 2 epochs in edge cases when
            # episode is longer than chunk_size and extends across 3 chunks

            fake_sampler = iter(EquiSampler(len(dataset),
                                            self.batch_length,
                                            self.batch_size,
                                            init_idx=0))
            if epoch == 0:
                in_states = wm.init_state(self.batch_size)
            else:
                # put the last hidden state of previous chunk
                # as the first hidden state of next chunk
                # and zero out first one (assume index starts at 0)
                in_states = [torch.roll(x, 1, 0) for x in in_states]
                in_states[0][0] = torch.zeros_like(in_states[0][0])
                in_states[1][0] = torch.zeros_like(in_states[1][0])
            for i, batch in enumerate(dataloader):
                obs = {k: v.to(self.device) for k, v in batch.items()}
                with autocast(enabled=True):
                    features, out_states = wm(obs, in_states)
                # T,B,Feats
                # T = 1 for inference
                in_states = out_states

                features = features.cpu().numpy().squeeze()

                idxs = np.array(next(fake_sampler))
                feats[idxs] = features

        print("Feature inference took", int(time.time()-s), "seconds")
        return feats

    @torch.inference_mode()
    def infer_features2(self, dataset):
        """Gets the learned state features for this 
        dataset using whichever WorldModel we loaded."""
        # size of the final feature array will be (transitions, feature_dim)
        s = time.time()
        fake_sampler = iter(EquiSampler(len(self.dataset),
                                        self.batch_length,
                                        self.batch_size,
                                        init_idx=0))
        feats = np.zeros((len(self.dataset), 2048), dtype=np.float32)

        in_states = self.wm.init_state(self.batch_size)

        chunk_size = self.chunk_size

        print('in infer_features2')
        for i, batch in enumerate(self.dataloader):
            action, image, reset = [x.to(self.device) for x in batch]

            obs = {"image": image, "reset": reset, "action": action}

            with autocast(enabled=True):
                features, out_states = self.wm(obs, in_states)
                # T,B,Feats
            in_states = out_states

            features = features.cpu().numpy().squeeze()

            feats[np.array(next(fake_sampler))] = features
        print("Feature2 inference took", int(time.time()-s), "seconds")
        return feats

    @torch.inference_mode()
    def infer_features_exact(self, dataset, dataloader, wm):
        """Gets the learned state features for this 
        dataset using whichever WorldModel we loaded."""

        # size of the final feature array will be (transitions, feature_dim)
        print("Begin inference")
        s = time.time()
        feature_list = []
        in_states = wm.init_state(1)
        for i in range(len(dataset)):
            action, image, reset = [x.to(self.device)
                                    for x in dataset.get_trans(i)]
            obs = {"image": image, "reset": reset, "action": action}

            with autocast(enabled=False):
                features, out_states = wm(obs, in_states)
            feature_list.append(features.cpu().numpy().squeeze())
            in_states = out_states
            if i % 10000 == 0:
                print("inferred", i, "features in",
                      int(time.time()-s), "seconds")

        print("got", len(feature_list), "features")

        print("Feature inference took", int(time.time()-s), "seconds")
        features = np.stack(feature_list)
        return features
