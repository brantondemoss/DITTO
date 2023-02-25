#!/usr/bin/env python3
import torch
import torch.optim as optim
import wandb
import yaml
from data.common import EquiSampler, transpose_collate
from data.d4rl_dataset import D4RLDataset
from data.wm_dataset import WorldModelDataset
from src.data.calvin_dataset import CalvinDataset
from models.world_model import WorldModelRSSM
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm, trange

from trainers.metrics import MetricsHelper


class RSSMTrainer(object):
    def __init__(self, root_conf):
        super().__init__()

        #print('root conf', root_conf)
        self.env_type = root_conf.raw.dreamer.env
        self.conf = root_conf.trainer_config
        self.device = self.conf.train_device
        self.batch_size = self.conf.batch_size
        self.seq_length = self.conf.seq_length
        self.do_val = self.conf.validation.do_val

        self.dataloaders = self.build_dataloaders(root_conf.dataset_config)

        self.model = WorldModelRSSM(root_conf.wm_config).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.conf.optimizer.lr, eps=self.conf.optimizer.eps)

        
        # this is here b/c I fucked up
        model_path = root_conf.ac_trainer_config.wm_path
        print('Got model path', model_path)
        # this dict contains keys: steps, model_state_dict, optimizer_state_dict
        checkpoint = torch.load(model_path)
        self.start_steps = checkpoint['steps']
        print('Got start_steps =',self.start_steps,'from loaded state dict')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loaded from state dict', model_path)

        self.scaler = GradScaler(enabled=True)

        self.metrics_helper = MetricsHelper(
            root_conf.dataset_config.pixel_mean, root_conf.dataset_config.pixel_std, do_val=self.do_val)
        self.pbar = tqdm()

        self.do_checkpoint = self.conf.checkpoints.do_checkpoint
        self.checkpoints = self.conf.checkpoints.savepoints
        self.checkpoint_path = self.conf.checkpoints.path
        self.c_idx = 0

    def val_train_split(self, dataset, split=0.9):
        print("Splitting data..")
        split_idx = int(len(dataset)*split)
        idxs = torch.arange(len(dataset))
        train_dataset = Subset(dataset, idxs[:split_idx])
        val_dataset = Subset(dataset, idxs[split_idx:])
        print("set lens:", len(train_dataset), len(val_dataset))
        return train_dataset, val_dataset

    def build_dataloader(self, dataset, dataset_config):
        batch_sampler = EquiSampler(
            len(dataset), dataset_config.seq_length, dataset_config.batch_size)
        dataloader = DataLoader(dataset,
                                pin_memory=True,
                                batch_sampler=batch_sampler,
                                collate_fn=transpose_collate)
        return dataloader

    def build_dataloaders(self, dataset_config):
        print("Building dataset..")
        if self.env_type == "calvin":
            train_dataset = CalvinDataset(dataset_config, train=True)
            val_dataset = CalvinDataset(dataset_config, train=False)
        else:
            dataset = D4RLDataset(dataset_config)
            train_dataset, val_dataset = self.val_train_split(dataset)

        print("Building dataloaders..")
        train_dataloader = self.build_dataloader(train_dataset, dataset_config)
        dataloaders = {"train": train_dataloader}
        if self.do_val:
            val_dataloader = self.build_dataloader(
                val_dataset, dataset_config.validation)
            dataloaders["val"] = val_dataloader
        return dataloaders

    def train(self):
        while True:
            in_states = self.model.init_state(self.batch_size)
            for batch in self.dataloaders["train"]:
                obs = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()

                with autocast(enabled=True):
                    batch_metrics, decoded_img, out_states, samples = self.model.training_step(
                        obs, in_states)
                in_states = out_states

                # put before param updates
                self.log_stats(batch_metrics, samples, decoded_img)

                self.scaler.scale(batch_metrics["loss"]).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 200)
                self.scaler.step(self.optimizer)
                self.scaler.update()

    def validate(self):
        self.model.eval()
        batch_size = self.conf.validation.batch_size
        running_metrics = {
            metric_name: 0 for metric_name in self.metrics_helper.batch_metrics}
        with torch.no_grad():
            in_states = self.model.init_state(batch_size)
            for batch in self.dataloaders["val"]:
                obs = {k: v.to(self.device) for k, v in batch.items()}
                batch_metrics, decoded_img, out_states, samples = self.model.training_step(
                    obs, in_states)
                in_states = out_states
                for key in batch_metrics.keys():
                    running_metrics[key] += batch_metrics[key].to(
                        "cpu").detach().numpy()

            pred_img = self.model.pred_img(*samples)
            self.metrics_helper.log_imgs(
                samples[-1], decoded_img, pred_img, "val")

        for key in batch_metrics.keys():
            running_metrics[key] /= len(self.dataloaders["val"])
        self.metrics_helper.update_stats("val", running_metrics)

        self.model.train()

    def log_stats(self, batch_metrics, samples, decoded_img):
        steps = self.metrics_helper.step_dict["train"]

        if steps % (len(self.dataloaders["train"])) == 0:
            self.validate() if self.do_val else None
            pred_img = self.model.pred_img(*samples)
            self.metrics_helper.log_imgs(
                samples[-1], decoded_img, pred_img, "train")

        self.metrics_helper.update_stats("train", batch_metrics)

        if steps % 25 == 0:
            self.pbar.update(25)

        if self.do_checkpoint:
            if self.c_idx == len(self.checkpoints):
                exit()
            elif steps > 0 and steps % self.checkpoints[self.c_idx] == 0:
                self.write_checkpoint(steps)

    def write_checkpoint(self, steps):
        torch.save({
            'steps': steps+self.start_steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f"{self.checkpoint_path}model-{steps+self.start_steps}_steps.pth")
        print('wrote',f"{self.checkpoint_path}model-{steps+self.start_steps}_steps.pth")
        self.c_idx += 1
