#!/usr/bin/env python3
import sys

import yaml

import wandb
from config.config import build_config
from trainers.ac_trainer import ACTrainer
from trainers.bc_trainer import BCTrainer
from trainers.rssm_trainer import RSSMTrainer


def load_conf(config_file, env_name):
    with open(config_file, "r") as f:
        raw_conf = yaml.safe_load(f)
    base_conf = raw_conf["base"]
    data_conf = raw_conf["data"]
    env_conf = raw_conf[env_name]
    dream_conf = raw_conf["dreamer"] if "dreamer" in raw_conf else None
    conf = {**base_conf, **env_conf}
    conf["data"] = data_conf
    conf["dreamer"] = dream_conf
    return conf


def wm_train(conf):
    wandb.login()
    wandb.init(project="world-model", config=conf.to_dict(),
               settings=wandb.Settings(code_dir="."))
    wandb.run.log_code(".")
    trainer = RSSMTrainer(conf)
    wandb.watch(trainer.model)
    trainer.train()


def bc_train(conf):
    wandb.login()
    wandb.init(project="bc", config=conf.to_dict(),
               settings=wandb.Settings(code_dir="."))
    conf = wandb.config
    trainer = BCTrainer(conf)
    wandb.watch(trainer.model)
    trainer.train()


def ac_train(conf):
    wandb.login()
    wandb.init(project="dreamer", config=conf.to_dict(),
               settings=wandb.Settings(code_dir="."))
    wandb.run.log_code(".")
    #conf = wandb.config
    # print('conf',conf)
    trainer = ACTrainer(conf)
    wandb.watch(trainer.policy)
    # wandb.watch(trainer.discrim)
    trainer.train()


def main(conf_path=None):
    config_file = "config/config.yaml" if conf_path is None else conf_path
    conf = build_config(config_file)
    # print(conf)
    # ac_train(conf)
    bc_train(conf)
    # wm_train(conf)


if __name__ == "__main__":
    conf_path = None
    if len(sys.argv) > 1:
        conf_path = sys.argv[1]
        print('got conf', conf_path)

    main(conf_path=conf_path)
