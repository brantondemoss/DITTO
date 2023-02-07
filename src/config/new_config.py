import argparse
from distutils.command.build import build
from signal import default_int_handler

import yaml
from ml_collections import config_dict

""" NEW RULES
1) each class has a dedicated builder/cfg
2) every class cfg variable is explicitly assigned
3) variables that determine conditionals must be defined globally in main, 
    OR must be passed as an inline parameter to a cfg builder
4) 
"""


def build_wm_cfg(**kwargs):
    encoder_config = config_dict.ConfigDict()
    encoder_config.image_channels = kwargs.get("image_channels", 1)
    encoder_config.cnn_depth = kwargs.get("cnn_depth", 48)

    rssm_config = config_dict.ConfigDict()
    # rssm_config.action_dim = 4
    rssm_config.deter_dim = kwargs.get("deter_dim", 1024)
    rssm_config.embed_dim = encoder_config.cnn_depth*32
    rssm_config.stoch_dim = kwargs.get("stoch_dim", 32)
    rssm_config.stoch_rank = kwargs.get("stoch_rank", 32)
    rssm_config.hidden_dim = kwargs.get("hidden_dim", 1000)
    rssm_config.gru_layers = kwargs.get("gru_layers", 1)

    decoder_config = config_dict.ConfigDict()
    decoder_config.image_channels = encoder_config.image_channels
    decoder_config.image_weight = kwargs.get("image_weight", 2.0)
    decoder_config.cnn_depth = encoder_config.cnn_depth

    wm_config = config_dict.ConfigDict()
    wm_config.encoder_config = encoder_config
    wm_config.rssm_config = rssm_config
    wm_config.decoder_config = decoder_config
    wm_config.kl_balance = kwargs.get("kl_balance", 0.8)
    wm_config.kl_weight = kwargs.get("kl_weight", 0.1)
    wm_config.features_dim = rssm_config.deter_dim + \
        rssm_config.stoch_dim*rssm_config.stoch_rank

    return wm_config


def build_rssm_trainer_config(env_name, train_device, **kwargs):
    optimizer = config_dict.ConfigDict()
    optimizer.lr = kwargs.get("lr", 3e-4)
    optimizer.eps = kwargs.get("eps", 1e-5)

    checkpoints = config_dict.ConfigDict()
    checkpoints.do_checkpoint = kwargs.get("do_checkpoint", True)
    checkpoints.savepoints = [1000, 5000, 10000, 50000,
                              100000, 200000, 500000, 1000000, 1500000, 2000000]
    checkpoints.path = "../checkpoints/bipedalwalker/" if env_name == "bipedalwalker" else "../checkpoints/breakout/"

    validation = config_dict.ConfigDict()
    validation.do_val = kwargs.get("do_val", True)
    validation.seq_len = kwargs.get("val_seq_len", 50)
    validation.batch_size = kwargs.get("val_batch_len", 128)

    trainer_config = config_dict.ConfigDict()
    trainer_config.checkpoints = checkpoints
    trainer_config.validation = validation
    trainer_config.seq_len = kwargs.get("train_seq_len", 50)
    trainer_config.batch_size = kwargs.get("train_batch_len", 50)
    trainer_config.train_device = train_device

    return trainer_config


def build_featurizer_cfg(wm_cfg, env_name):
    featurizer_cfg = config_dict.ConfigDict()
    featurizer_cfg.wm_cfg = wm_cfg
    featurizer_cfg.update(build_dataset_cfg(env_name))
    featurizer_cfg.seq_len = 1
    featurizer_cfg.batch_size = 500
    return featurizer_cfg


def build_dataset_cfg(env_name):
    dataset_cfg = config_dict.ConfigDict()
    dataset_cfg.data_keys = ["action", "obs", "reset"]
    if env_name == "breakout":
        dataset_cfg.pixel_mean = 33.0
        dataset_cfg.pixel_std = 55.0
        dataset_cfg.action_type = "discrete"

    elif env_name == "bipedalwalker":
        dataset_cfg.pixel_mean = 210.0
        dataset_cfg.pixel_std = 48.0
        dataset_cfg.action_type = "continuous"
    dataset_cfg.batch_size = 50
    dataset_cfg.seq_length = 50


def build_ac_trainer_cfg(env_name, args, ac_dataset_cfg):
    ac_trainer_config = config_dict.ConfigDict()
    # should split this up by trainer and policy classes
    ac_trainer_config.lr = 3e-4
    ac_trainer_config.gamma = 0.95
    ac_trainer_config.layers = 8
    ac_trainer_config.entropy = 0.003
    ac_trainer_config.use_sb3 = args.use_sb3
    ac_trainer_config.wm_path = ac_dataset_cfg.wm_path
    ac_trainer_config.actor_dist = "onehot"
    ac_trainer_config.actor_grad = "reinforce"
    ac_trainer_config.batch_size = 512
    ac_trainer_config.seq_length = 15
    ac_trainer_config.hidden_dim = 256
    ac_trainer_config.lambda_gae = 0.95

    ac_trainer_config.imag_horizon = 15
    ac_trainer_config.action_dim = 4

    ac_trainer_config.load_device = args.load_device
    ac_trainer_config.train_device = args.train_device

    ac_trainer_config.do_checkpoint = True
    ac_trainer_config.checkpoint_path = "../checkpoints/breakout/" if env_name == "breakout" else "../checkpoints/bipedalwalker/"
    ac_trainer_config.target_interval = 100

    validation = config_dict.ConfigDict()
    validation.n_games = 12
    validation.n_envs = 12
    ac_trainer_config.validation = validation


def build_ac_dataset_cfg(env_name, use_sb3):
    ac_dataset_cfg = config_dict.ConfigDict()
    if env_name == "breakout":
        if use_sb3:
            ac_dataset_cfg.wm_path = "../checkpoints/breakout/sb3/sb3-model-1000000_steps.pth"
            ac_dataset_cfg.cached_features_path = "../data/breakout/sb3-features-cached.npz"
        else:
            ac_dataset_cfg.wm_path = "../checkpoints/breakout/d4rl/model-1000000_steps.pth"
            ac_dataset_cfg.cached_features_path = "./data/breakout/exp-v2-cached.npz"
    elif env_name == "bipedalwalker":
        ac_dataset_cfg.wm_path = "../checkpoints/bipedalwalker/model-2000000_steps.pth"
        ac_dataset_cfg.cached_features_path = "../data/bipedalwalker/sb3-features-cached.npz"
    ac_dataset_cfg.episode_limit = 100
    ac_dataset_cfg.seq_length
    ac_dataset_cfg.load_device
    ac_dataset_cfg.use_cached_features


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_device', default="cuda:0")
    parser.add_argument('--load_device', default="cpu")
    parser.add_argument("--env_name", default="breakout")
    parser.add_argument("--use_d4rl", dest="use_sb3",
                        action="store_false")  # this defaults to use_sb3 == True
    return parser.parse_args()


def main():
    args = build_args()
    cfg = config_dict.ConfigDict()
    cfg.wm_config = build_wm_cfg()
    cfg.ac_dataset_config = build_ac_dataset_cfg(
        cfg.wm_config, args.env_name, use_sb3=args.use_sb3)
