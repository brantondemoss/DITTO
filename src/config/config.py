import yaml
from ml_collections import config_dict


def build_wm_trainer_config(raw_config, data_conf):
    trainer_config = config_dict.ConfigDict(raw_config["wm"]["training"])
    trainer_config.checkpoints.path = data_conf["checkpoint_path"]
    trainer_config.optimizer = {
        k: float(v) for k, v in trainer_config.optimizer.items()}
    del trainer_config.loss_weights

    return trainer_config


def build_wm_config(raw_config, env_conf):
    image_channels = raw_config["wm"]["architecture"]["cnn"]["image_channels"]
    cnn_depth = raw_config["wm"]["architecture"]["cnn"]["cnn_depth"]

    encoder_config = config_dict.ConfigDict()
    encoder_config.image_channels = image_channels
    encoder_config.cnn_depth = cnn_depth

    rssm_config = config_dict.ConfigDict(
        raw_config["wm"]["architecture"]["rssm"])
    rssm_config.action_dim = env_conf.action_dim
    rssm_config.embed_dim = cnn_depth*32
    rssm_config.update(raw_config["wm"]["architecture"]["latents"])

    decoder_config = config_dict.ConfigDict()
    decoder_config.image_channels = image_channels
    decoder_config.image_weight = raw_config["wm"]["training"]["loss_weights"]["image_weight"]
    decoder_config.cnn_depth = cnn_depth

    # weights_config = config_dict.ConfigDict(
    #     raw_config["wm"]["weights"])

    wm_config = config_dict.ConfigDict()
    wm_config.update(raw_config["wm"]["training"]["loss_weights"])
    wm_config.update(raw_config["wm"]["architecture"]["latents"])
    wm_config.rssm_config = rssm_config
    wm_config.encoder_config = encoder_config
    wm_config.decoder_config = decoder_config
    # wm_config.weights_config = weights_config

    return wm_config


def build_agent_config(raw_config, env_name):
    agent_conf = raw_config["agent"]
    data_conf = raw_config["data"]

    ac_trainer_config = config_dict.ConfigDict()
    ac_trainer_config.use_sb3 = data_conf[env_name]["use_sb3"]
    ac_trainer_config.load_device = data_conf["load_device"]
    ac_trainer_config.train_device = data_conf["train_device"]
    ac_trainer_config.update(agent_conf["architecture"])
    ac_trainer_config.update(agent_conf["training"])

    wm_cfg = data_conf[env_name]["world_models"]
    env_conf = wm_cfg["sb3"] if ac_trainer_config.use_sb3 else wm_cfg["d4rl"]

    ac_trainer_config.checkpoint_path = env_conf["checkpoint_path"]
    ac_trainer_config.wm_path = env_conf["checkpoint_path"] + \
        env_conf["wm_fname"]

    ac_dataset_config = config_dict.ConfigDict()
    ac_dataset_config.wm_path = ac_trainer_config.wm_path
    ac_dataset_config.seq_length = ac_trainer_config.seq_length
    ac_dataset_config.load_device = ac_trainer_config.load_device
    ac_dataset_config.episode_limit = data_conf[env_name]["episode_limit"]

    ac_dataset_config.use_cached_features = wm_cfg["use_cached_features"]
    if ac_dataset_config:
        ac_dataset_config.cached_features_path = env_conf["cached_features_path"]

    agent_config = config_dict.ConfigDict()
    agent_config.ac_trainer_config = ac_trainer_config
    agent_config.ac_dataset_config = ac_dataset_config

    return agent_config


def build_wm_dataset_config(trainer_config, env_conf):
    dataset_config = config_dict.ConfigDict()
    dataset_config.batch_size = trainer_config.batch_size
    dataset_config.seq_length = trainer_config.seq_length
    dataset_config.validation = config_dict.ConfigDict(
        {"batch_size": trainer_config.validation.batch_size,
         "seq_length": trainer_config.validation.seq_length}
    )
    # print(env_conf)
    dataset_config.pixel_mean = env_conf["pixel_stats"]["mean"]
    dataset_config.pixel_std = env_conf["pixel_stats"]["std"]
    dataset_config.action_type = env_conf.action_type
    dataset_config.validation.seq_length = trainer_config.validation.seq_length

    return dataset_config


def build_data_config(raw_config, env_name):
    data_conf = raw_config["data"]
    wm_cfg = data_conf[env_name]["world_models"]
    data_conf = wm_cfg["sb3"] if data_conf[env_name]["use_sb3"] else wm_cfg["d4rl"]
    return data_conf


def build_featurizer_config(dataset_config, wm_config):
    featurizer_config = config_dict.ConfigDict()
    featurizer_config.update(dataset_config)
    featurizer_config.wm_config = wm_config
    featurizer_config.seq_length = 1
    featurizer_config.batch_size = 500
    return featurizer_config


def build_config(config_file, checkpoint=True):
    with open(config_file, "r") as f:
        raw_config = yaml.safe_load(f)

    env_name = raw_config["dreamer"]["env"]
    print('got env name', env_name)
    env_config = config_dict.ConfigDict(raw_config["envs"][env_name])
    data_conf = build_data_config(raw_config, env_name)
    # print(data_conf)
    wm_config = build_wm_config(raw_config, env_config)
    trainer_config = build_wm_trainer_config(raw_config, data_conf)
    agent_config = build_agent_config(raw_config, env_name)

    dataset_config = build_wm_dataset_config(trainer_config, env_config)
    dataset_config.data_keys = raw_config["data"]["data_keys"]
    dataset_config.load_device = raw_config["data"]["load_device"]
    dataset_config.train_device = raw_config["data"]["train_device"]
    dataset_config.dataset_path = raw_config["data"][env_name]["dataset_path"]
    trainer_config.train_device = dataset_config.train_device

    config = config_dict.ConfigDict()
    config.raw = raw_config
    config.wm_config = wm_config
    config.dataset_config = dataset_config
    config.trainer_config = trainer_config
    config.ac_trainer_config = agent_config.ac_trainer_config
    config.ac_trainer_config.pixel_mean = dataset_config.pixel_mean
    config.ac_trainer_config.pixel_std = dataset_config.pixel_std
    config.ac_trainer_config.env_name = env_name
    config.ac_trainer_config.env_id = raw_config["envs"][env_name]["env_id"]
    config.ac_dataset_config = agent_config.ac_dataset_config
    featurizer_config = build_featurizer_config(
        dataset_config, wm_config)
    config.ac_dataset_config.featurizer_config = featurizer_config
    if not checkpoint:
        config.ac_trainer_config.do_checkpoint = False

    return config_dict.FrozenConfigDict(config)
