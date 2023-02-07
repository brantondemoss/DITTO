import torch
import torch.distributions as D
import torch.nn as nn

from models.decoders import DecoderModel
from models.encoders import EncoderModel
from models.rssm import RSSMCore


def init_weights(m):
    if type(m) in {nn.Conv2d, nn.ConvTranspose2d, nn.Linear}:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    if type(m) == nn.GRUCell:
        nn.init.xavier_uniform_(m.weight_ih.data)
        nn.init.orthogonal_(m.weight_hh.data)
        nn.init.zeros_(m.bias_ih.data)
        nn.init.zeros_(m.bias_hh.data)


class WorldModelRSSM(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.encoder = EncoderModel(conf.encoder_config)
        self.kl_balance = conf.kl_balance
        self.kl_weight = conf.kl_weight
        self.rssm_core = RSSMCore(**conf.rssm_config)
        features_dim = conf.deter_dim + \
             conf.stoch_dim * conf.stoch_rank
        self.decoder = DecoderModel(features_dim, conf.decoder_config)

        for m in self.modules():
            init_weights(m)

    def init_state(self, batch_size):
        return self.rssm_core.init_state(batch_size)

    def pred_img(self, prior, post_samples, features, cur_states):
        """Decodes a sample from parameterized prior"""
        with torch.no_grad():
            prior_samples = self.rssm_core.zdistr(
                prior).sample().reshape(post_samples.shape)
            features_prior = self.rssm_core.feature_replace_z(
                features, prior_samples)
            _, _, decoded_img = self.decoder(
                features_prior, cur_states)
            return decoded_img

    def unnormalize(self, img):
        img = img[0, 0, ...]
        img = img.to("cpu").detach().numpy()
        img = ((img + 0.5) * 255.0).clip(0, 255).astype('uint8')
        return img

    def forward(self, obs, in_state):
        embed = self.encoder(obs["obs"])
        prior, post, post_samples, features, hidden_states, out_states = \
            self.rssm_core.forward(embed,
                                   obs['action'],
                                   obs['reset'],
                                   in_state)
        return features, out_states

    def dream(self, action, in_state):
        _, (h, z) = self.rssm_core.cell.forward(action, in_state)
        return (h, z)

    def training_step(self,
                      obs,
                      in_state,
                      ):
        embed = self.encoder(obs["obs"])

        prior, post, post_samples, features, hidden_states, out_states = \
            self.rssm_core.forward(embed,
                                   obs['action'],
                                   obs['reset'],
                                   in_state)
        loss_reconstr, loss_image, decoded_img = self.decoder(
            features, obs["obs"])

        d = self.rssm_core.zdistr
        dprior = d(prior)
        dpost = d(post)

        loss_kl_exact = D.kl.kl_divergence(dpost, dprior)
        loss_kl_post = D.kl.kl_divergence(dpost, d(prior.detach()))
        loss_kl_prior = D.kl.kl_divergence(d(post.detach()), dprior)
        loss_kl = (1 - self.kl_balance) * loss_kl_post + \
            self.kl_balance * loss_kl_prior

        loss = self.kl_weight * loss_kl + loss_reconstr

        entropy_prior = dprior.entropy()
        entropy_post = dpost.entropy()

        batch_metrics = {"loss": loss, "loss_kl": loss_kl, "loss_kl_exact": loss_kl_exact,
                         "loss_kl_post": loss_kl_post, "loss_kl_prior": loss_kl_prior,
                         "loss_image": loss_image, "entropy_prior": entropy_prior,
                         "entropy_post": entropy_post}
        batch_metrics = {k: v.mean() for k, v in batch_metrics.items()}

        samples = (prior, post_samples, features, obs["obs"])
        return batch_metrics, decoded_img, out_states, samples
