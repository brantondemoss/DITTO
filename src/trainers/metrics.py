import wandb
import numpy as np
class MetricsHelper:
    def __init__(self, pixel_mean, pixel_std, do_val=True):
        # self.batch_size = batch_size
        self.step_dict = {"train": 0, "val": 0}
        self.batch_metrics = ["running_loss", "loss", "total_loss",
                              "loss_kl", "loss_kl_exact", "loss_kl_prior",
                              "loss_kl_post", "loss_image",
                              "entropy_prior", "entropy_post"]
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        train_metric_dict = {
            metric_name: 0 for metric_name in self.batch_metrics}
        val_metric_dict = {
            metric_name: 0 for metric_name in self.batch_metrics}

        self.metric_dicts = {
            "train": train_metric_dict, "val": val_metric_dict}

    def compute_ema(self, y_t, s_tt, alpha=0.1):
        ema = alpha * y_t + (1-alpha) * s_tt
        return ema

    def unnormalize(self, img):
        img = img[-1, 0, ...]  # get last image at first batch idx
        img = img.to("cpu").detach().numpy()
        img = ((img  * 55) + 33).clip(0, 255).astype('uint8')
        return img

    def update_stats(self, dict_name, metric_dict):
        """
        batch_metric_dict: (loss, loss_kl, loss_kl_exact,
         loss_kl_post,loss_kl_prior, loss_image,
         entropy_prior, entropy_post)
        """
        self.metric_dicts[dict_name] = metric_dict
        self.metric_dicts[dict_name]['epoch'] = self.step_dict['val']
        self.write_stats_to_log(dict_name)
        self.step_dict[dict_name] += 1

    def write_stats_to_log(self, dict_name):
        metric_dict = self.metric_dicts[dict_name]
        log_dict = {f"{dict_name}/{metric_name}": metric_value for metric_name,
                    metric_value in metric_dict.items()}
        wandb.log(log_dict, step=self.step_dict["train"])

    def log_imgs(self, cur_state, decoded_img, pred_img, dict_name):
        img1 = self.unnormalize(cur_state)
        img2 = self.unnormalize(decoded_img)
        img3 = self.unnormalize(pred_img)

        if img1.shape[0] == 3:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
            img3 = np.transpose(img3, (1, 2, 0))

        image1 = wandb.Image(img1, caption="orig")  # TODO TRANSPOSE SO THAT CHANNELS ARE LAST YAYA FUN TIMES
        image2 = wandb.Image(img2, caption="z reconst")
        image3 = wandb.Image(img3,  caption="z_hat reconst")

        images = [image1, image2, image3]
        wandb.log({f"{dict_name}/imgs": images}, step=self.step_dict["train"])
