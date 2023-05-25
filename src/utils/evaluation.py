from utils.utils import (
    has_int_squareroot,
    DeviceAwareDataLoader,
    get_date_str,
    sample_batched,
    dict_stats,
    handle_results_path
)
from torch.utils.data import Subset
from utils.logging import log, log_and_save_metrics
import torch
import math
from tqdm.auto import tqdm
from torchvision.utils import save_image
from ema_pytorch import EMA
import yaml
from collections import defaultdict
from diffusers.models.vae import DiagonalGaussianDistribution
from torchmetrics.image.fid import FrechetInceptionDistance
from models.encoder import Encoder
from utils.visualization import reconstruct_batched, visualize_reconstruction, visualize_sampling, get_tsne, get_pca, visualize_latent_manifold


class Evaluator:
    def __init__(
        self,
        diffusion_model,
        train_set,
        validation_set,
        config,
    ):
        self.cfg = config
        self.num_samples = config.num_samples
        self.n_sample_steps = config.num_sample_steps
        self.clip_samples = config.clip_samples
        self.device = config.device
        self.eval_batch_size = config.eval_batch_size
        self.n_samples_for_eval = config.n_samples_for_eval
        assert has_int_squareroot(
            self.num_samples
        ), "num_samples must have an integer sqrt"

        def make_dataloader(dataset, limit_size=None):
            # If limit_size is not None, only use a subset of the dataset
            if limit_size is not None:
                dataset = Subset(dataset, range(limit_size))
            return DeviceAwareDataLoader(
                dataset,
                self.eval_batch_size,
                device=self.device,
                shuffle=False,
                pin_memory=True,
                num_workers=config.num_workers,
                drop_last=True,
            )

        self.validation_dataloader = make_dataloader(validation_set)
        self.train_eval_dataloader = make_dataloader(train_set, len(validation_set))
        self.diffusion_model = diffusion_model.eval().to(self.device)
        # No need to set EMA parameters since we only use it for eval from checkpoint.
        self.ema = EMA(self.diffusion_model).to(self.device)
        self.ema.ema_model.eval()
        self.path = handle_results_path(config.results_path)
        self.eval_path = self.path / f"eval_{get_date_str()}"
        self.eval_path.mkdir(exist_ok=True)
        self.checkpoint_file = self.path / f"model.pt"
        with open(self.eval_path / "eval_config.yaml", "w") as f:
            eval_conf = {
                "n_sample_steps": self.n_sample_steps,
                "clip_samples": self.clip_samples,
                "n_samples_for_eval": self.n_samples_for_eval,
            }
            yaml.dump(eval_conf, f)
        self.load_checkpoint()

    def load_checkpoint(self):
        data = torch.load(self.checkpoint_file, map_location=self.device)
        log(f"Loading checkpoint '{self.checkpoint_file}'")
        self.diffusion_model.load_state_dict(data["model"])
        self.ema.load_state_dict(data["ema"])
        self.ckpt_step = data["step"]

    @torch.no_grad()
    def eval(self, cover_train_set=False):
        self.eval_model(self.diffusion_model, is_ema=False, cover_train_set=cover_train_set)
        self.eval_model(self.ema.ema_model, is_ema=True, cover_train_set=cover_train_set)

    def eval_model(self, model, *, is_ema, cover_train_set=False):
        log(f"\n *** Evaluating {'EMA' if is_ema else 'online'} model\n")
        self.sample_images(model, is_ema=is_ema)
        self.reconstruct_images(model, 0.4, 10, 250, 10, is_ema=is_ema)
        self.reconstruct_images(model, 0.6, 10, 250, 10, is_ema=is_ema)
        self.reconstruct_images(model, 0.8, 10, 250, 10, is_ema=is_ema)
        self.compute_latent_manifold('tsne', model.encoder, is_trained_encoder=True, is_ema=is_ema)
        self.compute_latent_manifold('pca', model.encoder, is_trained_encoder=True, is_ema=is_ema)
        random_encoder = model.encoder.clone_with_random_weights()
        self.compute_latent_manifold('tsne', random_encoder, is_trained_encoder=False, is_ema=is_ema)
        self.compute_latent_manifold('pca', random_encoder, is_trained_encoder=False, is_ema=is_ema)

        val = [True]
        if cover_train_set:
            val.append(False)

        for validation in val:
            evaluate_model_and_log(
                model,
                self.validation_dataloader
                if validation
                else self.train_eval_dataloader,
                self.eval_path / ("ema-metrics.jsonl" if is_ema else "metrics.jsonl"),
                "validation" if validation else "train",
                n=self.n_samples_for_eval,
                reduced_fid=False,
            )

    def sample_images(self, model, *, is_ema):
        samples = sample_batched(
            model,
            self.num_samples,
            self.eval_batch_size,
            self.n_sample_steps,
            self.clip_samples,
        )
        visualize_sampling(samples, self.eval_path, is_ema, self.ckpt_step)

    def reconstruct_images(self, model, noise_level, n_samples, n_recon_steps, n_recon_steps_retrieve, is_ema):
        orig, recon = reconstruct_batched(
            model,
            self.validation_dataloader,
            noise_level,
            n_samples,
            n_recon_steps,
            n_recon_steps_retrieve,
            self.device,
            self.clip_samples,
        )
        visualize_reconstruction(orig, recon, self.eval_path, noise_level, is_ema, self.ckpt_step)

    def compute_latent_manifold(self, method, encoder, is_trained_encoder, is_ema, n_points=10_000):
        if method == 'tsne':
            emb, lab = get_tsne(encoder, self.validation_dataloader, self.device, n_points)
        elif method == 'pca':
            emb, lab = get_pca(encoder, self.validation_dataloader, self.device, n_points)
        else:
            raise NotImplementedError(f'Unknown method {method}')
        visualize_latent_manifold(emb, lab, self.eval_path, method, is_trained_encoder, is_ema, self.ckpt_step)


def evaluate_model(model, dataloader):
    all_metrics = defaultdict(list)
    for batch in tqdm(dataloader, desc="evaluation"):
        loss, metrics = model(batch)
        for k, v in metrics.items():
            try:
                v = v.item()
            except AttributeError:
                pass
            all_metrics[k].append(v)
    return {k: sum(v) / len(v) for k, v in all_metrics.items()}  # average over dataset


def additional_metrics(model, dataloader, reduced_fid=True):
    def log_prob(sample, posterior):
        nll = posterior.nll(sample, dims=1)
        return -nll

    metrics = {}

    if model.encoder is not None:
        # Mutual information
        enc_out = []
        for batch in tqdm(dataloader, desc="mi"):
            x = batch[0]
            out = model.encoder(x)
            enc_out.append(out)
        enc_out = torch.cat(enc_out, dim=0)
        posterior = DiagonalGaussianDistribution(enc_out)
        samples = posterior.sample()
        log_probs = [log_prob(sample, posterior) for sample in samples]
        log_probs = torch.stack(log_probs, dim=0)
        M = log_probs.shape[0]
        mi = torch.tensor(
            [log_probs[i, i] - torch.sum(log_probs[i, :]) / M for i in range(M)]
        ).mean()
        metrics["mi"] = mi.item()

    # FID score
    if reduced_fid:
        n = int(math.ceil(200/dataloader.batch_size))  # over about 200 images
    else:
        n = len(dataloader)
    for i, batch in enumerate(dataloader):
        if i >= n:
            break

        x = batch[0]
        x_gen = model.sample(x.size(0), 100, False)  # 100 diffusion steps

        # convert from [0.,1.] to [0,255]
        x = (x.cpu() * 255).to(torch.uint8)
        x_gen = (x_gen.cpu() * 255).to(torch.uint8)
        if x.shape[1] == 1:
            # MNIST: repeat single bw channel to 3 rbg channels
            x = x.repeat(1, 3, 1, 1)
            x_gen = x_gen.repeat(1, 3, 1, 1)

        fid = FrechetInceptionDistance()
        fid.update(x, real=True)
        fid.update(x_gen, real=False)
    fid_score = fid.compute()
    metrics["fid"] = fid_score.item()

    return metrics


def evaluate_model_and_log(model, dataloader, filename, split, step=None, n=1, reduced_fid=True):
    # Call evaluate_model multiple times. Each call returns a dictionary of metrics, and
    # we then compute their average and standard deviation.
    if n > 1:
        log(f"\nRunning {n} evaluations to compute average metrics")
    metrics = dict_stats([evaluate_model(model, dataloader) for _ in range(n)])
    add_metrics = additional_metrics(model, dataloader, reduced_fid)
    metrics.update(add_metrics)
    log_and_save_metrics(metrics, split, step, filename)
