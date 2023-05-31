from utils.utils import (
    has_int_squareroot,
    DeviceAwareDataLoader,
    get_date_str,
    check_config_matches_checkpoint,
    sample_batched,
    handle_results_path,
    save_config,
)
from utils.logging import log
from utils.evaluation import evaluate_model_and_log
from ema_pytorch import EMA
from torch.utils.data import Subset
import yaml
import dataclasses
import torch
import math
from tqdm.auto import tqdm
from utils.visualization import visualize_sampling


class Trainer:
    def __init__(
        self,
        diffusion_model,
        train_set,
        validation_set,
        accelerator,
        make_opt,
        config,
    ):
        super().__init__()
        
        self.num_samples = config.num_samples
        self.save_and_eval_every = config.eval_every
        self.cfg = config
        self.train_num_steps = config.train_num_steps
        self.n_sample_steps = config.num_sample_steps
        self.clip_samples = config.clip_samples
        self.accelerator = accelerator
        self.step = 0
        assert has_int_squareroot(config.num_samples), "num_samples must have an integer sqrt"

        def make_dataloader(dataset, limit_size=None, *, train=False):
            if limit_size is not None:
                dataset = Subset(dataset, range(limit_size))
            dataloader = DeviceAwareDataLoader(
                dataset,
                config.batch_size,
                shuffle=train,
                pin_memory=True,
                num_workers=config.num_workers,
                drop_last=True,
                device=accelerator.device if not train else None,  # None -> standard DL
            )
            if train:
                dataloader = accelerator.prepare(dataloader)
            return dataloader

        self.train_dataloader = cycle(make_dataloader(train_set, train=True))
        self.validation_dataloader = make_dataloader(validation_set)
        self.train_eval_dataloader = make_dataloader(train_set, len(validation_set))

        self.path = handle_results_path(config.results_path)
        self.checkpoint_file = self.path / f"model.pt"
        if accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model.to(accelerator.device),
                beta=config.ema_decay,
                update_every=config.ema_update_every,
                power=config.ema_power,
            )
            self.ema.ema_model.eval()
            self.path.mkdir(exist_ok=True, parents=True)
        self.diffusion_model = accelerator.prepare(diffusion_model)
        self.opt = accelerator.prepare(make_opt(self.diffusion_model.parameters()))
        if config.resume:
            self.load_checkpoint()
        else:
            if len(list(self.path.glob("*.pt"))) > 0:
                raise ValueError(f"'{self.path}' contains checkpoints but resume=False")
            if accelerator.is_main_process:
                save_config(config, self.path)

    def save_checkpoint(self):
        tmp_file = self.checkpoint_file.with_suffix(f".tmp.{get_date_str()}.pt")
        if self.checkpoint_file.exists():
            self.checkpoint_file.rename(tmp_file)  # Rename old checkpoint to temp file
        checkpoint = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.diffusion_model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_file)
        tmp_file.unlink(missing_ok=True)  # Delete temp file

    def load_checkpoint(self):
        check_config_matches_checkpoint(self.cfg, self.path)
        data = torch.load(self.checkpoint_file, map_location=self.accelerator.device)
        self.step = data["step"]
        log(f"Resuming from checkpoint '{self.checkpoint_file}' (step {self.step})")
        model = self.accelerator.unwrap_model(self.diffusion_model)
        model.load_state_dict(data["model"])
        self.opt.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

    def train(self):
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            while self.step < self.train_num_steps:
                data = next(self.train_dataloader)
                self.opt.zero_grad()
                loss, _ = self.diffusion_model(data)
                self.accelerator.backward(loss)
                if self.cfg.clip_grad_norm:
                    self.accelerator.clip_grad_norm_(
                        self.diffusion_model.parameters(), 1.0
                    )
                self.opt.step()
                pbar.set_description(f"loss: {loss.item():.4f}")
                self.step += 1
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.ema.update()
                    if self.step % self.save_and_eval_every == 0:
                        self.eval()
                pbar.update()

    @torch.no_grad()
    def eval(self):
        self.sample_images(self.ema.ema_model, is_ema=True)
        self.sample_images(self.diffusion_model, is_ema=False)
        self.evaluate_ema_model_and_log(validation=True)
        self.evaluate_ema_model_and_log(validation=False)
        self.save_checkpoint()

    def evaluate_ema_model_and_log(self, *, validation):
        evaluate_model_and_log(
            self.ema.ema_model,
            self.validation_dataloader if validation else self.train_eval_dataloader,
            self.path / "metrics_log.jsonl",
            "validation" if validation else "train",
            self.step,
        )

    def sample_images(self, model, *, is_ema):
        train_state = model.training
        model.eval()
        samples = sample_batched(
            self.accelerator.unwrap_model(model),
            self.num_samples,
            self.cfg.batch_size,
            self.n_sample_steps,
            self.clip_samples,
        )
        visualize_sampling(samples, self.path, is_ema, self.step)
        model.train(train_state)


def cycle(dl):
    # We don't use itertools.cycle because it caches the entire iterator.
    while True:
        for data in dl:
            yield data
