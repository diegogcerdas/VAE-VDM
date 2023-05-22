import math
from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch
from datetime import datetime
from collections import defaultdict
import numpy as np
from torch import nn
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from utils.logging import log
from pathlib import Path
import yaml
import dataclasses


@dataclass
class Config:
    use_mnist: bool
    use_encoder: bool
    w_dim: int
    block_out_channels: int
    layers_per_block: int
    norm_num_groups: int
    embedding_dim: float
    n_blocks: int
    n_attention_heads: int
    dropout_prob: float
    norm_groups: int
    input_channels: int
    use_fourier_features: bool
    attention_everywhere: bool
    batch_size: int
    noise_schedule: str
    gamma_min: float
    gamma_max: float
    antithetic_time_sampling: bool
    lr: float
    weight_decay: float
    clip_grad_norm: bool
    encoder_loss_weight: float
    num_samples: int
    eval_every: int
    train_num_steps: int
    num_sample_steps: int
    clip_samples: bool
    num_workers: int
    results_path: str
    ema_decay: float
    ema_update_every: int
    ema_power: float
    resume: bool
    seed: int
    device: str
    eval_batch_size: int
    n_samples_for_eval: int


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def sample_batched(model, num_samples, batch_size, n_sample_steps, clip_samples):
    samples = []
    for i in range(0, num_samples, batch_size):
        corrected_batch_size = min(batch_size, num_samples - i)
        samples.append(model.sample(corrected_batch_size, n_sample_steps, clip_samples))
    return torch.cat(samples, dim=0)


def get_date_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class DeviceAwareDataLoader(DataLoader):
    """A DataLoader that moves batches to a device. If device is None, it is equivalent to a standard DataLoader."""

    def __init__(self, *args, device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def __iter__(self):
        for batch in super().__iter__():
            yield self.move_to_device(batch)

    def move_to_device(self, batch):
        if self.device is None:
            return batch
        if isinstance(batch, (tuple, list)):
            return [self.move_to_device(x) for x in batch]
        elif isinstance(batch, dict):
            return {k: self.move_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            return batch


def dict_stats(dictionaries: list[dict]) -> dict:
    """Computes the average and standard deviation of metrics in a list of dictionaries.

    Args:
        dictionaries: A list of dictionaries, where each dictionary contains the same keys,
            and the values are numbers.

    Returns:
        A dictionary of the same keys as the input dictionaries, with the average and
        standard deviation of the values. If the list has length 1, the original dictionary
        is returned instead.
    """
    if len(dictionaries) == 1:
        return dictionaries[0]

    # Convert the list of dictionaries to a dictionary of lists.
    lists = defaultdict(list)
    for d in dictionaries:
        for k, v in d.items():
            lists[k].append(v)

    # Compute the average and standard deviation of each list.
    stats = {}
    for k, v in lists.items():
        stats[f"{k}_avg"] = np.mean(v)
        stats[f"{k}_std"] = np.std(v)
    return stats


@torch.no_grad()
def zero_init(module: nn.Module) -> nn.Module:
    """Sets to zero all the parameters of a module, and returns the module."""
    for p in module.parameters():
        nn.init.zeros_(p.data)
    return module


def maybe_unpack_batch(batch):
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch
    else:
        return batch, None


def make_cifar(*, train, download, root_path):
    return CIFAR10(
        root=root_path,
        download=download,
        train=train,
        transform=transforms.Compose([transforms.ToTensor()]),
    )


def make_mnist(*, train, download, root_path):
    return MNIST(
        root=root_path,
        download=download,
        train=train,
        transform=transforms.Compose([transforms.ToTensor()]),
    )


def handle_results_path(res_path: str, default_root: str = "./results") -> Path:
    """Sets results path if it doesn't exist yet."""
    if res_path is None:
        results_path = Path(default_root) / get_date_str()
    else:
        results_path = Path(res_path)
    log(f"Results will be saved to '{results_path}'")
    return results_path


def unsqueeze_right(x, num_dims=1):
    """Unsqueezes the last `num_dims` dimensions of `x`."""
    return x.view(x.shape + (1,) * num_dims)


def init_config_from_args(cls, args):
    """Initializes a dataclass from a Namespace, ignoring unknown fields."""
    return cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls)})


def load_config_from_yaml(cls, args):
    with open(Path(args.results_path) / "config.yaml", "r") as f:
        cfg = cls(**yaml.safe_load(f))
    return cfg


def check_config_matches_checkpoint(config, checkpoint_path):
    with open(checkpoint_path / "config.yaml", "r") as f:
        ckpt_config = yaml.safe_load(f)
    config = dataclasses.asdict(config)
    if config != ckpt_config:
        config_str = "\n    ".join(f"{k}: {config[k]}" for k in sorted(config))
        ckpt_str = "\n    ".join(f"{k}: {ckpt_config[k]}" for k in sorted(ckpt_config))
        raise ValueError(
            f"Config mismatch:\n\n"
            f"> Config:\n    {config_str}\n\n"
            f"> Checkpoint:\n    {ckpt_str}\n\n"
        )
