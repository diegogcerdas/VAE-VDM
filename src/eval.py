import argparse
from pathlib import Path
from argparse import BooleanOptionalAction
import yaml
from accelerate.utils import set_seed
from utils.logging import print_model_summary
from utils.utils import make_cifar, make_mnist, Config
from utils.evaluation import Evaluator
from models.vdm import VDM
from models.vdm_unet import UNetVDM
from models.encoder import Encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--data-path", type=str, default="data")
    parser.add_argument("--results-path", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-sample-steps", type=int, default=250)
    parser.add_argument("--clip-samples", action=BooleanOptionalAction, default=True)
    parser.add_argument("--n-samples-for-eval", type=int, default=1)
    args = parser.parse_args()
    set_seed(args.seed)

    # Load config from YAML.
    with open(Path(args.results_path) / "config.yaml", "r") as f:
        cfg = Config(**yaml.safe_load(f))

    if cfg.use_mnist:
        cfg.input_channels = 1
        shape = (cfg.input_channels, 28, 28)
        train_set = make_mnist(train=True, download=True, root_path=args.data_path)
        validation_set = make_mnist(
            train=False, download=False, root_path=args.data_path
        )
    else:
        cfg.input_channels = 3
        shape = (cfg.input_channels, 32, 32)
        train_set = make_cifar(train=True, download=True, root_path=args.data_path)
        validation_set = make_cifar(
            train=False, download=False, root_path=args.data_path
        )

    model = UNetVDM(cfg)
    encoder = Encoder(shape, cfg) if cfg.use_encoder else None
    diffusion = VDM(model, cfg, image_shape=shape, encoder=encoder)
    print_model_summary(
        model, batch_size=None, shape=shape, w_dim=cfg.w_dim, encoder=encoder
    )

    Evaluator(
        diffusion,
        train_set,
        validation_set,
        config=cfg,
    ).eval()


if __name__ == "__main__":
    main()
