import argparse
from argparse import BooleanOptionalAction
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from utils.training import Trainer
from utils.logging import init_logger, print_model_summary
from utils.utils import make_cifar, make_mnist, Config, init_config_from_args
from models.vdm import VDM
from models.vdm_unet import UNetVDM
from models.encoder import Encoder


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--use-mnist", action=BooleanOptionalAction, default=True)

    # Architecture
    parser.add_argument("--use-encoder", action=BooleanOptionalAction, default=True)
    parser.add_argument("--w-dim", type=int, default=128)
    parser.add_argument("--block-out-channels", type=int, default=64)
    parser.add_argument("--layers-per-block", type=int, default=2)
    parser.add_argument("--norm-num-groups", type=int, default=32)

    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=32)
    parser.add_argument("--n-attention-heads", type=int, default=1)
    parser.add_argument("--dropout-prob", type=float, default=0.1)
    parser.add_argument("--norm-groups", type=int, default=8)
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument(
        "--use-fourier-features", action=BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--attention-everywhere", action=BooleanOptionalAction, default=False
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-schedule", type=str, default="fixed_linear")
    parser.add_argument("--gamma-min", type=float, default=-13.3)
    parser.add_argument("--gamma-max", type=float, default=5.0)
    parser.add_argument(
        "--antithetic-time-sampling", action=BooleanOptionalAction, default=True
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad-norm", action=BooleanOptionalAction, default=True)
    parser.add_argument("--encoder-loss-weight", type=float, default=1e4)

    parser.add_argument("--train-num-steps", type=int, default=10_000_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--data-path", type=str, default="data")
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--num-sample-steps", type=int, default=250)
    parser.add_argument("--clip-samples", action=BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--ema-update-every", type=int, default=1)
    parser.add_argument("--ema-power", type=float, default=0.75)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    args = parser.parse_args()
    return args


def get_datasets(cfg, args, accelerator):
    if cfg.use_mnist:
        cfg.input_channels = 1
        shape = (cfg.input_channels, 28, 28)
        with accelerator.local_main_process_first():
            train_set = make_mnist(
                train=True,
                download=accelerator.is_local_main_process,
                root_path=args.data_path,
            )
        validation_set = make_mnist(
            train=False, download=False, root_path=args.data_path
        )
    else:
        cfg.input_channels = 3
        shape = (cfg.input_channels, 32, 32)
        with accelerator.local_main_process_first():
            train_set = make_cifar(
                train=True,
                download=accelerator.is_local_main_process,
                root_path=args.data_path,
            )
        validation_set = make_cifar(
            train=False, download=False, root_path=args.data_path
        )
    return train_set, validation_set, shape
    


def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(split_batches=True)
    init_logger(accelerator)
    cfg = init_config_from_args(Config, args)

    train_set, validation_set, shape = get_datasets(cfg, args, accelerator)

    model = UNetVDM(cfg)
    encoder = Encoder(shape, cfg) if cfg.use_encoder else None
    diffusion = VDM(model, cfg, image_shape=train_set[0][0].shape, encoder=encoder)
    print_model_summary(
        model, batch_size=cfg.batch_size, shape=shape, w_dim=cfg.w_dim, encoder=encoder
    )

    Trainer(
        diffusion,
        train_set,
        validation_set,
        accelerator,
        make_opt=lambda params: torch.optim.AdamW(
            params, cfg.lr, betas=(0.9, 0.99), weight_decay=cfg.weight_decay, eps=1e-8
        ),
        config=cfg,
    ).train()


if __name__ == "__main__":
    main()
