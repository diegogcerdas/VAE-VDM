import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from argparse import BooleanOptionalAction
import yaml
from accelerate.utils import set_seed
from utils.logging import print_model_summary
from utils.utils import make_cifar, make_mnist, Config
from models.vdm import VDM
from models.vdm_unet import UNetVDM
from models.encoder import Encoder
import torch
from eval import Evaluator


def load_checkpoint(diffusion_model, checkpoint_file, device):
    data = torch.load(checkpoint_file, map_location=device)
    diffusion_model.load_state_dict(data["model"])
def vis_reconstruction(image, model, save_path=None):
    output = model.forward(image, return_reconstruction=True)
    #plot with image on left and output on right
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image.squeeze(0).permute(1, 2, 0))
    axs[0].set_title('Ground truth')
    axs[1].imshow(output.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    axs[1].set_title('Reconstruction1')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--data-path", type=str, default="data")
    parser.add_argument("--results-path", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
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
        validation_set = make_mnist(
            train=False, download=False, root_path=args.data_path
        )
    else:
        cfg.input_channels = 3
        shape = (cfg.input_channels, 32, 32)
        validation_set = make_cifar(
            train=False, download=False, root_path=args.data_path
        )

    model = UNetVDM(cfg)
    encoder = Encoder(shape, cfg) if cfg.use_encoder else None
    diffusion = VDM(model, cfg, image_shape=shape, encoder=encoder)
    print_model_summary(
        model, batch_size=None, shape=shape, w_dim=cfg.w_dim, encoder=encoder
    )
    load_checkpoint(diffusion, "model.pt", device)
    for i in range(3):
        image = validation_set[i][0].unsqueeze(0)
        vis_reconstruction(image, diffusion)

if __name__ == "__main__":
    #check if gpu is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)
    main()
