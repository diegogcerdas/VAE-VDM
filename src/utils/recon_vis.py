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
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def load_checkpoint(diffusion_model, checkpoint_file, device):
    data = torch.load(checkpoint_file, map_location=device)
    diffusion_model.load_state_dict(data["model"])
def vis_reconstruction(image, model, save_path=None):
    forward_output, output, noise_level = model.forward(image, return_reconstruction=True)
    #plot with image on left and output on right
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image.squeeze(0).permute(1, 2, 0))
    axs[0].set_title('Ground truth')
    axs[1].imshow(forward_output.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    axs[1].set_title(f'Forward pass, noise level:{noise_level.item():.2f}')
    axs[2].imshow(output.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    axs[2].set_title('Reconstruction')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def vis_reconstruct_grid(data_loader, model, grid_size=10, noise_level=0.7):
    original_images = []
    reconstructions = []
    for i, (image, _) in enumerate(data_loader):
        if i == grid_size**2:
            break
        output, _, _ = model.forward(image.unsqueeze(0), return_reconstruction=True, times=noise_level)
        original_images.append(image)
        reconstructions.append(output)

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    for i in range(grid_size):
        for j in range(grid_size):
            axs[i, j].imshow(original_images[i * grid_size + j].permute(1, 2, 0))
            axs[i, j].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    fig.suptitle('Original images', y=0.95, fontsize=16)

    canvas = FigureCanvas(fig)
    canvas.draw()
    # Convert the rendered figure to a PIL image
    original_plot = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    for i in range(grid_size):
        for j in range(grid_size):
            axs[i, j].imshow(reconstructions[i * grid_size + j].squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
            axs[i, j].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    fig.suptitle('Reconstructions', y=0.95, fontsize=16)

    canvas = FigureCanvas(fig)
    canvas.draw()
    # Convert the rendered figure to a PIL image
    reconstruction_plot = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    final_plot = get_concat_h(original_plot, reconstruction_plot)
    final_plot.save('reconstructions.png')
    final_plot.show()



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
    load_checkpoint(diffusion, "model.pt", device)
    samples = diffusion.sample_same_z(5, 3, 1)
    for s in samples:
        s = s.permute(1, 2, 0).cpu().detach().numpy()
        plt.imshow(s)
        plt.show()



    #vis_reconstruct_grid(validation_set, diffusion)

    # for i in range(10):
    #     image = validation_set[i][0].unsqueeze(0)
    #     vis_reconstruction(image, diffusion)

if __name__ == "__main__":
    #check if gpu is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)
    main()
