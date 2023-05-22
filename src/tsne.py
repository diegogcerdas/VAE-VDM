from pathlib import Path
import argparse

import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from accelerate.utils import set_seed

from eval import get_datasets
from utils.utils import load_config_from_yaml, Config
from models.vdm import VDM
from models.vdm_unet import UNetVDM
from models.encoder import Encoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", type=str, default="data")
    parser.add_argument("--results-path", type=str, required=True)
    parser.add_argument("--mode", type=str, default='tsne', help="tsne or pca")
    parser.add_argument("--n-points", type=int, default=10_000)
    args = parser.parse_args()
    return args


def load_checkpoint(path, model, device):
    data = torch.load(path, map_location=device)
    model.load_state_dict(data['model'])
    return data['step']


def setup():
    args = parse_args()
    args = argparse.Namespace(
        batch_size=128,
        seed=args.seed,
        data_path=Path(args.data_path),
        results_path=Path(args.results_path),
        num_workers=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_sample_steps=250,
        clip_samples=True,
        n_samples_for_eval=1,
        mode=args.mode,
        n_points=args.n_points,
    )
    cfg = load_config_from_yaml(Config, args)

    set_seed(args.seed)
    train_set, validation_set, shape = get_datasets(cfg, args)

    assert cfg.use_encoder, "Encoder must be used for this experiment"

    model = UNetVDM(cfg)
    encoder = Encoder(shape, cfg)
    diffusion = VDM(model, cfg, image_shape=shape, encoder=encoder)

    step = load_checkpoint(args.results_path / 'model.pt', diffusion, args.device)

    random_encoder = Encoder(shape, cfg)

    encoder = encoder.to(args.device)
    random_encoder = random_encoder.to(args.device)
    
    return args, encoder, random_encoder, validation_set, step


def get_embeddings(encoder, dataset, device, n_points=10_000):
    n_points = n_points if n_points > 0 else len(dataset)
    n_points = min(n_points, len(dataset))
    embeddings = []
    labels = []
    for i in range(n_points):
        x, y = dataset[i]
        x = x.to(device)
        z = encoder(x.unsqueeze(0))
        embeddings.append(z.squeeze().detach().cpu().numpy())
        labels.append(y)
    return np.array(embeddings), np.array(labels)


def fit_manifold(embeddings, perplexity=40, n_iter=300, mode='tsne'):
    if mode == 'tsne':
        fitter = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter)
    elif mode == 'pca':
        fitter = PCA(n_components=2)
    else:
        raise ValueError("mode must be one of ['tsne', 'pca']")
    tsne_embeddings = fitter.fit_transform(embeddings)
    return tsne_embeddings


def plot_manifold(manifold_embeddings, labels, title, save_path=None):
    plt.figure(figsize=(10, 10))
    plt.scatter(manifold_embeddings[:, 0], manifold_embeddings[:, 1], c=labels, cmap='tab10')
    plt.title(title)
    plt.colorbar()
    if save_path is not None:
        plt.savefig(save_path)
        print('saved at {}'.format(save_path))
    plt.show()


def generate_tsne_plot(enc, dset, device, n_points=10_000, step=0, mode='tsne'):
    enc_emb, enc_labels = get_embeddings(enc, dset, device, n_points=n_points)
    enc_manifold = fit_manifold(enc_emb, mode=mode)
    path = args.results_path / 'encoder_{mode}-{step}.png'.format(mode=mode, step=step)
    plot_manifold(enc_manifold, enc_labels, 'Encoder', save_path=path)


if __name__ == '__main__':
    args, enc, renc, vset, step = setup()
    print('[SETUP] done')

    # random encoder
    print('[RANDOM] start')
    renc_emb, renc_labels = get_embeddings(renc, vset, args.device, n_points=args.n_points)
    print('[RANDOM] done getting embeddings')
    renc_manifold = fit_manifold(renc_emb, mode=args.mode)
    print('[RANDOM] done fitting manifold')
    title = 'Random Encoder ({mode})'.format(mode=args.mode)
    plot_manifold(renc_manifold, renc_labels, title, save_path=args.results_path / 'random_encoder_{mode}.png'.format(mode=args.mode))

    # trained encoder
    print('[TRAINED] start')
    enc_emb, enc_labels = get_embeddings(enc, vset, args.device, n_points=args.n_points)
    print('[TRAINED] done getting embeddings')
    renc_manifold = fit_manifold(enc_emb, mode=args.mode)
    print('[TRAINED] done fitting manifold')
    title = 'Trained Encoder ({mode})'.format(mode=args.mode)
    plot_manifold(renc_manifold, enc_labels, title, save_path=args.results_path / 'encoder_{mode}-{step}.png'.format(mode=args.mode, step=step))
