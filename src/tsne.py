from pathlib import Path
import argparse

import torch
import numpy as np
from sklearn.manifold import TSNE
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
        n_samples_for_eval=1
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


def fit_tsne(embeddings, perplexity=40, n_iter=300):
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter)
    tsne_embeddings = tsne.fit_transform(embeddings)
    return tsne_embeddings


def plot_tsne(tsne_embeddings, labels, title, save_path=None):
    plt.figure(figsize=(10, 10))
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=labels, cmap='tab10')
    plt.title(title)
    plt.colorbar()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def generate_tsne_plot(enc, dset, device, n_points=10_000):
    enc_emb, enc_labels = get_embeddings(enc, dset, device, n_points=n_points)
    enc_tsne = fit_tsne(enc_emb)
    path = args.results_path / 'encoder_tsne-{}.png'.format(step)
    plot_tsne(enc_tsne, enc_labels, 'Encoder', save_path=path)
    print('saved at {}'.format(path))


if __name__ == '__main__':
    # generate trained random encoder vs trained encoder tsne plots for latest checkpoint
    args, enc, renc, vset, step = setup()
    print('[SETUP] done')

    # random encoder
    renc_emb, renc_labels = get_embeddings(renc, vset, args.device)
    print('[RANDOM] done getting embeddings')
    renc_tsne = fit_tsne(renc_emb)
    print('[RANDOM] done fitting tsne')
    plot_tsne(renc_tsne, renc_labels, 'Random Encoder', save_path=args.results_path / 'random_encoder_tsne.png')
    print('[RANDOM] saved tsne plot')

    # trained encoder
    enc_emb, enc_labels = get_embeddings(enc, vset, args.device)
    print('[TRAINED] done getting embeddings')
    enc_tsne = fit_tsne(enc_emb)
    print('[TRAINED] done fitting tsne')
    plot_tsne(enc_tsne, enc_labels, 'Encoder', save_path=args.results_path / 'encoder_tsne-{step}.png'.format(step=step))
    print('[TRAINED] saved tsne plot')