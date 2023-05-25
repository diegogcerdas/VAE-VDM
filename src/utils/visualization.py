import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import os
import json
import pandas as pd

from models.encoder import Encoder

from diffusers.models.vae import DiagonalGaussianDistribution
from tqdm import trange
import math

from torchvision.utils import save_image


# Latent: t-SNE, PCA

def get_latent_embeddings(encoder, dataloader, device, n_points=10_000):
    """
    Performs inference on encoder to get the output of the encoder for each image in the dataset.

    Args:
        encoder: the encoder to use
        dataloader: the dataloader to use
        device: the device to put the encoder on
        n_points: the number of points to get embeddings for, same as the number of images that will be loaded from the dataloader

    Returns:
        embeddings: the embeddings of the dataset, numpy array of shape (n_points, embedding_dim)
        labels: the labels of the dataset, numpy array of shape (n_points,)
    """
    encoder = encoder.to(device)

    training = encoder.training
    encoder.eval()

    n_points = min(n_points, len(dataloader.dataset))
    embeddings = []
    labels = []
    encoder = encoder.to(device)
    for x, y in dataloader:
        if len(embeddings) >= n_points:
            break
        x = x.to(device)

        enc_out = encoder(x)
        w = torch.chunk(enc_out, 2, dim=1)[0]  # mean

        embeddings.append(w.squeeze().detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    embeddings = embeddings[:n_points]
    labels = labels[:n_points]

    encoder.train(training)
    return np.array(embeddings), np.array(labels)


def get_tsne(encoder, dataloader, device, n_points=10_000, perplexity=40, n_iter=300):
    """
    Computes the t-SNE embedding of the latent space of the encoder.

    Args:
        encoder: the encoder to use
        dataloader: the dataloader to use
        device: the device to put the encoder and data on
        n_points: the number of points to get embeddings for, same as the number of images that will be loaded from the dataloader
        perplexity: the perplexity to use for t-SNE
        n_iter: the number of iterations to use for t-SNE

    Returns:
        tsne_emb: the t-SNE embeddings of the latent space of the encoder, numpy array of shape (n_points, 2)
        labels: the labels of the dataset, numpy array of shape (n_points,)
    """
    
    emb, labels = get_latent_embeddings(encoder, dataloader, device, n_points=n_points)
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter)
    tsne_emb = tsne.fit_transform(emb)
    return tsne_emb, labels

def get_pca(encoder, dataloader, device, n_points=10_000):
    """
    Computes the PCA embedding of the latent space of the encoder.

    Args:
        encoder: the encoder to use
        dataloader: the dataloader to use
        device: the device to put the encoder and data on
        n_points: the number of points to get embeddings for, same as the number of images that will be loaded from the dataloader

    Returns:
        pca_emb: the PCA embeddings of the latent space of the encoder, numpy array of shape (n_points, 2)
        labels: the labels of the dataset, numpy array of shape (n_points,)
    """
    emb, labels = get_latent_embeddings(encoder, dataloader, device, n_points=n_points)
    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(emb)
    return pca_emb, labels


def visualize_latent_manifold(reduced_emb, labels, save_path, method, is_trained_encoder, is_ema, ckpt_step=0):
    """
    Visualizes the latent manifold.

    Args:
        reduced_emb: the reduced embeddings of the subspace or manifold
        labels: the labels of the images each embedding corresponds to
        save_path: the path to save the image
        method: the method used to reduce the dimensionality of the latent space, one of ['tsne', 'pca']
        is_trained_encoder: whether the encoder is trained or random
        is_ema: whether the encoder is the EMA encoder
        ckpt_step: the checkpoint step of the trained encoder

    Returns:
        None
    """
    assert method in ['tsne', 'pca'], f"method must be one of ['tsne', 'pca'], got {method}"
    title = f"{'t-SNE' if method=='tsne' else 'PCA'} {'Trained' if is_trained_encoder else 'Random'} Encoder {'EMA' if is_ema else ''} {f'(Checkpoint Step {ckpt_step})'if is_trained_encoder else ''})"
    path = save_path / f"{method.lower()}_{'trained' if is_trained_encoder else 'random'}{'-ema' if is_ema else ''}{f'-{ckpt_step}' if is_trained_encoder else ''}.png"
    
    plt.figure(figsize=(10, 10))
    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=labels, cmap='tab10')
    plt.title(title)
    plt.colorbar()
    plt.savefig(path)
    print('saved at {}'.format(path))


# Reconstruction: Original, Reconstructed

def reconstruct_batched(vdm_model, dataloader, noise_level, n_samples, n_recon_steps, n_recon_steps_retrieve, device, clip_samples):
    """
    Reconstructs the images in the dataset.

    Args:
        vdm_model: The VDM model.
        dataloader: The dataloader from which to reconstruct.
        noise_level: The noise level to start the reconstruction from.
        n_samples: The number of samples from the dataset to reconstruct.
        n_recon_steps: The number of time steps to sample.
        n_recon_steps_retrieve: The number of n_sample_steps that will be retrieved, evenly spaced.
        device: The device to use.
        clip_samples: Whether to clip the samples.

    Returns:
        orig: The original images. (n_samples, C, H, W)
        recon: The reconstructed images. For each original image, there is an array with n_sample_steps_retrieve reconstructions steps. (n_samples, n_recon_steps_retrieve, C, H, W)
    """

    samples_so_far = 0
    orig = []
    recon = []

    for x, _ in dataloader:

        if samples_so_far >= n_samples:
            break

        x = x.to(device)
        samples_so_far += x.shape[0]

        orig.append(x.detach().cpu().numpy())

        times = torch.ones(x.shape[0], device=device) * noise_level
        x_t, _ = vdm_model.sample_q_t_0(x, times=times)

        enc_out = vdm_model.encoder(x)
        posterior = DiagonalGaussianDistribution(enc_out)
        w = posterior.sample()

        steps = torch.linspace(noise_level, 0.0, n_recon_steps + 1, device=device)
        # choose n_sample_steps_retrieve evenly spaced (approx) steps to retrieve
        retrieve_idx = np.linspace(0, n_recon_steps, n_recon_steps_retrieve + 1, dtype=int)

        recon.append([])
        z = x_t
        for t in trange(n_recon_steps, desc=f'reconstructing'):
            z = vdm_model.sample_p_s_t(z, steps[t], steps[t + 1], clip_samples, w=w)
            if t in retrieve_idx:
                recon[-1].append(z.detach().cpu().numpy())

    recon = np.asarray(recon).swapaxes(1, 2)
    recon = np.concatenate(recon, axis=0)
    orig = np.concatenate(orig, axis=0)
    recon = recon[:n_samples]
    orig = orig[:n_samples]
    orig = torch.from_numpy(orig)
    recon = torch.from_numpy(recon)

    return orig, recon


def visualize_reconstruction(original, reconstructed, save_path, noise_level, is_ema, ckpt_step):
    """
    Visualizes the reconstruction of the model.
    """
    images = torch.cat([original.unsqueeze(1), reconstructed], dim=1).flatten(0, 1)
    path = save_path / f"recon_{noise_level}{'-ema' if is_ema else ''}-{ckpt_step}.png"
    save_image(images, str(path), nrow=reconstructed.shape[1] + 1)


# Sampling

def visualize_sampling(samples, save_path, is_ema, step):
    path = save_path / f"sample-{'ema-' if is_ema else ''}{step}.png"
    save_image(samples, str(path), nrow=int(math.sqrt(samples.shape[0])))

def parse_metrics_to_csv(results_path):
    filename = os.path.join(results_path, 'metrics_log.jsonl')
    assert os.path.exists(filename), f'File {filename} does not exist.'
    metrics = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for l in lines:
            mydict = json.loads(l)
            step = mydict['step']
            stage = mydict['set']
            metrics.setdefault(step, {})
            for key, val in mydict.items():
                if key not in ['step', 'set']:
                    metrics[step][f'{stage}_{key}'] = val
    data = pd.DataFrame.from_dict(metrics, orient='index') 
    data.insert(0, 'step', data.index)
    data.to_csv(os.path.join(results_path, 'metrics.csv'), index=False)
    return data

def generate_metrics_plots(results_path):
    filename = os.path.join(results_path, 'metrics.csv')
    assert os.path.exists(filename), f'File {filename} does not exist.'
    data = pd.read_csv(filename)
    plots = [c.replace('train_', '').replace('validation_', '') for c in data.columns if c != 'step']
    for p in plots:
        train = data[f'train_{p}']
        val = data[f'validation_{p}']
        f = plt.figure(figsize=(6, 3))
        plt.plot(train, label='train')
        plt.plot(val, '--', label='validation')
        plt.title(p)
        plt.legend()
        plt.savefig(os.path.join(results_path, f'metric_{p}.png'))
        plt.close(f)