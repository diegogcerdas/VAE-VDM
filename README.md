# VAE-VDM: Representation Learning with Variational Diffusion Models

This repo contains the code for the project "Representation Learning with Variational Diffusion Models" for the course [Deep Learning 2](https://uvadl2c.github.io/) at the University of Amsterdam.

In [this blogpost](blogpost.md) we present the approach and discuss the results.
Additional material can be found in the [supplement](Supplement.pdf).


## Setup

Clone the repository:

```bash
git clone https://github.com/diegogcerdas/VAE-VDM.git
cd VAE-VDM
```

The environment can be set up with `requirements.txt`. For example, with conda:

```
conda create --name vdm python=3.9
conda activate vdm
pip install -r requirements.txt
```


## Training

To train with default parameters and options:

```bash
python src/train.py --results-path results/my_experiment/ --use-encoder
```

Append `--resume` to the command above to resume training from the latest checkpoint.

> :bulb: Tips:
> - You might need to reduce `batch-size` due to memory limits.
> - You also might want to decrease `eval-every` to do evaluation every less steps and save checkpoints.
> - Use `train-num-steps` to train for a specific number of steps.
> 
> See [`train.py`](src/train.py) for more training options.

### Reproducing the experiments

Find the scripts to reproduce the experiments in the paper in [`scripts/`](scripts/):

```bash
sh scripts/TVDM_OE.sh
```


## Evaluating from checkpoint

```bash
python src/eval.py --results-path results/my_experiment/ --n-sample-steps 1000
```

For each evaluation, a new folder is created in `results/my_experiment/eval_<timestamp>/` with the current timestamp.
Evaluation metrics and visualizations are saved in this folder.


## Acknowledgment

The code is based on [this repo](https://github.com/addtt/variational-diffusion-models), a PyTorch implementation of [Variational Diffusion Models](https://arxiv.org/abs/2107.00630). The official implementation in JAX can be found [here](https://github.com/google-research/vdm).
