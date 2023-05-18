# VAE-VDM: Representation Learning with Variational Diffusion Models

## Setup

The environment can be set up with `requirements.txt`. For example with conda:

```
conda create --name vdm python=3.9
conda activate vdm
pip install -r requirements.txt
```


## Training

To train with default parameters and options:

```bash
python src/train.py --use-encoder
```

Append `--resume` to the command above to resume training from the latest checkpoint. 
See [`train.py`](train.py) for more training options.


## Evaluating from checkpoint

```bash
python src/eval.py --results-path results/my_experiment/ --n-sample-steps 1000
```


## Acknowledgment

The code is based on [this repo](https://github.com/addtt/variational-diffusion-models), a PyTorch implementation of [Variational Diffusion Models](https://arxiv.org/abs/2107.00630). The official implementation in JAX can be found [here](https://github.com/google-research/vdm).
