## Setup

The environment can be set up with `requirements.txt`. For example with conda:

```
conda create --name vdm python=3.9
conda activate vdm
pip install -r requirements.txt
```


## Training with  🤗 Accelerate

To train with default parameters and options:

```bash
accelerate launch --config_file accelerate_config.yaml train.py --results-path results/my_experiment/
```

Append `--resume` to the command above to resume training from the latest checkpoint. 
See [`train.py`](train.py) for more training options.

Here we provide a sensible configuration for training on 2 GPUs in the file 
[`accelerate_config.yaml`](accelerate_config.yaml). This can be modified directly, or overridden 
on the command line by adding flags before "`train.py`" (e.g., `--num_processes N` 
to train on N GPUs).
See the [Accelerate docs](https://huggingface.co/docs/accelerate/index) for more configuration options.
After initialization, we print an estimate of the required GPU memory for the given 
batch size, so that the number of GPUs can be adjusted accordingly.
The training loop periodically logs train and validation metrics to a JSONL file,
and generates samples.


## Evaluating from checkpoint

```bash
python eval.py --results-path results/my_experiment/ --n-sample-steps 1000
```


## Credits

The code is based on [this repo](https://github.com/addtt/variational-diffusion-models), which at the same time is based on VDM [paper](https://arxiv.org/abs/2107.00630) and [official code](https://github.com/google-research/vdm).
