import json
import warnings
from typing import Optional
import torchinfo
from accelerate import Accelerator


def print_model_summary(
    model, *, batch_size, shape, w_dim, depth=4, batch_size_torchinfo=1, encoder=None
):
    if encoder is None:
        input = [(batch_size_torchinfo, *shape), (batch_size_torchinfo,)]
    else:
        input = [
            (batch_size_torchinfo, *shape),
            (batch_size_torchinfo,),
            (batch_size_torchinfo, w_dim),
        ]
    summary = torchinfo.summary(
        model,
        input,
        depth=depth,
        col_names=["input_size", "output_size", "num_params"],
        verbose=0,  # quiet
    )
    log(summary)
    if batch_size is None or batch_size == batch_size_torchinfo:
        return
    output_bytes_large = summary.total_output_bytes / batch_size_torchinfo * batch_size
    total_bytes = summary.total_input + output_bytes_large + summary.total_param_bytes
    log(
        f"\n--- With batch size {batch_size} ---\n"
        f"Forward/backward pass size: {output_bytes_large / 1e9:0.2f} GB\n"
        f"Estimated Total Size: {total_bytes / 1e9:0.2f} GB\n"
        + "=" * len(str(summary).splitlines()[-1])
        + "\n"
    )
    log("{:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))


def log_and_save_metrics(avg_metrics, dataset_split, step, filename):
    log(f"\n{dataset_split} metrics:")
    for k, v in avg_metrics.items():
        log(f"    {k}: {v}")

    avg_metrics = {"step": step, "set": dataset_split, **avg_metrics}
    with open(filename, "a") as f:
        json.dump(avg_metrics, f)
        f.write("\n")


_accelerator: Optional[Accelerator] = None


def init_logger(accelerator: Accelerator):
    global _accelerator
    if _accelerator is not None:
        raise ValueError("Accelerator already set")
    _accelerator = accelerator


def log(message):
    global _accelerator
    if _accelerator is None:
        warnings.warn("Accelerator not set, using print instead.")
        print_fn = print
    else:
        print_fn = _accelerator.print
    print_fn(message)
