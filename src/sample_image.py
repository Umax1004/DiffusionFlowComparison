from typing import Union
import torch
from torch import Tensor
from diffusers import DDPMScheduler
from tqdm import tqdm
import wandb
import torchvision

@torch.no_grad()
def sample_ddpm(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    device: Union[torch.device, str],
    num_samples: int = 8,
    image_size: int = 128,
) -> Tensor:
    """
    Sample images from a DDPM model using ancestral sampling.

    Args:
        model (torch.nn.Module):
            Trained UNet noise-prediction model (typically EMA weights).
        scheduler (DDPMScheduler):
            Diffusers DDPM scheduler used during training.
        device (torch.device | str):
            Device to run sampling on.
        num_samples (int, optional):
            Number of images to generate. Default: 8.
        image_size (int, optional):
            Spatial resolution of generated images. Default: 128.

    Returns:
        Tensor:
            Sampled images in range [0, 1] with shape
            (num_samples, 3, image_size, image_size).
    """

    model.eval()

    # Start from pure Gaussian noise
    x: Tensor = torch.randn(
        num_samples,
        3,
        image_size,
        image_size,
        device=device,
    )

    for t in scheduler.timesteps:
        model_output = model(x, t).sample
        x = scheduler.step(model_output, t, x).prev_sample

    # Map from [-1, 1] → [0, 1]
    x = (x.clamp(-1, 1) + 1) / 2

    return x

def log_samples_to_wandb(
    images: Tensor,
    step: int,
    prefix: str = "samples",
) -> None:
    """
    Log a grid of images to Weights & Biases.

    Args:
        images (Tensor):
            Image tensor with shape (N, C, H, W) and values in [0, 1].
        step (int):
            Global training step for W&B logging.
        prefix (str, optional):
            Key name under which images will appear in W&B. Default: "samples".

    Returns:
        None
    """

    grid: Tensor = torchvision.utils.make_grid(
        images,
        nrow=int(images.shape[0] ** 0.5),
        padding=2,
    )

    wandb.log(
        {
            prefix: wandb.Image(grid),
        },
        step=step,
    )

