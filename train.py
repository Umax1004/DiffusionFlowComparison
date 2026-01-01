import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import os
import glob
import random
import numpy as np
from torch.utils.data import DataLoader
from src.dataloader import CelebADataset
import wandb
from timm.utils.model_ema import ModelEmaV2
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, CosineAnnealingLR
from src.sample_image import sample_ddpm, log_samples_to_wandb

def set_seed(seed=42):
    """Full reproducibility for diffusion experiments"""
    # Python random
    random.seed(seed)
    # NumPy 
    np.random.seed(seed)
    # PyTorch CPU/GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Determinism (critical for CNNs/U-Nets)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Seed {seed} set for reproducible results")

def to_uint8(imgs: torch.Tensor) -> torch.Tensor:
    """Convert float32 [0,1] -> uint8 [0,255] safely"""
    return (imgs.clamp(0, 1) * 255).round().to(torch.uint8)

@hydra.main(version_base=None, config_path="configs", config_name="ddpm_fast")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(cfg.globals.seed)
    
    # Dataloaders
    train_dl = DataLoader(CelebADataset(split=cfg.data.train.split), 
                         batch_size=cfg.data.train.batch_size,
                         shuffle=True,
                         num_workers=cfg.data.train.num_workers,
                         persistent_workers=cfg.data.train.persistent_workers,
                         pin_memory=cfg.data.train.pin_memory)
    
    val_dl = DataLoader(CelebADataset(split=cfg.data.val.split),
                       batch_size=cfg.data.val.batch_size,
                       shuffle=False,
                       num_workers=cfg.data.val.num_workers,
                       pin_memory=cfg.data.train.pin_memory)  # Reuse pin_memory
    
    cfg.eval.fid_num_samples = len(val_dl.dataset)
    
    # wandb
    wandb.init(project=cfg.logging.wandb.project, 
              name=cfg.logging.wandb.name, 
              config=OmegaConf.to_container(cfg, resolve=True))
    
    # Model setup
    model = instantiate(cfg.model).to(device)
    model = torch.compile(model, **cfg.precision.compile)
    scheduler = instantiate(cfg.diffusion)
    optimizer = instantiate(cfg.training.optimizer, model.parameters())
    ema_model = ModelEmaV2(model, decay=cfg.training.ema_decay, device=device)

    steps_per_epoch = len(train_dl)
    total_steps = cfg.training.epochs * steps_per_epoch
    warmup_steps = int(cfg.training.scheduler.warmup_ratio * total_steps)  # 5% warmup
    min_lr_ratio = cfg.training.scheduler.min_lr_ratio

    # Linear warmup: 0 -> base_lr
    warmup_scheduler = ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=warmup_steps
    )

    # Cosine decay: base_lr -> base_lr * min_lr_ratio
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=optimizer.param_groups[0]["lr"] * min_lr_ratio
    )

    # SequentialLR: warmup first, then cosine
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]  # switch after warmup
    )
    
    # Paths
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)

    # Find latest checkpoint
    checkpoint_files = glob.glob(os.path.join(cfg.paths.checkpoint_dir, "ddpm_epoch_*.pth"))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Loading checkpoint {latest_checkpoint} ...")
        ckpt = torch.load(latest_checkpoint, map_location=device)

        model.load_state_dict(ckpt["model"])
        ema_model.module.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["scheduler"])
        
        start_epoch = ckpt["epoch"]


        # Extract last epoch number
        start_epoch = int(latest_checkpoint.split("_")[-1].split(".")[0])
    else:
        start_epoch = 0
        print("No checkpoint found, starting from scratch.")
    
    # Precision
    if cfg.precision.use_scaler:
        scaler = GradScaler(enabled=cfg.precision.use_scaler)
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    loss_fn = torch.nn.MSELoss()
    
    fid = FrechetInceptionDistance(feature=2048).to(device)
    
    print("Computing real features...")
    fid.eval()
    
    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Real features"):
            imgs = batch.to(device, non_blocking=True)
            imgs = (imgs + 1) / 2  # [-1,1] -> [0,1]
            fid.update(to_uint8(imgs), real=True)
    
    fid.reset_real_features = False
    
    # Training loop
    model.train()
    
    for epoch in range(start_epoch, cfg.training.epochs):
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{cfg.training.epochs}")
        
        for batch_idx, imgs in enumerate(pbar):
            global_step = epoch * steps_per_epoch + batch_idx

            imgs = imgs.to(device, non_blocking=True)
            noise = torch.randn_like(imgs)

            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (imgs.size(0),),
                device=device
            ).long()

            noisy_images = scheduler.add_noise(imgs, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)

            if cfg.precision.use_scaler:
                with autocast(device_type='cuda', dtype=amp_dtype):
                    model_output = model(noisy_images, timesteps).sample
                    if cfg.training.loss_fn == 'mse':
                        loss = loss_fn(model_output, noise)
                    elif cfg.training.loss_fn == 'snr':
                        mse = loss_fn(model_output, noise)
                        snr = scheduler.alphas_cumprod[timesteps] / (1 - scheduler.alphas_cumprod[timesteps])  # correct
                        gamma = torch.tensor(cfg.training.gamma, device=snr.device, dtype=snr.dtype)
                        min_snr_gamma = torch.minimum(snr, gamma)    
                        weight = min_snr_gamma / snr
                        loss = (weight * mse).mean()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 30.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                model_output = model(noisy_images, timesteps).sample
                loss = loss_fn(model_output, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            lr_scheduler.step()
            ema_model.update(model)

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/epoch": epoch + 1,
                    "train/step": global_step,
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
                step=global_step,
            )

            pbar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
            
            if global_step > 0 and global_step % cfg.eval.sample_every == 0:
                print(f"\nSampling at step {global_step}...")

                # Swap to EMA weights
                ema_model.module.eval()

                scheduler.set_timesteps(cfg.eval.sample_steps)

                samples = sample_ddpm(
                    model=ema_model.module,
                    scheduler=scheduler,
                    device=device,
                    num_samples=cfg.eval.num_sample_images,
                    image_size=cfg.data.image_size,
                )

                log_samples_to_wandb(
                    samples,
                    step=global_step,
                    prefix="ema_samples"
                )

                ema_model.module.train()
                model.train()
                
            if global_step > 0 and global_step % cfg.eval.fid_every == 0:
                # FID generation with cfg.eval.fid_num_samples, cfg.eval.fid_batch_size
                ema_model.module.eval()
                scheduler.set_timesteps(cfg.eval.sample_steps)

                with autocast(device_type='cuda', dtype=amp_dtype, enabled=cfg.precision.use_scaler):
                    with torch.no_grad():
                        for i in tqdm(range(0, cfg.eval.fid_num_samples, cfg.eval.fid_batch_size)):  # Smaller batch for generation to save mem
                            n = min(cfg.eval.fid_batch_size, cfg.eval.fid_num_samples - i)
                            noise = torch.randn((n, 3, 128, 128), device=device, dtype=amp_dtype)
                            samples = scheduler.add_noise(torch.zeros_like(noise), noise, torch.tensor(scheduler.config.num_train_timesteps-1, device=device).expand(n))
                            for t in scheduler.timesteps:
                                pred = ema_model.module(samples, t).sample
                                samples = scheduler.step(pred, t, samples).prev_sample
                            # generated_images.append(samples)
                            samples = (samples + 1) / 2
                            samples = to_uint8(samples)
                            # Compute FID - update with fakes
                            fid.update(samples, real=False)

                fid_score = fid.compute()
                wandb.log({"fid": fid_score.item()}, step=global_step)
                fid.reset()
                torch.cuda.empty_cache()
        
        # Save
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "ema": ema_model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
            },
            f"{cfg.paths.checkpoint_dir}/ddpm_epoch_{epoch+1}.pth"
        )
    
    wandb.finish()

if __name__ == "__main__":
    main()
