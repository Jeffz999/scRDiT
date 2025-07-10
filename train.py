import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from settings import args
from unet import Unet1d
from typing import List
import logging
from torch.utils.tensorboard import SummaryWriter
from loader import cell_dataloader
from torch.utils.data import DataLoader
from diffusion import DiffusionGene
from transformer import DiT

# Run this file to train your model.
# Change training parameters in settings.py.

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def train_ddpm(args):
    """
    Rewrite args parameters in settings.py
    :param args: Settings.
    :return: None
    """
    logging.info("Setting up training environment...")
    run_name: str = args.run_name
    device: str = args.device
    model: nn.Module = args.model.to(device)
    
    if args.ckpt:
        logging.info(f"Loading model checkpoint from epoch {args.ckpt_epoch}...")
        model.load_state_dict(torch.load(os.path.join("ckpts", args.run_name, f"{run_name}_epoch{args.ckpt_epoch}.pt")))
    
    dataloader: DataLoader = cell_dataloader
    optimizer: optim.Optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse: nn.MSELoss = nn.MSELoss()
    
    diffusion = DiffusionGene(gene_size=args.gene_size, device=device)

    if args.ckpt:
        try:
            optimizer.load_state_dict(torch.load(f'optim/{args.run_name}_AdamW.pt'))
            logging.info("Loaded optimizer state.")
        except FileNotFoundError:
            logging.warning("Optimizer state not found. Starting with a new optimizer.")

    logger = SummaryWriter(os.path.join("runs", args.run_name))
    logging.info(f"TensorBoard logs will be saved to: runs/{args.run_name}")

    l = len(dataloader)

    # --- NEW: Enable detailed logging in the model ---
    # This assumes the model has 'log_stats', 'writer', and 'global_step' attributes.
    # This is implemented in the updated DiT class in transformer.py
    if hasattr(model, 'log_stats'):
        logging.info("Enabling detailed model statistics logging to TensorBoard.")
        model.log_stats = True
        model.writer = logger

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_loss_list: List[float] = []

        for i, genes in enumerate(pbar):
            genes = genes.to(device)
            t: torch.Tensor = diffusion.sample_timesteps(genes.shape[0])
            x_t, noise = diffusion.noise_genes(genes, t)
            
            # --- NEW: Set global step for detailed logging ---
            if hasattr(model, 'global_step'):
                model.global_step = epoch * l + i

            predicted_noise = model(x_t, t)
            loss: torch.Tensor = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            
            # --- Monitor gradient norm ---
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            optimizer.step()

            # --- Log metrics to console and TensorBoard ---
            pbar.set_postfix(MSE=loss.item(), GradNorm=total_norm)
            logger.add_scalar("Loss/MSE", loss.item(), global_step=epoch * l + i)
            logger.add_scalar("Training/Gradient_Norm", total_norm, global_step=epoch * l + i)
            epoch_loss_list.append(loss.item())

        avg_epoch_loss: float = sum(epoch_loss_list) / len(epoch_loss_list)
        logging.info(f"Epoch {epoch} finished. Average Loss: {avg_epoch_loss:.6f}")
        logger.add_scalar("Loss/Epoch_Avg_MSE", avg_epoch_loss, global_step=epoch)

        if (epoch + 1) % args.save_frequency == 0:
            logging.info(f"Saving checkpoint at epoch {epoch+1}...")
            ckpt_dir = os.path.join("ckpts", args.run_name)
            optim_dir = "optim"
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(optim_dir, exist_ok=True)
            
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{run_name}_epoch{epoch}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(optim_dir, f"{run_name}_AdamW.pt"))
    
    logging.info("Training finished.")
    logger.close()

if __name__ == '__main__':
    train_ddpm(args)
