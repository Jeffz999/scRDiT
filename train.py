import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from settings import args
from typing import List
import logging
from torch.utils.tensorboard import SummaryWriter
from loader import cell_dataloader
from torch.utils.data import DataLoader
from diffusion import DiffusionGene
from copy import deepcopy
from collections import OrderedDict

# Run this file to train your model.
# Change training parameters in settings.py.

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay=0.999):
    """
    Step the EMA model towards the current model.

    Args:
        ema_model (nn.Module): The Exponential Moving Average model.
        model (nn.Module): The current training model.
        decay (float): The decay rate for the EMA.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def train_ddpm(args):
    """
    Main training function with EMA implementation.

    Args:
        args: An object containing training settings and parameters.
    """
    logging.info("Setting up training environment...")
    run_name: str = args.run_name
    device: str = args.device
    model: nn.Module = args.model.to(device)
    
    # --- EMA Implementation: Create EMA model ---
    ema_model = deepcopy(model).to(device)
    requires_grad(ema_model, False)
    ema_model.eval()  # EMA model is always in evaluation mode
    logging.info("EMA model created.")

    dataloader: DataLoader = cell_dataloader
    lr = args.lr
    optimizer: optim.Optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse: nn.MSELoss = nn.MSELoss()
    
    eta_min = lr * 2e-2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=eta_min)
    
    diffusion = DiffusionGene(gene_size=args.gene_size, device=device)

    # --- EMA Implementation: Modified Checkpoint Loading ---
    if args.ckpt:
        ckpt_path = os.path.join("ckpts", args.run_name, f"{run_name}_epoch{args.ckpt_epoch}.pt")
        logging.info(f"Loading checkpoint from {ckpt_path}...")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            # Load EMA model state if it exists in the checkpoint
            if 'ema' in checkpoint:
                ema_model.load_state_dict(checkpoint['ema'])
                logging.info("Loaded EMA model weights from checkpoint.")
            else:
                # If no EMA state, initialize it from the main model
                update_ema(ema_model, model, decay=0)
                requires_grad(ema_model, False)
                ema_model.eval()  # EMA model is always in evaluation mode
                logging.warning("EMA weights not found in checkpoint. Initializing from model weights.")

            # Load optimizer state if it exists
            if 'opt' in checkpoint:
                optimizer.load_state_dict(checkpoint['opt'])
                logging.info("Loaded optimizer state from checkpoint.")
            else:
                 logging.warning("Optimizer state not found in checkpoint.")

        except FileNotFoundError:
            logging.error(f"Checkpoint file not found at {ckpt_path}. Starting from scratch.")
        except Exception as e:
            logging.error(f"Could not load checkpoint: {e}. It might be an old format.")
            # Fallback for very old, non-dict checkpoints
            model.load_state_dict(torch.load(ckpt_path, map_location=device))


    logger = SummaryWriter(os.path.join("runs", args.run_name))
    logging.info(f"TensorBoard logs will be saved to: runs/{args.run_name}")

    l = len(dataloader)
    
    """
    # Needed for deep copy?
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    """

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

            # --- EMA Implementation: Update EMA model after each step ---
            update_ema(ema_model, model)

            # --- Log metrics to console and TensorBoard ---
            pbar.set_postfix(MSE=loss.item(), GradNorm=total_norm)
            logger.add_scalar("Loss/MSE", loss.item(), global_step=epoch * l + i)
            logger.add_scalar("Training/Gradient_Norm", total_norm, global_step=epoch * l + i)
            epoch_loss_list.append(loss.item())

        scheduler.step()
        logger.add_scalar("Training/Learning_Rate", scheduler.get_last_lr()[0], global_step=epoch)
        
        avg_epoch_loss: float = sum(epoch_loss_list) / len(epoch_loss_list)
        logging.info(f"Epoch {epoch} finished. Average Loss: {avg_epoch_loss:.6f}")
        logger.add_scalar("Loss/Epoch_Avg_MSE", avg_epoch_loss, global_step=epoch)

        # --- EMA Implementation: Modified Checkpoint Saving ---
        if (epoch + 1) % args.save_frequency == 0:
            logging.info(f"Saving checkpoint at epoch {epoch+1}...")
            ckpt_dir = os.path.join("ckpts", args.run_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            
            # Save model, EMA model, and optimizer in a single dictionary
            checkpoint = {
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "opt": optimizer.state_dict(),
            }
            ckpt_path = os.path.join(ckpt_dir, f"{run_name}_epoch{epoch}.pt")
            torch.save(checkpoint, ckpt_path)
            logging.info(f"Checkpoint saved to {ckpt_path}")
    
    logging.info("Training finished.")
    logger.close()

if __name__ == '__main__':
    train_ddpm(args)
