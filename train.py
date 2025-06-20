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
    # The model is dynamically chosen from the settings file (Unet1d or DiT)
    model: nn.Module = args.model.to(device)
    
    # Load model checkpoint if continuing training
    if args.ckpt:
        logging.info(f"Loading model checkpoint from epoch {args.ckpt_epoch}...")
        model.load_state_dict(torch.load(os.path.join("ckpts", args.run_name, f"{run_name}_epoch{args.ckpt_epoch}.pt")))
    
    
    dataloader: DataLoader = cell_dataloader
    optimizer: optim.Optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse: nn.MSELoss = nn.MSELoss()

    
    diffusion = DiffusionGene(gene_size=args.gene_size, device=device)

    # Load optimizer state if continuing trainings
    if args.ckpt:
        optimizer.load_state_dict(torch.load(f'optim/{args.run_name}_AdamW.pt'))

    # Initialize TensorBoard for logging
    logger = SummaryWriter(os.path.join("runs", args.run_name))

    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_loss_list: List[float] = []

        for i, genes in enumerate(pbar):
            genes = genes.to(device)
            # --- UPDATED: Using the new scheduler's methods ---
            t: torch.Tensor = diffusion.sample_timesteps(genes.shape[0])
            x_t, noise = diffusion.noise_genes(genes, t)
            
            # The model now also needs the timestep `t`
            predicted_noise = model(x_t, t)
            loss: torch.Tensor = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            epoch_loss_list.append(loss.item())

        avg_epoch_loss: float = sum(epoch_loss_list) / len(epoch_loss_list)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")
        logger.add_scalar("Average Epoch Loss", avg_epoch_loss, global_step=epoch)

        #print("epoch: ", epoch, " avg_loss: ", avg_epoch_loss)

        # Save a checkpoint periodicallys
        if (epoch + 1) % args.save_frequency == 0:
            logging.info(f"Saving checkpoint at epoch {epoch+1}...")
            if not os.path.exists(os.path.join("ckpts", args.run_name)):
                os.makedirs(os.path.join("ckpts", args.run_name))
            torch.save(model.state_dict(), os.path.join("ckpts", args.run_name, f"{run_name}_epoch{epoch}.pt"))
            if not os.path.exists(os.path.join("optim")):
                os.makedirs(os.path.join("optim"))
            torch.save(optimizer.state_dict(), os.path.join("optim", f"{run_name}_AdamW.pt"))
    
    logging.info("Training finished.")

if __name__ == '__main__':
    train_ddpm(args)
