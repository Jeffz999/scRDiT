import os
import torch
from diffusion import DiffusionGene
from settings import args
import numpy as np
from transformer import DiT
from unet import Unet1d
import logging

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def generate_samples(model_path: str, save_path: str, model_structure: torch.nn.Module, amount: int, inference_steps: int):
    """
    Generates and saves synthetic gene expression data using a trained model.

    Args:
        model_path (str): Path to the trained model checkpoint (.pt file).
        save_path (str): Path to save the generated samples (.npy file).
        model_structure (torch.nn.Module): The model architecture (e.g., DiT(), Unet1d()).
        amount (int): The number of samples to generate.
        inference_steps (int): The number of steps for the DPM-Solver.
    """
    logging.info(f"Loading model from: {model_path}")
    device = args.device

    # Load the model structure and move it to the correct device
    model = model_structure.to(device)
    
    # Load the entire checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # --- IMPORTANT: Load the EMA weights for best generation quality ---
    if 'ema' in checkpoint:
        logging.info("Found EMA weights in checkpoint. Loading them into the model.")
        model.load_state_dict(checkpoint['ema'])
    else:
        # Fallback for older checkpoints that only saved the main model
        logging.warning("EMA weights not found. Loading main model weights. For best results, use checkpoints from EMA-enabled training.")
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    
    model.eval()

    diffusion = DiffusionGene(gene_size=args.gene_size, device=device)

    logging.info(f"Generating {amount} samples with {inference_steps} inference steps...")
    with torch.no_grad():
        generated_samples = diffusion.sample(
            model,
            n=amount,
            num_inference_steps=inference_steps,
            clamp=True
        )
    
    # Squeeze the channel dimension before saving
    generated_samples = generated_samples.cpu().squeeze(1).numpy()

    # Ensure the directory for the save_path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    logging.info(f"Saving {generated_samples.shape[0]} samples to {save_path}")
    np.save(save_path, generated_samples)
    logging.info("Generation complete.")


if __name__ == '__main__':
    # === Configuration ===
    # 1. Set the path to your trained model checkpoint
    # This should be a .pt file from your "ckpts/your_run_name/" directory
    model_checkpoint_path = 'ckpts/malignant/malignant_epochfinal.pt'

    # 2. Set the path where you want to save the generated samples
    output_save_path = 'results/malignant_epochfinal_ddim3.npy'

    # 3. Choose the model structure that matches your checkpoint
    # This must be the same as the one used during training.
    model_architecture = args.model
    # model_architecture = Unet1d()

    # 4. Set the number of samples and inference steps
    num_samples_to_generate = 1024
    dpm_solver_steps = 100
    # =====================

    generate_samples(
        model_path=model_checkpoint_path,
        save_path=output_save_path,
        model_structure=model_architecture,
        amount=num_samples_to_generate,
        inference_steps=dpm_solver_steps
    )
