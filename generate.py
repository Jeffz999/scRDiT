import os
import torch
from diffusion import DiffusionGene
from settings import args
import numpy as np
from transformer import DiT
from unet import Unet1d
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def generate_samples(model_path: str, save_path: str, model_structure: torch.nn.Module, amount: int, inference_steps: int):
    """
    Generates and saves synthetic gene expression data using a trained 2 channel model.

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

    # --- MODIFIED: Initialize diffusion with 2 channels ---
    diffusion = DiffusionGene(gene_size=args.gene_size, device=device, num_channels=2)

    logging.info(f"Generating {amount} samples with {inference_steps} inference steps...")
    with torch.no_grad():
        # Generated samples will have shape (N, 2, gene_size)
        generated_samples = diffusion.sample(
            model,
            n=amount,
            num_inference_steps=inference_steps,
            clamp=True
        )
    
    generated_samples = generated_samples.cpu().numpy()

    # --- NEW: Decode the 2-channel output ---
    logging.info("Decoding 2-channel output...")
    
    # 1. Separate the expression and mask channels
    generated_expressions = generated_samples[:, 0, :]
    generated_mask = generated_samples[:, 1, :]

    # 2. Load the expression normalization statistics
    stats_path = 'data_stats_sparsity_bits.npy'
    try:
        stats = np.load(stats_path, allow_pickle=True).item()
        data_min, data_max = stats['min'], stats['max']
        logging.info(f"Loaded expression stats: min={data_min:.4f}, max={data_max:.4f}")
    except FileNotFoundError:
        logging.error(f"Error: Stats file not found at '{stats_path}'. Please run train.py first.")
        return

    # 3. De-normalize the expression channel from [-1, 1] back to the log-transformed range
    generated_expressions = (generated_expressions + 1) / 2 * (data_max - data_min) + data_min
    
    # 4. De-normalize the mask channel from [-1, 1] back to the [0, 1] range
    generated_mask = (generated_mask + 1) / 2.0
    
    # 5. Create a final binary mask by thresholding the generated mask
    # This is the key step for enforcing sparsity.
    final_binary_mask = (generated_mask > 0.5).astype(np.float32)
    
    # 6. Apply the final mask to the expression data.
    # This sets all genes where the mask is 0 to be 0.
    final_expressions = generated_expressions * final_binary_mask
    
    # 7. Reverse the log1p transformation on the now-sparse data
    final_expressions = np.expm1(final_expressions)
    
    # 8. Final clamp to ensure non-negativity
    final_expressions[final_expressions < 0] = 0
    logging.info("Decoding complete.")
    # --- END NEW ---

    # Ensure the directory for the save_path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    logging.info(f"Saving {final_expressions.shape[0]} samples to {save_path}")
    np.save(save_path, final_expressions)
    logging.info("Generation complete.")


if __name__ == '__main__':
    model_checkpoint_path = 'ckpts/malignant/malignant_epochfinal.pt'
    output_save_path = 'results/malignant_sbits_pdpm1.npy'
    model_architecture = args.model
    num_samples_to_generate = 1024
    solver_steps = 50

    generate_samples(
        model_path=model_checkpoint_path,
        save_path=output_save_path,
        model_structure=model_architecture,
        amount=num_samples_to_generate,
        inference_steps=solver_steps
    )
