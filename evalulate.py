import numpy as np
import argparse
import logging
from settings import args
from metrics import calculate_w_dist, calculate_kl_div
from definitions.mmd_loss import MMDLoss
import torch

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

gen_path = "results/malignant_sbits_pdpm1.npy"
dataset_path = "./datasets/malignant_datas.npy"

def evaluate(generated_path: str, original_path: str):
    """
    Calculates and prints evaluation metrics by comparing generated data to original data.

    Args:
        generated_path (str): The file path for the generated samples (.npy).
        original_path (str): The file path for the original dataset (.npy).
    """
    logging.info(f"Loading generated samples from: {generated_path}")
    try:
        generated_samples = np.load(generated_path, allow_pickle=True)
    except FileNotFoundError:
        logging.error(f"Generated samples file not found at {generated_path}. Please run generate.py first.")
        return

    logging.info(f"Loading original dataset from: {original_path}")
    try:
        original_samples = np.load(original_path, allow_pickle=True)
    except FileNotFoundError:
        logging.error(f"Original dataset not found at {original_path}. Check the path in settings.py.")
        return

    generated_samples = generated_samples.astype(np.float32)
    original_samples = original_samples.astype(np.float32)

    # Ensure generated samples are 2D
    if generated_samples.ndim == 3:
        generated_samples = generated_samples.squeeze(1)

    # For a fair comparison, use the same number of samples from each dataset
    num_samples = min(len(original_samples), len(generated_samples))
    if num_samples == 0:
        logging.error("No samples to compare. One of the datasets is empty.")
        return
        
    logging.info(f"Comparing {num_samples} generated samples against {num_samples} original samples.")
    
    original_samples = original_samples[:num_samples]
    generated_samples = generated_samples[:num_samples]

    # --- Calculate Metrics ---
    logging.info("Calculating Maximum Mean Discrepancy (MMD)...")
    mmd_score = MMDLoss().forward(torch.from_numpy(generated_samples), torch.from_numpy(original_samples))

    logging.info("Calculating Wasserstein Distance (on flattened data)...")
    w_dist_score = calculate_w_dist(generated_samples, original_samples, method='flatten')

    logging.info("Calculating KL Divergence...")
    # Note: KL Divergence can be sensitive and assumes data is like a probability distribution.
    # We apply a softmax to treat each gene expression profile as a distribution.
    kl_div_score = calculate_kl_div(generated_samples, original_samples)


    # --- Print Results ---
    print("\n--- Evaluation Results ---")
    print(f"  Maximum Mean Discrepancy (MMD): {mmd_score:.6f}")
    print(f"  Wasserstein Distance (W-dist): {w_dist_score:.6f}")
    print(f"  KL Divergence:                  {kl_div_score:.6f}")
    print("--------------------------")
    print("\nLower scores generally indicate better similarity between the generated and original data.")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Evaluate generated scRNA-seq data against an original dataset.")
    # parser.add_argument(
    #     "generated_path",
    #     type=str,
    #     help="Path to the generated samples .npy file (output from generate.py)."
    # )
    # cli_args = parser.parse_args()
    
    # The path to the original data is sourced from settings.py for consistency
    evaluate(gen_path, dataset_path)
