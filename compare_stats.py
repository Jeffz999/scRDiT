import numpy as np
import argparse
import logging
import pandas as pd

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

generated_path = "./results/malignant_epochfinal_ddim.npy"
original_path = "datasets/malignant_datas.npy"

def calculate_descriptive_stats(data: np.ndarray, name: str) -> dict:
    """
    Calculates a set of descriptive statistics for the given dataset.

    Args:
        data (np.ndarray): The input data array.
        name (str): The name of the dataset (e.g., "Original", "Generated").

    Returns:
        dict: A dictionary containing the calculated statistics.
    """
    stats = {
        'Dataset': name,
        'Mean': np.mean(data),
        'Std Dev': np.std(data),
        'Variance': np.var(data),
        'Median': np.median(data),
        'Min': np.min(data),
        'Max': np.max(data),
        'Sparsity (%)': (np.count_nonzero(data == 0) / data.size) * 100,
    }
    return stats

def compare_statistics(generated_path: str, original_path: str):
    """
    Loads, preprocesses, and compares descriptive statistics of generated and original data.

    Args:
        generated_path (str): The file path for the generated samples (.npy).
        original_path (str): The file path for the original dataset (.npy).
    """
    # --- Load Data ---
    logging.info(f"Loading generated samples from: {generated_path}")
    try:
        generated_samples = np.load(generated_path, allow_pickle=True)
    except FileNotFoundError:
        logging.error(f"Generated samples file not found at {generated_path}. Please check the path.")
        return

    logging.info(f"Loading original dataset from: {original_path}")
    try:
        original_samples = np.load(original_path, allow_pickle=True)
    except FileNotFoundError:
        logging.error(f"Original dataset not found at {original_path}. Please check the path.")
        return

    # --- Preprocessing for Fair Comparison (from evaluate.py) ---
    logging.info("Preprocessing data for a fair comparison...")

    generated_samples = generated_samples.astype(np.float32)
    original_samples = original_samples.astype(np.float32)

    # Reverse the "zero-negation" (-10 -> 0) on the original dataset
    if -10 in original_samples:
        logging.info("Reversing the 'zero-negation' (-10 -> 0) on the original dataset.")
        original_samples[original_samples == -10.] = 0.

    # Ensure generated samples are 2D
    if generated_samples.ndim == 3:
        generated_samples = generated_samples.squeeze(1)

    # Use the same number of samples from each dataset
    num_samples = min(len(original_samples), len(generated_samples))
    if num_samples == 0:
        logging.error("No samples to compare. One of the datasets is empty.")
        return

    logging.info(f"Comparing {num_samples} generated samples against {num_samples} original samples.")
    original_samples = original_samples[:num_samples]
    generated_samples = generated_samples[:num_samples]

    # --- Calculate Statistics ---
    logging.info("Calculating descriptive statistics...")
    original_stats = calculate_descriptive_stats(original_samples, "Original")
    generated_stats = calculate_descriptive_stats(generated_samples, "Generated")

    # --- Print Results Table ---
    df = pd.DataFrame([original_stats, generated_stats])
    df = df.set_index('Dataset')

    print("\n--- Descriptive Statistics Comparison ---")
    print(df.to_string(float_format="%.4f"))
    print("---------------------------------------")
    print("\nKey things to check:")
    print("  - Sparsity: Is the percentage of zeros similar? This is crucial for scRNA-seq data.")
    print("  - Min/Max: Are the value ranges comparable? Or is the generated data clamped or out of bounds?")
    print("  - Mean/Std Dev: Do the central tendency and spread of the data look alike?")


if __name__ == '__main__':
    compare_statistics(generated_path, original_path)