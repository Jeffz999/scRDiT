import numpy as np
import argparse
import logging
import pandas as pd

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# Default paths, can be overridden by command-line arguments
generated_path = "./results/malignant_epochfinal_dpmv2_1.npy" 
original_path = "datasets/malignant_datas.npy"

def calculate_descriptive_stats(data: np.ndarray, name: str) -> (dict, dict):
    """
    Calculates a set of descriptive statistics for the given dataset,
    both for the overall data and the non-zero subset.

    Args:
        data (np.ndarray): The input data array.
        name (str): The name of the dataset (e.g., "Original", "Generated").

    Returns:
        tuple[dict, dict]: A tuple containing dictionaries for overall stats and non-zero stats.
    """
    # Overall statistics
    overall_stats = {
        'Dataset': name,
        'Mean': np.mean(data),
        'Std Dev': np.std(data),
        'Variance': np.var(data),
        'Min': np.min(data),
        '25th Pct (Q1)': np.percentile(data, 25),
        'Median (50th Pct)': np.median(data),
        '75th Pct (Q3)': np.percentile(data, 75),
        'Max': np.max(data),
        'Sparsity (%)': (np.count_nonzero(data == 0) / data.size) * 100,
    }

    # --- Non-zero statistics ---
    non_zero_data = data[data > 0]
    if non_zero_data.size > 0:
        non_zero_stats = {
            'Dataset': name,
            'Mean (non-zero)': np.mean(non_zero_data),
            'Std Dev (non-zero)': np.std(non_zero_data),
            'Min (non-zero)': np.min(non_zero_data),
            '25th Pct (non-zero)': np.percentile(non_zero_data, 25),
            'Median (non-zero)': np.median(non_zero_data),
            '75th Pct (non-zero)': np.percentile(non_zero_data, 75),
            'Max (non-zero)': np.max(non_zero_data),
            'Tiny Values (<0.01) (%)': (np.count_nonzero((data > 0) & (data < 0.01)) / data.size) * 100
        }
    else:
        # Handle case where there are no non-zero values
        non_zero_stats = {
            'Dataset': name,
            'Mean (non-zero)': 0,
            'Std Dev (non-zero)': 0,
            'Min (non-zero)': 0,
            '25th Pct (non-zero)': 0,
            'Median (non-zero)': 0,
            '75th Pct (non-zero)': 0,
            'Max (non-zero)': 0,
            'Tiny Values (<0.01) (%)': 0
        }

    return overall_stats, non_zero_stats

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

    # --- Preprocessing ---
    logging.info("Preprocessing data for a fair comparison...")

    generated_samples = generated_samples.astype(np.float32)
    original_samples = original_samples.astype(np.float32)

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
    original_overall_stats, original_nonzero_stats = calculate_descriptive_stats(original_samples, "Original")
    generated_overall_stats, generated_nonzero_stats = calculate_descriptive_stats(generated_samples, "Generated")

    # --- Print Results Tables ---
    # Overall stats
    df_overall = pd.DataFrame([original_overall_stats, generated_overall_stats])
    df_overall = df_overall.set_index('Dataset')
    
    print("\n--- Overall Descriptive Statistics ---")
    print(df_overall.to_string(float_format="%.4f"))
    print("--------------------------------------")

    # Non-zero stats
    df_nonzero = pd.DataFrame([original_nonzero_stats, generated_nonzero_stats])
    df_nonzero = df_nonzero.set_index('Dataset')

    print("\n--- Non-Zero Value Statistics ---")
    print(df_nonzero.to_string(float_format="%.4f"))
    print("---------------------------------")

    print("\nKey things to check:")
    print("  - Sparsity: Is the percentage of zeros similar? This is crucial for scRNA-seq data.")
    print("  - Mean/Std Dev (non-zero): Do the expression levels of *expressed* genes look similar?")
    print("  - Tiny Values (%): A high percentage for 'Generated' indicates the model is creating a 'smear' of near-zero values instead of true zeros.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare descriptive statistics of scRNA-seq data.")
    parser.add_argument(
        "--generated_path",
        type=str,
        default=generated_path,
        help=f"Path to the generated samples .npy file. (Default: {generated_path})"
    )
    parser.add_argument(
        "--original_path",
        type=str,
        default=original_path,
        help=f"Path to the original dataset .npy file. (Default: {original_path})"
    )
    cli_args = parser.parse_args()

    compare_statistics(cli_args.generated_path, cli_args.original_path)
