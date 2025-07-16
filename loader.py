import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from settings import args
import os

datapath = args.dataset_path

# --- NEW: 2-Channel Preprocessing with Sparsity Bits ---
print("Loading and preprocessing data for 2-channel input...")
data = np.load(datapath, allow_pickle=True).astype(np.float32)

# 1. Create the sparsity mask (1 for non-zero, 0 for zero)
print("Creating sparsity mask...")
sparsity_mask = (data > 0).astype(np.float32)

# 2. Apply log1p transformation to the expression values
# This helps compress the range of high expression values.
print("Applying log1p transformation to expression values...")
log_data = np.log1p(data)

# 3. Normalize the log-transformed expression data to [-1, 1]
# IMPORTANT: We only calculate min/max on the non-zero values for a more stable range.
print("Normalizing expression data to [-1, 1]...")
non_zero_values = log_data[log_data > 0]
data_min = np.min(non_zero_values)
data_max = np.max(non_zero_values)

# Apply normalization. Zeros in log_data remain 0, which will be scaled to -1.
log_data = 2 * (log_data - data_min) / (data_max - data_min) - 1
# Any original zero values (which were 0 in log_data) are now -1.
# We explicitly set them to -1 to avoid any floating point inaccuracies.
log_data[sparsity_mask == 0] = -1.0

# 4. Normalize the sparsity mask to [-1, 1]
# This maps 0 -> -1 and 1 -> 1
print("Normalizing sparsity mask to [-1, 1]...")
sparsity_mask_normalized = 2.0 * sparsity_mask - 1.0

# 5. Save the expression stats for reversing the transformation later
stats_path = 'data_stats_sparsity_bits.npy'
print(f"Saving expression stats (min/max) to {stats_path}...")
np.save(stats_path, {'min': data_min, 'max': data_max})

# 6. Stack the two channels together
# The result is a single array of shape (num_samples, 2, num_genes)
combined_data = np.stack([log_data, sparsity_mask_normalized], axis=1)
print(f"Preprocessing complete. Final data shape: {combined_data.shape}")
# --- END NEW ---


class CellDataset(Dataset):
    def __init__(self, data, flag='train'):
        assert flag in ['train', 'test', 'valid']
        self.flag = flag
        self.data = data

    def __getitem__(self, index):
        # Return the 2-channel data for the given index
        return self.data[index]

    def __len__(self):
        return len(self.data)

cell_dataset = CellDataset(data=torch.Tensor(combined_data))
bs = args.batch_size
cell_dataloader = DataLoader(dataset=cell_dataset, batch_size=bs, shuffle=True)

if __name__ == "__main__":
    # Test code.
    print("\n--- Dataloader Test ---")
    for step, x in enumerate(cell_dataloader):
        print(f'Step: {step}, Batch shape: {x.shape}')
        # Channel 0 is expression, Channel 1 is sparsity mask
        print(f'Expression channel range: min={torch.min(x[:, 0]):.2f}, max={torch.max(x[:, 0]):.2f}')
        print(f'Sparsity mask channel range: min={torch.min(x[:, 1]):.2f}, max={torch.max(x[:, 1]):.2f}')
        if step == 0:
            print("First sample, expression channel, first 5 values:", x[0, 0, :5].numpy())
            print("First sample, sparsity channel, first 5 values:", x[0, 1, :5].numpy())
        break
    print("Dataloader test complete.")
