import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from settings import args

datapath = args.dataset_path
data = np.load(datapath, allow_pickle=True).astype(np.float32)


# 1. Apply log1p transformation
# This handles sparsity well, as log1p(0) = 0
print("Applying log1p transformation...")
data = np.log1p(data)

# 2. Normalize the data to the range [-1, 1]
print("Normalizing data to [-1, 1]...")
data_min = np.min(data)
data_max = np.max(data)
data = 2 * (data - data_min) / (data_max - data_min) - 1

# 3. Save the min/max statistics for reversing the transformation later
stats_path = 'data_stats.npy'
print(f"Saving normalization stats (min/max) to {stats_path}...")
np.save(stats_path, {'min': data_min, 'max': data_max})
print("Preprocessing complete.")
# --- END NEW ---

class CellDataset(Dataset):

    def __init__(self, data, flag='train'):
        assert flag in ['train', 'test', 'valid']
        self.flag = flag
        self.data = data

    def __getitem__(self, index):
        return self.data[index][None, :]

    def __len__(self):
        return len(self.data)

    def __load_data__(self, csv_paths: list):
        pass

    def preprocess(self, data):
        pass


cell_dataset = CellDataset(data=torch.Tensor(data))
bs = args.batch_size
cell_dataloader = DataLoader(dataset=cell_dataset, batch_size=bs, shuffle=True)  # used in train.py

if __name__ == "__main__":
    # Test code.
    for step, x in enumerate(cell_dataloader):
        print('step is :', step)
        data = x
        print('data is {}.'.format(data))
        print(data.shape)
