import argparse
import torch
from transformer import DiT
from unet import Unet1d
# Reset training parameters in this file.

parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

# choose dataset
# args.dataset_path = 'datasets/fibroblast_datas.npy'
args.dataset_path = 'datasets/malignant_datas.npy'

# training settings
# args.run_name = 'fibroblast'  # This will determine the savepath of checkpoints!
args.run_name = 'malignant'

# setting up model
# args.model = Unet1d()
args.model = DiT(depth=12)

args.epochs = 500  # epochs of training
args.batch_size = 128  # depends on your GPU memory size
args.gene_size = 2000  # size of gene set
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.lr = 1e-6  # learning rate
args.save_frequency = 20  # how many epochs to save a checkpoint
args.ckpt = False  # load checkpoint or not
args.ckpt_epoch = 0  # which checkpoint to load
