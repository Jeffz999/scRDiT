from tqdm import tqdm
from MMD import mmd
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance


def calculate_kl_div(sampled: np.ndarray, target: np.ndarray):
    target = torch.Tensor(target).to('cpu')
    sampled = torch.Tensor(sampled).to('cpu')

    sampled = F.log_softmax(sampled, dim=1)
    target = F.softmax(target, dim=1)
    kl_div = F.kl_div(sampled, target, reduction='batchmean')
    return kl_div


def calculate_w_dist(sampled: np.ndarray, target: np.ndarray, method='flatten'):
    assert method in ['avg', 'flatten']
    if method == 'flatten':
        sampled = sampled.flatten()
        target = target.flatten()
    else:
        sampled = np.mean(sampled, axis=0)
        target = np.mean(target, axis=0)
    w_dist = wasserstein_distance(sampled, target)
    return w_dist


def calculate_mmd(sampled: np.ndarray, target: np.ndarray, method='batch', batch_size=20):
    assert method in ['normal', 'batch', 'avg']
    if method == 'batch':
        assert sampled.shape[1] % batch_size == 0
        loss_list = list()
        b = batch_size
        for i in tqdm(range(2000 // b)):
            sample_clip = sampled[:, i * b:i * b + b]
            target_clip = target[:, i * b:i * b + b]
            loss = mmd(sample_clip, target_clip)
            loss_list.append(loss)

        mmd_loss = sum(loss_list) / len(loss_list)
    elif method == 'avg':
        sampled = np.mean(sampled, axis=0).reshape(1, -1)
        target = np.mean(target, axis=0).reshape(1, -1)
        mmd_loss = mmd(sampled, target)
    else:
        mmd_loss = mmd(sampled, target)
    return mmd_loss


if __name__ == '__main__':
    a1 = np.random.randn(1024, 2000)
    a2 = np.random.randn(1024, 2000)
    print(calculate_mmd(a1, a2, 'normal'))
