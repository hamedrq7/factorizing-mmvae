# Base MMVAE class definition

from itertools import combinations

import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision.utils import save_image, make_grid

import torch.nn.functional as F

from utils import get_mean, kl_divergence
from vis import embed_umap, tensors_to_df
from .mmvae_mnist_svhn import MNIST_SVHN
from .vae_mnist import MNIST
from .vae_svhn import SVHN
from .dirty_import_fails import custom_make_mnist_svhn_idx

class FUMMVAE(nn.Module):
    def __init__(self, params):
        super(FUMMVAE, self).__init__()  # to(
        vaes = [MNIST, SVHN]
        self.vaes = nn.ModuleList([vae(params) for vae in vaes])
        self.mmvae = MNIST_SVHN(params)
        self.modelName = 'fummvae' 
        self.params = params
        # self._pz_params = None  # defined in subclass

    # mnist_svhn dataloader? 
    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", max_d=10000, dm=30):
        if not (os.path.exists(f'../data/train-ms-mnist-idx_max_d_{max_d}_dm_{dm}.pt')
                and os.path.exists(f'../data/train-ms-svhn-idx_max_d_{max_d}_dm_{dm}.pt')
                and os.path.exists(f'../data/test-ms-mnist-idx_max_d_{max_d}_dm_{dm}.pt')
                and os.path.exists(f'../data/test-ms-svhn-idx_max_d_{max_d}_dm_{dm}.pt')):
            custom_make_mnist_svhn_idx(max_d, dm)

        # get transformed indices
        t_mnist = torch.load(f'../data/train-ms-mnist-idx_max_d_{max_d}_dm_{dm}.pt')
        t_svhn = torch.load(f'../data/train-ms-svhn-idx_max_d_{max_d}_dm_{dm}.pt')
        s_mnist = torch.load(f'../data/test-ms-mnist-idx_max_d_{max_d}_dm_{dm}.pt')
        s_svhn = torch.load(f'../data/test-ms-svhn-idx_max_d_{max_d}_dm_{dm}.pt')

        # load base datasets
        t1, s1 = self.vaes[0].getDataLoaders(batch_size, shuffle, device)
        t2, s2 = self.vaes[1].getDataLoaders(batch_size, shuffle, device)

        print('MNIST', f'train: {len(t1)}-{len(t1.dataset)}, test: {len(s1)}-{len(s1.dataset)}')
        print('SVHN', f'train: {len(t2)}-{len(t2.dataset)}, test: {len(s2)}-{len(s2.dataset)}')

        train_mnist_svhn = TensorDataset([
            ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
            ResampleDataset(t2.dataset, lambda d, i: t_svhn[i], size=len(t_svhn))
        ])
        # calling dataloader(train_mnist_svhn).get_item(idx) would return [dataset[0][idx], dataset[1][idx]]

        test_mnist_svhn = TensorDataset([
            ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
            ResampleDataset(s2.dataset, lambda d, i: s_svhn[i], size=len(s_svhn))
        ])

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_mnist_svhn, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_mnist_svhn, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train, test

    def forward(self, x, K=1):
        qz_xs_mmvae, px_zs_mmvae, zss_mmvae = self.mmvae(x, K=1)

        qz_x_mnist, px_z_mnist, zs_mnist = self.vaes[0](x, K=1)
        qz_x_svhn, px_z_svhn, zs_svhn = self.vaes[1](x, K=1)
        

        results = {
            'mmvae': [qz_xs_mmvae, px_zs_mmvae, zss_mmvae],
            'mnist': [qz_x_mnist, px_z_mnist, zs_mnist],
            'svhn':  [qz_x_svhn, px_z_svhn, zs_svhn]
        }
        
        return results

    def generate(self, runPath, epoch, only_mean: bool=False):
        self.mmvae.generate(runPath, epoch, only_mean)
        self.vaes[0].generate(runPath, epoch, only_mean)
        self.vaes[1].generate(runPath, epoch, only_mean)

    def reconstruct(self, data, runPath, epoch, is_train: bool=False):
        self.mmvae.reconstruct(data, runPath, epoch, is_train)
        self.vaes[0].reconstruct(data[0], runPath, epoch, is_train)
        self.vaes[1].reconstruct(data[1], runPath, epoch, is_train)
        
    def analyse(self, data, runPath, epoch):
        self.mmvae.analyse(data, runPath, epoch)
        self.vaes[0].analyse(data[0], runPath, epoch)
        self.vaes[1].analyse(data[1], runPath, epoch)

def resize_img(img, refsize):
    return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)
