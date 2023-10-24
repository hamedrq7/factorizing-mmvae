# MNIST-SVHN multi-modal model specification
import os

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt, prod
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision.utils import save_image, make_grid

from .dirty_import_fails import custom_make_mnist_svhn_idx
from vis import plot_embeddings, plot_kls_df
from .mmvae import MMVAE
from .vae_mnist import MNIST
from .vae_svhn import SVHN

class MNIST_SVHN(MMVAE):
    def __init__(self, params):
        super(MNIST_SVHN, self).__init__(dist.Normal, params, MNIST, SVHN)
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.vaes[0].llik_scaling = prod(self.vaes[1].dataSize) / prod(self.vaes[0].dataSize) \
            if params.llik_scaling == 0 else params.llik_scaling

        print('Mnist llik_scaling: ', self.vaes[0].llik_scaling)
        print('SVHN llik_scaling: ', self.vaes[1].llik_scaling)
        # print(self.vaes[0])
        # exit()

        self.modelName = 'mnist-svhn'

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', max_d=10000, dm=30):
        if not (os.path.exists(f'../data/train-ms-mnist-idx_max_d_{max_d}_dm_{dm}.pt')
                and os.path.exists(f'../data/train-ms-svhn-idx_max_d_{max_d}_dm_{dm}.pt')
                and os.path.exists(f'../data/test-ms-mnist-idx_max_d_{max_d}_dm_{dm}.pt')
                and os.path.exists(f'../data/test-ms-svhn-idx_max_d_{max_d}_dm_{dm}.pt')):


            # raise RuntimeError('Generate transformed indices with the script in bin')
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

    def generate(self, runPath, epoch, only_mean: bool=False):
        N = 64
        samples_list = super(MNIST_SVHN, self).generate(N)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            # wrangle things so they come out tiled
            samples = samples.view(N, *samples.size()[1:])
            save_image(samples,
                       '{}/mmvae_gen_samples_{}_{:03d}.png'.format(runPath, i, epoch),
                       nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch, is_train: bool=False):
        recons_mat = super(MNIST_SVHN, self).reconstruct([d[:8] for d in data])
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                _data = data[r][:8].cpu()
                recon = recon.squeeze(0).cpu()
                # resize mnist to 32 and colour. 0 => mnist, 1 => svhn
                _data = _data if r == 1 else resize_img(_data, self.vaes[1].dataSize)
                recon = recon if o == 1 else resize_img(recon, self.vaes[1].dataSize)
                comp = torch.cat([_data, recon])
                save_image(comp, '{}/mmvae_{}_recon_{}x{}_{:03d}.png'.format(runPath, 'train' if is_train else 'test', r, o, epoch))

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super(MNIST_SVHN, self).analyse(data, K=10)
        labels = ['Prior', *[vae.modelName.lower() for vae in self.vaes]]
        plot_embeddings(zemb, zsl, labels, '{}/mmvae_emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/mmvae_kl_distance_{:03d}.png'.format(runPath, epoch))


def resize_img(img, refsize):
    return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)
