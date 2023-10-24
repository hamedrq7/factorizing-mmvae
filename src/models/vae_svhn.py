# SVHN model specification

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt, prod
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from typing import List, Dict, Tuple
from utils import Constants
from vis import plot_embeddings, plot_kls_df
from .vae import VAE

# Constants for Conv Networks
dataSize = torch.Size([3, 32, 32])
imgChans = dataSize[0]
# fBase = 32  # base size of filter channels


# Constants for FC Networks
dataSizeFlat = int(prod(dataSize))

def extra_hidden_layer(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(True))


###################################### 
### FC Networks
###################################### 
class fcEnc(nn.Module):
    """ Generate latent parameters for SVHN image data. """
    def __init__(self, latent_dim, do_softmax: bool, hidden_dims=[400]):
        super(fcEnc, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(dataSizeFlat, hidden_dims[0]), nn.ReLU(True)))
        # modules.extend([extra_hidden_layer() for _ in range(num_hidden_layers - 1)])
        for layer_indx in range(1, len(hidden_dims)):
            modules.append(extra_hidden_layer(hidden_dims[layer_indx-1], hidden_dims[layer_indx]))

        self.enc = nn.Sequential(*modules)
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.sigma_layer = nn.Linear(hidden_dims[-1], latent_dim)
        
        self.do_softmax = do_softmax
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)

        e = self.enc(x)
        mu = self.mu_layer(e)
        sigma = self.sigma_layer(e)

        if self.do_softmax:
            return mu, F.softmax(sigma, dim=-1) * sigma.size(-1) + Constants.eta
        else: # Normal
            return mu, torch.exp(0.5*sigma) + Constants.eta


class fcDec(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """
    def __init__(self, latent_dim, scale_value: float, hidden_dims=[400]):
        super(fcDec, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(latent_dim, hidden_dims[0]), nn.ReLU(True)))
        # modules.extend([extra_hidden_layer() for _ in range(num_hidden_layers)])
        for layer_indx in range(1, len(hidden_dims)):
            modules.append(extra_hidden_layer(hidden_dims[layer_indx-1], hidden_dims[layer_indx]))

        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(hidden_dims[-1], dataSizeFlat)
        
        self.scale_value = scale_value

    def forward(self, z):
        p = self.fc3(self.dec(z))
        d = torch.sigmoid(p.view(*z.size()[:-1], *dataSize))
        d = d.clamp(Constants.eta, 1-Constants.eta) # clamp for mnist? 

        return d, torch.tensor(self.scale_value).to(z.device)

###################################### 
### Conv Networks
###################################### 
class Enc(nn.Module):
    """
    Generate latent parameters for SVHN image data using ConvNets 
    
    Attributes
    ----------
    latent_dim: int
        dimension of latent space
    do_softmax: bool
        if True, applies softmax into sigma/scale of latent space and multiplies by the size of latent space,
        as suggested by 'Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep Generative Models':
        "we employ Laplace priors and posteriors, constraining their scaling across the D dimensions to sum to D".
    """

    def __init__(self, fBase: int, latent_dim: int, do_softmax: bool):
        # fBase: base size of filter channels
        super(Enc, self).__init__()
        self.fBase = fBase
        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(imgChans, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        self.mu_layer = nn.Conv2d(fBase * 4, latent_dim, 4, 1, 0, bias=True)
        self.sigma_layer = nn.Conv2d(fBase * 4, latent_dim, 4, 1, 0, bias=True)
        # mu_layer, c2 sigma_layer: (latent_dim) x 1 x 1

        self.do_softmax = do_softmax
        print(self.do_softmax)
        print()

    def forward(self, x):
        e = self.enc(x) 
        mu = self.mu_layer(e).squeeze() # before .squeeze(): [bs, latent_dim, 1, 1], after .squeeze(): [bs, latent_dim]
        sigma = self.sigma_layer(e).squeeze() # before .squeeze(): [bs, latent_dim, 1, 1], after .squeeze(): [bs, latent_dim]

        if self.do_softmax:
            return mu, F.softmax(sigma, dim=-1) * sigma.size(-1) + Constants.eta
        else: # Normal
            # the 'torch.exp(0.5*sigma)' is suggested by? # TODO 
            return mu, torch.exp(0.5*sigma) + Constants.eta
            
class Dec(nn.Module):
    """ 
    Generate a SVHN image given a sample from the latent space using Convnets 
    
    Attributes
    ---------
    latent_dim: int
        dimension of latent space
    scale_value: float
        the scale of p(x|z) distribution # TODO experiment effect of this hyperparameter in training, not only generation and sampling
    """

    def __init__(self, fBase: int, latent_dim, scale_value: float):
        # fBase: base size of filter channels
        super(Dec, self).__init__()
        self.fBase = fBase
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, fBase * 4, 4, 1, 0, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.ConvTranspose2d(fBase, imgChans, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 3 x 32 x 32
        )
        self.scale_value = scale_value

    def forward(self, z, debug=False):
        # z: [1, bs, latent_dim] # TODO why dim1 at axis=0? -> probably cause of sampling batches of size K
        
        z = z.unsqueeze(-1).unsqueeze(-1)  # unsqueeze to fit to deconv layers
        # z: [1, bs, latent_dim, 1, 1]
        out = self.dec(z.view(-1, *z.size()[-3:])) # shape of input passed to dec: [bs, latent_dim, 1, 1]
                                                   # shape of out: [bs, 3, 32, 32]

        out = out.view(*z.size()[:-3], *out.size()[1:]) # [1, bs, 3, 32, 32] 

        # Author's comment: consider also predicting the length scale
        return out, torch.tensor(self.scale_value).to(z.device)  # mean, length scale, Author's choice: 0.75


class SVHN(VAE):
    """ 
    Derive a specific sub-class of a VAE for SVHN 
    
    Attributes
    ----------
    params: input arguments
    """
    def __init__(self, params):
        super(SVHN, self).__init__(
            enc=Enc(params.fBase, params.latent_dim, do_softmax=params.softmax) if not params.no_conv
                    else fcEnc(params.latent_dim, do_softmax=params.softmax, hidden_dims=params.hidden_dims),

            dec=Dec(params.fBase, params.latent_dim, scale_value=params.decoder_scale) if not params.no_conv
                else fcDec(params.latent_dim, scale_value=params.decoder_scale,  hidden_dims=params.hidden_dims),

            params=params
        )

        self.do_softmax = params.softmax
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar # '**' allows you to pass the key-value pairs of a dictionary as keyword arguments to a function
        ])
        self.modelName = 'svhn'
        self.dataSize = dataSize
        self.llik_scaling = 1.
        
    @property
    def pz_params(self): # TODO: why? 
        if self.do_softmax:
            return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)
        else: # Normal Distr
            return self._pz_params[0], torch.exp(0.5*self._pz_params[1])

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device='cuda'):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
        tx = transforms.ToTensor()
        train = DataLoader(datasets.SVHN('../data', split='train', download=True, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(datasets.SVHN('../data', split='test', download=True, transform=tx),
                          batch_size=batch_size, shuffle=False, **kwargs) # set test shuffle to False to keep track of the same batch
        
        print(len(train), len(train.dataset))

        return train, test

    def generate(self, runPath, epoch, only_mean: bool=False):
        # comments on parent's method
        N = 64
        K = 1 if only_mean else 9
        samples = super(SVHN, self).generate(N, K, only_mean).cpu()

        # wrangle things so they come out tiled
        samples = samples.view(K, N, *samples.size()[1:]).transpose(0, 1)
        s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples]
        save_image(torch.stack(s),
                   '{}/svhn_gen_samples_{}_{:03d}.png'.format(runPath, 'means' if only_mean else '', epoch),
                   nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch, is_train: bool=False):
        # comments on parent's method
        recon = super(SVHN, self).reconstruct(data[:8])
        comp = torch.cat([data[:8], recon]).data.cpu()
        save_image(comp, '{}/svhn_{}_recon_{:03d}.png'.format(runPath, 'train' if is_train else 'test', epoch))

    def analyse(self, data, runPath, epoch):
        # comments on parent's method
        zemb, zsl, kls_df = super(SVHN, self).analyse(data, K=10)
        labels = ['Prior', self.modelName.lower()]
        plot_embeddings(zemb, zsl, labels, '{}/svhn_emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/svhn_kl_distance_{:03d}.png'.format(runPath, epoch))


# DEEP SVHN models
"""
class deepEnc(nn.Module):
    def __init__(self, latent_dim, do_softmax: bool) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.do_softmax=do_softmax
        # channels = [16, 32, 64, 128, 256]
        channels = [32, 64, 128, 256]
        
        filter_sizes = [3, 3, 3, 3]
        stride_sizes = [1, 1, 1, 1]
        padding_sizes = [1, 1, 1, 1]

        # [3, 32, 32]
        bias_conv=False
        self.block0 = nn.Sequential(
            nn.Conv2d(3, channels[0], filter_sizes[0], stride_sizes[0], padding_sizes[0], bias=bias_conv),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),

            nn.Conv2d(channels[0], channels[0], filter_sizes[0], stride_sizes[0], padding_sizes[0], bias=bias_conv),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
        )
        self.pool0 = nn.AvgPool2d(2, 2)
        
        # [channels_0, 16, 16]
        self.block1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], filter_sizes[1], stride_sizes[1], padding_sizes[1], bias=bias_conv),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),

            nn.Conv2d(channels[1], channels[1], filter_sizes[1], stride_sizes[1], padding_sizes[1], bias=bias_conv),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
        )
        self.pool1 = nn.AvgPool2d(2, 2)
        
        # [channels_1, 8, 8]
        self.block2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], filter_sizes[2], stride_sizes[2], padding_sizes[2], bias=bias_conv),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),

            nn.Conv2d(channels[2], channels[2], filter_sizes[2], stride_sizes[2], padding_sizes[2], bias=bias_conv),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
        )
        self.pool2 = nn.AvgPool2d(2, 2)
        
        # [channels_2, 4, 4]
        self.block3 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], filter_sizes[3], stride_sizes[3], padding_sizes[3], bias=bias_conv),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(),

            nn.Conv2d(channels[3], channels[3], filter_sizes[3], stride_sizes[3], padding_sizes[3], bias=bias_conv),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(),
        ) # [channels_3, 4, 4]

        self.mu_block = nn.Sequential(
            ### use s=1, p=0, f=3 to make the feature maps 2x2 then avg pool
            nn.Conv2d(channels[3], self.latent_dim, 3, 1, 0),
            # nn.BatchNorm2d(self.latent_dim),
            # nn.Tanh(), # (latent_dim, 2, 2)
            nn.AvgPool2d(2, 2)
        ) # (latent_dim) x 1 x 1

        self.sigma_block = nn.Sequential(
            ### use s=1, p=0, f=3 to make the feature maps 2x2 then avg pool
            nn.Conv2d(channels[3], self.latent_dim, 3, 1, 0),
            # nn.BatchNorm2d(self.latent_dim),
            # nn.Tanh(), # (latent_dim, 2, 2)
            nn.AvgPool2d(2, 2)
        ) # (latent_dim) x 1 x 1


    def forward(self, x):
        # x.shape -> [3, 32, 32]
        x = self.pool0(self.block0(x)) # [c0, 16, 16]
        x = self.pool1(self.block1(x)) # [c1, 8, 8]
        x = self.pool2(self.block2(x)) # [c2, 4, 4]
        x = self.block3(x) # [c3, 4, 4]
        
        mu = self.mu_block(x).squeeze()
        sigma = self.sigma_block(x).squeeze()

        if self.do_softmax:
            return mu, F.softmax(sigma, dim=-1) * sigma.size(-1) + Constants.eta
        else:
            return mu, torch.exp(0.5*sigma) + Constants.eta

class deepDec(nn.Module):
    def __init__(self, latent_dim: int, scale_value: float) -> None:
        super().__init__()
        
        self.latent_dim = latent_dim
        self.scale_value = scale_value

        out_c = [256, 128, 64, 32, 16]

        # latent_dim, 1, 1
        self.block0 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, out_c[0], kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(out_c[0]),
            nn.Tanh(), # ? 
        )

        # 256, 3, 3
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(out_c[0], out_c[1], kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_c[1]),
            nn.ReLU(),
        )

        # 128, 4, 4
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(out_c[1], out_c[2], kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_c[2]),
            nn.ReLU(),
        )
        
        # 64, 8, 8
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(out_c[2], out_c[3], kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_c[3]),
            nn.ReLU(),
        )

        # 32, 16, 16
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(out_c[3], out_c[4], kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_c[4]),
            nn.Tanh(), # ? 
        )

        # 16, 32, 32
        self.block5 = nn.Sequential( 
            nn.Conv2d(out_c[4], 3, kernel_size=(1, 1), stride=1, padding=0, bias=True),
            nn.Sigmoid() 
        )
    

    def forward(self, z):
        # print(z.shape)
        
        z = z.unsqueeze(-1).unsqueeze(-1)
        # print(z.shape)
        x = self.block0(z.view(-1, *z.size()[-3:]))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        out = self.block5(x)
        out = out.view(*z.size()[:-3], *out.size()[1:]) # [1, bs, 3, 32, 32] 

        return out, torch.tensor(self.scale_value).to(z.device)
    

"""