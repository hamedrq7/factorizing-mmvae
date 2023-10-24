# MNIST model specification

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod, sqrt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from utils import Constants
from vis import plot_embeddings, plot_kls_df
from .vae import VAE

# Constants for FC Networks
dataSize = torch.Size([1, 28, 28])
data_dim = int(prod(dataSize))

def extra_hidden_layer(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(True))

# Constants for Conv Networks
imgChans = dataSize[0]
# fBase = 32  # base size of filter channels


###################################### 
### FC Networks 
###################################### 
class fcEnc(nn.Module):
    """ Generate latent parameters for MNIST image data. """
    def __init__(self, latent_dim, do_softmax: bool, hidden_dims=[400]):
        self.do_softmax = do_softmax
        super(fcEnc, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(data_dim, hidden_dims[0]), nn.ReLU(True)))
        for layer_indx in range(1, len(hidden_dims)):
            modules.append(extra_hidden_layer(hidden_dims[layer_indx-1], hidden_dims[layer_indx]))
        
        self.enc = nn.Sequential(*modules)
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.sigma_layer = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-3], -1))  # flatten data
        mu = self.mu_layer(e)
        sigma = self.sigma_layer(e)

        if self.do_softmax:
            return mu, F.softmax(sigma, dim=-1) * sigma.size(-1) + Constants.eta
        else: 
            return mu, torch.exp(0.5*sigma) + Constants.eta

class fcDec(nn.Module):
    """ Generate an MNIST image given a sample from the latent space. """
    def __init__(self, latent_dim, scale_value: float, hidden_dims=[400]):
        super(fcDec, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(latent_dim, hidden_dims[0]), nn.ReLU(True)))
        # modules.extend([extra_hidden_layer() for _ in range(num_hidden_layers - 1)])
        for layer_indx in range(1, len(hidden_dims)):
            modules.append(extra_hidden_layer(hidden_dims[layer_indx-1], hidden_dims[layer_indx]))

        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(hidden_dims[-1], data_dim)
        self.scale_value = scale_value

    def forward(self, z):
        p = self.fc3(self.dec(z)) 
        d = torch.sigmoid(p.view(*z.size()[:-1], *dataSize))  # reshape data
        
        d = d.clamp(Constants.eta, 1 - Constants.eta)
        
        return d, torch.tensor(self.scale_value).to(z.device)  # mean, length scale

###################################### 
### Conv Networks
###################################### 
class Enc(nn.Module):
    """ Generate latent parameters for MNIST image data. """
    def __init__(self, fBase: int, latent_dim: int, do_softmax: bool):
        # fBase: base size of filter channels
        super(Enc, self).__init__()
        self.fBase = fBase
        self.enc = nn.Sequential(
            # input size: 1 x 28 x 28
            nn.Conv2d(imgChans, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 14 x 14
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 7 x 7
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 3 x 3
        )
        self.mu_layer = nn.Conv2d(fBase * 4, latent_dim, 3, 1, 0, bias=True)
        self.sigma_layer = nn.Conv2d(fBase * 4, latent_dim, 3, 1, 0, bias=True)
        # c1, c2 size: latent_dim x 1 x 1

        self.do_softmax = do_softmax

    def forward(self, x):
        e = self.enc(x)
        mu = self.mu_layer(e).squeeze()
        sigma = self.sigma_layer(e).squeeze()

        if self.do_softmax:    
            return mu, F.softmax(sigma, dim=-1) * sigma.size(-1) + Constants.eta
        else: # complete
            return mu, torch.exp(0.5*sigma) + Constants.eta
            

class Dec(nn.Module):
    """ Generate a MNIST image given a sample from the latent space. """

    def __init__(self, fBase: int, latent_dim, scale_value: float):
        # fBase: base size of filter channels
        super(Dec, self).__init__() 
        self.fBase = fBase
        self.dec = nn.Sequential(
            # size: (latent_dim) x 1 x 1 
            nn.ConvTranspose2d(latent_dim, fBase * 4, 4, 1, 0, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 3, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 7 x 7
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 14 x 14
            nn.ConvTranspose2d(fBase, imgChans, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 1 x 28 x 28
        )
        self.scale_value = scale_value

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        # z = z.clamp(Constants.eta, 1-Constants.eta)

        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        return out, torch.tensor(self.scale_value).to(z.device)  # mean, length scale, 0.75 

class MNIST(VAE):
    """ Derive a specific sub-class of a VAE for MNIST. """

    def __init__(self, params):
        self.do_softmax = params.softmax
        super(MNIST, self).__init__(
            enc=Enc(params.fBase, params.latent_dim, do_softmax=params.softmax) if not params.no_conv
                else fcEnc(params.latent_dim, do_softmax=params.softmax, hidden_dims=params.hidden_dims),
            
            dec=Dec(params.fBase, params.latent_dim, scale_value=params.decoder_scale) if not params.no_conv
                else fcDec(params.latent_dim, scale_value=params.decoder_scale, hidden_dims=params.hidden_dims),
            
            params=params
        )
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'mnist'
        self.dataSize = dataSize
        self.llik_scaling = 1.

    @property
    def pz_params(self):
        if self.do_softmax:
            return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)
        else:
            return self._pz_params[0], torch.exp(0.5*self._pz_params[1])

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        tx = transforms.ToTensor()
        train = DataLoader(datasets.MNIST('../data', train=True, download=True, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(datasets.MNIST('../data', train=False, download=True, transform=tx),
                          batch_size=batch_size, shuffle=False, **kwargs) # set suffle of test to False, 
                                                                            # so you can track the same batch in .analysis()
        return train, test

    def generate(self, runPath, epoch, only_mean: bool=False):
        N = 64
        K = 1 if only_mean else 9
        samples = super(MNIST, self).generate(N, K, only_mean).cpu()
        # wrangle things so they come out tiled


        samples = samples.view(K, N, *samples.size()[1:]).transpose(0, 1)  # N x K x 1 x 28 x 28
        s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples]
        save_image(torch.stack(s),
                   '{}/mnist_gen_samples_{}_{:03d}.png'.format(runPath, 'mean' if only_mean else '', epoch),
                   nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch, is_train: bool=False):
        recon = super(MNIST, self).reconstruct(data[:8])
        comp = torch.cat([data[:8], recon]).data.cpu()
        save_image(comp, '{}/mnist_{}_recon_{:03d}.png'.format(runPath, 'train' if is_train else 'test', epoch))

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super(MNIST, self).analyse(data, K=10)
        labels = ['Prior Samples', f'posterior samples - {self.modelName.lower()}']
        plot_embeddings(zemb, zsl, labels, '{}/mnist_emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/mnist_kl_distance_{:03d}.png'.format(runPath, epoch))
