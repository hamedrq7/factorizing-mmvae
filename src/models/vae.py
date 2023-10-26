# Base VAE class definition

import torch
import torch.nn as nn

from utils import get_mean, kl_divergence
from vis import embed_umap, tensors_to_df
import torch.distributions as distributions


class VAE(nn.Module):
    def __init__(self, enc, dec, params):
        super(VAE, self).__init__()
        self.pz = getattr(distributions, params.distr) # prior
        self.px_z = getattr(distributions, params.distr) # likelihood
        self.qz_x = getattr(distributions, params.distr) # posterior
        print(params.distr)

        self.enc = enc
        self.dec = dec
        self.modelName = None
        self.params = params
        self._pz_params = None  # defined in subclass
        self._qz_x_params = None  # populated in `forward`
        self.llik_scaling = 1.0

    @property
    def pz_params(self):
        return self._pz_params

    @property
    def qz_x_params(self):
        if self._qz_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qz_x_params

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        raise NotImplementedError

    def forward(self, x, K=1):
        '''
        Add more Comments, explain more
        '''
        self._qz_x_params = self.enc(x) # mu([0]): [bs, latent_dim], sigma([1]): [bs, latent_dim]
        qz_x = self.qz_x(*self._qz_x_params) 
        
        zs = qz_x.rsample(torch.Size([K])) # [K=1, bs, latent_dim]

        # self.dec(zs)[0],[1]: [1, bs, imgChnl, H, W]
        px_z = self.px_z(*self.dec(zs)) # one batch of samples from px_z (equivelant to calling .rsample()) gives [1, bs, imgChnl, H, W]

        return qz_x, px_z, zs

    def generate(self, N, K, only_mean: bool=False):
        """
        Samples from pz and generates images using decoder, then saves it

        Params
            only_mean: bool
                if True, only use means of p(x|z) to plot
                if False, 
        """
        self.eval()
        with torch.no_grad():
            # pz_params[0] and [1]: [1, 20]
            pz = self.pz(*self.pz_params) # each sample from pz gives [1, 20]

            latents = pz.rsample(torch.Size([N])) # [N, 1, 20]
            # self.dec(latents)[0] -> [N, 1, imgChnl, H, W]
            # self.dec(latents)[1] -> [] an scalar (broadcasts and has same effect as [N, 1, imgChnl, H, W]

            # if you want to change scale if generated images manually, change it here:
            # pass an scale to self.px_z(), e.g:
            # px_z = self.px_z(self.dec(latents)[0], scale=0.1)
            px_z = self.px_z(*self.dec(latents)) # each sample from px_z is in shape of [N, 1, imgChnl, H, W]
            # in a way, there are 64(N) number of Normal distributions in px_z

            # for a better pictures, sample from the peak (mean) -> get_mean 
            # if not only_mean, then return each image with a variance noise along pixels, the 'variance' is the scale parameter given to `px_z`
            data = px_z.sample(torch.Size([K])) if not only_mean else get_mean(px_z) # [K, N, 1, imgChnl, H, W]

        if only_mean:
            data = data[None, :, :, :, :, :]

        return data.view(-1, *data.size()[3:]) # [N*k, imgChnl, H, W]

    def reconstruct(self, data):
        '''assmung data is in shape [N, imgChnl, H, W]'''
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data)) # each sample from qz_x is [N, latent_dim] 
            latents = qz_x.rsample()  # Author's comment: no dim expansion
            px_z = self.px_z(*self.dec(latents)) # each sampel from px_z is [N, imgChnl, H, W]
            recon = get_mean(px_z)
        return recon

    def analyse(self, data, K):
        """
        Passes the data through model, 
            zs: gets zs, K samples from qz_x (posterior dist, parametarized by encoder(data))
            qz_x: gets posterior distr. parametrized by data. 
        creates prior distr. using model's pz params,
            if learn_prior = False, the prior distr. is just a {normal} dist (mean0, var1)
        """
        self.eval()
        with torch.no_grad():
            qz_x, _, zs = self.forward(data, K=K) 
            # zs : [k, batch_size, 1, latent_dim]
            # qz_x: the posterior distr. 

            # create prior distr. using prior params of model (in a normal case, mu=0, sigma=1)
            pz = self.pz(*self.pz_params) # one sample: [1, latent_dim]
            pz_samples = pz.sample(torch.Size([K, data.size(0)])).view(-1, pz.batch_shape[-1])

            zss = [pz_samples, zs.view(-1, zs.size(-1))]
            # zss[0]: [k*batch_size, latent_dim]
            # zss[1]: [k*batch_size, latent_dim]
            
            # sets label 0 for prior samples and label 1 for posterior samples (mnist/svhn) 
            zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)] 
            
            kls_df = tensors_to_df(
                [kl_divergence(qz_x, pz).cpu().numpy()],
                head='KL',
                keys=[r'KL$(q(z|x)\,||\,p(z))$'],
                ax_names=['Dimensions', r'KL$(q\,||\,p)$']
            )
        
        # reutrn samples (reduced to 2D), samples labels and kl stuff
        return embed_umap(torch.cat(zss, 0).cpu().numpy()), \
            torch.cat(zsl, 0).cpu().numpy(), \
            kls_df
