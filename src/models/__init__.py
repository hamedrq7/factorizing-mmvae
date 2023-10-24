from .vae_mnist import MNIST as VAE_mnist
from .vae_svhn import SVHN as VAE_svhn
from .mmvae_mnist_svhn import MNIST_SVHN as VAE_mnist_svhn
from .fumvae import FUMMVAE as VAE_fummvae

__all__ = [VAE_mnist, VAE_svhn, VAE_mnist_svhn, VAE_fummvae]
