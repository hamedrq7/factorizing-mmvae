import argparse
import datetime
import sys
import json
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp


import numpy as np
import torch
from torch import optim
from tqdm import tqdm 

import models
import objectives
from utils import Logger, Timer, save_model, save_vars, unpack_data
from tqdm import trange
from torchsummary import summary
from vis import custom_plot_loss

parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
parser.add_argument('--experiment', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--model', type=str, default='mnist_svhn', metavar='M',
                    choices=[s[4:] for s in dir(models) if 'VAE_' in s],
                    help='model name (default: mnist_svhn)')

# parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H',
#                     help='number of hidden layers in enc and dec (default: 1)')

parser.add_argument('--obj', type=str, default='elbo', metavar='O',
                    choices=['elbo', 'iwae', 'dreg', 'elbo_naive', 'modified_elbo_naive'],
                    help='objective to use (default: elbo)')
parser.add_argument('--K', type=int, default=20, metavar='K',
                    help='number of particles to use for iwae/dreg (default: 10)')
parser.add_argument('--looser', action='store_true', default=False,
                    help='use the looser version of IWAE/DREG')
parser.add_argument('--llik_scaling', type=float, default=0.,
                    help='likelihood scaling for cub images/svhn modality when running in'
                         'multimodal setting, set as 0 to use default value')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size for data (default: 256)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--latent-dim', type=int, default=20, metavar='L',
                    help='latent dimensionality (default: 20)')
parser.add_argument('--pre-trained', type=str, default="",
                    help='path to pre-trained model (train from scratch if empty)')
parser.add_argument('--learn-prior', action='store_true', default=False,
                    help='learn model prior parameters')
parser.add_argument('--logp', action='store_true', default=False,
                    help='estimate tight marginal likelihood on completion')
parser.add_argument('--print-freq', type=int, default=0, metavar='f',
                    help='frequency with which to print stats (default: 0)')
parser.add_argument('--no-analytics', action='store_true', default=False,
                    help='disable plotting analytics')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# Added Arguments:
parser.add_argument('--distr', type=str, default='Normal', metavar='M',
                    choices=['Laplace', 'Normal'],
                    help='distribution used for modeling prior, posterior and likelihood (default: Normal)')
parser.add_argument('--decoder_scale', type=float, default=0.75,
                    help='decoder_scale, its a hyperparameter and needs tuning')
# parser.add_argument('--kl_alpha', type=float, default=1.00, # TODO add normalization
#                     help='coeff used to multiply KLD')
parser.add_argument('--no_conv', action='store_true', default=False,
                    help='use FC model for svhn')
parser.add_argument('--hidden_dims', metavar='N', type=int, nargs='+', 
                    default=[400], # TODO add this functionality to Conv networks
                    help='list of hidden layers dim of FC networks, default=[400]')
parser.add_argument('--fBase', type=int, default=32, metavar='L',
                    help='base size of filter channels (default: 32)')
parser.add_argument('--softmax', action='store_true', default=False,
                    help='apply softmax to scale of Laplace Distrubution')
parser.add_argument('--max_d', type=int, default=10000, metavar='N',
                    help='maximum number of datapoints per class (default: 10000) for multimodal setting ')
parser.add_argument('--dm', type=int, default=30, metavar='N',
                    help='data multiplier: random permutations to match (default: 30) - for multimodal setting ') 

# args
args = parser.parse_args()
# keep it?
args.softmax = True if args.distr == 'Laplace' else False


# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# load args from disk if pretrained model path is given
pretrained_path = ""
if args.pre_trained:
    pretrained_path = args.pre_trained
    args = torch.load(args.pre_trained + '/args.rar')

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(device)

# load model
modelC = getattr(models, 'VAE_{}'.format(args.model))
model = modelC(args).to(device)

if args.model == 'mnist':
    summary(model.enc, (1, 28, 28))
    summary(model.dec, (1, args.latent_dim))
elif args.model == 'svhn':
    summary(model.enc, (3, 32, 32))
    summary(model.dec, (1, args.latent_dim))

print(model) 

if pretrained_path:
    print('Loading model {} from {}'.format(model.modelName, pretrained_path))
    model.load_state_dict(torch.load(pretrained_path + '/model.rar'))
    model._pz_params = model._pz_params

if not args.experiment:
    args.experiment = model.modelName

# set up run path
runId = datetime.datetime.now().isoformat()
experiment_dir = Path('../experiments/' + args.experiment)
experiment_dir.mkdir(parents=True, exist_ok=True)
runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))

last_epoch_res = Path(f'{runPath}/final-gen-recons') # for last epoch gen/recons
last_epoch_res.mkdir(parents=True, exist_ok=True)

sys.stdout = Logger('{}/run.log'.format(runPath))
print('Expt:', runPath)
print('RunID:', runId)
print('args: ', args)

# save args to run
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
# -- also save object because we want to recover these for other things
torch.save(args, '{}/args.rar'.format(runPath))

# preparation for training
lr = 1e-3
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=lr, amsgrad=True)
if hasattr(model, 'vaes'):
    train_loader, test_loader = model.getDataLoaders(args.batch_size, device=device, max_d=args.max_d, dm=args.dm)
else:    
    train_loader, test_loader = model.getDataLoaders(args.batch_size, device=device)

objective = getattr(objectives,
                    ('m_' if hasattr(model, 'vaes') else '')
                    + args.obj
                    + ('_looser' if (args.looser and args.obj != 'elbo') else ''))

# Author originaly used IWAE loss for evaluating test
# t_objective = getattr(objectives, ('m_' if hasattr(model, 'vaes') else '') + 'iwae')
t_objective = getattr(objectives,
                    ('m_' if hasattr(model, 'vaes') else '')
                    + args.obj
                    + ('_looser' if (args.looser and args.obj != 'elbo') else ''))

def train_factorized(epoch, agg):
    model.train()
    b_loss = 0
    i=0
    for dataT in tqdm(train_loader):
        # dataT[0] -> images
        # dataT[1] -> class labels
        data = unpack_data(dataT, device=device) # [bs, num_channels, img_size, img_size]
        # for mnist_svhn multimodal setting: 
        # data[0] -> [bs, 1, 28, 28]
        # data[2] -> [bs, 3, 32, 32]

        optimizer.zero_grad()
        loss_mmvae, loss_mnist, loss_svhn = objective(model, data, agg, phase='train', K=args.K)
        loss_mmvae.backward()
        loss_mnist.backward()
        loss_svhn.backward()

        optimizer.step()

        if i == 0: 
            model.reconstruct(data, runPath, epoch, is_train=True)
        
        i += 1
        
def train(epoch, agg):
    model.train()
    b_loss = 0

    i = 0
    for dataT in tqdm(train_loader):
        # dataT[0] -> images
        # dataT[1] -> class labels
        data = unpack_data(dataT, device=device) # [bs, num_channels, img_size, img_size]
        
        # for mnist_svhn multimodal setting: 
        # data[0] -> [bs, 1, 28, 28]
        # data[1] -> [bs, 3, 32, 32]

        optimizer.zero_grad()
        loss = -objective(model, data, agg, phase='train', K=args.K)
        loss.backward()
        optimizer.step()
        
        b_loss += loss.item()

        if args.print_freq > 0 and i % args.print_freq == 0:
            print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / args.batch_size))

        if i == 0:
            model.reconstruct(data, runPath, epoch, is_train=True)

        i += 1

    num_batches, num_data = len(train_loader), len(train_loader.dataset)
    agg['train_epoch_lpx_z'].append(np.mean(agg['train_lpx_z'][-num_batches:]))
    agg['train_epoch_kl'].append(np.mean(agg['train_kl'][-num_batches:]))
    agg['train_epoch_loss'].append(b_loss / num_data)

    print('====> Epoch: {:03d} Train loss: {:.4f}, Train lpx_z: {:.4f}, Train kl: {:.4f}'.format(epoch, agg['train_epoch_loss'][-1], \
        agg['train_epoch_lpx_z'][-1], agg['train_epoch_kl'][-1]))

def test(epoch, agg):
    model.eval()
    b_loss = 0
    with torch.no_grad():
        
        i = 0
        for dataT in tqdm(test_loader):
            data = unpack_data(dataT, device=device)
            
            loss = -t_objective(model, data, agg, phase='test', K=args.K)
            b_loss += loss.item()

            if epoch == args.epochs and (i < 10): 
                model.reconstruct(data, f'{runPath}/final-gen-recons', epoch+i)
                # reconstruct and kl_map more samples in last epoch
            if i == 0:
                model.reconstruct(data, runPath, epoch)
                if not args.no_analytics:
                    model.analyse(data, runPath, epoch)
            i += 1

    num_batches, num_data = len(test_loader), len(test_loader.dataset)
    agg['test_epoch_lpx_z'].append(np.mean(agg['test_lpx_z'][-num_batches:]))
    agg['test_epoch_kl'].append(np.mean(agg['test_kl'][-num_batches:]))
    agg['test_epoch_loss'].append(b_loss / num_data)

    print('====> Epoch: {:03d} Test loss: {:.4f}, Test lpx_z: {:.4f}, Test kl: {:.4f}'.format(epoch, agg['test_epoch_loss'][-1], \
        agg['test_epoch_lpx_z'][-1], agg['test_epoch_kl'][-1]))


def estimate_log_marginal(K):
    """Compute an IWAE estimate of the log-marginal likelihood of test data."""
    model.eval()
    marginal_loglik = 0
    with torch.no_grad():
        for dataT in test_loader:
            data = unpack_data(dataT, device=device)
            marginal_loglik += -t_objective(model, data, K).item()

    marginal_loglik /= len(test_loader.dataset)
    print('Marginal Log Likelihood (IWAE, K = {}): {:.4f}'.format(K, marginal_loglik))

print('train #batches: ', len(train_loader))
print('test #batches: ', len(test_loader))
print(runPath)

if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        agg = defaultdict(list)

        for epoch in trange(1, args.epochs + 1):
            train(epoch, agg)
            test(epoch, agg)
            save_model(model, runPath + '/model.rar')
            save_vars(agg, runPath + '/losses.rar')
            # model.generate(runPath, epoch)
            model.generate(runPath, epoch, only_mean=True)
            
            custom_plot_loss(loss_dict=agg, keys_to_plot=['train_lp0_x0_z_0', 'train_lp1_x1_z_1'], title='normal reconstruction loss', name_to_save='train normal gen loss', runPath=runPath)
            custom_plot_loss(loss_dict=agg, keys_to_plot=['train_lp1_x1_z_0', 'train_lp0_x0_z_1'], title='cross reconstruction loss', name_to_save='train cross gen loss', runPath=runPath)
            custom_plot_loss(loss_dict=agg, keys_to_plot=['train_kl_q0_pz', 'train_kl_q1_pz'], title='kl losses', name_to_save='kl losses', runPath=runPath)
        
        # generate more samples
        for gen_indx in range(10):
            model.generate(f'{runPath}/final-gen-recons', epoch+gen_indx, only_mean=True)
            
        if args.logp:  # compute as tight a marginal likelihood as possible
            estimate_log_marginal(5000)

