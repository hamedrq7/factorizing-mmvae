import torch
from numpy import prod
from typing import Dict
from utils import log_mean_exp, is_multidata, kl_divergence
import math

# helper to vectorise computation
def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)

def elbo(model, x, loss_log: Dict, phase: str, K=1):
    # x: [bs, imgChnl, H, W]
    """Computes E_{p(x)}[ELBO] """
    qz_x, px_z, zs = model(x) # px_z.loc: [1, bs, imgChnl, H, W], 

    # px_z.log_prob(x).shape: [1, bs, imgChnl, H, W]
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.llik_scaling # [1, bs, imgChnl*H*W]

    kld = kl_divergence(qz_x, model.pz(*model.pz_params)) # [bs, latent_dim]

    lpx_z_sum = lpx_z.sum(-1) # [1, bs], sum over all variables==(pixels)
    kl_sum = kld.sum(-1) # [bs], sum over latent dim vars
    
    temp = (lpx_z_sum - kl_sum).mean(0) # [bs], the - operand broadcasts, the mean(0) is probably for avging over K batches (alhtough K=1)
    # TODO: the mean(0) is not across batch_size, is it okay?
    
    loss_log[f'{phase}_lpx_z'].append(lpx_z_sum.mean().item())
    loss_log[f'{phase}_kl'].append(kl_sum.mean().item())

    return temp.sum()

def _iwae(model, x, K):
    """IWAE estimate for log p_\theta(x) -- fully vectorised."""
    qz_x, px_z, zs = model(x, K)
    

    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.llik_scaling
    lqz_x = qz_x.log_prob(zs).sum(-1)
    return lpz + lpx_z.sum(-1) - lqz_x


def iwae(model, x, K):
    """Computes an importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    """
    S = compute_microbatch_split(x, K)
    lw = torch.cat([_iwae(model, _x, K) for _x in x.split(S)], 1)  # concat on batch
    return log_mean_exp(lw).sum()


def _dreg(model, x, K):
    """DREG estimate for log p_\theta(x) -- fully vectorised."""
    _, px_z, zs = model(x, K)
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.llik_scaling
    qz_x = model.qz_x(*[p.detach() for p in model.qz_x_params])  # stop-grad for \phi
    lqz_x = qz_x.log_prob(zs).sum(-1)
    lw = lpz + lpx_z.sum(-1) - lqz_x
    return lw, zs


def dreg(model, x, K, regs=None):
    """Computes a doubly-reparameterised importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    """
    S = compute_microbatch_split(x, K)
    lw, zs = zip(*[_dreg(model, _x, K) for _x in x.split(S)])
    lw = torch.cat(lw, 1)  # concat on batch
    zs = torch.cat(zs, 1)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 0, keepdim=True)).exp()
        if zs.requires_grad:
            zs.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).sum()


# multi-modal variants
def m_elbo_naive(model, x, agg, phase: str, K=1):
    """Computes E_{p(x)}[ELBO] for multi-modal vae --- NOT EXPOSED"""
    qz_xs, px_zs, zss = model(x, K=1) 

    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        klds.append(kld.sum(-1))
        for d, px_z in enumerate(px_zs[r]):
            lpx_z = px_z.log_prob(x[d]) * model.vaes[d].llik_scaling
            
            lpx_zs.append(lpx_z.view(*px_z.batch_shape[:2], -1).sum(-1))
    '''
    klds
    ----
        r = 0:
            klds[0]: Dkl(q_theta_1 (z|x_1) || p_z) [bs]
        
        r = 1:
            klds[1]: Dkl(q_theta_2 (z|x_2) || p_z) [bs]

    lpx_zs
    ------
        r = 0:
            d = 0, lpx_zs[0]: log prob (p_theta_1 (x_1|zss[0])) given (x_1) [k, bs]
            d = 1, lpx_zs[1]: log prob (p_theta_2 (x_2|zss[0])) given (x_2) [k, bs]

        r = 1:
            d = 0, lpx_zs[2]: log prob (p_theta_1 (x_1|zss[1])) given (x_1) [k, bs]
            d = 1, lpx_zs[3]: log prob (p_theta_2 (x_2|zss[1])) given (x_2) [k, bs]
    '''

    '''
    torch.stack(lpx_zs):            [4, k, bs]
    torch.stack(lpx_zs).sum(0):     [k, bs]
    torch.stack(klds):              [2, bs]
    torch.stack(klds).sum():        [bs]
    '''    
    
    agg[f'{phase}_lp0_x0_z_0'].append(lpx_zs[0].mean().item())
    agg[f'{phase}_lp1_x1_z_0'].append(lpx_zs[1].mean().item())
    agg[f'{phase}_lp0_x0_z_1'].append(lpx_zs[2].mean().item())
    agg[f'{phase}_lp1_x1_z_1'].append(lpx_zs[3].mean().item())
    agg[f'{phase}_kl_q0_pz'].append(klds[0].mean().item())
    agg[f'{phase}_kl_q1_pz'].append(klds[1].mean().item())
    
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    # [K, bs]
    
    return obj.mean(0).sum() # mean across K, the sum across batch_size


def m_elbo(model, x, agg, phase, K=1):
    """Computes importance-sampled m_elbo (in notes3) for multi-modal vae """

    qz_xs, px_zs, zss = model(x)

    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):

        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        klds.append(kld.sum(-1))
        for d in range(len(px_zs)):
            lpx_z = px_zs[d][d].log_prob(x[d]).view(*px_zs[d][d].batch_shape[:2], -1)
            lpx_z = (lpx_z * model.vaes[d].llik_scaling).sum(-1)
            if d == r:
                lwt = torch.tensor(0.0)
            else:
                zs = zss[d].detach()
                
                lwt = (qz_x.log_prob(zs) - qz_xs[d].log_prob(zs).detach()).sum(-1)
            
            if r==0 and d==1:
                agg[f'{phase}_lp1_x1_z_0'].append(lpx_z.mean().item())
            elif r==1 and d==0:
                agg[f'{phase}_lp0_x0_z_1'].append(lpx_z.mean().item())

            lpx_zs.append(lwt.exp() * lpx_z)

    '''
    klds
    ----
        r = 0, qz_x = q_theta_1 (z|x_1)
            klds[0] = Dkl(q_theta_1 (z|x_1), pz)
        r = 1, qz_x = q_theta_2 (z|x_2)
            klds[1] = Dkl(q_theta_2 (z|x_2), pz)
        
        lpx_zs:
            r = 0:
                qz_x = q_theta_1 (z|x_1)

                d = 0: 
                    lpx_z = log prob p_theta_1 (x_1|zss[0])
                    lwt = 0
            
                d = 1:
                    lpx_z = log prob p_theta_2 (x_2|zss[1])
                    lwt = log prob q_theta_1 (zss[1]) - log prob q_theta_2 (zss[1]) .sum()

            r = 1: 
                qz_x = q_theta_2 (z|x_2)

                d = 0:
                    lpx_z = log prob p_theta_1 (x_1|zss[0])
                    lwt = log prob q_theta_2 (zss[0]) - log prob q_theta_1 (zss[0])    
    '''
    
    agg[f'{phase}_lp0_x0_z_0'].append(lpx_zs[0].mean().item())
    # agg[f'{phase}_lp1_x1_z_0'].append(lpx_zs[1].mean().item()) # r = 0, d = 1
    # agg[f'{phase}_lp0_x0_z_1'].append(lpx_zs[2].mean().item()) # r = 1, d = 0
    agg[f'{phase}_lp1_x1_z_1'].append(lpx_zs[3].mean().item())
    agg[f'{phase}_kl_q0_pz'].append(klds[0].mean().item())
    agg[f'{phase}_kl_q1_pz'].append(klds[1].mean().item())

    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    return obj.mean(0).sum()



def m_elbo_all(model, x, loss_log, phase, K=1):
    # print('x[0] shape', x[0].shape)
    # print('x[1] shape', x[1].shape)

    # ELBO uni-mnist
    qz_x_mnist, px_z_mnist, zs_mnist = model.vaes[0](x[0])  
    lpx_z = px_z_mnist.log_prob(x[0]).view(*px_z_mnist.batch_shape[:2], -1) * model.vaes[0].llik_scaling 
    kld = kl_divergence(qz_x_mnist, model.vaes[0].pz(*model.vaes[0].pz_params)) 
    lpx_z_sum = lpx_z.sum(-1) 
    kl_sum = kld.sum(-1) 
    loss_log[f'mnist_{phase}_lpx_z'].append(lpx_z_sum.mean().item())
    loss_log[f'mnist_{phase}_kl'].append(kl_sum.mean().item())
    uni_mnist_loss = (lpx_z_sum - kl_sum).mean(0).sum()
    
    # ELBO uni-svhn
    qz_x_svhn, px_z_svhn, zs_svhn = model.vaes[1](x[1])  
    lpx_z = px_z_svhn.log_prob(x[1]).view(*px_z_svhn.batch_shape[:2], -1) * model.vaes[1].llik_scaling 
    kld = kl_divergence(qz_x_svhn, model.vaes[1].pz(*model.vaes[1].pz_params)) 
    lpx_z_sum = lpx_z.sum(-1) 
    kl_sum = kld.sum(-1) 
    loss_log[f'svhn_{phase}_lpx_z'].append(lpx_z_sum.mean().item())
    loss_log[f'svhn_{phase}_kl'].append(kl_sum.mean().item())
    uni_svhn_loss = (lpx_z_sum - kl_sum).mean(0).sum()
    
    # MMVAE naive ELBO: 
    qz_xs_mmvae, px_zs_mmvae, zss_mmvae = model.mmvae(x, K=1) 
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs_mmvae):
        kld = kl_divergence(qz_x, model.mmvae.pz(*model.mmvae.pz_params))
        klds.append(kld.sum(-1))
        for d, px_z in enumerate(px_zs_mmvae[r]):
            lpx_z = px_z.log_prob(x[d]) * model.mmvae.vaes[d].llik_scaling
            
            lpx_zs.append(lpx_z.view(*px_z.batch_shape[:2], -1).sum(-1))
    
    loss_log[f'mmvae_{phase}_lp0_x0_z_0'].append(lpx_zs[0].mean().item())
    loss_log[f'mmvae_{phase}_lp1_x1_z_0'].append(lpx_zs[1].mean().item())
    loss_log[f'mmvae_{phase}_lp0_x0_z_1'].append(lpx_zs[2].mean().item())
    loss_log[f'mmvae_{phase}_lp1_x1_z_1'].append(lpx_zs[3].mean().item())
    loss_log[f'mmvae_{phase}_kl_q0_pz'].append(klds[0].mean().item())
    loss_log[f'mmvae_{phase}_kl_q1_pz'].append(klds[1].mean().item())
    
    mmvae_loss = (1 / len(model.mmvae.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    mmvae_loss = mmvae_loss.mean(0).sum()

    # return -mmvae_loss, -uni_mnist_loss, -uni_svhn_loss
    total_loss = mmvae_loss + uni_mnist_loss + uni_svhn_loss
    return total_loss

def m_infoNCE_naive(model, x, loss_log, phase, K=1):
    # ELBO uni-mnist
    qz_x_mnist, px_z_mnist, zs_mnist = model.vaes[0](x[0])  
    lpx_z = px_z_mnist.log_prob(x[0]).view(*px_z_mnist.batch_shape[:2], -1) * model.vaes[0].llik_scaling 
    kld = kl_divergence(qz_x_mnist, model.vaes[0].pz(*model.vaes[0].pz_params)) 
    lpx_z_sum = lpx_z.sum(-1) 
    kl_sum = kld.sum(-1) 
    loss_log[f'mnist_{phase}_lpx_z'].append(lpx_z_sum.mean().item())
    loss_log[f'mnist_{phase}_kl'].append(kl_sum.mean().item())
    uni_mnist_loss = (lpx_z_sum - kl_sum).mean(0).sum()
    
    # ELBO uni-svhn
    qz_x_svhn, px_z_svhn, zs_svhn = model.vaes[1](x[1])  
    lpx_z = px_z_svhn.log_prob(x[1]).view(*px_z_svhn.batch_shape[:2], -1) * model.vaes[1].llik_scaling 
    kld = kl_divergence(qz_x_svhn, model.vaes[1].pz(*model.vaes[1].pz_params)) 
    lpx_z_sum = lpx_z.sum(-1) 
    kl_sum = kld.sum(-1) 
    loss_log[f'svhn_{phase}_lpx_z'].append(lpx_z_sum.mean().item())
    loss_log[f'svhn_{phase}_kl'].append(kl_sum.mean().item())
    uni_svhn_loss = (lpx_z_sum - kl_sum).mean(0).sum()
    
    # MMVAE naive ELBO: 
    qz_xs_mmvae, px_zs_mmvae, zss_mmvae = model.mmvae(x, K=1) 
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs_mmvae):
        kld = kl_divergence(qz_x, model.mmvae.pz(*model.mmvae.pz_params))
        klds.append(kld.sum(-1))
        for d, px_z in enumerate(px_zs_mmvae[r]):
            lpx_z = px_z.log_prob(x[d]) * model.mmvae.vaes[d].llik_scaling
            
            lpx_zs.append(lpx_z.view(*px_z.batch_shape[:2], -1).sum(-1))
    
    loss_log[f'mmvae_{phase}_lp0_x0_z_0'].append(lpx_zs[0].mean().item())
    loss_log[f'mmvae_{phase}_lp1_x1_z_0'].append(lpx_zs[1].mean().item())
    loss_log[f'mmvae_{phase}_lp0_x0_z_1'].append(lpx_zs[2].mean().item())
    loss_log[f'mmvae_{phase}_lp1_x1_z_1'].append(lpx_zs[3].mean().item())
    loss_log[f'mmvae_{phase}_kl_q0_pz'].append(klds[0].mean().item())
    loss_log[f'mmvae_{phase}_kl_q1_pz'].append(klds[1].mean().item())
    
    mmvae_loss = (1 / len(model.mmvae.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    mmvae_loss = mmvae_loss.mean(0).sum()

    # infoNCE: 
    loss_fn = torch.nn.CrossEntropyLoss() # reduction='none'

    ## MNIST
    # qz_x_mnist, zs_mnist 
    # qz_xs_mmvae[0], zss_mmvae[0]
    inf_mnist_1 = qz_xs_mmvae[0].log_prob(zss_mmvae[0]).exp() # ==1, # [K=1, bs, latent_dim] 
    inf_mnist_2 = qz_x_mnist.log_prob(zs_mnist).exp() # ==1, # [K=1, bs, latent_dim] 
    inf_mnist_3 = qz_x_mnist.log_prob(zss_mmvae[0]).exp() # ==0
    inf_mnist_4 = qz_xs_mmvae[0].log_prob(zs_mnist).exp() # ==0
    
    loss_log[f'{phase}_inf_mnist_1'].append(inf_mnist_1.mean().item()) # 1
    loss_log[f'{phase}_inf_mnist_2'].append(inf_mnist_2.mean().item()) # 1
    loss_log[f'{phase}_inf_mnist_3'].append(inf_mnist_3.mean().item()) # 1
    loss_log[f'{phase}_inf_mnist_4'].append(inf_mnist_4.mean().item()) # 1
    
    inf_mnist_1 = inf_mnist_1.mean(0)[:, None, :]
    inf_mnist_2 = inf_mnist_2.mean(0)[:, None, :]
    inf_mnist_3 = inf_mnist_3.mean(0)[:, None, :]
    inf_mnist_4 = inf_mnist_4.mean(0)[:, None, :]
    
    #### v1
    inf_mnist = torch.cat([inf_mnist_1, inf_mnist_2, inf_mnist_3, inf_mnist_4], dim=1) # [bs, 4, latent_dim]
    
    target_mnist = torch.cat([
        torch.ones((inf_mnist.shape[0], 2, inf_mnist.shape[-1]), device=inf_mnist.device),
        torch.zeros((inf_mnist.shape[0], 2, inf_mnist.shape[-1]), device=inf_mnist.device),
    ], dim=1)

    loss_fn = torch.nn.CrossEntropyLoss() # reduction='none'
    loss_inf_mnist = loss_fn(inf_mnist, target_mnist) # with reduction='none', size of the loss is [bs, latent_dim]

    #### v2
    # inf_mnist =  torch.cat([inf_mnist_1, inf_mnist_2, inf_mnist_3, inf_mnist_4], dim=1).mean(2) # MEAN ACROSS LATENT [bs, 4]
    # target_mnist = torch.cat([
    #     torch.ones((inf_mnist.shape[0], 2), device=inf_mnist.device),
    #     torch.zeros((inf_mnist.shape[0], 2), device=inf_mnist.device),
    # ], dim=1)

    # loss_inf_mnist = loss_fn(inf_mnist, target_mnist) # with reduction='none', size of the loss is [bs]

    #### v3
    # loss_inf_mnist = ((inf_mnist_1+inf_mnist_2)- (inf_mnist_3+ inf_mnist_4)).mean()

    ## SVHN
    # qz_x_svhn, zs_svhn
    # qz_xs_mmvae[1], zss_mmvae[1]
    inf_svhn_1 = qz_xs_mmvae[1].log_prob(zss_mmvae[1]).exp() # == 1
    inf_svhn_2 = qz_x_svhn.log_prob(zs_svhn).exp() # == 1 
    
    inf_svhn_3 = qz_x_svhn.log_prob(zss_mmvae[1]).exp() # ==0
    inf_svhn_4 = qz_xs_mmvae[1].log_prob(zs_svhn).exp() # ==0
    
    loss_log[f'{phase}_inf_svhn_1'].append(inf_svhn_1.mean().item()) # 1
    loss_log[f'{phase}_inf_svhn_2'].append(inf_svhn_2.mean().item()) # 1
    loss_log[f'{phase}_inf_svhn_3'].append(inf_svhn_3.mean().item()) # 1
    loss_log[f'{phase}_inf_svhn_4'].append(inf_svhn_4.mean().item()) # 1
    
    inf_svhn_1 = inf_svhn_1.mean(0)[:, None, :]
    inf_svhn_2 = inf_svhn_2.mean(0)[:, None, :]
    inf_svhn_3 = inf_svhn_3.mean(0)[:, None, :]
    inf_svhn_4 = inf_svhn_4.mean(0)[:, None, :]
    
    inf_svhn = torch.cat([inf_svhn_1, inf_svhn_2, inf_svhn_3, inf_svhn_4], dim=1) # [bs, 4, latent_dim]
    
    target_svhn = torch.cat([
        torch.ones((inf_svhn.shape[0], 2, inf_svhn.shape[-1]), device=inf_svhn.device),
        torch.zeros((inf_svhn.shape[0], 2, inf_svhn.shape[-1]), device=inf_svhn.device),
    ], dim=1)

    loss_inf_svhn = loss_fn(inf_svhn, target_svhn) # with reduction='none', size of the loss is [bs, latent_dim]

    loss_log[f'{phase}_loss_inf_mnist'].append(loss_inf_mnist.item())
    loss_log[f'{phase}_loss_inf_svhn'].append(loss_inf_svhn.item())
    
    alpha = 1.0
    total_loss = (mmvae_loss + uni_mnist_loss + uni_svhn_loss) - alpha*(loss_inf_mnist+loss_inf_svhn)
    return total_loss


def m_infoNCE_v2(model, x, loss_log, phase, K=1):
    # ELBO uni-mnist
    qz_x_mnist, px_z_mnist, zs_mnist = model.vaes[0](x[0])  
    lpx_z = px_z_mnist.log_prob(x[0]).view(*px_z_mnist.batch_shape[:2], -1) * model.vaes[0].llik_scaling 
    kld = kl_divergence(qz_x_mnist, model.vaes[0].pz(*model.vaes[0].pz_params)) 
    lpx_z_sum = lpx_z.sum(-1) 
    kl_sum = kld.sum(-1) 
    loss_log[f'mnist_{phase}_lpx_z'].append(lpx_z_sum.mean().item())
    loss_log[f'mnist_{phase}_kl'].append(kl_sum.mean().item())
    uni_mnist_loss = (lpx_z_sum - kl_sum).mean(0).sum()
    
    # ELBO uni-svhn
    qz_x_svhn, px_z_svhn, zs_svhn = model.vaes[1](x[1])  
    lpx_z = px_z_svhn.log_prob(x[1]).view(*px_z_svhn.batch_shape[:2], -1) * model.vaes[1].llik_scaling 
    kld = kl_divergence(qz_x_svhn, model.vaes[1].pz(*model.vaes[1].pz_params)) 
    lpx_z_sum = lpx_z.sum(-1) 
    kl_sum = kld.sum(-1) 
    loss_log[f'svhn_{phase}_lpx_z'].append(lpx_z_sum.mean().item())
    loss_log[f'svhn_{phase}_kl'].append(kl_sum.mean().item())
    uni_svhn_loss = (lpx_z_sum - kl_sum).mean(0).sum()
    
    # MMVAE naive ELBO: 
    qz_xs_mmvae, px_zs_mmvae, zss_mmvae = model.mmvae(x, K=1) 
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs_mmvae):
        kld = kl_divergence(qz_x, model.mmvae.pz(*model.mmvae.pz_params))
        klds.append(kld.sum(-1))
        for d in range(len(px_zs_mmvae)):
            lpx_z = px_zs_mmvae[d][d].log_prob(x[d]).view(*px_zs_mmvae[d][d].batch_shape[:2], -1)
            lpx_z = (lpx_z * model.mmvae.vaes[d].llik_scaling).sum(-1)
            if d == r:
                lwt = torch.tensor(0.0)
            else:
                zs = zss_mmvae[d].detach()
                
                lwt = (qz_x.log_prob(zs) - qz_xs_mmvae[d].log_prob(zs).detach()).sum(-1)

            lpx_zs.append(lwt.exp() * lpx_z)

    loss_log[f'mmvae_{phase}_lp0_x0_z_0'].append(lpx_zs[0].mean().item())
    loss_log[f'mmvae_{phase}_lp1_x1_z_0'].append(lpx_zs[1].mean().item())
    loss_log[f'mmvae_{phase}_lp0_x0_z_1'].append(lpx_zs[2].mean().item())
    loss_log[f'mmvae_{phase}_lp1_x1_z_1'].append(lpx_zs[3].mean().item())
    loss_log[f'mmvae_{phase}_kl_q0_pz'].append(klds[0].mean().item())
    loss_log[f'mmvae_{phase}_kl_q1_pz'].append(klds[1].mean().item())
    
    mmvae_loss = (1 / len(model.mmvae.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    mmvae_loss = mmvae_loss.mean(0).sum()

    # infoNCE: 
    loss_fn = torch.nn.CrossEntropyLoss() # reduction='none'

    ## MNIST
    # qz_x_mnist, zs_mnist 
    # qz_xs_mmvae[0], zss_mmvae[0]
    inf_mnist_1 = qz_xs_mmvae[0].log_prob(zss_mmvae[0]).exp() # ==1, # [K=1, bs, latent_dim] 
    inf_mnist_2 = qz_x_mnist.log_prob(zs_mnist).exp() # ==1, # [K=1, bs, latent_dim] 
    inf_mnist_3 = qz_x_mnist.log_prob(zss_mmvae[0]).exp() # ==0
    inf_mnist_4 = qz_xs_mmvae[0].log_prob(zs_mnist).exp() # ==0
    
    loss_log[f'{phase}_inf_mnist_1'].append(inf_mnist_1.mean().item()) # 1
    loss_log[f'{phase}_inf_mnist_2'].append(inf_mnist_2.mean().item()) # 1
    loss_log[f'{phase}_inf_mnist_3'].append(inf_mnist_3.mean().item()) # 1
    loss_log[f'{phase}_inf_mnist_4'].append(inf_mnist_4.mean().item()) # 1
    
    inf_mnist_1 = inf_mnist_1.mean(0)[:, None, :]
    inf_mnist_2 = inf_mnist_2.mean(0)[:, None, :]
    inf_mnist_3 = inf_mnist_3.mean(0)[:, None, :]
    inf_mnist_4 = inf_mnist_4.mean(0)[:, None, :]
    
    #### v1
    inf_mnist = torch.cat([inf_mnist_1, inf_mnist_2, inf_mnist_3, inf_mnist_4], dim=1) # [bs, 4, latent_dim]
    
    target_mnist = torch.cat([
        torch.ones((inf_mnist.shape[0], 2, inf_mnist.shape[-1]), device=inf_mnist.device),
        torch.zeros((inf_mnist.shape[0], 2, inf_mnist.shape[-1]), device=inf_mnist.device),
    ], dim=1)

    loss_fn = torch.nn.CrossEntropyLoss() # reduction='none'
    loss_inf_mnist = loss_fn(inf_mnist, target_mnist) # with reduction='none', size of the loss is [bs, latent_dim]

    #### v2
    # inf_mnist =  torch.cat([inf_mnist_1, inf_mnist_2, inf_mnist_3, inf_mnist_4], dim=1).mean(2) # MEAN ACROSS LATENT [bs, 4]
    # target_mnist = torch.cat([
    #     torch.ones((inf_mnist.shape[0], 2), device=inf_mnist.device),
    #     torch.zeros((inf_mnist.shape[0], 2), device=inf_mnist.device),
    # ], dim=1)

    # loss_inf_mnist = loss_fn(inf_mnist, target_mnist) # with reduction='none', size of the loss is [bs]

    #### v3
    # loss_inf_mnist = ((inf_mnist_1+inf_mnist_2)- (inf_mnist_3+ inf_mnist_4)).mean()

    ## SVHN
    # qz_x_svhn, zs_svhn
    # qz_xs_mmvae[1], zss_mmvae[1]
    inf_svhn_1 = qz_xs_mmvae[1].log_prob(zss_mmvae[1]).exp() # == 1
    inf_svhn_2 = qz_x_svhn.log_prob(zs_svhn).exp() # == 1 
    
    inf_svhn_3 = qz_x_svhn.log_prob(zss_mmvae[1]).exp() # ==0
    inf_svhn_4 = qz_xs_mmvae[1].log_prob(zs_svhn).exp() # ==0
    
    loss_log[f'{phase}_inf_svhn_1'].append(inf_svhn_1.mean().item()) # 1
    loss_log[f'{phase}_inf_svhn_2'].append(inf_svhn_2.mean().item()) # 1
    loss_log[f'{phase}_inf_svhn_3'].append(inf_svhn_3.mean().item()) # 1
    loss_log[f'{phase}_inf_svhn_4'].append(inf_svhn_4.mean().item()) # 1
    
    inf_svhn_1 = inf_svhn_1.mean(0)[:, None, :]
    inf_svhn_2 = inf_svhn_2.mean(0)[:, None, :]
    inf_svhn_3 = inf_svhn_3.mean(0)[:, None, :]
    inf_svhn_4 = inf_svhn_4.mean(0)[:, None, :]
    
    inf_svhn = torch.cat([inf_svhn_1, inf_svhn_2, inf_svhn_3, inf_svhn_4], dim=1) # [bs, 4, latent_dim]
    
    target_svhn = torch.cat([
        torch.ones((inf_svhn.shape[0], 2, inf_svhn.shape[-1]), device=inf_svhn.device),
        torch.zeros((inf_svhn.shape[0], 2, inf_svhn.shape[-1]), device=inf_svhn.device),
    ], dim=1)

    loss_inf_svhn = loss_fn(inf_svhn, target_svhn) # with reduction='none', size of the loss is [bs, latent_dim]

    loss_log[f'{phase}_loss_inf_mnist'].append(loss_inf_mnist.item())
    loss_log[f'{phase}_loss_inf_svhn'].append(loss_inf_svhn.item())
    
    alpha = 1.0
    total_loss = (mmvae_loss + uni_mnist_loss + uni_svhn_loss) - alpha*(loss_inf_mnist+loss_inf_svhn)
    return total_loss




def _m_iwae(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.cat(lws)  # (n_modality * n_samples) x batch_size, batch_size


def m_iwae(model, x, loss_log, phase, K=1):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw = [_m_iwae(model, _x, K) for _x in x_split]
    lw = torch.cat(lw, 1)  # concat on batch
    return log_mean_exp(lw).sum()


def _m_iwae_looser(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.stack(lws)  # (n_modality * n_samples) x batch_size, batch_size


def m_iwae_looser(model, x, K=1):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw = [_m_iwae_looser(model, _x, K) for _x in x_split]
    lw = torch.cat(lw, 2)  # concat on batch
    return log_mean_exp(lw, dim=1).mean(0).sum()


def _m_dreg(model, x, K=1):
    """DERG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
    qz_xs, px_zs, zss = model(x, K)
    qz_xs_ = [vae.qz_x(*[p.detach() for p in vae.qz_x_params]) for vae in model.vaes]
    lws = []
    for r, vae in enumerate(model.vaes):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.cat(lws), torch.cat(zss)


def m_dreg(model, x, loss_log, phase, K=1):
    """Computes dreg estimate for log p_\theta(x) for multi-modal vae """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw, zss = zip(*[_m_dreg(model, _x, K) for _x in x_split])
    lw = torch.cat(lw, 1)  # concat on batch
    zss = torch.cat(zss, 1)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 0, keepdim=True)).exp()
        if zss.requires_grad:
            zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).sum()


def _m_dreg_looser(model, x, K=1):
    """DERG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    qz_xs, px_zs, zss = model(x, K)
    qz_xs_ = [vae.qz_x(*[p.detach() for p in vae.qz_x_params]) for vae in model.vaes]
    lws = []
    for r, vae in enumerate(model.vaes):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.stack(lws), torch.stack(zss)


def m_dreg_looser(model, x, loss_log, phase, K=1):
    """Computes dreg estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw, zss = zip(*[_m_dreg_looser(model, _x, K) for _x in x_split])
    lw = torch.cat(lw, 2)  # concat on batch
    zss = torch.cat(zss, 2)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
        if zss.requires_grad:
            zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).mean(0).sum()


# def m_modified_elbo_naive_old(model, x, agg, phase: str, K=1):
#     """Computes E_{p(x)}[ELBO] for multi-modal vae --- NOT EXPOSED"""
#     qz_xs, px_zs, zss = model(x, K=1) 
    
#     lpx_zs, klds = [], []
#     for r, qz_x in enumerate(qz_xs):
#         kld = kl_divergence(qz_x, model.pz(*model.pz_params))
#         klds.append(kld.sum(-1))
#         for d, px_z in enumerate(px_zs[r]):
#             lpx_z = px_z.log_prob(x[d]) * model.vaes[d].llik_scaling
            
#             lpx_zs.append(lpx_z.view(*px_z.batch_shape[:2], -1).sum(-1))
#     '''
#     klds
#     ----
#         r = 0:
#             klds[0]: Dkl(q_theta_1 (z|x_1) || p_z) [bs]
        
#         r = 1:
#             klds[1]: Dkl(q_theta_2 (z|x_2) || p_z) [bs]

#     lpx_zs
#     ------
#         r = 0:
#             d = 0, lpx_zs[0]: log prob (p_theta_1 (x_1|zss[0])) given (x_1) [k, bs]
#             d = 1, lpx_zs[1]: log prob (p_theta_2 (x_2|zss[0])) given (x_2) [k, bs]

#         r = 1:
#             d = 0, lpx_zs[2]: log prob (p_theta_1 (x_1|zss[1])) given (x_1) [k, bs]
#             d = 1, lpx_zs[3]: log prob (p_theta_2 (x_2|zss[1])) given (x_2) [k, bs]
#     '''

#     '''
#     torch.stack(lpx_zs):            [4, k, bs]
#     torch.stack(lpx_zs).sum(0):     [k, bs]
#     torch.stack(klds):              [2, bs]
#     torch.stack(klds).sum():        [bs]
#     '''    
#     for _ in range(2): 
#         more_zs1 = qz_xs[0].rsample(torch.Size([1])).detach() # [K=1, bs, latent_dim]
#         more_zs2 = qz_xs[1].rsample(torch.Size([1])) # [K=1, bs, latent_dim]
#         ## px2_z1 = model.vaes[1].px_z(more_zs1)
#         ## px2_z2 = model.vaes[1].px_z(more_zs2)
#         px2_z1 = model.vaes[1].px_z(*model.vaes[1].dec(more_zs1))
#         px2_z2 = model.vaes[1].px_z(*model.vaes[1].dec(more_zs2))
#         lpx2_z1 = px2_z1.log_prob(x[1]) * model.vaes[d].llik_scaling
#         lpx2_z2 = px2_z2.log_prob(x[1]) * model.vaes[d].llik_scaling
#         lpx_zs.append(lpx2_z1.view(*px_z.batch_shape[:2], -1).sum(-1))
#         lpx_zs.append(lpx2_z2.view(*px_z.batch_shape[:2], -1).sum(-1))

#         # kld = kl_divergence(qz_x, model.pz(*model.pz_params))
#         # klds.append()


#     agg[f'{phase}_lp0_x0_z_0'].append(lpx_zs[0].mean().item())
#     agg[f'{phase}_lp1_x1_z_0'].append(lpx_zs[1].mean().item())
#     agg[f'{phase}_lp0_x0_z_1'].append(lpx_zs[2].mean().item())
#     agg[f'{phase}_lp1_x1_z_1'].append(lpx_zs[3].mean().item())
#     agg[f'{phase}_kl_q0_pz'].append(klds[0].mean().item())
#     agg[f'{phase}_kl_q1_pz'].append(klds[1].mean().item())
    
#     obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
#     # [K, bs]
    
#     return obj.mean(0).sum() # mean across K, the sum across batch_size


# def m_modified_elbo_naive(model, x, agg, phase: str, K=1):
#     qz_xs, px_zs, zss = model(x, K=1) 
    
#     lpx_zs, klds = [], []
#     for r, qz_x in enumerate(qz_xs):
#         kld = kl_divergence(qz_x, model.pz(*model.pz_params))
#         klds.append(kld.sum(-1))
#         for d, px_z in enumerate(px_zs[r]):
#             lpx_z = px_z.log_prob(x[d]) * model.vaes[d].llik_scaling
            
#             lpx_zs.append(lpx_z.view(*px_z.batch_shape[:2], -1).sum(-1))
#     '''
#     klds
#     ----
#         r = 0:
#             klds[0]: Dkl(q_theta_1 (z|x_1) || p_z) [bs]
        
#         r = 1:
#             klds[1]: Dkl(q_theta_2 (z|x_2) || p_z) [bs]

#     lpx_zs
#     ------
#         r = 0:
#             d = 0, lpx_zs[0]: log prob (p_theta_1 (x_1|zss[0])) given (x_1) [k, bs]
#             d = 1, lpx_zs[1]: log prob (p_theta_2 (x_2|zss[0])) given (x_2) [k, bs]

#         r = 1:
#             d = 0, lpx_zs[2]: log prob (p_theta_1 (x_1|zss[1])) given (x_1) [k, bs]
#             d = 1, lpx_zs[3]: log prob (p_theta_2 (x_2|zss[1])) given (x_2) [k, bs]
#     '''

#     '''
#     torch.stack(lpx_zs):            [4, k, bs]
#     torch.stack(lpx_zs).sum(0):     [k, bs]
#     torch.stack(klds):              [2, bs]
#     torch.stack(klds).sum():        [bs]
#     '''    
#     for _ in range(5): 
#         more_zs1 = qz_xs[0].rsample(torch.Size([1])).detach() # [K=1, bs, latent_dim]
#         # more_zs2 = qz_xs[1].rsample(torch.Size([1])) # [K=1, bs, latent_dim]
#         ## px2_z1 = model.vaes[1].px_z(more_zs1)
#         ## px2_z2 = model.vaes[1].px_z(more_zs2)
#         px2_z1 = model.vaes[1].px_z(*model.vaes[1].dec(more_zs1))
#         # px2_z2 = model.vaes[1].px_z(*model.vaes[1].dec(more_zs2))
#         lpx2_z1 = px2_z1.log_prob(x[1]) * model.vaes[d].llik_scaling
#         # lpx2_z2 = px2_z2.log_prob(x[1]) * model.vaes[d].llik_scaling
#         lpx_zs.append(lpx2_z1.view(*px_z.batch_shape[:2], -1).sum(-1))
#         # lpx_zs.append(lpx2_z2.view(*px_z.batch_shape[:2], -1).sum(-1))

#         # kld = kl_divergence(qz_x, model.pz(*model.pz_params))
#         # klds.append()


#     agg[f'{phase}_lp0_x0_z_0'].append(lpx_zs[0].mean().item())
#     agg[f'{phase}_lp1_x1_z_0'].append(lpx_zs[1].mean().item())
#     agg[f'{phase}_lp0_x0_z_1'].append(lpx_zs[2].mean().item())
#     agg[f'{phase}_lp1_x1_z_1'].append(lpx_zs[3].mean().item())
#     agg[f'{phase}_kl_q0_pz'].append(klds[0].mean().item())
#     agg[f'{phase}_kl_q1_pz'].append(klds[1].mean().item())
    
#     obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
#     # [K, bs]
    
#     return obj.mean(0).sum() # mean across K, the sum across batch_size
