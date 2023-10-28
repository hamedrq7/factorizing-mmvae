## Unimodal Objectives 

### `elbo()`: 

This is equivalent to the original VAE objective for the unimodal case:

```math
ELBO(x) = \underbrace{\mathbb{E}_{q_{\phi}(z|x)}(log \; p_{\theta}(x|z))}_\text{Reconstruction Term} - \underbrace{D_{KL}(q_{\phi}(z|x) \; || \; p_{\theta}(z))}_\text{Regularization Term}
```
such that $q_{\phi}(z|x)$, the encoder is an inference network, and the $p_{\theta}(x|z)$, the decoder is a deep neural network. The overview pipeline for uni-modal training is as follows: 

<p align="center" width="100%">
    <img width="70%" src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/unimodal.png">
</p>

### Effect of `decoder_scale`:
Various value for `decoder_scale` was tested (0.05, 0.2, 0.35, 0.5, 0.75, 0.9) and their effectiveness and optimal value were found. setting `decoder_scale` to `0.75` gave a good balance between generation quality and reconstruction quality. some of the generated images (from SVHN modality) and reconstructed images (from MNIST modality) for 0.2, 0.75, and 0.9 are show below: 


<p align="center"></p>
<table>
    <thead>
        <tr>
            <th align="left">0.2</th>
            <th align="center">0.75</th>
            <th align="right">0.9</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="left"><img src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/gen_samples_means_020 1.png" width="250"/></td>
            <td align="center"><img src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/gen_samples_means_020 5.png" width="250"/></td>
            <td align="right"><img src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/gen_samples_means_020 6.png" width="250"/></td>
        </tr>
        <tr>
            <td align="left"><img src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/test_recon_020 9.png" width="250"/></td>
            <td align="center"><img src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/test_recon_020 13.png" width="250"/></td>
            <td align="right"><img src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/test_recon_020 14.png" width="250"/></td>
        </tr>
    </tbody>
</table>
<p></p>

by looking at $q(z|x)$ for one batch of test data and `batch_size` number of samples from $p(z)$ (reduced to 2D using UMAP), it can be seen that using lower value for `decoder_value` results in dissimilarity of pz and qzx which explains the poor generation from lower values of `decoder_scale`:

<p align="center"></p>
<table>
    <thead>
        <tr>
            <th align="left">0.2</th>
            <th align="right">0.75</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="left"><img src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/emb_umap_020 1.png" width="400"/></td>
            <td align="center"><img src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/emb_umap_020 5.png" width="400"/></td>
        </tr>
    </tbody>
</table>
<p></p>

## Model Selection & Hyper-Parameters: 
### Model Architecture: 
Using `elbo` as objective, batch size of 256, latent dimension of 20, `decoder_scale` set to 0.75, and setting $p_z$ to a normal distribution, various architectures (listed in the table below) were tested for each MNIST and SVHN modality to select the best one. 

| Fully Connected  | Convolutional |
| ------------- | ------------- |
| in->512->20, 20->512->out|in->4x4(16)->4x4(32)->4x4(64)->3x3(20, 20)->4x4(64)->3x3(32)->2x2(16)->4x4 out|
| in->1024->512->20, 20->1024->512->out|in->4x4(32)->4x4(64)->4x4(128)->3x3(20, 20)->4x4(128)->3x3(64)->2x2(32)->4x4 out|
| in->1024->512->128->20, 20->1024->512->128->out|in->4x4(64)->4x4(128)->4x4(256)->3x3(20, 20)->4x4(256)->3x3(128)->2x2(64)->4x4 out|

Best model for MNIST was a fully connected model (input -> 512 -> latent -> 512 -> output) and the best model for svhn is show below. 
<p align="center">
    <img src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/Pasted image 20230718005227.png" width="800"/>
</p>

# Multi-Modal: 

## Metrics: 
We use the proposed metrics in the original paper which are depicted below: 
<p align="center" width="100%">
    <img width="70%" src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/image.png">
</p>
Note that synergy metric is proposed by [Relating by Contrasting: A Data-efficient Framework for Multimodal Generative Models](https://arxiv.org/abs/2007.01179) and its not used in here. 

## Objectives:
### `m_elbo_naive()`: 
based on Appendix. B, of the original mmvae paper, the basic MOE variational posterior for two modalities is: 
```math
\mathcal{L}_{ELBO} = 
\frac{1}{2}
\underbrace{(
\mathbb{E}_{z_1 \sim q_{\phi_1} (z | x_1)} 
\left[ log \frac{p_{\Theta} (z_1, x_1, x_2)}{q_{\Phi} (z_1|x_1, x_2)} \right])
}_{m=1}
+\frac{1}{2}\underbrace{(
\mathbb{E}_{z_2 \sim q_{\phi_2} (z|x_2)} 
\left[ log \frac{p_{\Theta} (z_2, x_1, x_2)}{q_{\Phi} (z_2|x_1, x_2)} \right])
}_{m=2} 
```
expanding: 

```math
=\frac{1}{2} (
\underbrace{\mathbb{E}_{z_1 \sim q_{\phi_1} (z | x_1)} 
\left[
log \frac{p_{\Theta} (z_1, x_1)}
{q_{\Phi} (z_1|x_1, x_2)} 
\right] 
}_{\text{A}}
+
\mathbb{E}_{z_1\sim{q_{\phi_1}}(z|x_1)}
\left[ 
\text{log} \; p_{\theta_2}(x_2|z_1) 
\right]
)
+ \frac{1}{2} 
(
\underbrace{\mathbb{E}_{z_2 \sim q_{\phi_2} (z | x_2)} 
\left[
log \frac{p_{\Theta} (z_2, x_2)}
{q_{\Phi} (z_2|x_1, x_2)} 
\right]}_{\text{B}}
+
\mathbb{E}_{z_2\sim{q_{\phi_2}}(z|x_2)}
\left[ 
\text{log} \; p_{\theta_1}(x_1|z_2) 
\right]
)
```

Using Equation.2 and Equation.3 from [the original VAE paper](https://arxiv.org/abs/1312.6114), we have: 
```math
\text{A} = \mathbb{E}_{z_1 \sim q_{\phi_1} (z | x_1)} 
\left[
\text{log} \; p_{\theta_1} (x_1|z_1)
\right]
- D_{KL} (q_{\phi_1} (z|x_1) || p(z))  
, \;\;\;
\text{B} = \mathbb{E}_{z_2 \sim q_{\phi_2} (z | x_2)} \left[
\text{log} \; p_{\theta_2} (x_2|z_2)
\right]
- D_{KL} (q_{\phi_2} (z|x_2) || p(z))  
```

so the objective is:  
```math
\begin{equation} \begin{split}
\mathcal{L}_{ELBO} = \frac{1}{2} (
\overbrace{\mathbb{E}_{z_1 \sim q_{\phi_1} (z | x_1)} 
\left[
\text{log} \; p_{\theta_1} (x_1|z_1)
\right]}^{C}
+
\overbrace{
\mathbb{E}_{z_1\sim{q_{\phi_1}}(z|x_1)}
\left[ 
\text{log} \; p_{\theta_2}(x_2|z_1) 
\right]
}^{D}
- 
\overbrace{D_{KL} (q_{\phi_1} (z|x_1) || p(z))}^{A} 
)
+ \\
\frac{1}{2} (
\underbrace{
\mathbb{E}_{z_2 \sim q_{\phi_2} (z | x_2)} 
\left[
\text{log} \; p_{\theta_2} (x_2|z_2)
\right]}_{E}
+
\underbrace{
\mathbb{E}_{z_2\sim{q_{\phi_2}}(z|x_2)}
\left[ 
\text{log} \; p_{\theta_1}(x_1|z_2) 
\right]
}_{F}
- 
\underbrace{
D_{KL} (q_{\phi_2} (z|x_2) || p(z))}_{B}
)
\end{split} \end{equation}
```

<p align="center" width="100%">
    <img width="40%" src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/path25.png">
    <img width="40%" src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/ezgif.com-gif-maker.gif">
</p>

#### Problem of naive elbo: 
When analyzing the latent space of the MMVAE trained with naive ELBO, the MNIST and SVHN are separated, making cross-modal generation difficult: 
<p align="middle" width="100%">
    <img width="40%" src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/mmvae_emb_umap_010.png">
</p>

This has a peculiar effect where MNIST-generated images appear to be fine and are supported by the MNIST latent accuracy of 93%; the SVHN-generated images, on the other hand, are half decent and half (almost) noise.
<p align="center" width="100%">
    <img width="40%" src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/mmvae_gen_samples_0_015.png">
    <img width="40%" src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/mmvae_gen_samples_1_014.png">
</p>
The cross-generation from SVHN to MNIST is accurate. However, the SVHN to MNIST transfer does not produce realistic images as the decoder is not adequately trained.
<p align="center" width="100%">
    <img width="40%" src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/mmvae_test_recon_0x1_018.png">
    <img width="40%" src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/mmvae_test_recon_1x0_017.png">
</p>
The encoder is doing a good job with SVHN, achieving 73% accuracy. However, the decoder produces unrealistic images due to the low 14% cross coherence from mnist to svhn.

We tried two different solutions to address the issue at hand. Firstly, we decreased the latent dimension from 20 to 10, which restricted the model's ability to separate each modality. Secondly, we attempted to improve the performance of the SVHN decoder by sampling K times from the SVHN posterior in each pass. However, neither of these solutions proved to be effective.

### Factorized Training: 
#### Proposed Method (`m_infoNCE_naive()`): 
Our proposed setup involves the mutual training of two VAEs that focus on individual modalities and an MMVAE that captures information present in both modalities. The primary objective is to encourage the unimodal VAE of modality A to capture information primarily in modality A and the unimodal VAE of modality B to capture information primarily in modality B. At the same time, the MMVAE is trained to capture information that is present in both modalities.

<img align="left" width="30%" src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/fission.png">

This factorization of representations lies in the "Representation Fission" sub-challenge presented in [Foundations and Trends in Multimodal Machine Learning: Principles, Challenges, and Open Questions](https://arxiv.org/abs/2209.03430):  **Representation Fission:** Learning a new set of representations that reflects multimodal internal structure such as data factorization or clustering.

<br>
<br>
<br>
<br>
<br>

To achieve this, we utilize infoNCE and some additional terms to `elbo_naive` loss (`m_infoNCE_naive()` function): 
<img align="center" width="99%" src="https://github.com/hamedrq7/mmvae/blob/main/readme%20media/infoNCE Loss.png">

To evaluate the proposed method, we conducted an evaluation based on the same metrics that were used in the original MMVAE paper: 

| Metric                     | elbo naive    | elbo naive infoNCE (disentangled)  | elbo naive infoNCE (Uni) |  
| :---                             |         :---: |     :---:                          |  :---:                   |
| Cross Coherence (SVHN -> MNIST)  | 77.35%        | 76.70%                             | - |
| Cross Coherence (MNIST -> SVHN)  | 14.50%        | 12.22%                             | - |
| Joint Coherence                  | 36.63%        | 31.90%                          |-|
| latent accuracy (SVHN)           | 73.08%        | 29.02%                           |13.87%|
| latent accuracy (MNIST)          | 93.13%        | 93.24%                           |93.20%|

See following for the arguments we used for this results:
- **elbo_naive**: `python main.py --experiment comparasion_final --model mnist_svhn --obj elbo_naive --batch-size 128 --epochs 10 --fBase 32 --max_d 10000 --dm 30`
- **m_infoNCE_naive**: `python main.py --experiment comparasion_final --model fummvae --obj infoNCE_naive --batch-size 128 --epochs 10 --fBase 32 --max_d 10000 --dm 30 `
