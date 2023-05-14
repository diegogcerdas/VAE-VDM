This blog post focuses on our work that came as an extension from the paper Diffusion Based Representation Learning \[1\]. We provide an analysis, discuss its core components, look at the weaknesses and strength of the paper. Furthermore we extend the paper to analytically derive the ELBO in the context of Variational Diffusion Models \[3\], to allow for sampling over the priors in the latent space.

## Introduction

\[1\] introduce a novel way to leverage diffusion models for representation learning. They build forward on the work of \[2\], where diffusion based methods using stochastic differential equations (SDE) on continuous time domains are discussed. The training of these models is dependent on score matching methods, of which denoising score matching \[5\] and sliced score matching \[4\] are commonly used. The authors of the original paper enable representation learning by augmenting the denoising score matching framework without any supervised signal. While GANs and VAEs directly transform latent codes to data samples, diffusion-based representation learning relies on a new formulation of the denoising score matching objective, which encodes the information needed for denoising. This difference allows for manual control of the level of details encoded in the representation. The proposed approach involves learning an infinite-dimensional latent code that achieves improved performance on semi-supervised image classification compared to state-of-the-art models. Moreover, the authors compare the quality of the learned representations of diffusion score matching to other methods such as autoencoders and contrastively trained systems through their performances on downstream tasks. 

The proposed alternative formulation of the Denoising Score Matching (DSM) formula is given by:

$$J\_t^{D S M}(\theta)=\mathbf{E}\_{x\_0}\left\\{\mathbf{E}\_{x\_t \mid x\_0}\left[\left\\|\nabla\_{x\_t} \log p\_{0 t}\left(x\_t \mid x\_0\right)-\nabla\_{x\_t} \log p\_t\left(x\_t\right)\right\\|\_2^2+\left\\|s\_\theta\left(x\_t, t\right)-\nabla\_{x\_t} \log p\_t\left(x\_t\right)\right\\|\_2^2\right]\right\\}$$<div align="right">(1)</div>

The non-vanishing constant that has been added opposed to the original denoising function \[5\] is what enables the model for latent representation learning. The proposed learning objective for Diffusion-based Representation Learning (DRL) is given by: 

$$J^{D R L}(\theta, \phi)=\mathbf{E}\_{t, x\_0, x\_t}\left[\lambda(t)\left\\|s\_\theta\left(x\_t, t, E\_\phi\left(x\_0\right)\right)-\nabla\_{x\_t} \log p\_{0 t}\left(x\_t \mid x\_0\right)\right\\|\_2^2\right]$$<div align="right">(2)</div>

Where $E_\phi\left(x_0\right)$ is a trainable encoder that represents the labeling function. Intuitively, $E_\phi\left(x_0\right)$ chooses the direction in which the recovery of $x_0$ from $x_t$ is maximized. Moreover, the authors suggest that the encoder's purpose is to learn how to represent the essential information required to eliminate the noise in $x_0$, which varies depending on the noise level $\sigma(t)$. They claim that by modifying the weighting function $\lambda(t)$, the level of granularity in the encoded features can be manually controlled. Thus, they include the $E_\phi$ to obtain a low-dimensional representation to condition the generation of the diffusion model. The specified encoder $E_\phi$ is not enforced to be deterministic, intuitively it any function that can influence $I(x_t, x_0)$, the mutual information between $x_0$ and $x_t$ information channel, can be used. This naturally leads to a VAE-like encoder resulting in Variational Diffusion-based Representation Learning (VDRL). The VDRL objective function is given by:

$$J^{V D R L}(\theta, \phi)=\mathbf{E}\_{t, x\_0, x\_t}\left[\mathbf{E}\_{z \sim E\_\phi\left(Z \mid x\_0\right)}\left[\lambda(t)\left\\|s\_\theta\left(x\_t, t, z\right)-\nabla\_{x\_t} \log p\_{0 t}\left(x\_t \mid x\_0\right)\right\\|\_2^2\right]+\mathcal{D}\_{\mathrm{KL}}\left(E\_\phi\left(Z \mid x\_0\right) \| \mathcal{N}(Z ; 0, I)\right)\right]$$<div align="right">(3)</div>

Additionaly, the authors introduce a variation of DRL that incorporates time-varying representations. In contrast to the previous approach, which used weighted training objectives to account for varying noise levels, this new method takes the time t as an input to the encoder.  $E_\phi(x_0)$ in Equation (2) is now replaced by  $E_\phi(x_0, t)$, leading to the following objective: 

$$\mathbf{E}\_{t, x\_0, x\_t}\left[\lambda(t)\left\\|s\_\theta\left(x\_t, t, E\_\phi\left(x\_0, t\right)\right)-\nabla\_{x\_t} \log p\_{0 t}\left(x\_t \mid x\_0\right)\right\\|\_2^2\right]$$<div align="right">(4)</div>

Intuitively, this enables the encoder to extract the essential information of $x_0$ for denoising $x_t$ at any noise level. This approach allows for richer representation learning, which is not typically achievable with traditional auto-encoders or other static representation learning methods. Finally, the authors show that using adversarial training and choosing the right noise schedule improves the performance of score-based matching. 

## Review of the paper and steps moving forward

The proposed methods that are introduced are straightforward to follow and novel extensions on non-adversial generative modelling. The experiments that were provided show promising results and empirically justify the statements given the context achieving SoTA results on semi-supervised classification. Moreover, the authors provide a clear ablation study and also provide additional contribution to the training and sampling of diffusion models. 

However, the paper does not place the significance in representation learning into context, especially considering recent advances during that time, thus making it difficult for the reader to gauge what the true significance is of the proposed methods. Moreover, the paper occasionally lacks derivations of certain objective functions as well as in the appendix, making the reader question where the statement comes from. The weaknesses in our paper initialized our extension. We build upon DRL and present a VAE where the decoder is parameterized by a conditional VDM. Our method allows for sampling from the prior over the latent variables. These samples provide information about the global features present in the images generated by the conditional VDM. We study the effect of introducing the representation learning task for different capacities of the VDM in terms of log-likelihood and FID.

## References

[1] Korbinian Abstreiter, Sarthak Mittal, Stefan Bauer, Bernhard Schölkopf, and Arash Mehrjou. Diffusion-based representation learning. 2022.

[2] Ajay Jain Jonathan Ho and Pieter Abbeel. Denoising diffusion probabilistic models, 2020.

[3] Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. Advances in neural information processing systems, 2021.

[4] Yang Song, Sahaj Garg, Jiaxin Shi, and Stefano Ermon. Sliced score matching: A scalable approach to density and score estimation. In Uncertainty in Artificial Intelligence, pages 574–584. PMLR, 2020.

[5] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661–1674, 2011
