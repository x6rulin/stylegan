## StyleGAN &mdash; Pytorch Implementation
![Python 3.6](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![Pytorch 1.20](https://img.shields.io/badge/pytorch-1.20-green.svg?style=plastic)

This repository contains the Pytorch implementation of the following paper:

> **A Style-Based Generator Architecture for Generative Adversarial Networks**<br>
> Tero Karras (NVIDIA), Samuli Laine (NVIDIA), Timo Aila (NVIDIA)<br>
> http://stylegan.xyz/paper
>
> **Abstract:** *We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis. The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. To quantify interpolation quality and disentanglement, we propose two new, automated methods that are applicable to any generator architecture. Finally, we introduce a new, highly varied and high-quality dataset of human faces.*

For more implementation details, please reference to the [official repository](https://github.com/NVlabs/stylegan)
