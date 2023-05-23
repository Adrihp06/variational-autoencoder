# Variational Autoencoder for Chest X-ray Images

This project implements a Variational Autoencoder (VAE) trained on chest X-ray images. The VAE is a generative model that can learn to encode and reconstruct images while also generating new samples from a learned latent space. In this specific application, the VAE is trained on grayscale chest X-ray images.

## Loss Functions

The VAE loss function consists of two components: the Kullback-Leibler (KL) loss and the reconstruction loss.

1. KL Loss:
   - The KL loss measures the divergence between the learned latent space distribution and a desired distribution (usually a standard Gaussian distribution). The formula for the KL loss is:
     - L_kl = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

2. Grayscale Images Reconstruction Loss:
   - The reconstruction loss measures the dissimilarity between the input grayscale image and the reconstructed image generated by the VAE. The Mean Squared Error (MSE) is commonly used as the reconstruction loss function. The formula for the grayscale image reconstruction loss is:
     - L_recon = MSE(input_image, reconstructed_image)

The total loss used for training the VAE is a combination of the KL loss and the grayscale image reconstruction loss, with appropriate scaling factors.

## Dataset

The chest X-ray dataset used for training the VAE consists of a collection of grayscale images of chest X-rays. The dataset should be preprocessed and normalized before training the VAE.

## Results

Include any relevant results and insights obtained from training and evaluating the VAE on the chest X-ray dataset. Provide visual examples of reconstructed images and any other relevant analysis.
