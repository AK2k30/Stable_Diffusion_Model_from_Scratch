# Stable Diffusion Model from Scratch

![image](https://github.com/user-attachments/assets/548ecf20-97a8-49e9-a363-9880b7c4e448)

## Overview

This project implements a Stable Diffusion model from scratch using PyTorch. The model is capable of generating images from textual descriptions (text-to-image) and transforming existing images based on textual prompts (image-to-image). The implementation is based on the foundational principles of diffusion models and includes features like classifier-free guidance and latent space representations.

![Stable_Diffusion_Diagrams_V2_page-0003](https://github.com/user-attachments/assets/edff397c-b6fc-4519-9f4a-1367d44c6c96)

## Features

- **Text-to-Image Generation**: Create images from textual descriptions.
- **Image-to-Image Transformation**: Modify existing images based on textual prompts.
- **Inpainting**: Fill in missing parts of images using textual descriptions.
- **Classifier-Free Guidance**: Improved generation quality by balancing conditioned and unconditioned outputs.
- **Latent Space Representation**: Efficient computation using a variational autoencoder.
  
![Stable_Diffusion_Diagrams_V2_page-0007](https://github.com/user-attachments/assets/c65f3f5a-9059-499a-95d0-67f211b14d74)

## Prerequisites

- Basic understanding of probability and statistics (e.g., multivariate Gaussian, conditional probability, Bayes' rule).
- Basic knowledge of PyTorch and neural networks.
- Familiarity with attention mechanisms and convolution layers.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AK2k30/Stable_Diffusion_model_from_scratch.git
   cd Stable_Diffusion_model_from_scratch
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage

Training

To train the model from scratch, run:
  ```bash
    python train.py --config configs/config.yaml
  ```
Customize the configuration file (configs/config.yaml) to suit your dataset and training preferences.

## Generating Images

Text-to-Image

To generate an image from a text prompt:
```bash
python generate.py --mode text-to-image --prompt "A dog with glasses"
```

![Stable_Diffusion_Diagrams_V2_page-0023](https://github.com/user-attachments/assets/3bda259e-6704-4d30-959e-67d19919e70e)

Image-to-Image

To transform an existing image based on a text prompt:
```bash
python generate.py --mode image-to-image --input_image path/to/image.jpg --prompt "A dog with glasses"
```

![Stable_Diffusion_Diagrams_V2_page-0024](https://github.com/user-attachments/assets/93de9efc-5547-4f24-8c79-e2c082cea568)

Inpainting

To inpaint a missing part of an image using a text prompt:
```bash
python inpaint.py --input_image path/to/image.jpg --prompt "A dog running"
```

![Stable_Diffusion_Diagrams_V2_page-0025](https://github.com/user-attachments/assets/057d5c32-fa14-4025-866c-a38340f299da)

# Architecture

## Variational Autoencoder (VAE)
The VAE is responsible for encoding the input images into a latent space. 

It consists of two main parts:

- **Encoder:** Compresses the input image into a latent vector.

- **Decoder:** Reconstructs the image from the latent vector.

The VAE helps in reducing the dimensionality of the data, making it computationally efficient to process through the diffusion model.

![Stable_Diffusion_Diagrams_V2_page-0022](https://github.com/user-attachments/assets/64347791-7d58-470f-a131-4d075e8977b1)

## Latent Diffusion Model (LDM)
The LDM operates on the latent representations obtained from the VAE. It is designed to learn the data distribution in the latent space. The key idea is to progressively denoise a sample from a simple distribution (e.g., Gaussian noise) to match the target data distribution.

![Stable_Diffusion_Diagrams_V2_page-0019](https://github.com/user-attachments/assets/ead70c5b-6eee-44a3-89e9-d367a72befba)

## UNet Backbone
The UNet architecture is used as the core neural network within the diffusion model. 

It consists of:

- **Encoder Path:** A series of convolutional layers that downsample the input.

- **Bottleneck:** The middle part of the network that captures the most compressed representation.

- **Decoder Path:** A series of convolutional layers that upsample the representation back to the original size.
- 
The UNet allows for efficient handling of high-resolution images by capturing multi-scale features through its symmetric design.

![Stable_Diffusion_Diagrams_V2_page-0011](https://github.com/user-attachments/assets/e01ac407-907d-47a3-bbfd-2d30458a3ab9)

## Attention Mechanisms
Attention mechanisms are integrated within the UNet to enhance the model's ability to focus on relevant parts of the image during processing. 

This includes:

- **Self-Attention**: Helps the model to consider dependencies between different parts of the image.

- **Cross-Attention**: Facilitates the interaction between the text embeddings and the image features, crucial for text-to-image generation.
  
![Stable_Diffusion_Diagrams_V2_page-0017](https://github.com/user-attachments/assets/d1230bc0-4313-40cf-96c6-3be93dc3ae7f)

## Classifier-Free Guidance
Classifier-free guidance improves the generation quality by balancing conditioned (text-prompt) and unconditioned (random noise) outputs. This technique allows the model to generate more coherent and high-quality images by guiding the diffusion process with the text prompt while still considering the general data distribution.

![Stable_Diffusion_Diagrams_V2_page-0014](https://github.com/user-attachments/assets/41e95d3e-f629-4ea5-99b7-a94a12c4a2b5)

## References

**Denoising Diffusion Probabilistic Models** by **Ho et al**.

**U-Net: Convolutional Networks for Biomedical Image Segmentation** by **Ronneberger et al**.

## Conclusion

#### If you like this project, show your support & love!

[![buy me a coffee](https://res.cloudinary.com/customzone-app/image/upload/c_pad,w_200/v1712840190/bmc-button_wl78gx.png)](https://www.buymeacoffee.com/akashsunile)
