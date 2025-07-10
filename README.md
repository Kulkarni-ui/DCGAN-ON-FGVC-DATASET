# 🌌 DCGAN for Synthetic Image Generation on FGVC Dataset

This project implements a **Deep Convolutional GAN (DCGAN)** to generate synthetic images based on the **FGVC (Fine-Grained Visual Classification)** dataset, specifically focused on flight images.The dataset contains 10,000 images of aircraft, The data is split into 3334 training images, 3333 validation and 3333 testing images Aircraft models are organized in a four-levels hierarchy. The four levels, from finer to coarser, are:

Model, e.g. Boeing 737-76J. Since certain models are nearly visually indistinguishable, this level is not used in the evaluation. Variant, e.g. Boeing 737-700. A variant collapses all the models that are visually indistinguishable into one class. The dataset comprises 100 different variants.

Family, e.g. Boeing 737. The dataset comprises 70 different families. Manufacturer, e.g. Boeing. The dataset comprises 41 different manufacturers

## Objective

To train a DCGAN model for generating realistic-looking synthetic flight images that can support augmentation in fine-grained classification tasks.

## Dataset

- **FGVC-Aircraft** dataset (subset used)
- Preprocessed images resized and normalized
- Categories include various aircraft models and orientations

## Technologies Used

- **Python**, **PyTorch**
- **Torchvision** for dataset handling
- **Matplotlib** and **NumPy** for visualization
- **Google Colab** for GPU training

## Model Details

- **Generator** and **Discriminator** contain 4 Conv2D layers
- Training optimized with **Adam optimizer**
- Batch normalization and LeakyReLU activation
- Image resolution: 64x64 pixels

## Key Features

- Training progression shown via real vs generated image grids
- Use of latent noise vector (z) for image synthesis
- Model checkpoints and output samples saved
- Visual validation of convergence over epochs

## Future Work

- Integrate **Conditional DCGAN** for class-aware generation
- Compare DCGAN with **WGAN-GP** or **StyleGAN** on same dataset
- Use generated images for downstream classification task performance boost

## Repository Structure
📦 DCGAN_FGVC_ImageGen/
┣ 📄 DCGAN_FGVC_DATASET_FINAL.ipynb
┣ 📄 README.md
┗ 📁 results/ (generated images, training outputs)


