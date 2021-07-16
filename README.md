This repository contains an implementation of several Generative Adversarial Models such as:
- Conditional GANs 
- Pix2Pix
- Cycle GANs

# Conditional GANs
Conditional GANs were trained on the MNIST dataset and conditioned on the digits.
The result is that after 8-9 epochs the results are very satisfying while after epoch 12 we start encountering worsening results which are probably due to vanishing gradiet problems. 

# Pix2Pix
Trained to simulate the map->aerial photo application illustrated in the original paper. 

# Cycle GANs
Trained to simulate the photo->ukiyoe application illustrated in the paper. Therefore, identity loss is included by default in the config file. 

In both cases the discriminator is a $70 \times 70$ PatchGAN discriminator while the generator is based on the U-Net architecture. 

All parameters were chosen to simulate the results achieved in the paper.


In collaboration with:
https://github.com/dbasso98

This implementation was the focus of the final project of the Deep Learning course held by Prof. Alessio Ansuini and Prof. Marco Zullich (DSSC).
