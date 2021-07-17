# Conditional GANs
Conditional GANs were trained on the MNIST dataset and conditioned on the digits.
The result is that after 8-9 epochs the results are very satisfying while after epoch 12 we start encountering worsening results which are probably due to vanishing gradient problems.

FILES: 

- config: setting of all hyperparameters and number of epochs. Include transformation of images. 
- dataset: read the data and create the dataloaders.
- discriminator: the PatchGAN discriminator.
- generator: U-Net based generator.
- train: training of the CycleGAN architecture.
- utils: checkpoint saves.  
