# Cycle Gans

Implementation following the results in the original paper. For more information on how to download the dataset see:


FILES: 

- config: setting of all hyperparameters and number of epochs. Include transformation of images. 
- dataset: read the data and create the dataloaders.
- discriminator: the PatchGAN discriminator.
- generator: U-Net based generator.
- train: training of the CycleGAN architecture.
- utils: checkpoint saves.  
