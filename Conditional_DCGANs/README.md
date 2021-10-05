# Conditional DCGANs

Implementation of Conditional Deep Convolutional GANs following https://arxiv.org/abs/1511.06434 paper. All the details are present in form of comments inside *conditional_gans.py* file.

### FILES
* conditional_gans.py: Contains Generator, Discriminator, weight initialization, training and testing of the network.
* c_dcgans.ipynb: Notebook executing the previous file.
### FOLDERS
* images-conditional_dcgan: Contains 2 gifs showing good and bad behaviour (much probably due to vanishing gradient problem) of the network in generating synthetic images. Each gif is annotated with an Epoch counter for the sake of clarity.
