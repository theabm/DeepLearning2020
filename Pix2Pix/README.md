# Pix2Pix
Implementation of Pix2Pix, an image-to-image translation using GANs following specifics from [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf.
Among the different applications we opted for the map to aerial scenario.

### FILES
* pix2pix.py: Contains U-Net Generator, PatchGAN Discriminator implementation. More details about the parameters are present in the form of comments.
* pix2pix.ipynb: Notebook with same code as previous file. In addition there are summaries about Generator and Discriminator showing the total number of parameters.
* train.py: Training file for the network.
* test.py: Test file for the network.
* merged.png: Image shwoing the results after 100 epochs. In the first column we have the input image, in the second one the output of the network and in the third one the ground truth image. 
  #### _Note that in the paper it was suggested to perform 200 epochs but due to computational resources availabilty we stopped at 100._
  
### FODLERS
* script: Contains scripts for downloading the dataset and launching the program on a cluster.
