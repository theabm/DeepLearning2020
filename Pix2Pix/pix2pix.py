import torch
import torch.nn as nn

##### Pix2pix is implemented using a U-Net for the generator and a "PatchGAN" for the
##### discriminator. With PatchGAN we are actually restricting our interest only on 
##### patches of the original image, namely the discriminator will classify if each
##### patch considered belongs to a real or fake image.

##### For the generator model we have defined both encoder and decoder blocks.
##### Each encoder block is made up of Convolution-BatchNorm-ReLU layer with all
##### ReLUs that are leaky (negative slope = 0.2). For the convolution we take
##### as parameters kernel size = 4, stride = 2, padding = 1.
 
##### The overall structure for the encoder part can be summarized in this form:
##### C64-C128-C256-C512-C512-C512-C512-C512

class encoder_block(nn.Module):
    def __init__(self, in_channels, num_filters, batch_size, activ=nn.LeakyReLU, norm=True):
        super().__init__()
        
        if norm is True:
            normalization = None
            if batch_size == 1:
                normalization = nn.InstanceNorm2d(num_filters)
            else:
                normalization = nn.BatchNorm2d(num_filters)
            
            self.layers = nn.Sequential(
                activ(0.2, inplace=True),
                nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1),
                normalization
            )
        
        else:
            self.layers = nn.Sequential(
                activ(0.2, inplace=True),
                nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1)
            )

    def forward(self, X):
        out = self.layers(X)
        return out

##### The decoder block differs from the encoder one mainly for the following:
##### • Dropout is added in some layers with dropout rate = 0.5;
##### • All the ReLUs are simple ReLUs.

##### Also in this case we can summarize the decoder block as:
##### CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128

class decoder_block(nn.Module):
    def __init__(self, in_channels, num_filters, batch_size, activ=nn.ReLU, dropout=True, attach=True):
        super().__init__()

        normalization = None
        if batch_size == 1:
            normalization = nn.InstanceNorm2d(num_filters)
        else:
            normalization = nn.BatchNorm2d(num_filters)

        if dropout is True:
            drop = nn.Dropout(0.5)
            if attach is True:
                self.layers = nn.Sequential(
                    activ(inplace=True),
                    nn.ConvTranspose2d(in_channels*2, num_filters, kernel_size=4, stride=2, padding=1),
                    normalization,
                    drop
                )
            else:
                self.layers = nn.Sequential(
                activ(inplace=True),
                nn.ConvTranspose2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1),
                normalization,
                drop
            )

        else:
            self.layers = nn.Sequential(
                activ(inplace=True),
                nn.ConvTranspose2d(in_channels*2, num_filters, kernel_size=4, stride=2, padding=1),
                normalization
            )

    def forward(self, X, encoder):
        out = self.layers(X)
        out = torch.cat((encoder,out), dim=1)
        return out

##### So the Generator can be implemented in this way
class Generator(nn.Module):

    def __init__(self, batch_size):
        super().__init__()

        # Batch-Norm is not applied to the first C64 layer in the encoder
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv2 = encoder_block(64, 128, batch_size=batch_size)
        self.conv3 = encoder_block(128, 256, batch_size=batch_size)
        self.conv4 = encoder_block(256, 512, batch_size=batch_size)
        self.conv5 = encoder_block(512, 512, batch_size=batch_size)
        self.conv6 = encoder_block(512, 512, batch_size=batch_size)
        self.conv7 = encoder_block(512, 512, batch_size=batch_size)
        self.conv8 = encoder_block(512, 512, batch_size=batch_size, norm=False)

        self.deconv8 = decoder_block(512,512, batch_size=batch_size, attach=False)
        self.deconv7 = decoder_block(512,512, batch_size=batch_size)
        self.deconv6 = decoder_block(512,512, batch_size=batch_size)
        self.deconv5 = decoder_block(512,512, batch_size=batch_size, dropout=False)
        self.deconv4 = decoder_block(512,256, batch_size=batch_size, dropout=False)
        self.deconv3 = decoder_block(256,128, batch_size=batch_size, dropout=False)
        self.deconv2 = decoder_block(128,64, batch_size=batch_size, dropout=False)
        self.deconv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * 2, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, X):
        x1 = self.conv1(X)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)

        out = self.deconv8(x8, x7)
        out = self.deconv7(out, x6)
        out = self.deconv6(out, x5)
        out = self.deconv5(out, x4)
        out = self.deconv4(out, x3)
        out = self.deconv3(out, x2)
        out = self.deconv2(out, x1)
        out = self.deconv1(out)
    
        return out

##### The discriminator tries to classify if each NxN patch in an image
##### is real or fake. The case in which N is 70 has been proved, experimentally,
##### to be the most suitable choice.
##### After the last layer, a convolution is applied to map to a 1-dimensional 
##### output, followed by a Sigmoid function.

##### This architecture can be seen as:
##### C64-C128-C256-C512

class Discriminator(nn.Module):
    def __init__(self, batch_size):
        super().__init__()

        normalization = None
        if batch_size == 1:
            normalization = nn.InstanceNorm2d
        else:
            normalization = nn.BatchNorm2d

        self.layers = nn.Sequential(
            # BatchNorm is not applied to the first layer
            nn.Conv2d(3*2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            normalization(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            normalization(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            normalization(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, real, fake):
        out = torch.cat((real, fake), dim=1)
        out = self.layers(out)
        return out