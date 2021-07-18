import argparse
import os
import numpy as np
import math
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch import optim

img_save_path = 'images-conditional_dcgan'
os.makedirs(img_save_path, exist_ok=True)

parser = argparse.ArgumentParser(description='Our Implementation of Conditional GANs')

parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

parser.add_argument('--latent_dim', type=int, default=100)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--sample_interval', type=int, default=400)
parser.add_argument('--log_step', type=int, default=100)

args = parser.parse_args()

C,H,W = args.channels, args.img_size, args.img_size

##### Custom weights initialization called on discrim and generator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

##### Building block of the generator, it is made up of:
##### • A deconvolution layer;
##### • batch normalization layer;
##### • ReLU activation.
class gen_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding, kernel_size=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, X):
        out = self.layers(X)
        return out

##### In order to build the generator we will follow the specifics present in the
##### paper.
##### We will concatenate the 10-dimensional encoding (1 per digit) and the noise
##### to get a 110-dimensional input that will be fed to the first hidden layer.
##### In the last layer we won't apply any batch normalization and the activation 
##### function that we use is a the Tanh function.
class Generator(nn.Module):
    def __init__(self, dim_latent=args.latent_dim, base_width=128, input_ch=C):
        super().__init__()
        self.deconv_z1 = gen_block(dim_latent, base_width*2, stride=1, padding=0)
        self.deconv_y1 = gen_block(10, base_width*2, stride=1, padding=0)
        self.deconv_2 = gen_block(base_width*4, base_width*2, stride=2, padding=1)
        self.deconv_3 = gen_block(base_width*2, base_width, stride=2, padding=1)
        self.deconv_4 = nn.Sequential(
            nn.ConvTranspose2d(base_width, input_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, X, label):
        out_z = self.deconv_z1(X)
        out_y = self.deconv_y1(label)
        out = torch.cat((out_z,out_y), dim=1)
        out = self.deconv_2(out)
        out = self.deconv_3(out)
        out = self.deconv_4(out)

        return out

##### Building block of the discriminator, it is made up of:
##### • A convolution layer;
##### • batch normalization layer;
##### • LeakyReLU activation with alpha=0.2.
class discr_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        if norm is True:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
    
    def forward(self, X):
        out = self.layers(X)
        return out

##### Discriminator follows the same idea of the generator. We can notice that in 
##### this case for the last layer we've substituted the LeakyReLU with the sigmoid.
class Discriminator(nn.Module):
    def __init__(self, base_width=128, input_ch=C):
        super().__init__()
        self.conv_x1 = discr_block(input_ch, base_width//2, norm=False)
        self.conv_y1 = discr_block(10, base_width//2, norm=False)
        self.conv_2 = discr_block(base_width, base_width*2)
        self.conv_3 = discr_block(base_width*2, base_width*4)
        self.conv_4 = nn.Sequential(
            nn.Conv2d(base_width*4, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, X, label):
        out_z = self.conv_x1(X)
        out_y = self.conv_y1(label)
        out = torch.cat((out_z,out_y), dim=1)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)

        return out

##### Let's load now the MNIST dataset
transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        # Normalization for better training performances
        transforms.Normalize((0.5), (0.5))
])
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "datasets",
        train=True,
        download=True,
        transform=transform
    ),
    batch_size=args.batchSize,
    shuffle=True,
    drop_last=True
)

##### Checking for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

##### We can initialize both generator and discriminator with random weights
##### and pass them to the GPU, if available.
generator = Generator()
generator.apply(weights_init)
generator.to(device)

discriminator = Discriminator()
discriminator.apply(weights_init)
discriminator.to(device)

##### Loss function is the usual Binary Cross Entropy
loss_fn = nn.BCELoss().to(device)

##### Let's set up also the optimizers with the correspondent hyperparameters
g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

##### And now we can start with the training itself
generator.train()
discriminator.train()
total_step = len(dataloader)
for epoch in range(args.num_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = args.batchSize
        n_class = args.n_classes
        img_size = args.img_size

        # Defining ground truth for real and fake data
        true_label = torch.full([batch_size], 1.0, dtype=torch.float).to(device)
        fake_label = torch.full([batch_size], 0.0, dtype=torch.float).to(device)

        imgs = imgs.to(device)

        # Creating an image to pass as real one to the generator (filled with ones)
        real_y = torch.zeros(batch_size, n_class)
        real_y = real_y.scatter_(1, labels.view(batch_size, 1), 1).view(batch_size, n_class, 1, 1).contiguous()
        real_y = real_y.expand(-1, -1, img_size, img_size).to(device)

        # Generating the noise
        noise = torch.randn(batch_size, args.latent_dim, 1, 1).to(device)

        # Creating an image to pass as fake one to the generator (filled with zeros)
        gen_labels = (torch.rand(batch_size, 1) * n_class).type(torch.LongTensor)
        gen_y = torch.zeros(batch_size, n_class)
        gen_y = gen_y.scatter_(1, gen_labels.view(batch_size, 1), 1).view(batch_size, n_class,1,1).to(device)
        
        # Synthetic data from generator
        synthetic_data = generator(noise, gen_y)

        # Finally we can procede with the training of the discriminator
        d_optimizer.zero_grad()

        pred_real = discriminator(imgs, real_y)
        error_real = loss_fn(pred_real.squeeze(), true_label)

        gen_y_for_D = gen_y.view(batch_size, n_class, 1, 1).contiguous().expand(-1, -1, img_size, img_size)
        pred_fake = discriminator(synthetic_data.detach(), gen_y_for_D)
        error_fake = loss_fn(pred_fake.squeeze(), fake_label)

        loss_D = (error_fake + error_real)
        loss_D.backward()
        d_optimizer.step()

        # And then with the generator
        generator.zero_grad()

        pred_fake = discriminator(synthetic_data, gen_y_for_D)
        loss_G = loss_fn(pred_fake.squeeze(), true_label)
        loss_G.backward()
        g_optimizer.step()

        # print some informations
        if (i + 1) % args.log_step == 0:
            print(f'Epoch [{epoch+1}/{args.num_epochs}], BatchStep[{i + 1}/{total_step}], D_Real_loss: {error_real.item():.4f}, D_Fake_loss: {error_fake.item():.4f}, G_loss: {loss_G.item():.4f}')
        
        # We can now save the output of generated image
        batches_done = epoch * total_step + i
        if batches_done % args.sample_interval == 0:
            noise = torch.FloatTensor(np.random.normal(0, 1, (n_class**2, args.latent_dim,1,1))).to(device)
            #fixed labels
            y_ = torch.LongTensor(np.array([num for num in range(n_class)])).view(n_class,1).expand(-1,n_class).contiguous()
            y_fixed = torch.zeros(n_class**2, n_class)
            y_fixed = y_fixed.scatter_(1,y_.view(n_class**2,1),1).view(n_class**2, n_class,1,1).to(device)

            gen_imgs = generator(noise, y_fixed).view(-1,C,H,W)

            # saving the generated images in a grid, in the i-th row we place the i-th digit (0-9)
            save_image(gen_imgs.data, img_save_path + f'/{epoch}-{batches_done}.png', nrow=n_class, normalize=True)