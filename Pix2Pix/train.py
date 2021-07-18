import argparse
import os
import random
import torch
import torchvision
import torch.nn as nn
from torch.backends import cudnn
from torch import optim
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from PIL import Image
from pix2pix import *

##### The training phase is pretty straightforward and follows the usual
##### path of a general cGAN architecture:
##### • train the discriminator on the real data;
##### • train the discriminator on the synthetic data;
##### • train the generator using synthetic data.

##### It's important to notice that the entire network is trying to learn
##### in a structured way. In fact, from previous approaches it has been 
##### found that it is beneficial to mix the GAN objective with a more 
##### traditional loss, such as L2 distance. The discriminator’s job remains 
##### unchanged, but the generator is tasked to not only fool the discriminator 
##### but also to be near the ground truth output in an L2 sense.
##### Note: L1 distance is used rather than L2 as L1 encourages less blurring.

##### Here below some code to efficiently deal with command line input.

parser = argparse.ArgumentParser(description='Our Implementation of Pix2Pix')

parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders train, val, etc)')
parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')

parser.add_argument('--no_resize_or_crop', action='store_true', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

##### Hyperparameters follow the values present in the paper:
##### • learning rate = 2e-4;
##### • Adam optimizer with beta1 = 0.5 and beta2 = 0.999
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
parser.add_argument('--lambda_A', type=float, default=100.0)

parser.add_argument('--model_path', type=str, default='./models') 
parser.add_argument('--sample_path', type=str, default='./results') 
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--sample_step', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=2)

##### Class for loading in the proper way the dataset
class ImageFolder(data.Dataset):
    def __init__(self, opt):
        # os.listdir Function gives all lists of directory
        self.root = opt.dataroot
        self.no_resize_or_crop = opt.no_resize_or_crop
        self.no_flip = opt.no_flip
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5),
                                                                  (0.5, 0.5, 0.5))])
        self.dir_AB = os.path.join(opt.dataroot, 'train') 
        self.image_paths = list(map(lambda x: os.path.join(self.dir_AB, x), os.listdir(self.dir_AB)))

    def __getitem__(self, index):
        AB_path = self.image_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        if(not self.no_resize_or_crop):
            AB = AB.resize((286 * 2, 286), Image.BICUBIC)
            AB = self.transform(AB)

            w = int(AB.size(2) / 2)
            h = AB.size(1)
            w_offset = random.randint(0, max(0, w - 256 - 1))
            h_offset = random.randint(0, max(0, h - 256 - 1))

            A = AB[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
            B = AB[:, h_offset:h_offset + 256, w + w_offset:w + w_offset + 256]
        else:
            AB = self.transform(AB)
            w_total = AB.size(2)
            w = int(w_total / 2)

            A = AB[:, :256, :256]
            B = AB[:, :256, w:w + 256]

        if (not self.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        return {'A': A, 'B': B}

    def __len__(self):
        return len(self.image_paths)

##### Helper Function for denormalization of an image
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

##### Custom weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

##### GAN loss helper function, marks the images passed as true or fake
def GAN_Loss(input, target, criterion, device):
    if target == True:
        labels = torch.full(input.size(), fill_value = 1.0, dtype=torch.float)
    else:
        labels = torch.full(input.size(), fill_value = 0.0, dtype=torch.float)

    labels = labels.to(device)

    return criterion(input, labels)



def main():
    # enable the built-in cudnn auto-tuner to find the best algorithm 
    # to use for your hardware
    cudnn.benchmark = True
    global args
    args = parser.parse_args()

    dataset = ImageFolder(args)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=args.batchSize,
                                  shuffle=True,
                                  num_workers=args.num_workers)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

    generator = Generator(args.batchSize)
    discriminator = Discriminator(args.batchSize)
    
    # Apply init weights
    weights_init(generator)
    weights_init(discriminator)
    
    # Loss functions
    loss_fn = nn.BCELoss()
    loss_L1 = nn.L1Loss()

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator, device_ids=[0,1])
        discriminator = nn.DataParallel(discriminator, device_ids=[0,1])

    generator.to(device)
    discriminator.to(device)
    
    # Training phase
    generator.train()
    discriminator.train()
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, sample in enumerate(data_loader):

            AtoB = args.which_direction == 'AtoB'
            input_A = sample['A' if AtoB else 'B']
            input_B = sample['B' if AtoB else 'A']

            # Discriminator training
            discriminator.zero_grad()

            real_A = input_A.to(device)
            fake_B = generator(real_A)
            real_B = input_B.to(device)

            # train discriminator first on comparing real input to synthetic one
            pred_fake = discriminator(real_A, fake_B)
            error_fake = GAN_Loss(pred_fake, False, loss_fn, device)
            
            # then compare real input to the conditioning one
            pred_real = discriminator(real_A, real_B)
            error_real = GAN_Loss(pred_real, True, loss_fn, device)

            # Combined loss
            loss_D = (error_fake + error_real) * 0.5

            # Retaining the graph in order to perform backpropagation since
            # D is independent w.r.t. G.
            # We can call the backward here only once since we have already
            # summed the two losses
            loss_D.backward(retain_graph=True)
            d_optimizer.step()

            # Generator training
            generator.zero_grad()

            pred_fake = discriminator(real_A, fake_B)
            loss_G_GAN = GAN_Loss(pred_fake, True, loss_fn, device)

            loss_G_L1 = loss_L1(fake_B, real_B)

            loss_G = loss_G_GAN + loss_G_L1 * args.lambda_A
            loss_G.backward()
            g_optimizer.step()

            # print some informations
            if (i + 1) % args.log_step == 0:
                print(f'Epoch [{epoch+1}/{args.num_epochs}], BatchStep[{i + 1}/{total_step}], D_Real_loss: {error_real.item():.4f}, D_Fake_loss: {error_fake.item():.4f}, G_loss: {loss_G_GAN.item():.4f}, G_L1_loss: {loss_G_L1.item():.4f}')

            # save the sampled images
            if (i + 1) % args.sample_step == 0:
                res = torch.cat((torch.cat((real_A, fake_B), dim=3), real_B), dim=3)
                torchvision.utils.save_image(denorm(res.data), os.path.join(args.sample_path, f'Generated-{epoch+1}-{i+1}.png' ))

        # save the model parameters for each epoch
        g_path = os.path.join(args.model_path, f'generator-{epoch+1}.pkl')
        torch.save(generator.state_dict(), g_path)

if __name__ == "__main__":
    main()