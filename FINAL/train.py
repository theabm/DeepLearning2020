import torch
from dataset import ukiyoe2photo
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t * m + s
        return tensor

def train_fn(disc_P, disc_U, gen_U, gen_P, loader, opt_disc, opt_gen, l1, mse, epoch, sD= None, sG = None):
    loop = tqdm(loader, leave=True)

    for idx, (uki, pic) in enumerate(loop):
        uki = uki.to(config.DEVICE)
        pic = pic.to(config.DEVICE)

        # Train Discriminators P and U
        with torch.cuda.amp.autocast():
            opt_disc.zero_grad()

            fake_pic = gen_P(uki) #generate fake photo from uki painting
            D_P_real = disc_P(pic) #discriminate real photo
            D_P_fake = disc_P(fake_pic.detach()) #discriminate fake photo
            D_P_real_loss = mse(D_P_real, torch.ones_like(D_P_real))
            D_P_fake_loss = mse(D_P_fake, torch.zeros_like(D_P_fake))
            D_P_loss = D_P_real_loss + D_P_fake_loss

            fake_uki = gen_U(pic)
            D_U_real = disc_U(uki)
            D_U_fake = disc_U(fake_uki.detach())
            D_U_real_loss = mse(D_U_real, torch.ones_like(D_U_real))
            D_U_fake_loss = mse(D_U_fake, torch.zeros_like(D_U_fake))
            D_U_loss = D_U_real_loss + D_U_fake_loss

            # put it togethor
            D_loss = (D_P_loss + D_U_loss)/2 #divide by 2 to slow down training compared
                                             # to G.
        D_loss.backward()
        opt_disc.step()
        if epoch >= 100:
            sD.step()
        

        # Train Generators P and U
        with torch.cuda.amp.autocast():
            opt_gen.zero_grad()

            # adversarial loss for both generators
            D_P_fake = disc_P(fake_pic)
            D_U_fake = disc_U(fake_uki)
            loss_G_P = mse(D_P_fake, torch.ones_like(D_P_fake))
            loss_G_U = mse(D_U_fake, torch.ones_like(D_U_fake))

            # cycle loss
            cycle_uki = gen_U(fake_pic)
            cycle_pic = gen_P(fake_uki)
            cycle_uki_loss = l1(uki, cycle_uki)
            cycle_pic_loss = l1(pic, cycle_pic)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_uki = gen_U(uki)
            identity_pic = gen_P(pic)
            identity_uki_loss = l1(uki, identity_uki)
            identity_pic_loss = l1(pic, identity_pic)

            # add all together
            G_loss = (
                loss_G_U
                + loss_G_P
                + cycle_uki_loss * config.LAMBDA_CYCLE
                + cycle_pic_loss * config.LAMBDA_CYCLE
                + identity_pic_loss * config.LAMBDA_IDENTITY
                + identity_uki_loss * config.LAMBDA_IDENTITY
            )

        G_loss.backward()
        opt_gen.step()
        if epoch >= 100:
            sG.step()
        
        if (idx + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}], BatchStep[{idx + 1}/{len(loader)}], D_loss: {D_loss.item():.4f}, G_loss: {G_loss.item():.4f}')

        if idx % 200 == 0:
            unormP = UnNormalize((0.3926, 0.4092, 0.4123), (0.2778, 0.2486, 0.2710))
            unormU = UnNormalize((0.5261, 0.5975, 0.6413), (0.2350, 0.2589, 0.2811))

            save_image(unormP(fake_pic), f"saved_images/{epoch}pic_{idx}.png")
            save_image(unormU(fake_uki), f"saved_images/{epoch}uki_{idx}.png")




def main():
    disc_P = Discriminator().to(config.DEVICE)
    disc_U = Discriminator().to(config.DEVICE)
    gen_U = Generator().to(config.DEVICE)
    gen_P = Generator().to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_P.parameters()) + list(disc_U.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_U.parameters()) + list(gen_P.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_P, gen_P, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_U, gen_U, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_P, disc_P, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_U, disc_U, opt_disc, config.LEARNING_RATE,
        )

    train_dataset = ukiyoe2photo(
        root_pic=config.TRAIN_DIR+"/trainB", root_uki=config.TRAIN_DIR+"/trainA", transformU=config.transformU, transformP = config.transformP
    )
    test_dataset = ukiyoe2photo(
       root_pic=config.TRAIN_DIR+"/testB", root_uki=config.TRAIN_DIR+"/testA", transformU=config.transformU, transformP = config.transformP
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )


    schedulerD = torch.optim.lr_scheduler.StepLR(opt_disc, step_size=1, gamma=config.LEARNING_RATE/100)
    schedulerG = torch.optim.lr_scheduler.StepLR(opt_gen, step_size=1, gamma=config.LEARNING_RATE/100)


    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_P, disc_U, gen_U, gen_P, train_loader,opt_disc, 
        opt_gen, L1, mse, epoch, schedulerD, schedulerG)


        if config.SAVE_MODEL:
            save_checkpoint(gen_P, opt_gen, filename=config.CHECKPOINT_GEN_P)
            save_checkpoint(gen_U, opt_gen, filename=config.CHECKPOINT_GEN_U)
            save_checkpoint(disc_P, opt_disc, filename=config.CHECKPOINT_DISC_P)
            save_checkpoint(disc_U, opt_disc, filename=config.CHECKPOINT_DISC_U)

if __name__ == "__main__":
    main()