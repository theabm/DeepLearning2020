import torch 
import torch.nn as nn

# define the basic block of the discriminator
# from the paper we have 70x70 PatchGANs

class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, n_filter=64):
        super().__init__()
        ks = 4 #vkernel size
        pw = 1 #pad 
        model = [
            nn.Conv2d(in_channels, out_channels=n_filter, kernel_size = ks, stride = 2, padding = pw, padding_mode = "reflect"),
            nn.LeakyReLU(0.2, True)
            # no instance norm for first conv layer
        ]

        layers = 3
        channels = int(n_filter)
        for i in range(1, layers):
            mult = 2**(i)
            model += [
                nn.Conv2d(channels, channels*mult, kernel_size = ks, stride = 2, padding = pw, bias = True, padding_mode = "reflect"),
                nn.InstanceNorm2d(channels*mult),
                nn.LeakyReLU(0.2, True)
            ]
            channels = channels*mult
         
        model += [nn.Conv2d(channels, channels*2, kernel_size = ks, stride = 1, padding = pw, bias = True, padding_mode = "reflect"),
        nn.InstanceNorm2d(channels*2),
        nn.LeakyReLU(0.2, True)
        ]

        model += [nn.Conv2d(channels*2, 1, kernel_size= ks, stride = 1, padding = pw, padding_mode = "reflect")]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
