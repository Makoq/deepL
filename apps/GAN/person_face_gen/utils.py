import torch.nn as nn
from generator import Generator
from discriminator import Discriminator

from const import ngpu
from device import device



# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def test_generator(nz, ngf, nc):
   
    netG = get_generator(nz, ngf, nc)
    # Print the model
    print("Generator",netG)

    # Print the model's state_dict
def get_generator(nz, ngf, nc):
     # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)


    return netG


def test_discriminator(nc, ndf):

    netD=get_discriminator(nc, ndf)
    # Print the model
    print("Discriminator",netD)

def get_discriminator(nc, ndf):
     # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)
    return netD