import torch
import torchvision.utils as vutils
from torch.cuda.amp import autocast, GradScaler

from const import ngpu, nz, ngf, nc, ndf, batch_size, workers, image_size, num_epochs

def train(dataloader, netD, netG, fixed_noise, criterion, real_label, fake_label, optimizerD, optimizerG, device):
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    scalerD = GradScaler()
    scalerG = GradScaler()

    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            with autocast():
                output = netD(real_cpu).view(-1)
                errD_real = criterion(output, label)
            scalerD.scale(errD_real).backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            with autocast():
                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
            scalerD.scale(errD_fake).backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake

            scalerD.step(optimizerD)
            scalerD.update()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            with autocast():
                output = netD(fake).view(-1)
                errG = criterion(output, label)
            scalerG.scale(errG).backward()
            D_G_z2 = output.mean().item()

            scalerG.step(optimizerG)
            scalerG.update()

            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    return G_losses, D_losses, img_list
