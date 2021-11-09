import numpy as np
import torch
import os
import shutil
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST 
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime 
from checkpoint_tracker import CheckpointTracker
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, configs):
        # configs
        self.lr = configs['lr']
        self.batch_size = configs['batch_size']
        self.num_of_epochs = configs['num_of_epochs']
        self.architecture = configs['architecture']
        self.resume = configs['resume']
        self.display_step = configs['display_step']
        self.zdim = configs['zdim']
        
        torch.manual_seed(0)
        
        #device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # checkpoint tracker to save model checkpoint
        self.checkpoint_tracker = CheckpointTracker(self.architecture, self.device, self.zdim)

        # load model weights
        self.generator, self.discriminator = self.checkpoint_tracker.load_checkpoint(self.resume)

        # dataset
        # Load MNIST dataset as tensors
        if self.architecture == 'cnn-bn':
            self.dataloader = DataLoader(
                MNIST(root='.', download=True,
                        transform=transforms.Compose([
                            transforms.Resize(28),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ])),

                batch_size=self.batch_size,
                shuffle=True)
        
        else:
            self.dataloader = DataLoader(
                MNIST('.', download=True, transform=transforms.ToTensor()),
                batch_size=self.batch_size,
                shuffle=True)

        # optimizer
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        # loss
        self.criterion = nn.BCEWithLogitsLoss()

        # ckpt folder
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')

        # 
        if os.path.exists(self.architecture + '-result-samples'):
            shutil.rmtree(self.architecture + '-result-samples')
        os.mkdir(self.architecture + '-result-samples')
        
        # 
        if not os.path.exists('plots'):
            os.mkdir('plots')

        
    def show_tensor_images(self, image_tensor, img_name, num_images=25, size=(1, 28, 28)):
        '''
        Function for visualizing images: Given a tensor of images, number of images, and
        size per image, plots and prints the images in a uniform grid.
        '''
        image_unflat = image_tensor.detach().cpu().view(-1, *size)
        image_grid = make_grid(image_unflat[:num_images], nrow=5)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.savefig(self.architecture + '-result-samples/'+img_name)
        
        plt.close()

    def train(self):
        cur_step = 0
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        gen_losses = []
        disc_losses = []
        mean_epoch_gen_losses = []
        mean_epoch_disc_losses = []

        for epoch in range(self.num_of_epochs):
            epoch_gen_losses = []
            epoch_disc_losses = []
            # Dataloader returns the batches
            for real, _ in tqdm(self.dataloader):
                cur_batch_size = len(real)
                real = real.to(self.device)

                # Flatten the batch of real images from the dataset
                if self.architecture in ['linear', 'linear-bn']:
                    real = real.view(cur_batch_size, -1).to(self.device)



                #### discriminator
                # zero grad
                self.disc_opt.zero_grad()

                # forward path
                fake = self.generator(self.batch_size)
                disc_fake_pred = self.discriminator(fake)
                disc_real_pred = self.discriminator(real)
                # loss
                disc_fake_loss = self.criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = self.criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                epoch_disc_losses.append(disc_loss.item())

                # backward path
                disc_loss.backward(retain_graph=True)

                # update optimizer
                self.disc_opt.step()
                ####



                #### generator
                # zero grad
                self.gen_opt.zero_grad()
                
                # forward path
                fake = self.generator(self.batch_size)

                # loss
                disc_fake_pred = self.discriminator(fake)
                gen_loss = self.criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
                gen_losses.append(gen_loss.item())
                epoch_gen_losses.append(gen_loss.item())

                # backward path
                gen_loss.backward()

                # update optimizer
                self.gen_opt.step()
                ####



                # Keep track of the average discriminator loss
                mean_discriminator_loss += disc_loss.item() / self.display_step

                # Keep track of the average generator loss
                mean_generator_loss += gen_loss.item() / self.display_step

                ### Visualization code ###
                if cur_step % self.display_step == 0 and cur_step > 0:
                    print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                    self.show_tensor_images(fake, 'fake-'+str(cur_step)+'.png')
                    self.show_tensor_images(real, 'real-'+str(cur_step)+'.png')
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0
                cur_step += 1
            
            # save checkpoint
            self.checkpoint_tracker.save_checkpoint(self.generator, self.discriminator)

            # mean gen losses
            gen_mean = sum(epoch_gen_losses) / len(epoch_gen_losses)
            mean_epoch_gen_losses.append(gen_mean)
            # mean disc losses
            disc_mean = sum(epoch_disc_losses) / len(epoch_disc_losses)
            mean_epoch_disc_losses.append(disc_mean)

        
        # plot losses
        plt.plot(mean_epoch_disc_losses, color='blue', label='Discriminator')
        plt.plot(mean_epoch_gen_losses, color='red', label='Generator')
        plt.savefig('plots/'+self.architecture+'-epoch-mean-losses.png')
        plt.close()

        plt.plot(disc_losses, color='blue', label='Discriminator')
        plt.plot(gen_losses, color='red', label='Generator')
        plt.savefig('plots/'+self.architecture+'-batch-losses.png')
        plt.close()



