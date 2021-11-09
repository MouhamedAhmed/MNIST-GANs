import os
import torch

from linear_bn import Generator as LBN_Generator
from linear_bn import Discriminator as LBN_Discriminator

from linear import Generator as L_Generator
from linear import Discriminator as L_Discriminator

from cnn_bn import Generator as CNN_BN_Generator
from cnn_bn import Discriminator as CNN_BN_Discriminator

from cnn_sn import Generator as CNN_SN_Generator
from cnn_sn import Discriminator as CNN_SN_Discriminator

class CheckpointTracker():
    def __init__(self, architecture, device, zdim):
        self.architecture = architecture
        self.device = device
        self.zdim = zdim
        
    def save_checkpoint(self, generator, discriminator):
        torch.save(generator.state_dict(), './checkpoints/' + 'gen-' + self.architecture + '.pth')
        torch.save(discriminator.state_dict(), './checkpoints/' + 'disc-' + self.architecture + '.pth')
    
    def load_checkpoint(self, resume):
        # load model weights
        if self.architecture == 'linear':
            gen = L_Generator(z_dim=self.zdim, device=self.device).to(self.device)
            disc = L_Discriminator().to(self.device)

        if self.architecture == 'linear-bn':
            gen = LBN_Generator(z_dim=self.zdim, device=self.device).to(self.device)
            disc = LBN_Discriminator().to(self.device)

        if self.architecture == 'cnn-bn':
            gen = CNN_BN_Generator(z_dim=self.zdim, device=self.device).to(self.device)
            disc = CNN_BN_Discriminator().to(self.device)

        if self.architecture == 'cnn-sn':
            gen = CNN_SN_Generator(z_dim=self.zdim, device=self.device).to(self.device)
            disc = CNN_SN_Discriminator().to(self.device)




        if resume and os.path.exists('./checkpoints/' + 'gen-' + self.architecture + '.pth'):
            gen.load_state_dict(torch.load('./checkpoints/' + 'gen-' + self.architecture + '.pth'))
        
        if resume and os.path.exists('./checkpoints/' + 'disc-' + self.architecture + '.pth'):
            disc.load_state_dict(torch.load('./checkpoints/' + 'disc-' + self.architecture + '.pth'))

        return gen, disc
        
    