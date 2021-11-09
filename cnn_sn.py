import torch
from torch import nn
from spectral_norm import SpectralNorm

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_dim=28, hidden_dim=128, device='cpu'):
        super(Generator, self).__init__()
        self.device = device
        self.zdim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(z_dim, im_dim*8, 4, 1, 0)),
            nn.ReLU(inplace=True),
            self.get_generator_block(im_dim*8, im_dim*4),
            self.get_generator_block(im_dim*4, im_dim*2),
            self.get_generator_block(im_dim*2, im_dim),
            nn.ConvTranspose2d(im_dim, 1, kernel_size=1, stride=1, padding=2),
            nn.Tanh()
        )

    def get_noise(self, n_samples):
        '''
        Function for creating noise vectors: Given the dimensions (n_samples),
        creates a tensor of that shape filled with random numbers from the normal distribution.
        Parameters:
            n_samples: the number of samples to generate, a scalar
            device: the device type
        '''
        return torch.randn(n_samples,self.zdim,device=self.device)


    def get_generator_block(self, input_dim, output_dim):
        '''
        Function for returning a block of the generator's neural network
        given input and output dimensions.
        Parameters:
            input_dim: the dimension of the input vector, a scalar
            output_dim: the dimension of the output vector, a scalar
        Returns:
            a generator neural network layer, with a linear transformation 
            followed by a batch normalization and then a relu activation
        '''
        return nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(input_dim, output_dim, 4, 2, 1)),
            nn.ReLU(inplace=True),
        )

    def forward(self, batch_size):
        noise = self.get_noise(batch_size)
        noise = torch.unsqueeze(noise, 2)
        noise = torch.unsqueeze(noise, 2)
        return self.gen(noise)
    
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_dim=28, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, im_dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            self.get_discriminator_block(im_dim, im_dim*2),
            self.get_discriminator_block(im_dim*2, im_dim*4),
            nn.Conv2d(im_dim*4, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def get_discriminator_block(self, input_dim, output_dim):
        '''
        Discriminator Block
        Function for returning a neural network of the discriminator given input and output dimensions.
        Parameters:
            input_dim: the dimension of the input vector, a scalar
            output_dim: the dimension of the output vector, a scalar
        Returns:
            a discriminator neural network layer, with a linear transformation 
            followed by an nn.LeakyReLU activation with negative slope of 0.2 
            (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
        '''
        return nn.Sequential(
            SpectralNorm(nn.Conv2d(input_dim, output_dim, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, image):
        return self.disc(image)
    