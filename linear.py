import torch
from torch import nn

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128, device='cpu'):
        super(Generator, self).__init__()
        self.device = device
        self.zdim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.get_generator_block(z_dim, hidden_dim),
            self.get_generator_block(hidden_dim, hidden_dim * 2),
            self.get_generator_block(hidden_dim * 2, hidden_dim * 4),
            self.get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
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
        '''
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, batch_size):
        noise = self.get_noise(batch_size)
        return self.gen(noise)
    
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.get_discriminator_block(im_dim, hidden_dim * 4),
            self.get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            self.get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
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
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, image):
        return self.disc(image)
    
