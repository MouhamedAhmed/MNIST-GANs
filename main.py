import argparse
import os
from trainer import Trainer

def main():
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-arch', '--architecture', type=str, help='model architecture, choose from [linear, linear-bn, cnn-bn, cnn-sn], default is linear!', default = 'linear')
    argparser.add_argument('-lr', '--learning_rate', type=float, help='starting learning rate', default=0.00001)
    argparser.add_argument('-batch', '--batch_size', type=int, help='batch size', default=128)
    argparser.add_argument('-display_step', '--display_step', type=int, help='validate each how many steps', default=500)
    argparser.add_argument('-epochs', '--num_of_epochs', type=int, help='num of epochs', default=200)
    argparser.add_argument("--resume_from_last_trial", action='store_true')
    argparser.add_argument('-zdim', '--zdim', type=int, help='dimension of input noise vector', default=64)
    args = argparser.parse_args()
    
    if not (args.architecture in ['linear', 'linear-bn', 'cnn-bn', 'cnn-sn']):
        print('incorrect model architecture! choose from [linear, linear-bn, cnn-bn, cnn-sn].')
        exit()
    
    configs = {
        'lr': args.learning_rate,
        'batch_size': args.batch_size,
        'num_of_epochs': args.num_of_epochs,
        'architecture': args.architecture,
        'resume': args.resume_from_last_trial, 
        'display_step': args.display_step,
        'zdim': args.zdim
    }
    trainer = Trainer(configs)
    trainer.train()


if __name__ == '__main__':
    main()
        
        
                                                                
                            



 

