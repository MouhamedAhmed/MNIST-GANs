# MNIST-GANs
In this Repository, there're 4 different simple GAN architectures trained on `MNIST` dataset
- `linear`: Fully Connected Layers only
- `linear-bn`: Fully Connected Layers with BatchNorm
- `cnn-bn`: CNN Layers with BatchNorm (DCGAN)
- `cnn-sn`: CNN Layers with SpectralNorm (DCGAN + SpectralNorm)

## Installations
```bash
conda create --name MnistGan --file spec-file.txt
conda activate MnistGan
```

## Train
```bash
python main.py\ 
  -arch <gan_version>\
  -lr <learning_rate>\
  -batch <batch_size>\
  -display_step <losses_display_step>\
  -epochs <number_of_epochs>\
  -zdim <dim_of_z_noise_vector>\
```

- **gan_version**: any version of the 4 stated above, default is `linear`
- **learning_rate**: learning rate used to train bert model, default is `0.00001`
- **batch_size**: number of samples per batch, default is `128`
- **losses_display_step**: display losses after how many steps, default is `500`
- **number_of_epochs**: number of epochs, default is `200`
- **dim_of_z_noise_vector**: number of epochs, default is `64`


you can add `--resume_from_last_trial` to resume from last checkpoint

For Example:
```bash
python main.py --architecture linear --batch_size 32 -lr 0.00001 -epochs 20
```

```bash
python main.py --architecture linear-bn --batch_size 32 -lr 0.00001 -epochs 20
```

```bash
python main.py --architecture cnn-bn --batch_size 32 -lr 0.00001 -epochs 20
```

```bash
python main.py --architecture cnn-sn --batch_size 32 -lr 0.00001 -epochs 20
```

## Results
Here are the plots of loss function after 20 epochs of training
-`linear`
![Alt text](plots/linear-epoch-mean-losses.png?raw=true "Title")
