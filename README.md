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

## Results (Losses Plots)
Here are the plots of loss function after 20 epochs of training, where `blue` is for `Discriminator` and `red` is for `Generator`.

-`linear`

![Alt text](plots/linear-epoch-mean-losses.png?raw=true "Title")


-`linear-bn`

![Alt text](plots/linear-bn-epoch-mean-losses.png?raw=true "Title")


-`cnn-bn`

![Alt text](plots/cnn-bn-epoch-mean-losses.png?raw=true "Title")


-`cnn-sn`

![Alt text](plots/cnn-sn-epoch-mean-losses.png?raw=true "Title")


## Results (Visual Validation)
Here are the outputs after 5, 10, 15 and 20 epochs. You can find the samples after each 500 steps in `<model>-result-samples` folders.

- `linear`

1. 5 epochs

![Alt text](linear-result-samples/fake-9500.png?raw=true "Title")

2. 10 epochs

![Alt text](linear-result-samples/fake-19000.png?raw=true "Title")

3. 15 epochs

![Alt text](linear-result-samples/fake-28500.png?raw=true "Title")

4. 20 epochs

![Alt text](linear-result-samples/fake-37000.png?raw=true "Title")



- `linear-bn`

1. 5 epochs

![Alt text](linear-bn-result-samples/fake-9500.png?raw=true "Title")

2. 10 epochs

![Alt text](linear-bn-result-samples/fake-19000.png?raw=true "Title")

3. 15 epochs

![Alt text](linear-bn-result-samples/fake-28500.png?raw=true "Title")

4. 20 epochs

![Alt text](linear-bn-result-samples/fake-37000.png?raw=true "Title")



- `cnn-bn`

1. 5 epochs

![Alt text](cnn-bn-result-samples/fake-9500.png?raw=true "Title")

2. 10 epochs

![Alt text](cnn-bn-result-samples/fake-19000.png?raw=true "Title")

3. 15 epochs

![Alt text](cnn-bn-result-samples/fake-28500.png?raw=true "Title")

4. 20 epochs

![Alt text](cnn-bn-result-samples/fake-37000.png?raw=true "Title")




- `cnn-sn`

1. 5 epochs

![Alt text](cnn-sn-result-samples/fake-9500.png?raw=true "Title")

2. 10 epochs

![Alt text](cnn-sn-result-samples/fake-19000.png?raw=true "Title")

3. 15 epochs

![Alt text](cnn-sn-result-samples/fake-28500.png?raw=true "Title")

4. 20 epochs

![Alt text](cnn-sn-result-samples/fake-37000.png?raw=true "Title")




