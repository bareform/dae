## Denoising AutoEncoder 

### Table of Contents
1. [Architecture](#Architecture)
1. [Datasets](#Datasets)
1. [Training](#Training)
1. [References](#References)

### Architecture
AutoEncoders consist of two networks (an encoder and a decoder) connected together in latent space. The encoder encodes the input data into a lower dimensional space, while the decoder decodes the encoded data into higher dimensional space. The Denoising AutoEncoder (DAE) is a variation on the AutoEncoder architecture, where instead of feeding in the original input to the encoder, the encode is instead feed a noisy version of it instead [<a href="#vincent2008denoisingautoencoder">2</a>]. The task of the DAE is to reconstruct the original non-noisy input.

The DAE implementation in this repository is intentionally kept simple: rather than requiring the exact dimensions of each layer, you only need to specify the number of encoder and decoder layers. The dimension of each layer is then automatically calculated in evenly spaced increments.

### Datasets

#### `MNIST`

You can download the MNIST dataset [<a href="#lecun1998mnist">1</a>] using the following command:

```console
python3 data/download_fashionmnist.py
```

This command will download place the dataset at the following path: `./data/datasets`.

The dataset consist of 70000 28x28 grayscale images split into 10 classes with each class represent a digit from 0 to 9. There are a total of 60000 training images and 10000 test images.

| Category | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Total |
| -------- | - | - | - | - | - | - | - | - | - | - | ----- |
| Training | 5923 | 6742 | 5958 | 6131 | 5842 | 5421 | 5918 | 6265 | 5851 | 5949 | 60000 |
| Test | 980 | 1135 | 1032 | 1010 | 982 | 892 | 958 | 1028 | 974 | 1009 | 10000 |

---

#### `FashionMNIST`

You can download the FashionMNIST dataset [<a href="#xia2017fashionmnist">3</a>] using the following command:

```console
python3 data/download_fashionmnist.py
```

The dataset consist of 70000 28x28 grayscale images split into 10 classes with each class representing a different article of clothing. There are a total of 60000 training images and 10000 test images.

| Category | T-shirt/Top | Trouser | Pullover | Dress | Coat | Sandal | Shirt | Sneaker | Bag | Ankle Boot | Total |
| -------- | - | - | - | - | - | - | - | - | - | - | ----- |
| Training | 6000 | 6000 | 6000 | 6000 | 6000 | 6000 | 6000 | 6000 | 6000 | 6000 | 60000 |
| Test | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 10000 |

This command will download place the dataset at the following path: `./data/datasets`.

### Training

#### `MNIST`

You can train a feedforward AutoEncoder model on the MNIST dataset using the following command:

```console
python3 -m utils.trainer \
  --root './data/datasets' \
  --dataset 'MNIST' \
  --batch_size 512 \
  --num_workers 1 \
  --num_epochs 5000 \
  --lr 0.001 \
  --optimizer 'AdamW' \
  --lr_scheduler 'StepLR' \
  --step_size 200 \
  --gamma 0.9 \
  --model_type 'ffn' \
  --num_encoder_layers 3 \
  --num_decoder_layers 3 \
  --latent_features 128 \
  --random_seed 0 \
  --ckpt_dir './checkpoints' \
  --results_dir './results'
  --save_interval 1000
  --compile \
  --pin_memory
```

The command above will produce the following results:

<p align="center">
  <img src="./docs/MNIST/gif/denoising.gif"/>
</p>

---

#### `FashionMNIST`

You can train a feedforward AutoEncoder model on the FashionMNIST dataset using the following command:

```console
python3 -m utils.trainer \
  --root './data/datasets' \
  --dataset 'FashionMNIST' \
  --batch_size 512 \
  --num_workers 1 \
  --num_epochs 5000 \
  --lr 0.001 \
  --optimizer 'AdamW' \
  --lr_scheduler 'StepLR' \
  --step_size 200 \
  --gamma 0.9 \
  --model_type 'ffn' \
  --num_encoder_layers 3 \
  --num_decoder_layers 3 \
  --latent_features 128 \
  --random_seed 0 \
  --ckpt_dir './checkpoints' \
  --results_dir './results'
  --save_interval 3000
  --compile \
  --pin_memory
```

The command above will produce the following results:

<p align="center">
  <img src="./docs/FashionMNIST/gif/denoising.gif"/>
</p>

Denoising images from FashionMNIST results in the image losing fine-grained details for the article of clothing (e.g. a plaid shirt loses its plaid strips), but maintains the overall shape.

### References
<a name="lecun1998mnist"></a>[1] Yann LeCun, Corinna Cortes, and Christopher J.C. Burges. The MNIST database of handwritten digits. 1998. 

<a name="vincent2008denoisingautoencoder"></a>[2] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and Composing Robust Features with Denoising Autoencoders. In ICML, pages 1096-1103, 2008.

<a name="xia2017fashionmnist"></a>[3] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. *arXiv preprint arXiv:1708.07747*, 2017.
