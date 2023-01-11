#### SpikingTPU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YigitDemirag/spikingTPU/blob/master/SHD_SNN_fast.ipynb)

A simple Colab notebook supporting the training of recurrent spiking neural networks (RSNN) on multi-devices (TPU, GPUs).
On a single A100 GPU on Colab, it trains a small size RSNN (512 LIF neurons) on Spiking Heidelberg Digits dataset (SHD) under <50 seconds and achieves 76.3% accuracy. 


```
Epoch: 0/250 - Loss: 17.07 - Training acc: 5.96
Epoch: 10/250 - Loss: 2.90 - Training acc: 5.65
Epoch: 20/250 - Loss: 2.42 - Training acc: 21.16
Epoch: 30/250 - Loss: 1.56 - Training acc: 46.11
Epoch: 40/250 - Loss: 1.03 - Training acc: 63.17
Epoch: 50/250 - Loss: 0.72 - Training acc: 73.54
Epoch: 60/250 - Loss: 0.52 - Training acc: 80.41
Epoch: 70/250 - Loss: 0.70 - Training acc: 77.75
Epoch: 80/250 - Loss: 0.42 - Training acc: 84.32
Epoch: 90/250 - Loss: 0.30 - Training acc: 88.67
Epoch: 100/250 - Loss: 0.28 - Training acc: 90.65
Epoch: 110/250 - Loss: 0.27 - Training acc: 90.54
Epoch: 120/250 - Loss: 0.19 - Training acc: 92.69
Epoch: 130/250 - Loss: 0.16 - Training acc: 94.46
Epoch: 140/250 - Loss: 0.13 - Training acc: 95.76
Epoch: 150/250 - Loss: 0.12 - Training acc: 95.44
Epoch: 160/250 - Loss: 0.13 - Training acc: 96.48
Epoch: 170/250 - Loss: 0.15 - Training acc: 96.61
Epoch: 180/250 - Loss: 0.25 - Training acc: 91.85
Epoch: 190/250 - Loss: 0.27 - Training acc: 90.15
Epoch: 200/250 - Loss: 0.09 - Training acc: 96.94
Epoch: 210/250 - Loss: 0.07 - Training acc: 96.00
Epoch: 220/250 - Loss: 0.06 - Training acc: 98.79
Epoch: 230/250 - Loss: 0.05 - Training acc: 99.08
Epoch: 240/250 - Loss: 0.03 - Training acc: 99.33
Training completed in 48.46 seconds (5.16 epoch/s)
SHD Test Accuracy: 76.3%
```

##### features

- written in [JAX](https://github.com/google/jax) for `vmap`, `pmap` <3.
- [tfds](https://github.com/tensorflow/datasets) as a dataloader
- prefetching batches to devices.
