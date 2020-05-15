# Pytorch-Practice
Some machine learning and deep learning practice with PyTorch.

## Table of Contents
* [FullyConnectedNet_MNIST](#FullyConnectedNet_MNIST)
* [AlexNet_CIFAR10](#AlexNet_CIFAR10)
* [Problems to solve](#Problems-to-solve)
* [Solved problems](#Solved-problems)

### FullyConnectedNet_MNIST
Use a simple fully connected network to classify MNIST, which can achieve 97% test accuracy. But MNIST is too easy, even linear regression can achieve 90% test accuracy.

### AlexNet_CIFAR10
Use the original AlexNet and my modified AlexNet with global average pooling (GAP) to classify CIFAR10. Original AlexNet can achieve 76% test accuracy and AlexNet with GAP can achieve 80% test accuracy. CIFAR10 is much more difficult than MNIST.

### Problems to solve

### Solved problems
1. Problem: Don't know why dataLoader parameter: num_worker>0 in .py file will lead to runtime error. While it can be set and works fine in .ipynb file. 
    * Solution: it's just a Windows10 problem, don't sweat on that.
