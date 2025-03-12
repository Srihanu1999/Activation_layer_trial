# Activation_layer_trial

## Overview

This repository implements CUDA-accelerated activation functions for an image processing GAN. The activation layer includes Leaky ReLU, Tanh, and Sigmoid, optimized for GPU execution using CUDA kernels.
The test_activation file is used to check if all the functions are working properly by giving a input array. The output would be something like:
Original Data: 0 -0.1 0.2 -0.3 0.4 -0.5 0.6 -0.7 0.8 -0.9
After Leaky ReLU: 0 -0.01 0.2 -0.03 0.4 -0.05 0.6 -0.07 0.8 -0.09
After Tanh: 0 -0.0997 0.1974 -0.2913 0.3799 -0.4621 0.5370 -0.6043 0.6640 -0.7163
After Sigmoid: 0.5 0.4750 0.5498 0.4256 0.5987 0.3775 0.6457 0.3318 0.6899 0.2890
