# Activation_layer_trial

## Overview

This repository implements CUDA-accelerated activation functions for an image processing GAN. The activation layer includes Leaky ReLU, Tanh, and Sigmoid, optimized for GPU execution using CUDA kernels.

The test_activation file processes an input array and applies multiple activation functions, ensuring they work as expected.


**Original Data**: [-1.0, 0.0, 1.0, -2.0, 2.0]

The output would be something like:

**After Leaky ReLU**: [-0.1, 0.0, 1.0, -0.2, 2.0](with alpha = 0.1)

**After Tanh**: 0 -0.0997 0.1974 -0.2913 0.3799 -0.4621 0.5370 -0.6043 0.6640 -0.7163

**After Sigmoid**: [0.26894142, 0.5, 0.73105858, 0.11920305, 0.88079708]
