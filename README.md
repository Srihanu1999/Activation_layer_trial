# Activation_layer_trial

## Overview

This repository implements CUDA-accelerated activation functions for an image processing GAN. The activation layer includes Leaky ReLU, Tanh, and Sigmoid, optimized for GPU execution using CUDA kernels.

The test_activation file processes an input array and applies multiple activation functions, ensuring they work as expected.


**Original Data**: [-1.0, 0.0, 1.0, -2.0, 2.0]

The output would be something like:

**After Leaky ReLU**: [-0.1, 0.0, 1.0, -0.2, 2.0](with alpha = 0.1)

**After Tanh**:[-0.761594, 0.0, 0.761594, -0.964027, 0.964027]

**After ReLU**:[0.0, 0.0, 1.0, 0.0, 2.0]

**After Sigmoid**: [0.26894142, 0.5, 0.73105858, 0.11920305, 0.88079708]
