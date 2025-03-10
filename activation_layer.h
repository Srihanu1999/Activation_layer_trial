// activation_layer.h - Header file for activation functions
#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(float* input, int size, float alpha);
__global__ void tanh_kernel(float* input, int size);
__global__ void sigmoid_kernel(float* input, int size);

void apply_leaky_relu(float* input, int size, float alpha);
void apply_tanh(float* input, int size);
void apply_sigmoid(float* input, int size);

#endif
