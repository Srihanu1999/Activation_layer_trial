// activation_layer.cu - CUDA implementation of activation functions
#include "activation_layer.h"
#include <cmath>

__global__ void leaky_relu_kernel(float* input, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = (input[idx] > 0) ? input[idx] : alpha * input[idx];
    }
}

__global__ void tanh_kernel(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = tanhf(input[idx]);
    }
}

__global__ void sigmoid_kernel(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

void apply_leaky_relu(float* input, int size, float alpha) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    leaky_relu_kernel<<<numBlocks, blockSize>>>(input, size, alpha);
    cudaDeviceSynchronize();
}

void apply_tanh(float* input, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    tanh_kernel<<<numBlocks, blockSize>>>(input, size);
    cudaDeviceSynchronize();
}

void apply_sigmoid(float* input, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    sigmoid_kernel<<<numBlocks, blockSize>>>(input, size);
    cudaDeviceSynchronize();
}
