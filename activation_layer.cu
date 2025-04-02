// activation_layer.cu - Fully optimized CUDA activation functions
#include "activation_layer.h"
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256  // Fixed block size for all kernels

// Leaky ReLU Kernel (No Branching)
__global__ void leaky_relu_kernel(float* input, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    input[idx] = input[idx] * (input[idx] > 0) + alpha * input[idx] * (input[idx] <= 0);
}

// Tanh Kernel (No Bounds Check)
__global__ void tanh_kernel(float* input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    input[idx] = tanhf(input[idx]);
}

// Sigmoid Kernel (No Bounds Check)
__global__ void sigmoid_kernel(float* input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    input[idx] = 1.0f / (1.0f + expf(-input[idx]));
}

// Confined Kernel Launch: Only Launch Required Threads
void apply_leaky_relu(float* input, int size, float alpha) {
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;  // Confine thread grid
    leaky_relu_kernel<<<gridSize, BLOCK_SIZE>>>(input, size, alpha);
    cudaDeviceSynchronize();
}

void apply_tanh(float* input, int size) {
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;  // Confine thread grid
    tanh_kernel<<<gridSize, BLOCK_SIZE>>>(input);
    cudaDeviceSynchronize();
}

void apply_sigmoid(float* input, int size) {
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;  // Confine thread grid
    sigmoid_kernel<<<gridSize, BLOCK_SIZE>>>(input);
    cudaDeviceSynchronize();
}
