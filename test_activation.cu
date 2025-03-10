#include <iostream>
#include <cuda_runtime.h>
#include "activation_layer.h"

__global__ void initialize_test_data(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (idx % 2 == 0) ? idx * 0.1f : -idx * 0.1f;  // Alternating positive and negative values
    }
}

void print_array(float* data, int size) {
    float* h_data = new float[size];
    cudaMemcpy(h_data, data, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;
    delete[] h_data;
}

int main() {
    int size = 10;
    float* d_input;

    cudaMalloc(&d_input, size * sizeof(float));

    initialize_test_data<<<1, size>>>(d_input, size);
    cudaDeviceSynchronize();

    std::cout << "Original Data: ";
    print_array(d_input, size);

    apply_leaky_relu(d_input, size, 0.1f);
    std::cout << "After Leaky ReLU: ";
    print_array(d_input, size);

    initialize_test_data<<<1, size>>>(d_input, size);
    cudaDeviceSynchronize();

    apply_tanh(d_input, size);
    std::cout << "After Tanh: ";
    print_array(d_input, size);

    initialize_test_data<<<1, size>>>(d_input, size);
    cudaDeviceSynchronize();

    apply_sigmoid(d_input, size);
    std::cout << "After Sigmoid: ";
    print_array(d_input, size);

    cudaFree(d_input);
    return 0;
}
