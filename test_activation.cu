#include <iostream>
#include <cuda_runtime.h>
#include "activation_layer.h"

#define SIZE 10  // Number of test elements

void print_array(const char* label, float* arr, int size) {
    std::cout << label << ": ";
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    // Test input values
    float h_input[SIZE] = { -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, -3.0f, 3.0f, -4.0f };
    float *d_input;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_input, SIZE * sizeof(float));

    // Copy input data to GPU
    cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Apply Leaky ReLU (alpha = 0.1)
    apply_leaky_relu(d_input, SIZE, 0.1f);
    cudaMemcpy(h_input, d_input, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    print_array("Leaky ReLU", h_input, SIZE);

    // Restore input values and copy again
    float h_input2[SIZE] = { -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, -3.0f, 3.0f, -4.0f };
    cudaMemcpy(d_input, h_input2, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Apply Tanh
    apply_tanh(d_input, SIZE);
    cudaMemcpy(h_input, d_input, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    print_array("Tanh", h_input, SIZE);

    // Restore input values
    cudaMemcpy(d_input, h_input2, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Apply Sigmoid
    apply_sigmoid(d_input, SIZE);
    cudaMemcpy(h_input, d_input, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    print_array("Sigmoid", h_input, SIZE);

    // Free GPU memory
    cudaFree(d_input);

    return 0;
}
