#include <iostream>
#include <cassert>
#include "activation_layer.h"

#define SIZE 5  // Size of the input array

// Helper function to print arrays
void print_array(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

// Test the activation layer
int main() {
    float input[SIZE] = {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f};  // Example input

    // Test ReLU
    float* d_input;
    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMemcpy(d_input, input, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Testing ReLU activation:" << std::endl;
    apply_activation(d_input, SIZE, ActivationType::RELU, 0.0f);  // ReLU does not need alpha

    cudaMemcpy(input, d_input, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    print_array(input, SIZE);  // Expected output: 0.0 0.0 1.0 0.0 2.0

    // Test Leaky ReLU
    cudaMemcpy(d_input, input, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "\nTesting Leaky ReLU activation with alpha=0.1:" << std::endl;
    apply_activation(d_input, SIZE, ActivationType::LEAKY_RELU, 0.1f);

    cudaMemcpy(input, d_input, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    print_array(input, SIZE);  // Expected output: -0.1 0.0 1.0 -0.2 2.0

    // Test Sigmoid
    cudaMemcpy(d_input, input, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "\nTesting Sigmoid activation:" << std::endl;
    apply_activation(d_input, SIZE, ActivationType::SIGMOID, 0.0f);  // Sigmoid does not need alpha

    cudaMemcpy(input, d_input, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    print_array(input, SIZE);  // Expected output: 0.268941 0.5 0.731059 0.119203 0.880797

    cudaFree(d_input);
    return 0;
}
