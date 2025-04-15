#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include <cuda_runtime.h>

enum class ActivationType {
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH
};

void apply_activation(float* input, int size, ActivationType activation_type, float alpha);
void apply_activation_backward(float* input, float* grad_output, int size, ActivationType activation_type, float alpha, float* grad_input);

#endif  // ACTIVATION_LAYER_H
