// activation_layer.h - Header file for activation functions
#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

void apply_leaky_relu(float* input, int size, float alpha);
void apply_tanh(float* input, int size);
void apply_sigmoid(float* input, int size);

#endif // ACTIVATION_LAYER_H
