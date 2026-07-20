#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include <stdio.h>
#include <arm_neon.h>
#include "feature_extraction.h"

int vector_multiplication_bias_relu(
    const float32_t *input,
    size_t in_size,
    const float32_t *weights,
    size_t w_size,
    const float32_t *bias,
    size_t bias_size,
    float32_t *output,
    size_t out_size);

int vector_multiplication_bias(
    const float32_t *input,
    size_t in_size,
    const float32_t *weights,
    size_t w_size,
    const float32_t *bias,
    size_t bias_size,
    float32_t *output,
    size_t out_size);

int auto_encoder(all_features *input_features, all_features *output_features);

#endif //AUTOENCODER_H
