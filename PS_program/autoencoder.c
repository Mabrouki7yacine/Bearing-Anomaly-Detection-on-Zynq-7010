#include "autoencoder.h"
#include "weights.h"

int vector_multiplication_bias_relu(
    const float32_t *input,
    size_t in_size,
    const float32_t *weights,
    size_t w_size,
    const float32_t *bias,
    size_t bias_size,
    float32_t *output,
    size_t out_size)
{
    if (input == NULL ||
        weights == NULL ||
        bias == NULL ||
        output == NULL)
    {
        return -1;
    }

    if (in_size == 0U || out_size == 0U) {
        return -1;
    }

    if (in_size * out_size != w_size) {
        xil_printf(
            "Wrong weight dimensions: in_size * out_size = %lu, "
            "but w_size = %lu\r\n",
            (unsigned long)(in_size * out_size),
            (unsigned long)w_size
        );
        return -1;
    }

    if (bias_size != out_size) {
        xil_printf(
            "Wrong bias dimensions: bias_size = %lu, "
            "but out_size = %lu\r\n",
            (unsigned long)bias_size,
            (unsigned long)out_size
        );
        return -1;
    }

    if ((out_size % 4U) != 0U) {
        xil_printf(
            "out_size must be a multiple of 4, but out_size = %lu\r\n",
            (unsigned long)out_size
        );
        return -1;
    }

    const float32x4_t vZero = vdupq_n_f32(0.0f);

    for (size_t j = 0U; j < out_size; j += 4U) {
        /*
         * vOutput = {
         *     bias[j + 0],
         *     bias[j + 1],
         *     bias[j + 2],
         *     bias[j + 3]
         * };
         */
        float32x4_t vOutput = vld1q_f32(bias + j);

        for (size_t i = 0U; i < in_size; ++i) {
            /*
             * Expected weight layout:
             *
             * weights[i * out_size + j + 0]
             * weights[i * out_size + j + 1]
             * weights[i * out_size + j + 2]
             * weights[i * out_size + j + 3]
             */
            float32x4_t vWeight = vld1q_f32(weights + i * out_size + j);

            // vOutput += vWeight * input[i]
            vOutput = vmlaq_n_f32(vOutput, vWeight, input[i]);
        }

        /* ReLU: max(weighted_sum + bias, 0). */
        vOutput = vmaxq_f32(vOutput, vZero);

        vst1q_f32(output + j, vOutput);
    }

    return 0;
}

int vector_multiplication_bias(
    const float32_t *input,
    size_t in_size,
    const float32_t *weights,
    size_t w_size,
    const float32_t *bias,
    size_t bias_size,
    float32_t *output,
    size_t out_size)
{
    if (input == NULL ||
        weights == NULL ||
        bias == NULL ||
        output == NULL)
    {
        return -1;
    }

    if (in_size == 0U || out_size == 0U) {
        return -1;
    }

    if (in_size * out_size != w_size) {
        xil_printf(
            "Wrong weight dimensions: in_size * out_size = %lu, "
            "but w_size = %lu\r\n",
            (unsigned long)(in_size * out_size),
            (unsigned long)w_size
        );
        return -1;
    }

    if (bias_size != out_size) {
        xil_printf(
            "Wrong bias dimensions: bias_size = %lu, "
            "but out_size = %lu\r\n",
            (unsigned long)bias_size,
            (unsigned long)out_size
        );
        return -1;
    }

    if ((out_size % 4U) != 0U) {
        xil_printf(
            "out_size must be a multiple of 4, but out_size = %lu\r\n",
            (unsigned long)out_size
        );
        return -1;
    }

    for (size_t j = 0U; j < out_size; j += 4U) {
        /*
         * vOutput = {
         *     bias[j + 0],
         *     bias[j + 1],
         *     bias[j + 2],
         *     bias[j + 3]
         * };
         */
        float32x4_t vOutput = vld1q_f32(bias + j);

        for (size_t i = 0U; i < in_size; ++i) {
            /*
             * Expected weight layout:
             *
             * weights[i * out_size + j + 0]
             * weights[i * out_size + j + 1]
             * weights[i * out_size + j + 2]
             * weights[i * out_size + j + 3]
             */
            float32x4_t vWeight = vld1q_f32(weights + i * out_size + j);

            // vOutput += vWeight * input[i]
            vOutput = vmlaq_n_f32(vOutput, vWeight, input[i]);
        }

        /* ReLU: max(weighted_sum + bias, 0). */
        vst1q_f32(output + j, vOutput);
    }

    return 0;
}

int auto_encoder(
    all_features *input_features,
    all_features *output_features)
{
    if (input_features == NULL || output_features == NULL) {
        return -1;
    }

    float32_t out_layer0[16] __attribute__((aligned(16)));
    float32_t out_layer1[8]  __attribute__((aligned(16)));
    float32_t out_layer2[16] __attribute__((aligned(16)));
    float32_t out_layer3[24] __attribute__((aligned(16)));

    /*
     * Layer 0:
     * 24 inputs -> 16 outputs
     */
    int retval = vector_multiplication_bias_relu(
        (const float32_t *)&input_features->data_mean,
        24,
        &W1[0][0],
        24 * 16,
        b1,
        16,
        out_layer0,
        16
    );

    if (retval != 0) {
        xil_printf("Problem Layer 0\r\n");
        return -1;
    }

    /*
     * Layer 1:
     * 16 inputs -> 8 outputs
     */
    retval = vector_multiplication_bias_relu(
        out_layer0,
        16,
        &W2[0][0],
        16 * 8,
        b2,
        8,
        out_layer1,
        8
    );

    if (retval != 0) {
        xil_printf("Problem Layer 1\r\n");
        return -2;
    }

    /*
     * Layer 2:
     * 8 inputs -> 16 outputs
     */
    retval = vector_multiplication_bias_relu(
        out_layer1,
        8,
        &W3[0][0],
        8 * 16,
        b3,
        16,
        out_layer2,
        16
    );

    if (retval != 0) {
        xil_printf("Problem Layer 2\r\n");
        return -3;
    }

    /*
     * Layer 3:
     * 16 inputs -> 24 outputs
     */
    retval = vector_multiplication_bias(
        out_layer2,
        16,
        &W4[0][0],
        16 * 24,
        b4,
        24,
        out_layer3,
        24
    );

    if (retval != 0) {
        xil_printf("Problem Layer 3\r\n");
        return -4;
    }

    float32_t *out_ptr = (float32_t *)output_features;

    for (int i = 0; i < 24; i++) {
        out_ptr[i] = out_layer3[i];
    }

    output_features->reserved           =0.0f;

    return 0;
}
