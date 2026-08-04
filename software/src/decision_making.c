#include "decision_making.h"

float32_t reconstruct_error(all_features* features, all_features* x_hat)
{
    const float32_t *features_ptr = (const float32_t *)features;
    const float32_t *xhat_ptr    = (const float32_t *)x_hat;
    float32x4_t vAcc  = vdupq_n_f32(0.0f);

    for (uint32_t i = 0U; i < ALL_FEATURES_COUNT; i += 4U) {
        float32x4_t v_features = vld1q_f32(features_ptr + i);
        float32x4_t v_xhat     = vld1q_f32(xhat_ptr + i);

        v_features = vsubq_f32(v_features, v_xhat);
        v_features = vmulq_f32(v_features, v_features);
        vAcc       = vaddq_f32(vAcc , v_features);
    }
    float32_t result[4] = {};
    vst1q_f32(result, vAcc);
    result[0] = result[0] + result[1] + result[2] + result[3];
    return result[0] / 23;
}

Decision  threshold_comparison(float32_t error_r, float32_t threshold)
{
    if (error_r >= threshold) return ABNORMAL;
    return NORMAL;
}
