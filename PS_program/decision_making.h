#ifndef DECISION_MAKING_H
#define DECISION_MAKING_H

#include <stdio.h>
#include <arm_neon.h>
#include "feature_extraction.h"

typedef enum {
    NORMAL,
    ABNORMAL
} Decision;

float32_t reconstruct_error(all_features* features, all_features* x_hat);
Decision  threshold_comparison(float32_t error_r, float32_t threshold);

#endif // DECISION_MAKING_H