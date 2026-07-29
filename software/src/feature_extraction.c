#include "feature_extraction.h"

float mean(float *data, int length) {
    assert((length % 4) == 0);
    float sum = 0.0f;
	float32x4_t vMean = vld1q_f32(data);
    for (int i = 4; i < length; i += 4) {
        float32x4_t vData = vld1q_f32(&data[i]);
        vMean = vaddq_f32(vMean, vData);
    }
    float32_t result[4] = {};
    vst1q_f32(result, vMean);
    result[0] = result[0] + result[1] + result[2] + result[3];
    return result[0] / (float) length;
}

float standard_deviation(float *data, int length, float mean_value) {
    assert((length % 4) == 0);
    float32x4_t vMean = vdupq_n_f32(mean_value);
    float32x4_t vAcc  = vdupq_n_f32(0.0f);
    for (int i = 0; i < length; i+=4) {
        float32x4_t vData = vld1q_f32(&data[i]); 
        vData = vsubq_f32(vData, vMean);
        vData = vmulq_f32(vData, vData);
        vAcc  = vaddq_f32(vAcc , vData);
    }
    float32_t result[4] = {};
    vst1q_f32(result, vAcc);
    result[0] = result[0] + result[1] + result[2] + result[3];
    result[0] = result[0] / length;
    return sqrtf(result[0]);
}

float variance(float standard_deviation_value)
{
    return standard_deviation_value * standard_deviation_value;
}

float RMS(float *data, int length)
{
    assert((length % 4) == 0);
    float32x4_t vAcc  = vdupq_n_f32(0.0f);
    for (int i = 0; i < length; i+=4) {
        float32x4_t vData = vld1q_f32(&data[i]); 
        vData = vmulq_f32(vData, vData);
        vAcc  = vaddq_f32(vAcc , vData);
    }
    float32_t result[4] = {};
    vst1q_f32(result, vAcc);
    result[0] = result[0] + result[1] + result[2] + result[3];
    result[0] = result[0] / length;
    return sqrtf(result[0]);
}

float peak(float *data, int length) {
    assert((length % 4) == 0);
    float32x4_t vMax = vld1q_f32(data);
    for (int i = 4; i < length; i += 4) {
        float32x4_t vData = vld1q_f32(&data[i]);
        float32x4_t vAbs  = vabsq_f32(vData);
        vMax = vmaxq_f32(vMax, vAbs);
    }
    float32_t result[4] = {};
    vst1q_f32(result, vMax);
    result[0] = fmaxf(fmaxf(result[0], result[1]), fmaxf(result[2], result[3]));
    return result[0];
}

float peak_to_peak(float *data, int length) {
    assert((length % 4) == 0);
    float32x4_t vMax = vld1q_f32(data);
    float32x4_t vMin = vld1q_f32(data);
    for (int i = 4; i < length; i += 4) {
        float32x4_t vData = vld1q_f32(&data[i]);
        vMax = vmaxq_f32(vMax, vData);
        vMin = vminq_f32(vMin, vData);
    }
    float32_t max_result[4] = {};
    float32_t min_result[4] = {};
    vst1q_f32(max_result, vMax);
    vst1q_f32(min_result, vMin);
    float max_value = fmaxf(fmaxf(max_result[0], max_result[1]), fmaxf(max_result[2], max_result[3]));
    float min_value = fminf(fminf(min_result[0], min_result[1]), fminf(min_result[2], min_result[3]));
    return max_value - min_value;
}

float crest_factor(float peak_value, float RMS_value) {
    return peak_value / RMS_value;
}

float skewness(float *data, int length, float mean_value, float standard_deviation_value)
{
    assert((length % 4) == 0);
    float32x4_t vMean = vdupq_n_f32(mean_value);
    float32x4_t vAcc  = vdupq_n_f32(0.0f);
    for (int i = 0; i < length; i+=4) {
        float32x4_t vData = vld1q_f32(&data[i]); 
        vData = vsubq_f32(vData, vMean);
        float32x4_t vData2 = vmulq_f32(vData, vData);
        vData = vmulq_f32(vData2, vData);
        vAcc  = vaddq_f32(vAcc , vData);
    }
    float32_t result[4] = {};
    vst1q_f32(result, vAcc);
    result[0] = result[0] + result[1] + result[2] + result[3];
    result[0] = result[0] / length;
    float std3 = standard_deviation_value * standard_deviation_value * standard_deviation_value;
    result[0] = result[0] / std3;
    return result[0];
}

float kurtosis(float *data, int length, float mean_value, float standard_deviation_value)
{
    assert((length % 4) == 0);
    float32x4_t vMean = vdupq_n_f32(mean_value);
    float32x4_t vAcc  = vdupq_n_f32(0.0f);
    for (int i = 0; i < length; i+=4) {
        float32x4_t vData = vld1q_f32(&data[i]); 
        vData = vsubq_f32(vData, vMean);
        vData = vmulq_f32(vData, vData);
        vData = vmulq_f32(vData, vData);
        vAcc  = vaddq_f32(vAcc , vData);
    }
    
    float32_t result[4] = {};
    vst1q_f32(result, vAcc);
    result[0] = result[0] + result[1] + result[2] + result[3];
    result[0] = result[0] / length;
    float std4 = standard_deviation_value * standard_deviation_value * standard_deviation_value * standard_deviation_value;
    result[0] = result[0] / std4;
    return result[0];
}

static inline float32x4_t divq_f32_approx(float32x4_t a, float32x4_t b)
{
    // initial approximation of 1 / b
    float32x4_t recip = vrecpeq_f32(b);
    // improve approximation
    recip = vmulq_f32(vrecpsq_f32(b, recip), recip);
    // improve approximation again
    recip = vmulq_f32(vrecpsq_f32(b, recip), recip);
    // a / b = a * (1 / b)
    return vmulq_f32(a, recip);
}

float hann(int n, int NumSamples)
{
    float cos_in = 2.0f * PI_F * (float)n / (float)(NumSamples - 1);
    float hann_value = 1.0f - cosf(cos_in);
    return hann_value * 0.5f;
}

void preprocess_window(
    float *data,
    int length,
    float *preprocessed_data,
    float mean_value,
    float standard_deviation_value)
{
    if (data == NULL || preprocessed_data == NULL) {
        xil_printf("ERROR: NULL pointer in preprocess_window\r\n");
        return;
    }

    if (length <= 1 || (length % 4) != 0) {
        xil_printf("ERROR: invalid preprocessing length\r\n");
        return;
    }

    if (fabsf(standard_deviation_value) <= 1.0e-8f) {
        xil_printf("ERROR: standard deviation is zero\r\n");

        for (int i = 0; i < length; ++i) {
            preprocessed_data[i] = 0.0f;
        }

        return;
    }

    float32x4_t vMean   = vdupq_n_f32(mean_value);
    float32x4_t vInvStd = vdupq_n_f32(1.0f / standard_deviation_value);

    for (int i = 0; i < length; i += 4) {
        float32x4_t vData = vld1q_f32(&data[i]);

        vData = vsubq_f32(vData, vMean);
        vData = vmulq_f32(vData, vInvStd);

        float32_t hann_values[4] = {
            hann(i + 0, length),
            hann(i + 1, length),
            hann(i + 2, length),
            hann(i + 3, length)
        };

        float32x4_t vHann = vld1q_f32(hann_values);

        vData = vmulq_f32(vData, vHann);

        vst1q_f32(&preprocessed_data[i], vData);
    }
}

void fft_frequency_bins(int fft_size, float sample_rate_hz, float *bins_hz)
{
    if (bins_hz == NULL || fft_size <= 0 || sample_rate_hz <= 0.0f) {
        return;
    }

    int num_bins = (fft_size / 2) + 1;

    for (int k = 0; k < num_bins; k++) {
        bins_hz[k] = ((float)k * sample_rate_hz) / (float)fft_size;
    }
}

energy_features extract_energy_features(float *data)
{
    energy_features energy_features_values = {
        .energy_0_500     = 0.0f,
        .energy_500_1500  = 0.0f,
        .energy_1500_3000 = 0.0f,
        .energy_3000_6000 = 0.0f,
        .total_energy     = 0.0f
    };

    if (data == NULL) {
        return energy_features_values;
    }

    float32_t result[4] = {0.0f};

    float32x4_t v0_500 = vdupq_n_f32(0.0f);
    for (int i = 0; i < 44; i += 4) {
        float32x4_t vData = vld1q_f32(&data[i]);
        v0_500 = vaddq_f32(v0_500, vData);
    }

    vst1q_f32(result, v0_500);
    energy_features_values.energy_0_500 =
        result[0] + result[1] + result[2] + result[3] - data[43];


    float32x4_t v500_1500 = vdupq_n_f32(0.0f);
    for (int i = 44; i < 128; i += 4) {
        float32x4_t vData = vld1q_f32(&data[i]);
        v500_1500 = vaddq_f32(v500_1500, vData);
    }

    vst1q_f32(result, v500_1500);
    energy_features_values.energy_500_1500 =
        result[0] + result[1] + result[2] + result[3] + data[43];


    float32x4_t v1500_3000 = vdupq_n_f32(0.0f);
    for (int i = 128; i < 256; i += 4) {
        float32x4_t vData = vld1q_f32(&data[i]);
        v1500_3000 = vaddq_f32(v1500_3000, vData);
    }

    vst1q_f32(result, v1500_3000);
    energy_features_values.energy_1500_3000 =
        result[0] + result[1] + result[2] + result[3];


    float32x4_t v3000_6000 = vdupq_n_f32(0.0f);
    for (int i = 256; i < 512; i += 4) {
        float32x4_t vData = vld1q_f32(&data[i]);
        v3000_6000 = vaddq_f32(v3000_6000, vData);
    }

    vst1q_f32(result, v3000_6000);
    energy_features_values.energy_3000_6000 =
        result[0] + result[1] + result[2] + result[3] + data[512];

    energy_features_values.total_energy =
        energy_features_values.energy_0_500 +
        energy_features_values.energy_500_1500 +
        energy_features_values.energy_1500_3000 +
        energy_features_values.energy_3000_6000;

    return energy_features_values;
}

ratio_features extract_energy_ratio_features(energy_features *energy_features_values)
{
    ratio_features ratio_values = {
        .ratio_0_500     = 0.0f,
        .ratio_500_1500  = 0.0f,
        .ratio_1500_3000 = 0.0f,
        .ratio_3000_6000 = 0.0f
    };

    if (energy_features_values == NULL) {
        return ratio_values;
    }

    if (energy_features_values->total_energy == 0.0f) {
        return ratio_values;
    }

    float32x4_t vEnergy = {
        energy_features_values->energy_0_500,
        energy_features_values->energy_500_1500,
        energy_features_values->energy_1500_3000,
        energy_features_values->energy_3000_6000
    };

    float32x4_t vInvTotal =
        vdupq_n_f32(1.0f / energy_features_values->total_energy);

    float32x4_t vRatio = vmulq_f32(vEnergy, vInvTotal);

    float32_t result[4];
    vst1q_f32(result, vRatio);

    ratio_values.ratio_0_500     = result[0];
    ratio_values.ratio_500_1500  = result[1];
    ratio_values.ratio_1500_3000 = result[2];
    ratio_values.ratio_3000_6000 = result[3];

    return ratio_values;
}

dominance_features extract_dominant_frequency_features(float *data, int length, float *bins_hz)
{
    assert((length % 4) == 0);

    dominance_features dominance_features_values = {
        .dominant_frequency = 0.0f,
        .dominant_magnitude = 0.0f,
        .index = 0,
    };

    float32x4_t vMax = vld1q_f32(data);

    uint32x4_t vMaxIdx = {
        0,
        1,
        2,
        3
    };
    for (int i = 4; i < length; i += 4) {
        float32x4_t vData = vld1q_f32(&data[i]);

        uint32_t idx_array[4] __attribute__((aligned(16)))  = {
            (uint32_t) i + 0,
            (uint32_t) i + 1,
            (uint32_t) i + 2,
            (uint32_t) i + 3            
        };

        uint32x4_t vIdx = vld1q_u32(idx_array);

        uint32x4_t mask = vcgtq_f32(vData, vMax);
        vMax = vbslq_f32(mask, vData, vMax);
        vMaxIdx = vbslq_u32(mask, vIdx, vMaxIdx);
    }

    float max_values[4] __attribute__((aligned(16)));
    uint32_t max_indexes[4] __attribute__((aligned(16)));

    vst1q_f32(max_values, vMax);
    vst1q_u32(max_indexes, vMaxIdx);

    float final_max = max_values[0];
    int final_index = (int)max_indexes[0];

    if (max_values[1] > final_max) {
        final_max = max_values[1];
        final_index = (int)max_indexes[1];
    }

    if (max_values[2] > final_max) {
        final_max = max_values[2];
        final_index = (int)max_indexes[2];
    }

    if (max_values[3] > final_max) {
        final_max = max_values[3];
        final_index = (int)max_indexes[3];
    }

    dominance_features_values.index = final_index;
    dominance_features_values.dominant_frequency = bins_hz[final_index];
    dominance_features_values.dominant_magnitude = sqrtf(data[final_index]);

    // xil_printf("final index %d\r\n", final_index);

    return dominance_features_values;
}

all_features scale_features(all_features *features, float32_t *scalar_mean, float32_t *inv_scalar_std)
{
    all_features scaled_features = {0.0f};

    const float32_t *features_ptr = (const float32_t *)features;
    float32_t *scaled_ptr = (float32_t *)&scaled_features;

    for (uint32_t i = 0U; i < ALL_FEATURES_COUNT; i += 4U) {
        float32x4_t v_features       = vld1q_f32(features_ptr + i);
        float32x4_t v_scalar_mean    = vld1q_f32(scalar_mean + i);
        float32x4_t v_inv_scalar_std = vld1q_f32(inv_scalar_std + i);

        v_features = vsubq_f32(v_features, v_scalar_mean);

        v_features = vmulq_f32(v_features, v_inv_scalar_std);

        vst1q_f32(scaled_ptr + i, v_features);
    }

    return scaled_features;
}

spectral_features extract_spectral_features(float *data, int length, float *bins_hz)
{
    spectral_features result = {
        .spectral_bandwidth = 0.0f,
        .spectral_centroid  = 0.0f,
        .spectral_flatness  = 0.0f
    };

    if (data == NULL || bins_hz == NULL || length <= 0) {
        return result;
    }

    int vec_len = length & ~3;

    float32_t mag_array[513];

    float32x4_t vWeightedFreqSum = vdupq_n_f32(0.0f);
    float32x4_t vMagnitudeSum    = vdupq_n_f32(0.0f);

    float log_sum = 0.0f;

    for (int i = 0; i < vec_len; i += 4) {
        float e0 = fmaxf(data[i + 0], 0.0f);
        float e1 = fmaxf(data[i + 1], 0.0f);
        float e2 = fmaxf(data[i + 2], 0.0f);
        float e3 = fmaxf(data[i + 3], 0.0f);

        mag_array[i + 0] = sqrtf(e0);
        mag_array[i + 1] = sqrtf(e1);
        mag_array[i + 2] = sqrtf(e2);
        mag_array[i + 3] = sqrtf(e3);

        float32x4_t vFreq = vld1q_f32(&bins_hz[i]);
        float32x4_t vMag  = vld1q_f32(&mag_array[i]);

        vWeightedFreqSum = vmlaq_f32(vWeightedFreqSum, vMag, vFreq);
        vMagnitudeSum    = vaddq_f32(vMagnitudeSum, vMag);

        log_sum += logf(mag_array[i + 0] + EPSILON_F);
        log_sum += logf(mag_array[i + 1] + EPSILON_F);
        log_sum += logf(mag_array[i + 2] + EPSILON_F);
        log_sum += logf(mag_array[i + 3] + EPSILON_F);
    }

    float numerator_lanes[4] __attribute__((aligned(16)));
    float denominator_lanes[4] __attribute__((aligned(16)));

    vst1q_f32(numerator_lanes, vWeightedFreqSum);
    vst1q_f32(denominator_lanes, vMagnitudeSum);

    float centroid_numerator =
        numerator_lanes[0] +
        numerator_lanes[1] +
        numerator_lanes[2] +
        numerator_lanes[3];

    float magnitude_sum =
        denominator_lanes[0] +
        denominator_lanes[1] +
        denominator_lanes[2] +
        denominator_lanes[3];

    for (int i = vec_len; i < length; i++) {
        float e = fmaxf(data[i], 0.0f);
        float mag = sqrtf(e);

        mag_array[i] = mag;

        centroid_numerator += mag * bins_hz[i];
        magnitude_sum += mag;

        log_sum += logf(mag + EPSILON_F);
    }

    if (magnitude_sum <= 0.0f) {
        return result;
    }

    result.spectral_centroid = centroid_numerator / magnitude_sum;

    float32x4_t vBandwidthNumerator = vdupq_n_f32(0.0f);
    float32x4_t vCentroid = vdupq_n_f32(result.spectral_centroid);

    for (int i = 0; i < vec_len; i += 4) {
        float32x4_t vFreq = vld1q_f32(&bins_hz[i]);
        float32x4_t vMag  = vld1q_f32(&mag_array[i]);

        float32x4_t vDiff = vsubq_f32(vFreq, vCentroid);
        float32x4_t vDiff2 = vmulq_f32(vDiff, vDiff);

        vBandwidthNumerator =
            vmlaq_f32(vBandwidthNumerator, vMag, vDiff2);
    }

    vst1q_f32(numerator_lanes, vBandwidthNumerator);

    float bandwidth_numerator =
        numerator_lanes[0] +
        numerator_lanes[1] +
        numerator_lanes[2] +
        numerator_lanes[3];

    for (int i = vec_len; i < length; i++) {
        float diff = bins_hz[i] - result.spectral_centroid;
        bandwidth_numerator += mag_array[i] * diff * diff;
    }

    result.spectral_bandwidth =
        sqrtf(bandwidth_numerator / magnitude_sum);

    float inv_length = 1.0f / (float)length;

    float geometric_mean =
        expf(log_sum * inv_length);

    float arithmetic_mean =
        magnitude_sum * inv_length + EPSILON_F;

    result.spectral_flatness =
        geometric_mean / arithmetic_mean;

    return result;
}

void print_all_features(const all_features *features)
{
    if (features == NULL) {
        xil_printf("all_features pointer is NULL\r\n");
        return;
    }

    printf("=== Data Features ===\r\n");
    printf("data_mean           : %.6f\n", (double)features->data_mean);
    printf("data_std            : %.6f\n", (double)features->data_std);
    printf("data_var            : %.6f\n", (double)features->data_var);
    printf("data_rms            : %.6f\n", (double)features->data_rms);
    printf("data_peak           : %.6f\n", (double)features->data_peak);
    printf("data_peak2peak      : %.6f\n", (double)features->data_peak2peak);
    printf("data_crest          : %.6f\n", (double)features->data_crest);
    printf("data_skewness       : %.6f\n", (double)features->data_skewness);
    printf("data_kurtosis       : %.6f\n", (double)features->data_kurtosis);

    printf("\n=== Energy Features ===\n");
    printf("energy_total        : %.6f\n", (double)features->energy_total);
    printf("energy_0_500        : %.6f\n", (double)features->energy_0_500);
    printf("energy_500_1500     : %.6f\n", (double)features->energy_500_1500);
    printf("energy_1500_3000    : %.6f\n", (double)features->energy_1500_3000);
    printf("energy_3000_6000    : %.6f\n", (double)features->energy_3000_6000);

    printf("\n=== Energy Ratio Features ===\n");
    printf("ratio_0_500         : %.6f\n", (double)features->ratio_0_500);
    printf("ratio_500_1500      : %.6f\n", (double)features->ratio_500_1500);
    printf("ratio_1500_3000     : %.6f\n", (double)features->ratio_1500_3000);
    printf("ratio_3000_6000     : %.6f\n", (double)features->ratio_3000_6000);

    printf("\n=== Dominance Features ===\n");
    printf("dominant_frequency  : %.6f\n", (double)features->dominant_frequency);
    printf("dominant_magnitude  : %.6f\n", (double)features->dominant_magnitude);

    printf("\n=== Spectral Features ===\n");
    printf("spectral_centroid   : %.6f\n", (double)features->spectral_centroid);
    printf("spectral_bandwidth  : %.6f\n", (double)features->spectral_bandwidth);
    printf("spectral_flatness   : %.6f\n", (double)features->spectral_flatness);

    printf("\n=== Reserved ===\n");
    printf("reserved            : %.6f\n", (double)features->reserved);
}
