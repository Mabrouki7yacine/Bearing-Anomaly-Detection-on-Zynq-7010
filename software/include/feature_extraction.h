#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <arm_neon.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

#define PI_F      3.14159265358979323846f
#define EPSILON_F 0.0000000000001f
#define ALL_FEATURES_COUNT 24U

typedef struct {
    float32_t mean_value;
    float32_t standard_deviation_value;
    float32_t variance_value;
    float32_t peak_value;
    float32_t peak_to_peak_value;
    float32_t crest_factor_value;
    float32_t skewness_value;
    float32_t kurtosis_value;
} data_features;

typedef struct {
    float32_t total_energy;
    float32_t energy_0_500;
    float32_t energy_500_1500;
    float32_t energy_1500_3000;    
    float32_t energy_3000_6000;    
} energy_features;

typedef struct {
    float32_t ratio_0_500;
    float32_t ratio_500_1500;
    float32_t ratio_1500_3000;    
    float32_t ratio_3000_6000;    
} ratio_features;

typedef struct {
    int index;
    float32_t dominant_frequency;
    float32_t dominant_magnitude;
} dominance_features;

typedef struct {
    float32_t spectral_centroid;
    float32_t spectral_bandwidth;
    float32_t spectral_flatness;
} spectral_features;

typedef struct __attribute__((packed, aligned(4))) {
    /* Data-related features */
    float32_t data_mean;
    float32_t data_std;
    float32_t data_var;
    float32_t data_rms;
    float32_t data_peak;
    float32_t data_peak2peak;
    float32_t data_crest;
    float32_t data_skewness;
    float32_t data_kurtosis;

    /* Energy-related features */
    float32_t energy_total;    
    float32_t energy_0_500;    
    float32_t energy_500_1500; 
    float32_t energy_1500_3000;
    float32_t energy_3000_6000;

    /* Energy-ratio features */
    float32_t ratio_0_500;
    float32_t ratio_500_1500; 
    float32_t ratio_1500_3000;
    float32_t ratio_3000_6000;

    /* Dominance-related features */
    float32_t dominant_frequency;

    /* Spectral-related features */
    float32_t spectral_centroid;
    float32_t spectral_bandwidth;
    float32_t spectral_flatness;

    /* Dominance-related features */
    float32_t dominant_magnitude;

    float32_t reserved;
} all_features;

void print_all_features(const all_features *features);

float mean(float *data, int length);
float standard_deviation(float *data, int length, float mean_value);
float variance(float standard_deviation_value);
float RMS(float *data, int length);
float peak(float *data, int length);
float peak_to_peak(float *data, int length);
float crest_factor(float peak_value, float RMS_value);
float skewness(float *data, int length, float mean_value, float standard_deviation_value);
float kurtosis(float *data, int length, float mean_value, float standard_deviation_value);

void fft_frequency_bins(int fft_size, float sample_rate_hz, float *bins_hz);
void preprocess_window(float *data, int length, float *preprocessed_data, float mean_value, float standard_deviation_value);

energy_features extract_energy_features(float *data);
ratio_features extract_energy_ratio_features(energy_features *energy_features_values);
dominance_features extract_dominant_frequency_features(float *data, int length, float *bins_hz);
spectral_features extract_spectral_features(float *data, int length, float *bins_hz);
all_features scale_features(all_features *features, float32_t *scalar_mean, float32_t *inv_scalar_std);

#endif // FEATURE_EXTRACTION_H