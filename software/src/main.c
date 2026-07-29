#include "xaxidma.h"
#include "xil_types.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xtime_l.h"

#include "autoencoder.h"
#include "decision_making.h"
#include "feature_extraction.h"
#include "weights.h"
#include "freq_bins.h"
#include "pl_helpers.h"

#define DMA_DEV_ID      XPAR_AXIDMA_0_DEVICE_ID

#define FFT_LEN         1024
#define ENERGY_LEN      513

#define TX_BYTE_SIZE    (FFT_LEN * sizeof(u64))
#define RX_BYTE_SIZE    (ENERGY_LEN * sizeof(u32))

XAxiDma AxiDma;

static u64 TxBuffer[FFT_LEN]    __attribute__ ((aligned(64)));
static u32 RxBuffer[ENERGY_LEN] __attribute__ ((aligned(64)));

static double elapsed_seconds(XTime start, XTime end)
{
    return ((double)(end - start)) / ((double)COUNTS_PER_SECOND);
}

int main()
{
    int Status;
    XAxiDma_Config *CfgPtr;

    XTime feature_start, feature_end;
    XTime ae_start, ae_end;

    double feature_time_s;
    double ae_time_s;

    xil_printf("\r\nTest Started\r\n");

    CfgPtr = XAxiDma_LookupConfig(DMA_DEV_ID);
    if (!CfgPtr) {
        xil_printf("ERROR: No DMA config found\r\n");
        return XST_FAILURE;
    }

    Status = XAxiDma_CfgInitialize(&AxiDma, CfgPtr);
    if (Status != XST_SUCCESS) {
        xil_printf("ERROR: DMA initialization failed\r\n");
        return XST_FAILURE;
    }

    if (XAxiDma_HasSg(&AxiDma)) {
        xil_printf("ERROR: DMA is in Scatter-Gather mode\r\n");
        return XST_FAILURE;
    }

    float32_t signal[1024];
    for (int i = 0; i < FFT_LEN; i++) {
        signal[i] =
            sinf(2.0f * PI_F * 500.0f * i / 12000.0f);
    }

    XTime_GetTime(&feature_start);

    float32_t signal_preprocessed[1024];

    all_features input_features = {0.0f};

    input_features.data_mean      = mean(signal, 1024);
    input_features.data_std       = standard_deviation(signal, 1024, input_features.data_mean);
    input_features.data_var       = variance(input_features.data_std);
    input_features.data_rms       = RMS(signal, 1024);
    input_features.data_peak      = peak(signal, 1024);
    input_features.data_peak2peak = peak_to_peak(signal, 1024);
    input_features.data_crest     = crest_factor(input_features.data_peak, input_features.data_rms);
    input_features.data_skewness  = skewness(signal, 1024, input_features.data_mean, input_features.data_std);
    input_features.data_kurtosis  = kurtosis(signal, 1024, input_features.data_mean, input_features.data_std);

    float32_t FFT[516] = {0.0f};

    preprocess_window(
        signal,
        1024,
        signal_preprocessed,
        input_features.data_mean,
        input_features.data_std
    );

    for (int i = 0; i < FFT_LEN; i++) {
        TxBuffer[i] = pack_complex_float(signal_preprocessed[i], 0.0f);
    }

    Xil_DCacheFlushRange((UINTPTR)TxBuffer, TX_BYTE_SIZE);
    Xil_DCacheFlushRange((UINTPTR)RxBuffer, RX_BYTE_SIZE);

    Status = XAxiDma_SimpleTransfer(
        &AxiDma,
        (UINTPTR)RxBuffer,
        RX_BYTE_SIZE,
        XAXIDMA_DEVICE_TO_DMA
    );

    if (Status != XST_SUCCESS) {
        xil_printf("ERROR: S2MM transfer failed to start\r\n");
        return XST_FAILURE;
    }

    Status = XAxiDma_SimpleTransfer(
        &AxiDma,
        (UINTPTR)TxBuffer,
        TX_BYTE_SIZE,
        XAXIDMA_DMA_TO_DEVICE
    );

    if (Status != XST_SUCCESS) {
        xil_printf("ERROR: MM2S transfer failed to start\r\n");
        return XST_FAILURE;
    }

    Status = wait_dma_done(XAXIDMA_DMA_TO_DEVICE);
    if (Status != XST_SUCCESS) {
        xil_printf("ERROR: MM2S did not finish\r\n");
        return XST_FAILURE;
    }

    Status = wait_dma_done(XAXIDMA_DEVICE_TO_DMA);
    if (Status != XST_SUCCESS) {
        xil_printf("ERROR: S2MM did not finish\r\n");
        return XST_FAILURE;
    }

    Xil_DCacheInvalidateRange((UINTPTR)RxBuffer, RX_BYTE_SIZE);

    for (int i = 0; i < ENERGY_LEN; i++) {
        FFT[i] = u32_to_float(RxBuffer[i]);
    }

    energy_features    energy_values    = extract_energy_features(FFT);
    ratio_features     ratio_values     = extract_energy_ratio_features(&energy_values);
    spectral_features  spectral_values  = extract_spectral_features(FFT, 513, freq_bins);
    dominance_features dominance_values = extract_dominant_frequency_features(FFT, 516, freq_bins);

    input_features.energy_total     = energy_values.total_energy;
    input_features.energy_0_500     = energy_values.energy_0_500;
    input_features.energy_500_1500  = energy_values.energy_500_1500;
    input_features.energy_1500_3000 = energy_values.energy_1500_3000;
    input_features.energy_3000_6000 = energy_values.energy_3000_6000;

    input_features.ratio_0_500     = ratio_values.ratio_0_500;
    input_features.ratio_500_1500  = ratio_values.ratio_500_1500;
    input_features.ratio_1500_3000 = ratio_values.ratio_1500_3000;
    input_features.ratio_3000_6000 = ratio_values.ratio_3000_6000;

    input_features.dominant_magnitude = dominance_values.dominant_magnitude;
    input_features.dominant_frequency = dominance_values.dominant_frequency;

    input_features.spectral_centroid  = spectral_values.spectral_centroid;
    input_features.spectral_bandwidth = spectral_values.spectral_bandwidth;
    input_features.spectral_flatness  = spectral_values.spectral_flatness;

    input_features.reserved = 0.0f;

    all_features scaled_features = scale_features(&input_features, scaler_mean, scaler_std);

    scaled_features.reserved = 0.0f;

    XTime_GetTime(&feature_end);

    XTime_GetTime(&ae_start);

    all_features output_features = {0.0f};
    output_features.reserved = 0.0f;

    if (auto_encoder(&scaled_features, &output_features) != 0) {
        xil_printf("ERROR: Autoencoder failed\r\n");
        return -1;
    }

    output_features.reserved = 0.0f;

    float32_t error_r = reconstruct_error(&scaled_features, &output_features);

    XTime_GetTime(&ae_end);

    feature_time_s = elapsed_seconds(feature_start, feature_end);
    ae_time_s      = elapsed_seconds(ae_start, ae_end);

    xil_printf("Results\r\n");

    printf("Feature extraction time          = %.9f s\r\n", feature_time_s);
    printf("Autoencoder + reconstruct time   = %.9f s\r\n", ae_time_s);
    printf("Total measured time              = %.9f s\r\n", feature_time_s + ae_time_s);

    xil_printf("Raw Features : \r\n");
    print_all_features(&input_features);

    xil_printf("Scaled Features : \r\n");
    print_all_features(&scaled_features);

    xil_printf("Output Features : \r\n");
    print_all_features(&output_features);

    printf("reconstruct_error = %.6f\r\n", (double)error_r);

    Decision state = threshold_comparison(error_r, ANOMALY_THRESHOLD);

    if (state == NORMAL)
        xil_printf("NORMAL\r\n");
    else
        xil_printf("ABNORMAL\r\n");

    return XST_SUCCESS;
}
