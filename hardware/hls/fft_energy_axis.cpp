#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>

#define FFT_LEN       1024
#define OUTPUT_BINS   513   // bins 0..512 for real-input 1024-point FFT

typedef ap_axiu<64, 0, 0, 0> axis64_t;
typedef ap_axiu<32, 0, 0, 0> axis32_t;

static float bits_to_float(ap_uint<32> bits)
{
    union {
        unsigned int u;
        float f;
    } conv;

    conv.u = bits.to_uint();
    return conv.f;
}

static ap_uint<32> float_to_bits(float value)
{
    union {
        unsigned int u;
        float f;
    } conv;

    conv.f = value;
    return conv.u;
}

void fft_energy_axis(
    hls::stream<axis64_t> &s_axis,
    hls::stream<axis32_t> &m_axis
)
{
#pragma HLS INTERFACE axis port=s_axis
#pragma HLS INTERFACE axis port=m_axis
#pragma HLS INTERFACE ap_ctrl_none port=return

    while (1) {
        for (int i = 0; i < FFT_LEN; i++) {
#pragma HLS PIPELINE II=1

            axis64_t in_word;
            axis32_t out_word;

            in_word = s_axis.read();

            ap_uint<32> real_bits = in_word.data.range(31, 0);
            ap_uint<32> imag_bits = in_word.data.range(63, 32);

            float real_value = bits_to_float(real_bits);
            float imag_value = bits_to_float(imag_bits);

            float energy = real_value * real_value + imag_value * imag_value;

            if (i < OUTPUT_BINS) {
                out_word.data = float_to_bits(energy);
                out_word.keep = 0xF;
                out_word.strb = 0xF;

                if (i == OUTPUT_BINS - 1) {
                    out_word.last = 1;
                } else {
                    out_word.last = 0;
                }

                m_axis.write(out_word);
            }
        }
    }
}