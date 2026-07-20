#ifndef PL_HELPERS_H
#define PL_HELPERS_H

#include "xaxidma.h"
#include "xil_types.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xparameters.h"

#define TIMEOUT_LIMIT   10000000

extern XAxiDma AxiDma;

static u32 float_to_u32(float value)
{
    union {
        float f;
        u32 u;
    } conv;

    conv.f = value;
    return conv.u;
}

static float u32_to_float(u32 value)
{
    union {
        float f;
        u32 u;
    } conv;

    conv.u = value;
    return conv.f;
}

static u64 pack_complex_float(float real, float imag)
{
    u32 real_bits = float_to_u32(real);
    u32 imag_bits = float_to_u32(imag);

    return ((u64)imag_bits << 32) | real_bits;
}

static int wait_dma_done(int direction)
{
    int timeout = TIMEOUT_LIMIT;

    while (XAxiDma_Busy(&AxiDma, direction)) {
        timeout--;

        if (timeout == 0) {
            xil_printf("ERROR: DMA timeout\r\n");
            return XST_FAILURE;
        }
    }

    return XST_SUCCESS;
}


#endif