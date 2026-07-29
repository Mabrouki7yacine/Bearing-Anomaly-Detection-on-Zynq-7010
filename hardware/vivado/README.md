# Vivado design

This folder contains the exported Vivado block design for the Zynq-7010 implementation.

![Block design](../../docs/images/bearing_anomaly_block_design.png)

## Files

- `block_design.tcl`: recreates the Vivado block design
- `../hls/fft_energy_axis.cpp`: HLS IP used after the FFT

## Recreate the design

Create a Vivado project for the same Zynq-7010 target, add the exported HLS IP to the IP repository, then run:

```tcl
source hardware/vivado/block_design.tcl
validate_bd_design
save_bd_design
```

The hardware path is:

```text
PS → AXI DMA → FFT → fft_energy_axis → AXI DMA → PS
```

The input is a 1024-sample complex stream. The output contains 513 floating-point energy bins.
