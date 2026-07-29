# Vivado block design

The repository currently stores the block-design screenshot at:

```text
docs/images/bearing_anomaly_block_design.png
```

For exact reproduction, open the completed block design in Vivado and export it from the Tcl console:

```tcl
write_bd_tcl -force hardware/vivado/block_design.tcl
```

Commit `block_design.tcl`, but do not commit generated project runs, IP caches, or `.Xil` directories.

Essential stream path:

```text
AXI DMA MM2S → Xilinx FFT → fft_energy_axis → AXI DMA S2MM
```
