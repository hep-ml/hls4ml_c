# hls4ml on Alveo U250 (HLS C/C++ Kernel)

Setup tools, licenses, check connection to FPGA card

Check out packages
```bash
# check out hls4ml_c SDAccel project
git clone https://github.com/drankincms/hls4ml_c -b alveo_deepcalo
```
Compile SDAccel project
```bash
cd hls4ml_c
make check TARGET=sw DEVICE=xilinx_u250_xdma_201830_2 all # software emulation
make check TARGET=hw_emu DEVICE=xilinx_u250_xdma_201830_2 all # hardware emulation
make TARGET=hw DEVICE=xilinx_u250_xdma_201830_2 all # build
```

Run project
```bash
./host 
```
