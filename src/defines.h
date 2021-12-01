#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 64
#define N_INPUT_2_1 64
#define N_INPUT_3_1 1
#define OUT_HEIGHT_16 66
#define OUT_WIDTH_16 66
#define N_CHAN_16 1
#define OUT_HEIGHT_2 64
#define OUT_WIDTH_2 64
#define N_FILT_2 8
#define OUT_HEIGHT_5 16
#define OUT_WIDTH_5 16
#define N_FILT_5 8
#define OUT_HEIGHT_17 18
#define OUT_WIDTH_17 18
#define N_CHAN_17 8
#define OUT_HEIGHT_6 16
#define OUT_WIDTH_6 16
#define N_FILT_6 16
#define OUT_HEIGHT_9 4
#define OUT_WIDTH_9 4
#define N_FILT_9 16
#define N_LAYER_10 12
#define N_LAYER_13 3

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> input_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer16_t;
typedef nnet::array<ap_fixed<16,6>, 8*1> layer2_t;
typedef ap_fixed<7,1> weight2_t;
typedef nnet::array<ap_fixed<7,1,AP_RND,AP_SAT>, 8*1> layer4_t;
typedef ap_fixed<16,6> max_pooling2d_default_t;
typedef nnet::array<ap_fixed<16,6>, 8*1> layer5_t;
typedef nnet::array<ap_fixed<16,6>, 8*1> layer17_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer6_t;
typedef ap_fixed<7,1> weight6_t;
typedef nnet::array<ap_fixed<7,1,AP_RND,AP_SAT>, 16*1> layer8_t;
typedef ap_fixed<16,6> max_pooling2d_1_default_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer9_t;
typedef nnet::array<ap_fixed<16,6>, 12*1> layer10_t;
typedef ap_fixed<7,1> weight10_t;
typedef ap_fixed<7,1> bias10_t;
typedef nnet::array<ap_fixed<7,1,AP_RND,AP_SAT>, 12*1> layer12_t;
typedef nnet::array<ap_fixed<16,6>, 3*1> layer13_t;
typedef ap_fixed<7,1> weight13_t;
typedef ap_fixed<7,1> bias13_t;
typedef ap_fixed<16,6> softmax_default_t;
typedef nnet::array<ap_fixed<16,6>, 3*1> result_t;

#endif
