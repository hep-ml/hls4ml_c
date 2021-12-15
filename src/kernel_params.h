#include "ap_fixed.h"
#include <parameters.h>

//how many consecutive sets of inputs to run over per kernel execution
#define COMPRESSION 32
#define STREAMSIZE 5
#define BIGSTREAMSIZE_IN  128
#define BIGSTREAMSIZE_OUT 1

#define IN_STREAM_LEN  (N_INPUT_1_1*N_INPUT_2_1)
#define OUT_STREAM_LEN  1

#define DATA_SIZE_IN  N_INPUT_3_1
#define DATA_SIZE_OUT  N_LAYER_13

typedef ap_fixed<16,6> data_t;
typedef ap_uint<512>    bigdata_t;
