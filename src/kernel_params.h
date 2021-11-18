#include "ap_fixed.h"
#include <parameters.h>

//how many consecutive sets of inputs to run over per kernel execution
#define COMPRESSION 32
#define STREAMSIZE 1
#define BIGSTREAMSIZE_IN  77
#define BIGSTREAMSIZE_OUT 1

#define IN_STREAM_LEN  (N_INPUT_1_1*N_INPUT_2_1)
#define OUT_STREAM_LEN  1

#define DATA_SIZE_IN  (N_INPUT_3_1-1)
#define DATA_SIZE_OUT  N_LAYER_52

typedef ap_fixed<16,6> data_t;
typedef ap_uint<512>    bigdata_t;

#define NW1 73728
#define NW2 147456
#define NW3 294912
#define NW4 589824
#define NW5 589824
#define NW6 65536
