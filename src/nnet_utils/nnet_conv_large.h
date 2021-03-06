#ifndef NNET_CONV_LARGE_H_
#define NNET_CONV_LARGE_H_

#include "nnet_common.h"
#include "nnet_conv.h"
#include "nnet_dense_large.h"

namespace nnet {

template<class data_T, typename CONFIG_T>
void im2col_1d(data_T data[CONFIG_T::n_in * CONFIG_T::n_chan], data_T data_col[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_out]) {
    //int index = 0;
    for (int channel = CONFIG_T::n_chan; channel--; data += CONFIG_T::n_in) {
        #pragma HLS PIPELINE II=1 rewind
		for (int kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
            #pragma HLS UNROLL
            int input_col = -CONFIG_T::pad_left + kernel_col * CONFIG_T::dilation;
            for (int output_col = CONFIG_T::n_out; output_col; output_col--) {
                #pragma HLS UNROLL
                if (input_col >= 0 && input_col < CONFIG_T::n_in) {
                    *(data_col++) = data[input_col];
                    //data_col[index] = data[input_col];
                } else {
                    *(data_col++) = 0;
                    //data_col[index] = 0;
                }
                //index++;
                input_col += CONFIG_T::stride;
            }
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_1d_full(
    data_T data[CONFIG_T::n_in * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::n_out * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{
    data_T data_conv[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_out];
    data_T data_col[CONFIG_T::filt_width * CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];

    //#pragma HLS ARRAY_PARTITION variable=data_conv complete
    #pragma HLS ARRAY_PARTITION variable=data_col complete
    #pragma HLS ARRAY_PARTITION variable=res_col complete

    im2col_1d<data_T, CONFIG_T>(data, data_conv);

    for (int i = 0; i < CONFIG_T::n_out; i++) {
        #pragma HLS UNROLL
        for (int j = 0; j < CONFIG_T::filt_width * CONFIG_T::n_chan; j++) {
            data_col[j] = data_conv[j * CONFIG_T::n_out + i];
        }
        dense_large<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
        for (int j = 0; j < CONFIG_T::n_filt; j++) {
            //res[i * CONFIG_T::n_filt + j] = res_col[j];
            res[j * CONFIG_T::n_out + i] = res_col[j]; // Transposed order
        }
    }
}

template<class data_T, typename CONFIG_T>
void im2col_1d_cf(data_T data[CONFIG_T::n_in * CONFIG_T::n_chan], data_T data_col[CONFIG_T::n_chan * CONFIG_T::filt_width], const int col) {
    #pragma HLS function_instantiate variable=col
    int index = 0;
    ChannelLoop:
    for (int channel = CONFIG_T::n_chan; channel--; data += CONFIG_T::n_in) {
		#pragma HLS UNROLL
        KernelLoop:
        for (int kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
            int input_col = -CONFIG_T::pad_left + kernel_col * CONFIG_T::dilation + col * CONFIG_T::stride;
            if (input_col >= 0 && input_col < CONFIG_T::n_in) {
                //*(data_col++) = data[input_col];
                data_col[index] = data[input_col];
            } else {
                //*(data_col++) = 0;
                data_col[index] = 0;
            }
            index++;
        }
    }
}

template<class res_T, typename CONFIG_T>
void collect_res_cf(res_T res[CONFIG_T::n_out * CONFIG_T::n_filt], res_T res_col[CONFIG_T::n_filt], int col) {
    #pragma HLS function_instantiate variable=col
    for (int j = 0; j < CONFIG_T::n_filt; j++) {
        res[j * CONFIG_T::n_out + col] = res_col[j]; // Transposed order
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_1d_large_cf(
    data_T data[CONFIG_T::n_chan * CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_out * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    data_T data_col[CONFIG_T::filt_width * CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];

    #pragma HLS ARRAY_PARTITION variable=data_col complete
    #pragma HLS ARRAY_PARTITION variable=res_col complete

    ColLoop:
    for (int i = 0; i < CONFIG_T::n_out; i++) {
        #pragma HLS PIPELINE
        im2col_1d_cf<data_T, CONFIG_T>(data, data_col, i);
        dense_large<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
        collect_res_cf<res_T, CONFIG_T>(res, res_col, i);
    }
}

template<class data_T, typename CONFIG_T>
void im2col_1d_cl(data_T data[CONFIG_T::n_in * CONFIG_T::n_chan], data_T data_col[CONFIG_T::filt_width * CONFIG_T::n_chan], const int col) {
    #pragma HLS function_instantiate variable=col
    int index = 0;
    ChannelLoop:
    for (int channel = CONFIG_T::n_chan; channel--; data++) {
		#pragma HLS UNROLL
        KernelLoop:
        for (int kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
            int input_col = -CONFIG_T::pad_left + kernel_col * CONFIG_T::dilation + col * CONFIG_T::stride;
            if (input_col >= 0 && input_col < CONFIG_T::n_in) {
                //*(data_col++) = data[input_col * CONFIG_T::n_chan];
                data_col[index] = data[input_col * CONFIG_T::n_chan];
            } else {
                //*(data_col++) = 0;
                data_col[index] = 0;
            }
            index++;
        }
    }
}

template<class res_T, typename CONFIG_T>
void collect_res_cl(res_T res[CONFIG_T::n_out * CONFIG_T::n_filt], res_T res_col[CONFIG_T::n_filt], int col) {
    #pragma HLS function_instantiate variable=col
    for (int j = 0; j < CONFIG_T::n_filt; j++) {
        res[col * CONFIG_T::n_filt + j] = res_col[j];
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_1d_large_cl(
    data_T data[CONFIG_T::n_in * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::n_out * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    data_T data_col[CONFIG_T::filt_width * CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];

    #pragma HLS ARRAY_PARTITION variable=data_col complete
    #pragma HLS ARRAY_PARTITION variable=res_col complete

    ColLoop:
    for (int i = 0; i < CONFIG_T::n_out; i++) {
        //#pragma HLS PIPELINE II=1 rewind
        im2col_1d_cl<data_T, CONFIG_T>(data, data_col, i);
        dense_large<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
        collect_res_cl<res_T, CONFIG_T>(res, res_col, i);
    }
}

}
#endif
