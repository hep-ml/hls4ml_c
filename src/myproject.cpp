//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &q_conv2d_input,
    hls::stream<result_t> &layer15_out,
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=q_conv2d_input,layer15_out 
    #pragma HLS DATAFLOW 

    const_size_in_1 = N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1;
    const_size_out_1 = N_LAYER_13;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 72>(w2, "w2.txt");
        nnet::load_weights_from_txt<model_default_t, 8>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight6_t, 1152>(w6, "w6.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(b6, "b6.txt");
        nnet::load_weights_from_txt<weight10_t, 3072>(w10, "w10.txt");
        nnet::load_weights_from_txt<bias10_t, 12>(b10, "b10.txt");
        nnet::load_weights_from_txt<weight13_t, 36>(w13, "w13.txt");
        nnet::load_weights_from_txt<bias13_t, 3>(b13, "b13.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    hls::stream<layer16_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=4356
    nnet::zeropad2d_cl<input_t, layer16_t, config16>(q_conv2d_input, layer16_out); // zp2d_q_conv2d

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=4096
    nnet::conv_2d_cl<layer16_t, layer2_t, config2>(layer16_out, layer2_out, w2, b2); // q_conv2d

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=4096
    nnet::relu<layer2_t, layer4_t, relu_config4>(layer2_out, layer4_out); // relu1

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=256
    nnet::pooling2d_cl<layer4_t, layer5_t, config5>(layer4_out, layer5_out); // max_pooling2d

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=324
    nnet::zeropad2d_cl<layer5_t, layer17_t, config17>(layer5_out, layer17_out); // zp2d_q_conv2d_1

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=256
    nnet::conv_2d_cl<layer17_t, layer6_t, config6>(layer17_out, layer6_out, w6, b6); // q_conv2d_1

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=256
    nnet::relu<layer6_t, layer8_t, relu_config8>(layer6_out, layer8_out); // relu2

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=16
    nnet::pooling2d_cl<layer8_t, layer9_t, config9>(layer8_out, layer9_out); // max_pooling2d_1

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=1
    nnet::dense<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // q_dense

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=1
    nnet::relu<layer10_t, layer12_t, relu_config12>(layer10_out, layer12_out); // relu5

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=1
    nnet::dense<layer12_t, layer13_t, config13>(layer12_out, layer13_out, w13, b13); // q_dense_1

    nnet::softmax<layer13_t, result_t, softmax_config15>(layer13_out, layer15_out); // softmax

}
