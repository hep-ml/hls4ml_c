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

//hls-fpga-machine-learning insert weights
#include "weights/s3.h"
#include "weights/b3.h"
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w9.h"
#include "weights/b9.h"
#include "weights/w13.h"
#include "weights/b13.h"
#include "weights/w18.h"
#include "weights/b18.h"
#include "weights/w22.h"
#include "weights/b22.h"
#include "weights/b27.h"
#include "weights/b31.h"
#include "weights/b36.h"
#include "weights/b40.h"
#include "weights/b44.h"
#include "weights/s46.h"
#include "weights/b46.h"
#include "weights/b48.h"
#include "weights/s50.h"
#include "weights/b50.h"
#include "weights/w52.h"
#include "weights/b52.h"


void myproject(
	       hls::stream<input_t>   em_barrel[N_INPUT_3_1],
	       hls::stream<result_t>  layer54_out[N_LAYER_52],
	       //hls::stream<result_t>  layer54_out[N_LAYER_52],
	       model_weightdefault_t w27[73728],
	       model_weightdefault_t w31[147456],
	       model_weightdefault_t w36[294912],
	       model_weightdefault_t w40[589824],
	       model_weightdefault_t w44[589824],
	       model_weightdefault_t w48[65536]
	       ) { 

#ifndef __SYNTHESIS__
  static bool loaded_weights = false;
  if (!loaded_weights) {

    nnet::load_weights_from_txt<model_weightdefault_t, 4>(s3, "s3.txt");
    nnet::load_weights_from_txt<model_default_t, 4>(b3, "b3.txt");
    nnet::load_weights_from_txt<model_weightdefault_t, 1600>(w4, "w4.txt");
    nnet::load_weights_from_txt<model_default_t, 16>(b4, "b4.txt");
    nnet::load_weights_from_txt<model_weightdefault_t, 4608>(w9, "w9.txt");
    nnet::load_weights_from_txt<model_default_t, 32>(b9, "b9.txt");
    nnet::load_weights_from_txt<model_weightdefault_t, 9216>(w13, "w13.txt");
    nnet::load_weights_from_txt<model_default_t, 32>(b13, "b13.txt");
    nnet::load_weights_from_txt<model_weightdefault_t, 18432>(w18, "w18.txt");
    nnet::load_weights_from_txt<model_default_t, 64>(b18, "b18.txt");
    nnet::load_weights_from_txt<model_weightdefault_t, 36864>(w22, "w22.txt");
    nnet::load_weights_from_txt<model_default_t, 64>(b22, "b22.txt");  
    nnet::load_weights_from_txt<model_weightdefault_t, 9216>(w13, "w13.txt");
    nnet::load_weights_from_txt<model_default_t, 32>(b13, "b13.txt");
    nnet::load_weights_from_txt<model_weightdefault_t, 18432>(w18, "w18.txt");
    nnet::load_weights_from_txt<model_default_t, 64>(b18, "b18.txt");
    nnet::load_weights_from_txt<model_weightdefault_t, 36864>(w22, "w22.txt");
    nnet::load_weights_from_txt<model_default_t, 64>(b22, "b22.txt");
    nnet::load_weights_from_txt<model_weightdefault_t, 73728>(w27, "w27.txt");
    nnet::load_weights_from_txt<model_default_t, 128>(b27, "b27.txt");
    nnet::load_weights_from_txt<model_weightdefault_t, 147456>(w31, "w31.txt");
    nnet::load_weights_from_txt<model_default_t, 128>(b31, "b31.txt");
    nnet::load_weights_from_txt<model_weightdefault_t, 294912>(w36, "w36.txt");
    nnet::load_weights_from_txt<model_default_t, 256>(b36, "b36.txt");
    nnet::load_weights_from_txt<model_weightdefault_t, 589824>(w40, "w40.txt");
    nnet::load_weights_from_txt<model_default_t, 256>(b40, "b40.txt");
    nnet::load_weights_from_txt<model_weightdefault_t, 589824>(w44, "w44.txt");
    nnet::load_weights_from_txt<model_default_t, 257>(b44, "b44.txt");
    nnet::load_weights_from_txt<model_weightdefault_t, 256>(s46, "s46.txt");
    nnet::load_weights_from_txt<model_default_t, 256>(b46, "b46.txt");
    nnet::load_weights_from_txt<model_weightdefault_t, 65536>(w48, "w48.txt");
    nnet::load_weights_from_txt<model_default_t, 257>(b48, "b48.txt");
    nnet::load_weights_from_txt<model_weightdefault_t, 256>(s50, "s50.txt");
    nnet::load_weights_from_txt<model_default_t, 256>(b50, "b50.txt");
    nnet::load_weights_from_txt<model_weightdefault_t, 256>(w52, "w52.txt");
    nnet::load_weights_from_txt<model_default_t, 1>(b52, "b52.txt");
    loaded_weights = true;
  }
#endif

  #pragma HLS DATAFLOW

  //  for(int iEvent = 0; iEvent < 1; iEvent++) { 
  unsigned nevent=5;

  hls::stream<layer2_t> layer2_out[N_CHANNEL_2];
  #pragma HLS STREAM variable=layer2_out depth=3080 dim=1
  for(int i0 = 0; i0 < 56*11*nevent; i0++) {    
    #pragma HLS PIPELINE 
    nnet::upsampling2d_stream<input_t, layer2_t, config2>(em_barrel, layer2_out);
  }
  hls::stream<layer3_t> layer3_out[N_CHANNEL_2];
  #pragma HLS STREAM variable=layer3_out depth=3080 dim=1
  for(int i0 = 0; i0 < 56*55*nevent; i0++) {    
    #pragma HLS PIPELINE 
    nnet::normalize_stream<layer2_t, layer3_t, config3>(layer2_out, layer3_out, s3, b3);
  }  
  hls::stream<layer7_t> layer6_out[N_CHANNEL_2];
  #pragma HLS STREAM variable=layer6_out depth=3080 dim=1
  unsigned iN=0;
  for(int i0 = 0; i0 < (56)*(55)*nevent; i0++) {    
    #pragma HLS PIPELINE 
    //#pragma HLS PIPELINE II=8
    nnet::zeropad<layer3_t, layer7_t, config4>(iN,layer3_out, layer6_out);
  }
  hls::stream<layer7_t> layer7_out[N_FILT_4];
  #pragma HLS STREAM variable=layer7_out depth=3080 dim=1
  for(int i0 = 0; i0 < (56+4)*(55+4)*nevent; i0++) {    
    //#pragma HLS PIPELINE 
    #pragma HLS PIPELINE II=5
    nnet::conv_2d_large_cl_nopad_pad<layer3_t, layer7_t, config4>(layer6_out, layer7_out, w4, b4);
  }

  hls::stream<layer8_t> layer8_out[N_FILT_8];
  #pragma HLS STREAM variable=layer8_out depth=3080 dim=1
  for(int i0 = 0; i0 < 56*55*nevent; i0++) {
    //#pragma HLS PIPELINE
    #pragma HLS PIPELINE II=2
    nnet::pooling2d_cl<layer7_t, layer8_t, config8>(layer7_out, layer8_out);
  }
  hls::stream<layer8_t> layer9_out[N_FILT_8];
  #pragma HLS STREAM variable=layer9_out depth=3080 dim=1
  for(int i0 = 0; i0 < (28)*(27)*nevent; i0++) {    
    #pragma HLS PIPELINE 
    //#pragma HLS PIPELINE II=8
    nnet::zeropad<layer8_t, layer8_t, config9>(iN,layer8_out, layer9_out);
  }
  hls::stream<layer12_t> layer12_out[N_FILT_9];
  #pragma HLS STREAM variable=layer12_out depth=378 dim=1
  for(int i0 = 0; i0 < (28+2)*(27+2)*nevent; i0++) {
    //#pragma HLS PIPELINE II=30
    //nnet::conv_2d_large_cl<layer8_t, layer12_t, config9>(layer8_out, layer12_out, w9, b9);        
    //#pragma HLS PIPELINE II=24
    nnet::conv_2d_large_cl_nopad_pad<layer8_t, layer12_t, config9>(layer9_out, layer12_out, w9, b9);        
  }

  hls::stream<layer12_t> layer13_out[N_FILT_9];
  #pragma HLS STREAM variable=layer13_out depth=378 dim=1
  for(int i0 = 0; i0 < (28)*(27)*nevent; i0++) {    
    #pragma HLS PIPELINE 
    //#pragma HLS PIPELINE II=8
    nnet::zeropad<layer12_t, layer12_t, config13>(iN,layer12_out, layer13_out);
  }
  hls::stream<layer16_t> layer16_out[N_FILT_13];
  #pragma HLS STREAM variable=layer16_out depth=378 dim=1
  for(int i0 = 0; i0 < (28+2)*(27+2)*nevent; i0++) {
    //#pragma HLS PIPELINE //II=30
    //nnet::conv_2d_large_cl<layer12_t, layer16_t, config13>(layer12_out, layer16_out, w13, b13);
    //#pragma HLS PIPELINE II=24
   nnet::conv_2d_large_cl_nopad_pad<layer12_t, layer16_t, config13>(layer13_out, layer16_out, w13, b13);
  }
  hls::stream<layer17_t> layer17_out[N_FILT_17];
  #pragma HLS STREAM variable=layer17_out depth=378 dim=1
  for(int i0 = 0; i0 < 28*27*nevent; i0++) {
    //#pragma HLS PIPELINE II=30
    #pragma HLS PIPELINE
    nnet::pooling2d_cl<layer16_t, layer17_t, config17>(layer16_out, layer17_out);
  }

  hls::stream<layer17_t> layer18_out[N_FILT_18];
  #pragma HLS STREAM variable=layer13_out depth=378 dim=1
  for(int i0 = 0; i0 < 14*13*nevent; i0++) {    
    #pragma HLS PIPELINE 
    //#pragma HLS PIPELINE II=8
    nnet::zeropad<layer17_t, layer21_t, config18>(iN,layer17_out, layer18_out);
  }
  hls::stream<layer21_t> layer21_out[N_FILT_18];
  #pragma HLS STREAM variable=layer21_out depth=91 dim=1
  for(int i0 = 0; i0 < (14+2)*(13+2)*nevent; i0++) {
    //#pragma HLS PIPELINE //II=120
    //nnet::conv_2d_large_cl<layer17_t, layer21_t, config18>(layer17_out, layer21_out, w18, b18);
    // #pragma HLS PIPELINE II=96
    nnet::conv_2d_large_cl_nopad_pad<layer17_t, layer21_t, config18>(layer18_out, layer21_out, w18, b18);
  }
 
  hls::stream<layer25_t> layer25_out;//[N_FILT_22];
  #pragma HLS STREAM variable=layer25_out depth=91 dim=1
  for(int i0 = 0; i0 < 14*13*nevent; i0++) {
    nnet::conv_2d_large_cl_ss1<layer21_t, layer25_t, config22>(layer21_out, layer25_out, w22, b22);
  }
  hls::stream<layer26_t> layer26_out;//[N_FILT_26];
  #pragma HLS STREAM variable=layer26_out depth=91 dim=1
  for(int i0 = 0; i0 < 14*13*nevent; i0++) {  
   #pragma HLS PIPELINE II=64
   nnet::pooling2d_cl_ss<layer25_t, layer26_t, config26>(layer25_out, layer26_out);
  }

  hls::stream<layer30_t> layer30_out;
  #pragma HLS STREAM variable=layer30_out depth=42 dim=1
  for(int i0 = 0; i0 < 7*6*nevent; i0++) { 
    nnet::conv_2d_large_cl_ss<layer26_t, layer30_t, config27>(layer26_out, layer30_out, w27, b27);
  }

  hls::stream<layer34_t> layer34_out;
  #pragma HLS STREAM variable=layer34_out depth=21 dim=1
  for(int i0 = 0; i0 < 7*6*nevent; i0++) { 
    nnet::conv_2d_large_cl_ss<layer30_t, layer34_t, config31>(layer30_out, layer34_out, w31, b31);
  }
  
  hls::stream<layer35_t> layer35_out;
  #pragma HLS STREAM variable=layer35_out depth=21 dim=1
  for(int i0 = 0; i0 < 7*6*nevent; i0++) { 
    #pragma HLS PIPELINE 
    nnet::pooling2d_cl_ss<layer34_t, layer35_t, config35>(layer34_out, layer35_out);
  }

  hls::stream<layer39_t> layer39_out;
  #pragma HLS STREAM variable=layer39_out depth=9 dim=1
  for(int i0 = 0; i0 < 3*3*nevent; i0++) { 
    nnet::conv_2d_large_cl_ss<layer35_t, layer39_t, config36>(layer35_out, layer39_out, w36, b36);
  }

  hls::stream<layer43_t> layer43_out;
  #pragma HLS STREAM variable=layer43_out depth=9 dim=1
  for(int i0 = 0; i0 < 3*3*nevent; i0++) { 
    nnet::conv_2d_large_cl_ss<layer39_t, layer43_t, config40>(layer39_out, layer43_out, w40, b40);
  }

  hls::stream<layer44_t> layer44_out;
  #pragma HLS STREAM variable=layer44_out depth=45 dim=1
  for(int i0 = 0; i0 < 3*3*nevent; i0++) { 
    nnet::dense_large_stream_ss<layer43_t, layer44_t, config44>(layer43_out, layer44_out, w44, b44);
  }

  hls::stream<layer48_t> layer48_out;
  #pragma HLS STREAM variable=layer48_out depth=45 dim=1
  for(int i0 = 0; i0 < nevent; i0++) {
    nnet::dense_large_stream_ss<layer47_t, layer48_t, config48>(layer44_out, layer48_out, w48, b48);
  }
  hls::stream<layer52_t> layer52_out;
  #pragma HLS STREAM variable=layer52_out depth=45 dim=1
  for(int i0 = 0; i0 < nevent; i0++) {
    nnet::dense_large_stream_ss<layer51_t, layer52_t, config52>(layer48_out, layer52_out, w52, b52);
  }

  hls::stream<layer52_t> layer54b_out[N_LAYER_52];
  #pragma HLS STREAM variable=layer54b_out depth=45 dim=1
  for(int i0 = 0; i0 < nevent; i0++) {
    nnet::relu_stream_ss<layer52_t, result_t, relu_config54>(layer52_out, layer54b_out);
  }
  for(int i0 = 0; i0 < nevent; i0++) {
    for(int iX = 0; iX < 1; iX++) { 
      for(int i1 = 0; i1 < N_LAYER_52; i1++) {//layer12_out[N_FILT_9]
       #pragma HLS UNROLL
       result_t pTmp = (result_t) layer54b_out[i1].read();
       layer54_out[i1].write(pTmp);
      }
    }
  }
}

/*
void myproject(
	       hls::stream<input_t>   em_barrel[N_INPUT_3_1],
	       hls::stream<result_t>  layer54_out[N_LAYER_52],
	       model_weightdefault_t w27[73728],
	       model_weightdefault_t w31[147456],
	       model_weightdefault_t w36[294912],
	       model_weightdefault_t w40[589824],
	       model_weightdefault_t w44[589824],
	       model_weightdefault_t w48[65536]) { 
  
  #pragma HLS PIPELINE
  for(int i0 = 0; i0 < 5; i0++) { 
    myproject_in(em_barrel,layer54_out,w27,w31,w36,w40,w44,w48);
  }
}


*/
