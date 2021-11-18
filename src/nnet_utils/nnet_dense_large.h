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

#ifndef NNET_LARGE_LAYER_H_
#define NNET_LARGE_LAYER_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "hls_stream.h"

#include <math.h>
#include <assert.h>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void dense_large_rf_leq_nin(
    data_T data[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]) {

    const int rufactor = CONFIG_T::reuse_factor;
    const int multfactor = MIN(CONFIG_T::n_in,CONFIG_T::reuse_factor);
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, multfactor);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int multscale = multiplier_limit/CONFIG_T::n_out;
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((multiplier_limit == block_factor) && "This function is correct only for RF <= N_IN");

    //#pragma HLS function_instantiate variable=biases
    //#pragma HLS RESOURCE variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose correctly
    //if (!CONFIG_T::store_weights_in_bram){
    // #pragma HLS RESOURCE variable=weights  core=XPM_MEMORY uram
    //}
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    /*
    typename CONFIG_T::accum_t tmpacc[block_factor];
    #pragma HLS ARRAY_PARTITION variable=tmpacc complete  
    for (int iacc = 0; iacc < block_factor; iacc++) {
        #pragma HLS UNROLL
        tmpacc[iacc] = 0;
    }
    */
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 
        typename CONFIG_T::accum_t tmpmult[block_factor];
        #pragma HLS ARRAY_PARTITION variable=tmpmult complete  
	unsigned in_index=ir;
	for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            unsigned w_index  =  ir + (rufactor*im); 
	    tmpmult[im] = product<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::accum_t>(data[in_index], weights[w_index]);
	    in_index += rufactor;
	    if (in_index >= nin) {
             in_index = 0;
            }
        }
	for (int im = 0; im < block_factor; im++) {
         #pragma HLS UNROLL
	 //tmpacc[im] += tmpmult[im];
	 unsigned out_index =   im/multscale;
	 acc[out_index] += tmpmult[im];
       }
    }

    /*    
    ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 
        typename CONFIG_T::accum_t tmpmult[block_factor];
        #pragma HLS ARRAY_PARTITION variable=tmpmult complete  

        MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            unsigned w_index  =  (ir+rufactor*im); 
	    unsigned in_index =  ir+rufactor*im - (int((ir+rufactor*im)/nin)*nin); 
	    tmpmult[im] = data[in_index];//product<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::accum_t>(data[in_index], weights[w_index]);
        }
	for (int im = 0; im < block_factor; im++) {
 	    unsigned out_index =   im/multscale;
            tmpacc[im] += tmpmult[im];
        }
    }
    for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
 	    unsigned out_index =   im/multscale;
            acc[out_index] = tmpacc[im];
    }
    */
    // Cast to "res_t" type
    Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_large_rf_gt_nin_rem0(
    data_T data[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]) {

    const int rufactor = MIN(CONFIG_T::reuse_factor, CONFIG_T::n_in * CONFIG_T::n_out);
    const int multfactor = MIN(CONFIG_T::n_in,CONFIG_T::reuse_factor);
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, multfactor);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int multscale = multiplier_limit/CONFIG_T::n_out;
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((rufactor > nin && rufactor % nin == 0) && "This function is correct only for RF > N_IN && RF % N_IN == 0");

    #pragma HLS function_instantiate variable=biases
    //#pragma HLS RESOURCE variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose correctly
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    int w_index;
    int in_index = 0;
    int out_index;
    int outstep = 0;
    const int outscale = rufactor / nin;

    int outidx[rufactor];
    IndexLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        outidx[ir] = outstep;
        if ((ir + 1) % nin == 0) {
            outstep++;
        }
    }

    ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        //w_index = ir;
        //out_index = outidx[ir]/*outstep*/;
        
	unsigned in_index  = ir - int(ir/nin)*nin;
	unsigned out_idx_r = int(ir/nin);
	//if(in_index  != in_index1) std::cout << " check in_index "  << in_index  << " -- " << in_index1 << std::endl;
	//if(out_index != out_idx_r) std::cout << " check out_index " << out_index << " -- " << out_idx_r << std::endl;
	
        MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            unsigned w_index     = ir + rufactor*im;
	    unsigned out_index   = im*outscale+out_idx_r;
	    //if(w_index   != w_index1)   std::cout << "----> w_index " << w_index     << " -- " << w_index1 << std::endl;
	    //if(out_index != out_index1) std::cout << "----> out_index " << out_index << " -- " << out_index1 << std::endl;
	    acc[out_index] += product<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::accum_t>(data[in_index], weights[w_index]);

	    // w_index += rufactor;
            //if (w_index >= CONFIG_T::n_in * CONFIG_T::n_out) break; // check out of bounds
            //out_index += outscale;
        }

        //in_index++;
        //if (in_index >= nin) {
	//   in_index = 0;
            //outstep++; // This causes a huge increase in scheduling and RTL generation times, hence the above workaround.
        //}
    }

    // Cast to "res_t" type
    Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_large_rf_gt_nin(
    data_T data[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]) {

    const int rufactor = CONFIG_T::reuse_factor;
    const int multfactor = MIN(CONFIG_T::n_in,CONFIG_T::reuse_factor);
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, multfactor);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int multscale = multiplier_limit/CONFIG_T::n_out;
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((rufactor > nin) && "This function is correct only for RF > N_IN");

    #pragma HLS function_instantiate variable=biases
    //#pragma HLS RESOURCE variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose correctly
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind
        typename CONFIG_T::accum_t tmpmult[block_factor];
        #pragma HLS ARRAY_PARTITION variable=tmpmult complete

        MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            int w_index = ir + rufactor * im;
            int in_index = w_index % nin;
            if (w_index >= CONFIG_T::n_in*CONFIG_T::n_out) continue; // check out of bounds
            tmpmult[im] = product<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::accum_t>(data[in_index], weights[w_index]);
        }

        typename CONFIG_T::accum_t mult[multiplier_limit];
        #pragma HLS ARRAY_PARTITION variable=mult complete

        ResetMult:
        for (int imult = 0; imult < multiplier_limit; imult++) {
            #pragma HLS UNROLL
            mult[imult] = 0;
        }

        AccumLoop1:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            int w_index = ir + rufactor * im;
            int out_index = w_index / multfactor;
            if (out_index >= multiplier_limit) continue; // check out of bounds
            mult[out_index] += tmpmult[im];
        }

        AccumLoop2:
        for (int im = 0; im < multiplier_limit; im++) {
            #pragma HLS UNROLL
            //int out_index = im/multscale; // This is the general case
            //acc[out_index] += mult[im];
            acc[im] += mult[im]; // If RF > N_IN then multiplier_limit == n_out
        }
    }

    // Cast to "res_t" type
    Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_large_rf_leq_nin_merge(
    data_T data[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out/CONFIG_T::merge_factor],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]) {

    const int rufactor = CONFIG_T::reuse_factor;
    const int multfactor = MIN(CONFIG_T::n_in,CONFIG_T::reuse_factor);
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out/CONFIG_T::merge_factor, multfactor);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out/CONFIG_T::merge_factor, CONFIG_T::reuse_factor);
    const int multscale = CONFIG_T::merge_factor*multiplier_limit/CONFIG_T::n_out;
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % (nout/CONFIG_T::merge_factor) == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((multiplier_limit == block_factor) && "This function is correct only for RF <= N_IN");

    #pragma HLS function_instantiate variable=biases
    //#pragma HLS RESOURCE variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose correctly
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_RESHAPE variable=biases complete

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }
    ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        int w_index = ir;
        int in_index = ir;
        int out_index = 0;
        int acc_step = 0;
        MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            typename CONFIG_T::accum_t tmpmult[CONFIG_T::merge_factor];
	    product_merge_split<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::weightmult_t>(data[in_index], weights[w_index],tmpmult);
	    w_index += rufactor;
	    in_index += rufactor;
            if (in_index >= nin) {
                in_index = ir;
            }
            acc[out_index]   += tmpmult[0];
            acc[out_index+1] += tmpmult[1];
	    // Increment out_index
            if (acc_step + 1 >= multscale) {
                acc_step = 0;
		out_index+=CONFIG_T::merge_factor;
            } else {
                acc_step++;
            }
       }
	
    }
    // Cast to "res_t" type
    Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_large_rf_gt_nin_rem0_merge(
    data_T data[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out/CONFIG_T::merge_factor],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]) {

    const int rufactor = MIN(CONFIG_T::reuse_factor, CONFIG_T::n_in * CONFIG_T::n_out/CONFIG_T::merge_factor);
    const int multfactor = MIN(CONFIG_T::n_in,CONFIG_T::reuse_factor);
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out/CONFIG_T::merge_factor, multfactor);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out/CONFIG_T::merge_factor, CONFIG_T::reuse_factor);
    const int multscale = CONFIG_T::merge_factor*multiplier_limit/CONFIG_T::n_out;
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((rufactor > nin && rufactor % nin == 0) && "This function is correct only for RF > N_IN && RF % N_IN == 0");

    #pragma HLS function_instantiate variable=biases
    //#pragma HLS RESOURCE variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose correctly
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    int w_index;
    int in_index = 0;
    int out_index;
    int outstep = 0;
    const int outscale = rufactor / nin;

    int outidx[rufactor];
    IndexLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        outidx[ir] = outstep;
        if ((ir + 1) % nin == 0) {
            outstep++;
        }
    }

    ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        w_index = ir;
        out_index = outidx[ir]/*outstep*/;

        MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
   	    typename CONFIG_T::weightmult_t tmp  = product_merge<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::weightmult_t>(data[in_index], weights[w_index]);

            acc[out_index] += tmp.range(7,0);
            acc[out_index+1] += tmp.range(25,18);

            w_index += rufactor;
            if (w_index >= CONFIG_T::n_in * CONFIG_T::n_out) break; // check out of bounds
            out_index += CONFIG_T::merge_factor*outscale;
        }

        in_index++;
        if (in_index >= nin) {
            in_index = 0;
            //outstep++; // This causes a huge increase in scheduling and RTL generation times, hence the above workaround.
        }
    }

    // Cast to "res_t" type
    Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_large_rf_gt_nin_merge(
    data_T data[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out/CONFIG_T::merge_factor],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]) {

    const int rufactor = CONFIG_T::reuse_factor;
    const int multfactor = MIN(CONFIG_T::n_in,CONFIG_T::reuse_factor);
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out/CONFIG_T::merge_factor, multfactor);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out/CONFIG_T::merge_factor, CONFIG_T::reuse_factor);
    const int multscale = CONFIG_T::merge_factor*multiplier_limit/CONFIG_T::n_out;
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((rufactor > nin) && "This function is correct only for RF > N_IN");

    #pragma HLS function_instantiate variable=biases
    //#pragma HLS RESOURCEs variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose correctly
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_RESHAPE variable=biases complete

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_RESHAPE variable=acc complete

    InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind
        typename CONFIG_T::weightmult_t tmpmult[block_factor];
        #pragma HLS ARRAY_PARTITION variable=tmpmult complete

        MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            int w_index = ir + rufactor * im;
            int in_index = w_index % nin;
            if (w_index >= CONFIG_T::n_in*CONFIG_T::n_out) continue; // check out of bounds
            tmpmult[im] = product_merge<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::weightmult_t>(data[in_index], weights[w_index]);
        }

        typename CONFIG_T::accum_t mult[multiplier_limit];
        #pragma HLS ARRAY_PARTITION variable=mult complete

        ResetMult:
        for (int imult = 0; imult < multiplier_limit; imult++) {
            #pragma HLS UNROLL
            mult[imult] = 0;
        }

        AccumLoop1:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            int w_index = ir + rufactor * im;
            int out_index = CONFIG_T::merge_factor*w_index / multfactor;
            if (out_index >= multiplier_limit) continue; // check out of bounds
            mult[out_index]   += tmpmult[im].range(7,0);
            mult[out_index+1] += tmpmult[im].range(25,18);
        }

        AccumLoop2:
        for (int im = 0; im < multiplier_limit; im++) {
            #pragma HLS UNROLL
            //int out_index = im/multscale; // This is the general case
            //acc[out_index] += mult[im];
            acc[im] += mult[im]; // If RF > N_IN then multiplier_limit == n_out
        }
    }

    // Cast to "res_t" type
    Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_large(
    data_T data[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out/CONFIG_T::merge_factor],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]) {

  #pragma HLS INLINE region
  //  if(CONFIG_T::merge_factor == 1) { 
    if (CONFIG_T::reuse_factor <= CONFIG_T::n_in) {
       dense_large_rf_leq_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else if (CONFIG_T::reuse_factor % CONFIG_T::n_in == 0) {
        dense_large_rf_gt_nin_rem0<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_large_rf_gt_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
    /* 
 } else if (CONFIG_T::merge_factor == 2) {
    if (CONFIG_T::reuse_factor <= CONFIG_T::n_in) {
       dense_large_rf_leq_nin_merge<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else if (CONFIG_T::reuse_factor % CONFIG_T::n_in == 0) {
        dense_large_rf_gt_nin_rem0_merge<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_large_rf_gt_nin_merge<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } 
  } // can't merge more weights to use DSPs just yet
    */
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_large_stream(
      hls::stream<data_T> data[CONFIG_T::n_input],
      hls::stream<res_T>  res[CONFIG_T::n_output],
      typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
      typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]) {

      static unsigned pX = 0; 
      data_T pStatus = data[0].read();
      if(pStatus == 0) { 
       pX = 0;
      }
      static data_T tmpdata[CONFIG_T::n_in];
      #pragma HLS ARRAY_PARTITION variable=tmpdata complete
      for(int i0 = 0; i0 < CONFIG_T::n_input-1; i0++) { 
       #pragma HLS UNROLL
       data_T pTmp = data[i0+1].read();
       //if(pX == 0) tmpdata[i0] = pTmp;
       unsigned index = i0+pX*(CONFIG_T::n_input-1);
       tmpdata[index] = pTmp;
      }
      pX = pX+1;
      if(pX == CONFIG_T::block_factor) {
	data_T tmpres[CONFIG_T::n_out];
        #pragma HLS ARRAY_PARTITION variable=tmpdres complete
	dense_large<data_T, res_T, CONFIG_T>(tmpdata,tmpres,weights,biases);
	res_T pOStatus = 0;
	res[0].write(pOStatus);
	for(int i0 = 0; i0 < CONFIG_T::n_out; i0++) { 
         #pragma HLS UNROLL
	 res_T pTmp = tmpres[i0];
         //res_T pTmp = tmpdata[CONFIG_T::n_in+i0-CONFIG_T::n_out];
         res[i0+1].write(pTmp);
        }
	pX = 0;
     }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_large_stream_ss(
      hls::stream<data_T> &data,
      hls::stream<res_T>  &res,
      typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
      typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]) {

      static unsigned pX = 0; 
      data_T pStatus = data.read();
      if(pStatus == 0) { 
       pX = 0;
      }
      static data_T tmpdata[CONFIG_T::n_in];
      #pragma HLS ARRAY_PARTITION variable=tmpdata complete
      for(int i0 = 0; i0 < CONFIG_T::n_input-1; i0++) { 
       #pragma HLS UNROLL
       data_T pTmp = data.read();
       //if(pX == 0) tmpdata[i0] = pTmp;
       unsigned index = i0+pX*(CONFIG_T::n_input-1);
       tmpdata[index] = pTmp;
      }
      pX = pX+1;
      if(pX == CONFIG_T::block_factor) {
	data_T tmpres[CONFIG_T::n_out];
        #pragma HLS ARRAY_PARTITION variable=tmpdres complete
	dense_large<data_T, res_T, CONFIG_T>(tmpdata,tmpres,weights,biases);
	res_T pOStatus = 0;
	res.write(pOStatus);
	for(int i0 = 0; i0 < CONFIG_T::n_out; i0++) { 
         #pragma HLS UNROLL
	 res_T pTmp = tmpres[i0];
         //res_T pTmp = tmpdata[CONFIG_T::n_in+i0-CONFIG_T::n_out];
         res.write(pTmp);
        }
	pX = 0;
     }
}


}



#endif
