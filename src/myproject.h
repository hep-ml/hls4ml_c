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

#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"

#include "parameters.h"


// Prototype of top level function for C-synthesis
void myproject(
	       hls::stream<input_t>   em_barrel[N_INPUT_3_1],
	       //hls::stream<result_t>  layer54_out[N_FILT_18],
	       hls::stream<result_t>  layer54_out[N_LAYER_52],
	       model_weightdefault_t w27[73728],
	       model_weightdefault_t w31[147456],
	       model_weightdefault_t w36[294912],
	       model_weightdefault_t w40[589824],
	       model_weightdefault_t w44[589824],
	       model_weightdefault_t w48[65536]
	       );

#endif
