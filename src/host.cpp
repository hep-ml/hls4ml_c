/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <memory>
#include <string>
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::system_clock SClock;

#include "xcl2.hpp"
#include <vector>
#include "kernel_params.h"

#include "weights/w27.h"
#include "weights/w31.h"
#include "weights/w36.h"
#include "weights/w40.h"
#include "weights/w44.h"
#include "weights/w48.h"

#include <thread>
#include <sstream>

#define NBUFFER 1
#define NUM_CU 1

#define STRINGIFY2(var) #var
#define STRINGIFY(var) STRINGIFY2(var)

void print_nanoseconds(std::string prefix, std::chrono::time_point<std::chrono::system_clock> now, int ik) {
    auto duration = now.time_since_epoch();
    
    typedef std::chrono::duration<int, std::ratio_multiply<std::chrono::hours::period, std::ratio<8>
    >::type> Days; /* UTC: +8:00 */
    
    Days days = std::chrono::duration_cast<Days>(duration);
        duration -= days;
    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
        duration -= hours;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
        duration -= minutes;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        duration -= seconds;
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        duration -= milliseconds;
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);
        duration -= microseconds;
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);

    std::cout << "KERN" << ik << ", " << prefix << hours.count() << ":"
          << minutes.count() << ":"
          << seconds.count() << ":"
          << milliseconds.count() << ":"
          << microseconds.count() << ":"
          << nanoseconds.count() << std::endl;
}

void print_nanoseconds(std::string prefix, std::chrono::time_point<std::chrono::system_clock> now, int ik, std::stringstream &ss) {
    auto duration = now.time_since_epoch();
    
    typedef std::chrono::duration<int, std::ratio_multiply<std::chrono::hours::period, std::ratio<8>
    >::type> Days; /* UTC: +8:00 */
    
    Days days = std::chrono::duration_cast<Days>(duration);
        duration -= days;
    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
        duration -= hours;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
        duration -= minutes;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        duration -= seconds;
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        duration -= milliseconds;
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);
        duration -= microseconds;
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);

    ss << "KERN" << ik << ", " << prefix << hours.count() << ":"
          << minutes.count() << ":"
          << seconds.count() << ":"
          << milliseconds.count() << ":"
          << microseconds.count() << ":"
          << nanoseconds.count() << "\n";
}

// An event callback function that prints the operations performed by the OpenCL
// runtime.
void event_cb(cl_event event1, cl_int cmd_status, void *data) {
    cl_int err;
    cl_command_type command;
    cl::Event event(event1, true);
    OCL_CHECK(err, err = event.getInfo(CL_EVENT_COMMAND_TYPE, &command));
    cl_int status;
    OCL_CHECK(err,
              err = event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status));
    const char *command_str;
    const char *status_str;
    switch (command) {
    case CL_COMMAND_READ_BUFFER:
        command_str = "buffer read";
        break;
    case CL_COMMAND_WRITE_BUFFER:
        command_str = "buffer write";
        break;
    case CL_COMMAND_NDRANGE_KERNEL:
        command_str = "kernel";
        break;
    case CL_COMMAND_MAP_BUFFER:
        command_str = "kernel";
        break;
    case CL_COMMAND_COPY_BUFFER:
        command_str = "kernel";
        break;
    case CL_COMMAND_MIGRATE_MEM_OBJECTS:
        command_str = "buffer migrate";
        break;
    default:
        command_str = "unknown";
    }
    switch (status) {
    case CL_QUEUED:
        status_str = "Queued";
        break;
    case CL_SUBMITTED:
        status_str = "Submitted";
        break;
    case CL_RUNNING:
        status_str = "Executing";
        break;
    case CL_COMPLETE:
        status_str = "Completed";
        break;
    }
    printf("[%s]: %s %s\n",
           reinterpret_cast<char *>(data),
           status_str,
           command_str);
    fflush(stdout);
}

// Sets the callback for a particular event
void set_callback(cl::Event event, const char *queue_name) {
    cl_int err;
    OCL_CHECK(err,
              err =
                  event.setCallback(CL_COMPLETE, event_cb, (void *)queue_name));
}

class fpgaObj {
  public:
    std::stringstream ss;
    int ithr;
    int nevents;
    int ikern;
    std::vector<bigdata_t,aligned_allocator<bigdata_t>> source_in;
    std::vector<bigdata_t,aligned_allocator<bigdata_t>> source_hw_results;
    std::vector<model_default_t,aligned_allocator<model_default_t>> source_w27_in;
    std::vector<model_default_t,aligned_allocator<model_default_t>> source_w31_in;
    std::vector<model_default_t,aligned_allocator<model_default_t>> source_w36_in;
    std::vector<model_default_t,aligned_allocator<model_default_t>> source_w40_in;
    std::vector<model_default_t,aligned_allocator<model_default_t>> source_w44_in;
    std::vector<model_default_t,aligned_allocator<model_default_t>> source_w48_in;
    cl::Program program;
    std::vector<cl::CommandQueue> q;
    std::vector<cl::Kernel> krnl_xil;
    std::vector<std::vector<cl::Event>>   writeList;
    std::vector<std::vector<cl::Event>>   kernList;
    std::vector<std::vector<cl::Event>>   readList;
    std::vector<cl::Buffer> buffer_in;
    std::vector<cl::Buffer> buffer_out;
    std::vector<cl::Buffer> buffer_wvec_in;
    std::vector<cl::Event>   write_event;
    std::vector<cl::Event>   kern_event;
    //std::vector<cl::Event>   read_event;
    std::vector<bool> isFirstRun;
    cl_int err;

    std::pair<int,bool> get_info_lock() {
      int i;
      bool first;
      mtx.lock();
      //i = rand() % 1;
      i = ikern++;
      if (ikern==NUM_CU*NBUFFER) ikern = 0;
      first = isFirstRun[i];
      if (first) isFirstRun[i]=false;
      mtx.unlock();
      return std::make_pair(i,first);
    }
    void get_ilock(int ik) {
      mtxi[ik].lock();
    }
    void release_ilock(int ik) {
      mtxi[ik].unlock();
    }
    void write_ss_safe(std::string newss) {
      smtx.lock();
      ss << "Thread " << ithr << "\n" << newss << "\n";
      ithr++;
      smtx.unlock();
    }

    std::stringstream runFPGA() {
        auto t0 = Clock::now();
        auto t1 = Clock::now();
        auto t1a = Clock::now();
        auto t1b = Clock::now();
        auto t2 = Clock::now();
        auto t3 = Clock::now();
        std::stringstream ss;
        for (int i = 0 ; i < nevents ; i++){
            t0 = Clock::now();
            auto ikf = get_info_lock();
            int ikb = ikf.first;
            int ik = ikb%NUM_CU;
            bool firstRun = ikf.second;

            t1 = Clock::now();
            auto ts1 = SClock::now();
            print_nanoseconds("        start:  ",ts1, ik, ss);
            std::string queuename = "ooo_queue "+std::to_string(ikb);
        
            get_ilock(ikb);
            //Copy input data to device global memory
            if (!firstRun) {
                OCL_CHECK(err, err = kern_event[ikb].wait());
            }
            OCL_CHECK(err,
                      err =
		      q[ik].enqueueMigrateMemObjects({buffer_in[ikb],buffer_wvec_in[0],buffer_wvec_in[1],buffer_wvec_in[2],buffer_wvec_in[3],buffer_wvec_in[4],buffer_wvec_in[5]},
                                                     0 /* 0 means from host*/,
                                                     NULL,
                                                     &(write_event[ikb])));
            t1a = Clock::now();
            writeList[ikb].clear();
            writeList[ikb].push_back(write_event[ikb]);
            //Launch the kernel
            OCL_CHECK(err,
                      err = q[ik].enqueueNDRangeKernel(
                          krnl_xil[ikb], 0, 1, 1, &(writeList[ikb]), &(kern_event[ikb])));
            t1b = Clock::now();
            kernList[ikb].clear();
            kernList[ikb].push_back(kern_event[ikb]);
            cl::Event read_event;
            OCL_CHECK(err,
                      err = q[ik].enqueueMigrateMemObjects({buffer_out[ikb]},
                                                       CL_MIGRATE_MEM_OBJECT_HOST,
                                                       &(kernList[ikb]),
                                                       &(read_event)));

            release_ilock(ikb);
        
            OCL_CHECK(err, err = kern_event[ikb].wait());
            OCL_CHECK(err, err = read_event.wait());
            auto ts2 = SClock::now();
            print_nanoseconds("       finish:  ",ts2, ik, ss);
            t2 = Clock::now();
	    t3 = Clock::now();
            std::cout << " Prep time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() << " ns" << std::endl;
            std::cout << " FPGA time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " ns" << std::endl;
            std::cout << "    inputs: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1a - t1).count() << " ns" << std::endl;
            std::cout << "    kernel: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1b - t1a).count() << " ns" << std::endl;
            std::cout << "   outputs: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1b).count() << " ns" << std::endl;
            ss << "KERN"<<ik<<"   Total time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t0).count() << " ns\n";
        }
        return ss;
    }

  private:
    mutable std::mutex mtx;
    mutable std::mutex mtxi[NUM_CU*NBUFFER];
    mutable std::mutex smtx;
};

void FPGA(fpgaObj& theFPGA) {
    std::stringstream ss;
    ss << (theFPGA.runFPGA()).str();
    theFPGA.write_ss_safe(ss.str());
}

int main(int argc, char** argv)
{

    int nevents = 5;
    std::string datadir = STRINGIFY(HLS4ML_DATA_DIR);
    std::string xclbinFilename = "";
    if (argc > 1) xclbinFilename = argv[1];
    if (argc > 2) nevents = atoi(argv[2]);
    if (argc > 3) datadir = argv[3];
    std::cout << "Will run " << nevents << " time(s), using " << datadir << " to get input features and output predictions (tb_input_features.dat and tb_output_predictions.dat)" << std::endl;

    size_t vector_size_in_bytes = sizeof(bigdata_t) * STREAMSIZE * BIGSTREAMSIZE_IN;
    size_t vector_size_out_bytes = sizeof(bigdata_t) * STREAMSIZE * BIGSTREAMSIZE_OUT;
    size_t vector_size_in_w27_bytes = sizeof(model_default_t) * NW1;
    size_t vector_size_in_w31_bytes = sizeof(model_default_t) * NW2;
    size_t vector_size_in_w36_bytes = sizeof(model_default_t) * NW3;
    size_t vector_size_in_w40_bytes = sizeof(model_default_t) * NW4;
    size_t vector_size_in_w44_bytes = sizeof(model_default_t) * NW5;
    size_t vector_size_in_w48_bytes = sizeof(model_default_t) * NW6;
    fpgaObj fpga;
    fpga.nevents = nevents;
    fpga.ikern = 0;
    fpga.source_in.reserve(STREAMSIZE*BIGSTREAMSIZE_IN*NUM_CU*NBUFFER);
    fpga.source_hw_results.reserve(STREAMSIZE*BIGSTREAMSIZE_OUT*NUM_CU*NBUFFER);
    fpga.source_w27_in.reserve(NW1);
    fpga.source_w31_in.reserve(NW2);
    fpga.source_w36_in.reserve(NW3);
    fpga.source_w40_in.reserve(NW4);
    fpga.source_w44_in.reserve(NW5);
    fpga.source_w48_in.reserve(NW6);

    //initialize
    for(int j = 0 ; j < STREAMSIZE*BIGSTREAMSIZE_IN*NUM_CU*NBUFFER ; j++){
      if(j != 0) fpga.source_in[j] = 1;
      if(j == 0) fpga.source_in[j] = 1;
    }
    for(int j = 0 ; j < STREAMSIZE*BIGSTREAMSIZE_OUT*NUM_CU*NBUFFER ; j++){
      data_t in=(data_t) j;
      fpga.source_hw_results[j] = in;
    }
    for(int j = 0; j < NW1; j++){
      fpga.source_w27_in[j] = w27[j];
    }
    for(int j = 0; j < NW2; j++){
      fpga.source_w31_in[j] = w31[j];
    }
    for(int j = 0; j < NW3; j++){
      fpga.source_w36_in[j] = w36[j];
    }
    for(int j = 0; j < NW4; j++){
      fpga.source_w40_in[j] = w40[j];
    }
    for(int j = 0; j < NW5; j++){
      fpga.source_w44_in[j] = w44[j];
    }
    for(int j = 0; j < NW6; j++){
      fpga.source_w48_in[j] = w48[j];
    }

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);
    for (int i = 0; i < NUM_CU; i++) {
        cl::CommandQueue q_tmp(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
        fpga.q.push_back(q_tmp);
    }
    std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    cl::Program::Binaries bins;
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);

    // Creating Program from Binary File
    bins.push_back({buf,nb});

    devices.resize(1);
    cl::Program tmp_program(context, devices, bins);
    fpga.program = tmp_program;

    for (int ib = 0; ib < NBUFFER; ib++) {
        for (int i = 0; i < NUM_CU; i++) {
            std::string cu_id = std::to_string(NUM_CU>1 ? i : 1);
            std::string krnl_name_full =
                "alveo_hls4ml:{alveo_hls4ml_" + cu_id + "}";
            printf("Creating a kernel [%s] for CU(%d)\n",
                   krnl_name_full.c_str(),
                   i);
            //Here Kernel object is created by specifying kernel name along with compute unit.
            //For such case, this kernel object can only access the specific Compute unit
            cl::Kernel krnl_tmp = cl::Kernel(
                   fpga.program, krnl_name_full.c_str(), &fpga.err);
            fpga.krnl_xil.push_back(krnl_tmp);
        }
    }
    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and 
    // Device-to-host communication

    cl::Buffer buffer_in_w27_tmp    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,   vector_size_in_w27_bytes, fpga.source_w27_in.data());
    cl::Buffer buffer_in_w31_tmp    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,   vector_size_in_w31_bytes, fpga.source_w31_in.data());
    cl::Buffer buffer_in_w36_tmp    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,   vector_size_in_w36_bytes, fpga.source_w36_in.data());
    cl::Buffer buffer_in_w40_tmp    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,   vector_size_in_w40_bytes, fpga.source_w40_in.data());
    cl::Buffer buffer_in_w44_tmp    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,   vector_size_in_w44_bytes, fpga.source_w44_in.data());
    cl::Buffer buffer_in_w48_tmp    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,   vector_size_in_w48_bytes, fpga.source_w48_in.data());
    fpga.buffer_wvec_in.push_back(buffer_in_w27_tmp);
    fpga.buffer_wvec_in.push_back(buffer_in_w31_tmp);
    fpga.buffer_wvec_in.push_back(buffer_in_w36_tmp);
    fpga.buffer_wvec_in.push_back(buffer_in_w40_tmp);
    fpga.buffer_wvec_in.push_back(buffer_in_w44_tmp);
    fpga.buffer_wvec_in.push_back(buffer_in_w48_tmp);
    
    fpga.writeList.reserve(NUM_CU*NBUFFER);
    fpga.kernList.reserve(NUM_CU*NBUFFER);
    fpga.readList.reserve(NUM_CU*NBUFFER);
    for (int ib = 0; ib < NBUFFER; ib++) {
        for (int ik = 0; ik < NUM_CU; ik++) {
	  cl::Buffer buffer_in_tmp    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,   vector_size_in_bytes, fpga.source_in.data());
	  cl::Buffer buffer_out_tmp(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, vector_size_out_bytes, fpga.source_hw_results.data());
	  fpga.buffer_in.push_back(buffer_in_tmp);
	  fpga.buffer_out.push_back(buffer_out_tmp);
	  
	  cl::Event tmp_write = cl::Event();
	  cl::Event tmp_kern = cl::Event();
	  cl::Event tmp_read = cl::Event();
	  fpga.write_event.push_back(tmp_write);
	  fpga.kern_event.push_back(tmp_kern);
	  //fpga.read_event.push_back(tmp_read);

	  int narg = 0;
	  fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_in[ib*NUM_CU+ik]);
	  fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_wvec_in[0]);
	  fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_wvec_in[1]);
	  fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_wvec_in[2]);
	  fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_wvec_in[3]);
	  fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_wvec_in[4]);
	  fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_wvec_in[5]);

	  fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_out[ib*NUM_CU+ik]);
	  fpga.isFirstRun.push_back(true);
	  std::vector<cl::Event> tmp_write_vec(1);
	  std::vector<cl::Event> tmp_kern_vec(1);
	  std::vector<cl::Event> tmp_read_vec(1);
	  fpga.writeList.push_back(tmp_write_vec);
	  fpga.kernList.push_back(tmp_kern_vec);
	  fpga.readList.push_back(tmp_read_vec);
        }
    }
    auto t0 = Clock::now();
    auto t1 = Clock::now();
    auto t1a = Clock::now();
    auto t1b = Clock::now();
    auto t2 = Clock::now();
    auto t3 = Clock::now();
    
    int index = 0;
    for (int ib = 0; ib < NBUFFER; ib++) {

        for (int i = 0 ; i < NUM_CU ; i++){

            for (int istream = 0; istream < STREAMSIZE; istream++) {
                for (int ij = 0; ij < BIGSTREAMSIZE_IN; ij++) {
		  // Create the test data if no data files found or if end of files has been reached
		  fpga.source_in[ib*NUM_CU*STREAMSIZE*BIGSTREAMSIZE_IN+i*STREAMSIZE*BIGSTREAMSIZE_IN+istream*BIGSTREAMSIZE_IN+ij] = (bigdata_t)(12354.37674*(ij+istream*BIGSTREAMSIZE_IN+STREAMSIZE*BIGSTREAMSIZE_IN*(ib+i+1)));
		  //fpga.source_in[ib*NUM_CU*STREAMSIZE*BIGSTREAMSIZE_IN+i*STREAMSIZE*BIGSTREAMSIZE_IN+istream*BIGSTREAMSIZE_IN+ij] = 12345678-12345684*istream;//(bigdata_t)(ij*32);
		  //fpga.source_in[index] = (bigdata_t) index;
		  
		  bigdata_t tmp = 0;
		  for(int i0 = 0; i0 < COMPRESSION; i0++) {
		    data_t inTmpL = (data_t) 100;//index;
		    tmp.range((i0+1)*16-1,(i0)*16) = inTmpL.range(15,0);
		  } 
		  fpga.source_in[index] = tmp;
		  index++;
                }
            }
        }
    }

    auto ts0 = SClock::now();
    print_nanoseconds("      begin:  ",ts0, 0);

    fpga.ithr = 0;

    FPGA(std::ref(fpga));
    auto ts4 = SClock::now();
    print_nanoseconds("       done:  ",ts4, 0);
    for (int i = 0 ; i < NUM_CU ; i++){
        OCL_CHECK(fpga.err, fpga.err = fpga.q[i].flush());
        OCL_CHECK(fpga.err, fpga.err = fpga.q[i].finish());
    }
// OPENCL HOST CODE AREA END
    auto ts5 = SClock::now();
    print_nanoseconds("       end:   ",ts5, 0);
    std::cout << fpga.ss.str();
    for (int ib = 0; ib < NBUFFER; ib++) {
        for (int i = 0 ; i < NUM_CU ; i++){
            for (int istream = 0; istream < STREAMSIZE; istream++) {
                std::cout<<"STREAM - "<<istream<<"\n\t";
                for (int ij = 0; ij < BIGSTREAMSIZE_OUT; ij++) {
                // Create the test data if no data files found or if end of files has been reached

		  bigdata_t outTmp = fpga.source_hw_results[ib*NUM_CU*STREAMSIZE*BIGSTREAMSIZE_OUT+i*STREAMSIZE*BIGSTREAMSIZE_OUT+istream*BIGSTREAMSIZE_OUT+ij];
		  std::cout << "----> ";
		  for(int ik = 0; ik < COMPRESSION; ik++) { 
		    data_t outTmpL;
		    outTmpL.range(15,0) = outTmp.range((ik+1)*16-1,(ik)*16);
		    std::cout << outTmpL << "  ";
		  }
		  std::cout << std::endl;
		  std::cout << " out num " <<outTmp << std::endl;
                }
                std::cout<<std::endl;
            }
        }
    }

    return EXIT_SUCCESS;
}

