// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


//////////////////////////////////////////////
// Rob Riggleman                8/18/2021   //
// Defining compute group to calculate      //
// various quantities at regular intervals. //
// Examples: Average S(k) for a group, avg  //
// density field, etc.                      //
//////////////////////////////////////////////

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <complex>


extern int ns, *molecID, extra_ns_memory;

#ifndef _COMPUTE
#define _COMPUTE

class Compute {
protected:
    int GRID;               // Size of thread grid for cell operations
    int BLOCK;              // Num of blocks per thread
    std::string group_name;      // Name of the group in the n list
    int group_index;        // Index of the group
    std::string out_name;        // Name of the output file
    std::string style;           // Compute style (e.g., avg_sk, avg_rho)
    int compute_id;
    float* fstore1;         // Variable to store data
    float* fstore2;         // Variable to store data
    float* d_fdata;         // Device data
    std::vector<std::complex<float>> cpx;   // Variable to store data
    int num_data_pts;      // Number of data points accumulated through doCompute
    std::string input_command;
    static int total_computes;
    void set_optional_args(std::istringstream&);
public:


    int compute_freq;       // Frequency for calling doCompute (timesteps), default 100
    int compute_wait;       // Time to wait before calling doCompute (timesteps), default 0
    virtual void allocStorage()=0;
    virtual void doCompute(void) =0;
    virtual void writeResults()=0;
    std::string printCommand(){return input_command;}
    Compute(std::istringstream&);
    virtual ~Compute();

};

#endif


__global__ void d_copyPositions(float*, float*, int, int);  // device_array_utils.cu
__global__ void d_removeMolecule(float*, const float*,      // Compute.cu, bottom
                                 const int, const int, const int, const int);
__global__ void d_removeMolecFromFields(float*, const int,  // Compute.cu, bottom
                                        const int*, const float*, const int*, const int*,
                                        const float, const int, const int, const int, const int);
__global__ void d_restoreMolecToFields(float*, const int,  // Compute.cu, bottom
                                       const int*, const float*, const int*, const int*,
                                       const float, const int, const int, const int, const int);