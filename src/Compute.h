// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#ifndef _COMPUTE
#define _COMPUTE

#include "include_libs.h"


class Box;


class Compute {
protected:
    int GRID;                   // Size of thread grid for cell operations
    int BLOCK;                  // Num of blocks per thread
    int group_index;            // Index of the group
    std::string output_name;    // Name of the output file
    std::string style;          // Compute style (e.g., avg_sk, avg_rho)
    int compute_id;
    static int total_computes;  // size of computes vector: mybox->Computes.size()

    int num_data_pts;           // Number of data points accumulated through doCompute
    Box* mybox;

public:

    int compute_freq;       // Frequency for calling doCompute (timesteps), default 100
    int compute_wait;       // Time to wait before calling doCompute (timesteps), default 0
    virtual void alloc_storage()=0;
    virtual void do_compute(void) =0;
    virtual void write_output()=0;


    Compute(std::istringstream&, Box*);
    virtual ~Compute();

};

#endif