// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_compute.h"
#include "PS_Box.h"

PS_Compute::PS_Compute(std::istringstream &iss, PS_Box* box) : mybox(box) {
    this->compute_id = this->total_computes++;

    // Set defaults for optional arguments
    this->compute_wait = 0;
    this->compute_freq = 100;
    this->num_data_pts = 0;
};

int PS_Compute::total_computes = 0;

PS_Compute::~PS_Compute() {}