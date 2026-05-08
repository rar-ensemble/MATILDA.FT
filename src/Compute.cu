// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Compute.h"
#include "Box.h"

Compute::Compute(std::istringstream &iss, Box* box) {
    this->compute_id = this->total_computes++;

    // Set defaults for optional arguments
    this->compute_wait = 0;
    this->compute_freq = 100;
    this->num_data_pts = 0;
};

int Compute::total_computes = 0;

Compute::~Compute() {}