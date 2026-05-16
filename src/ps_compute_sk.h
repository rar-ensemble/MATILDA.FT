// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_compute.h"

#ifndef _PSCOMPUTESK
#define _PSCOMPUTESK


class PS_ComputeSK : public PS_Compute {
protected:

    float *sk_real;             // [M] Storage for real part of structure factor
    float *d_sk;            // [M] device storage for structure factor

public:

    void do_compute(int) override;
    void write_output() override;
    void initialize_compute() override;
    

    PS_ComputeSK(std::istringstream&, PS_Box*);
    ~PS_ComputeSK(void);

};

#endif