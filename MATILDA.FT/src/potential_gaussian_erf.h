// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "potential.h"

#ifndef _PAIR_GAUSSIAN_ERF
#define _PAIR_GAUSSIAN_ERF


// This class is of type Potential and inherits all Potential
// routines. Potential is initialized first, then the Gaussian
// initialization is performed.
class GaussianErf : public Potential {
private:
	static int num;
public:
    GaussianErf();
    ~GaussianErf();
    GaussianErf(std::istringstream&);
    void Initialize();
    void ReportEnergies(int&)  override;
};


#endif
