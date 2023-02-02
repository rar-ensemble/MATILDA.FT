// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "potential.h"
#include <string>

#ifndef _PAIR_GAUSSIAN
#define _PAIR_GAUSSIAN


// This class is of type Potential and inherits all Potential
// routines. Potential is initialized first, then the Gaussian
// initialization is performed.
class Gaussian : public Potential {
private:
	static int num;

public:
    Gaussian();
    Gaussian(std::istringstream&);
    ~Gaussian();
    void Initialize();
    void ReportEnergies(int&)  override;
};


#endif