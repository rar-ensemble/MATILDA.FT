// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "potential.h"


#ifndef _PAIR_CHARGES
#define _PAIR_CHARGES

void calc_electrostatic_energy(void);
void cuda_collect_electric_field(void);

class Charges : public Potential {
private: 
	void Allocate_Memory(void);
public:
	Charges();
	Charges(std::istringstream&);
	~Charges();
	void CalcCharges();
	void CalcForces(){CalcCharges();};
    void CalcVirial() override;
	void Initialize();
	void ReportEnergies(int&) override;
	float CalcEnergy() override;
	static int do_charges;
};


#endif

__global__ void d_prepareChargeDensity(float*, cufftComplex*, int);
__global__ void d_prepareElectrostaticPotential(cufftComplex*, cufftComplex*,
    float, float, const int, const int, const float*, const int*);
__global__ void d_prepareElectricField(cufftComplex*, cufftComplex*,
    float, const int, const int, const float*, const int*, const int);
__global__ void d_divideByDimension(cufftComplex*, const int);
__global__ void d_accumulateGridForceWithCharges(cufftComplex*,
    float*, float*, const int);
__global__ void d_setElectrostaticPotential(cufftComplex*,
    float*, const int);
__global__ void d_resetComplexes(cufftComplex*, cufftComplex*, const int);
__global__ void d_setElectricField(cufftComplex*, float*, int, const int);