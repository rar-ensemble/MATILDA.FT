// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "timing.h"
#include <cmath>


__global__ void d_bondStressEnergy(int*, int*, int*, float*,
	float*, float*, float*, float*, float*, float*, int, int, int, int);

float calc_nbEnergy(void);
void calc_nbVirial(void);
void calc_bondedProps(int);
void calc_electrostatic_energy_directly(void);
void host_bonds(void);
void host_angles(void);
void cuda_collect_x(void);


// flag indicates whether to calculate virial contribution
void calc_properties(int virFlag) {

	Unb = calc_nbEnergy();

    Upe = Unb;

	// Ptens is zeroed in nbVirial
	if ( virFlag ) calc_nbVirial();

	bond_t_in = int(time(0));
	calc_bondedProps(virFlag);
	bond_t_out = int(time(0));
	bond_tot_time += bond_t_out - bond_t_in;

	//cout << "Bond Pxx: " << bondVir[0] << " tin, tout: " << bond_t_in 
	//	<< ", " << bond_t_out << endl;
}

void calc_bondedProps(int virFlag) {

	cuda_collect_x();
      
    for ( int i=0 ; i<n_molecules ; i++ ) 
        molec_Ubond[i] = 0.0f;

	host_bonds();

	Upe += Ubond;
	
    if ( virFlag ) {
        for (int i = 0; i < n_P_comps; i++) {
		    Ptens[i] += bondVir[i];
        }
	}

	host_angles();

	Upe += Uangle;

    if ( virFlag ) {
	    for (int i = 0; i < n_P_comps; i++) {
		    Ptens[i] += angleVir[i];    
        }
	}

	
}


float calc_nbEnergy() {
	float nbE = 0.0f;

	for (auto Iter: Potentials){
		nbE += Iter->CalcEnergy();
	}

    return nbE;

}

void calc_nbVirial() {
	for (int i = 0; i < n_P_comps; i++)
		Ptens[i] = 0.0f;

	for (auto Iter: Potentials){
		Iter->CalcVirial();
		for (int j = 0; j < n_P_comps; j++)
			Ptens[j] += Iter->total_vir[j];
	}

}




float moleculeBondEnergy(int);  // bonds.cu
float moleculeAngleEnergy(int); // angles.cu

// Calculates the total bonded energy for molecule ``molec''
float calc_moleculeBondedEnergy(int molec) {

    float molecEnergy = 0.0f;

    molecEnergy += moleculeBondEnergy(molec);

    molecEnergy += moleculeAngleEnergy(molec);

    return molecEnergy;

}