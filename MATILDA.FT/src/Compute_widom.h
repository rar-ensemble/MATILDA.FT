// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Compute.h"
#include <string>
#include <sstream>
#include <iostream>

void prepareDensityFields(void);       // grid_utils.cu
void calc_properties(int);             // calc_properties.cu
double ran2(void);                     // random.cu
double gasdev2(void);             // random.cu
float calc_nbEnergy(void);             // calc_properties.cu
float calc_moleculeBondedEnergy(int);  // calc_properties.cu
void calc_bondedProps(int);
void update_device_positions(float**, float*);              // device_comm_utils.cu

#ifndef _COMPUTE_WIDOM_H_
#define _COMPUTE_WIDOM_H_

class Widom : public Compute {


public:

    // The Widom compute will estimate the excess chemical potential by 
    // growing a copy of a specified molecule at random and calculating the
    // energy of the system. 

    int mole_id;      // molecule ID to use as the template for the ghost molecule
    int num_configs;  // number of configurations to generate each time compute is called
    int num_sites;    //  (calculated) number of sites in the molecule
    int first_site;   // (calculated) first site index 

    void allocStorage() override;
    void doCompute(void);
    void writeResults();
    Widom(std::istringstream&);
    ~Widom();

};

#endif
