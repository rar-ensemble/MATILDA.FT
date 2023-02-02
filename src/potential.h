// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#ifndef POTENTIAL
#define POTENTIAL
#include <cufft.h>
#include <cufftXt.h>

#include <cmath>
#include <complex>
#include <sstream>
#include <string>
#include <vector>

#include "field_component.h"


// create an enum of four types of pair styles

class Potential {
protected:
    int size;
    std::string input_command;
    void ramp_check_input(std::istringstream&);
    void check_types();
public:
    int type_specific_id;
    std::string potential_type;
    int type1, type2;
    float* u, ** f, ** vir, energy, * total_vir;
    float* rho1, * rho2, ** force1, ** force2;
    float* d_rho1, * d_rho2;
    std::complex<float>* u_k, ** f_k, ** vir_k;
    
    float* d_u, * d_f, * d_vir;
    cufftComplex* d_u_k, * d_f_k, * d_vir_k, * d_master_u_k, * d_master_f_k, * d_master_vir_k;

    double sigma_squared, Rp, U, initial_prefactor, final_prefactor;
    bool ramp = false;
    bool allocated = false;

    Potential();
    Potential(std::istringstream &iss);
    virtual ~Potential();

    void Initialize_Potential();
    virtual float CalcEnergy();
    virtual void CalcVirial();
    void InitializeVirial();
    virtual void CalcForces();

	virtual void Initialize() = 0;
    void AddForces();
    virtual void Update();
    virtual void ReportEnergies(int&)  = 0;
    std::string printCommand(){return input_command;}
    static void determine_types_to_fft();
    static std::vector<int> types_to_fft;
};

#endif

extern std::ofstream dout;