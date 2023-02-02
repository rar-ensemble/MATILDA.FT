// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "potential_charges.h"
#include "device_comm_utils.h"

using namespace std;

extern int ns_alloc, read_charge_flag, config_read_flag;

void Charges::Initialize() {
    Charges::Allocate_Memory();
    Potential::Initialize_Potential();
}

Charges::Charges(){}
Charges::~Charges(){}

Charges::Charges(istringstream& iss) : Potential(iss){
    potential_type = "Charges";
    type_specific_id = 0;

    if ( config_read_flag == 1 )
        die("charges keyword must come before read_data command!");

    read_charge_flag = 1;

    iss >> charge_bjerrum_length;

    string tofloatlengthscale;
    iss >> charge_smearing_length;

    if (Charges::do_charges++ > 1){
        die("Charges have been doubly defined! This is improper.");
    }

}

void Charges::Allocate_Memory(){

    cudaMalloc(&d_charge_density_field, M * sizeof(float));
    cudaMalloc(&d_electrostatic_potential, M * sizeof(float));
    // cudaMalloc(&d_charges, ns_alloc * sizeof(float));
    cudaMalloc(&d_electric_field, M * Dim * sizeof(float));

    cudaMalloc(&d_all_fx_charges, M * sizeof(float));
    cudaMalloc(&d_all_fy_charges, M * sizeof(float));
    device_mem_use += sizeof(float) * (2 * M);

    if (Dim == 3) {
        cudaMalloc(&d_all_fz_charges, M * sizeof(float));
        device_mem_use += sizeof(float*) * (M);
    }
    
    electrostatic_potential = (float*)calloc(M, sizeof(float));
    charge_density_field = (float*)calloc(M, sizeof(float));
    electric_field = (float*)calloc(M * Dim, sizeof(float));

}



// Calculates the electrostatic contribution to the pressure using 
// the Maxwell stress tensor:
// P_i,j = kT / (V*4*pi*lB) * \int E_i E_j - 0.5 * |E|^2 \delta_i,j
// with lB = Bjerrum length, E_i is ith component of electric field
// E_i = -dpsi / dri, psi = electrostatic potential, delta_ij = Kronecker
void Charges::CalcVirial() {
    // electric_field = d_electric_field
    cuda_collect_electric_field();

    // Compute tmp(r) = |E(r)|^2
    for ( int i=0 ; i < M ; i++ ) {
        tmp[i] = 0.0;
        for ( int j=0 ; j<Dim ; j++ ) {
            tmp[i] += electric_field[j*M+i] * electric_field[j*M+i];
        }
    }

    
    // xx contribution
    for ( int i=0 ; i<M ; i++ ) {
        tmp2[i] = electric_field[0*M+i] * electric_field[0*M+i] - 0.5 * tmp[i];
    }
    this->total_vir[0] = integ_trapPBC(tmp2) / (V * PI4 * charge_bjerrum_length);

    // yy contribution
    for ( int i=0 ; i<M ; i++ ) {
        tmp2[i] = electric_field[1*M+i] * electric_field[1*M+i] - 0.5 * tmp[i];
    }
    this->total_vir[1] = integ_trapPBC(tmp2) / (V * PI4 * charge_bjerrum_length);

    // zz (Dim=3) or xy (Dim=2)
    for ( int i=0 ; i<M ; i++ ) {
        if ( Dim == 2 ) 
            tmp2[i] = electric_field[0*M+i] * electric_field[1*M+i];
        else if ( Dim == 3 )
            tmp2[i] = electric_field[2*M+i] * electric_field[2*M+i] - 0.5 * tmp[i];
    }
    this->total_vir[2] = integ_trapPBC(tmp2) / (V * PI4 * charge_bjerrum_length);

    // 3D off-diagonal terms
    if ( Dim == 3 ) {
        // xy
        for ( int i=0 ; i<M ; i++ ) {
            tmp2[i] = electric_field[0*M+i] * electric_field[1*M+i];
        }
        this->total_vir[3] = integ_trapPBC(tmp2) / (V * PI4 * charge_bjerrum_length);

        // xz
        for ( int i=0 ; i<M ; i++ ) {
            tmp2[i] = electric_field[0*M+i] * electric_field[2*M+i];
        }
        this->total_vir[4] = integ_trapPBC(tmp2) / (V * PI4 * charge_bjerrum_length);
        
        // yz
        for ( int i=0 ; i<M ; i++ ) {
            tmp2[i] = electric_field[1*M+i] * electric_field[2*M+i];
        }
        this->total_vir[5] = integ_trapPBC(tmp2) / (V * PI4 * charge_bjerrum_length);
    }

}






float Charges::CalcEnergy(){
    calc_electrostatic_energy();
    cuda_collect_electric_field();

    energy = 0;
    *electrostatic_energy = 0;

	for (int i = 0; i < M; i++) {
		*electrostatic_energy += electrostatic_potential[i] * charge_density_field[i]; // *M; //* d_x[i];
		energy += electrostatic_potential[i] * charge_density_field[i]; // *M; //* d_x[i];
	}
    return energy;
}

void Charges::CalcCharges() {
    //zero d_cpx1 and d_cpx2
    d_resetComplexes<<<M_Grid, M_Block>>>(d_cpx1, d_cpx2, M);

    //fft charge density
    d_prepareChargeDensity<<<M_Grid, M_Block>>>(d_charge_density_field, d_cpx1, M);

    cufftExecC2C(fftplan, d_cpx1, d_cpx2, CUFFT_FORWARD);//now fourier transformed density data in cpx2

    check_cudaError("cufftExec1");

    d_divideByDimension<<<M_Grid, M_Block>>>(d_cpx2, M);//normalizes the charge density field

    //electric potential in cpx1
    d_prepareElectrostaticPotential<<<M_Grid, M_Block>>>(d_cpx2, d_cpx1, charge_bjerrum_length, charge_smearing_length,
        M, Dim, d_L, d_Nx); 


    check_cudaError("d_prepareElectrostaticPotential");

    for (int j = 0; j < Dim; j++) {
        d_prepareElectricField<<<M_Grid, M_Block>>>(d_cpx2, d_cpx1, charge_smearing_length, M, Dim, d_L, d_Nx, j);//new data for electric field in cpx2

        check_cudaError("d_prepareElectrostaticField");

        cufftExecC2C(fftplan, d_cpx2, d_cpx2, CUFFT_INVERSE); //d_cpx2 now holds the electric field, in place transform

        check_cudaError("cufftExec2");

        if (j == 0)
            d_accumulateGridForceWithCharges<<<M_Grid, M_Block>>>(d_cpx2,
                d_charge_density_field, d_all_fx_charges, M);
        if (j == 1)
            d_accumulateGridForceWithCharges<<<M_Grid, M_Block>>>(d_cpx2,
                d_charge_density_field, d_all_fy_charges, M);
        if (j == 2)
            d_accumulateGridForceWithCharges<<<M_Grid, M_Block>>>(d_cpx2,
                d_charge_density_field, d_all_fz_charges, M);

        check_cudaError("d_accumulateGridForceWithCharges");

        d_setElectricField<<<M_Grid, M_Block>>>(d_cpx2, d_electric_field, j, M);
    }

    //prepares d_electrostatic_potential to be copied onto host
    cufftExecC2C(fftplan, d_cpx1, d_cpx1, CUFFT_INVERSE);


    check_cudaError("cufftExec3");

    d_setElectrostaticPotential<<<M_Grid, M_Block>>>(d_cpx1, d_electrostatic_potential, M);

    check_cudaError("d_setElectrostaticPotential");

}

void Charges::ReportEnergies(int &die_flag){
    
    if (std::isnan(energy)) { die_flag = 1; }
    cout<< "Electrostatic Energy: " + to_string(energy) + " ";
}

void calc_electrostatic_energy() {

	cuda_collect_charge_density_field();
	cuda_collect_electrostatic_potential();
}

void calc_electrostatic_energy_directly() {
	*electrostatic_energy_direct_computation = 0.0f;

	float distance = sqrt(((x[0][0] - x[1][0]) * (x[0][0] - x[1][0])) + ((x[0][1] - x[1][1]) * (x[1][0] - x[1][1])));

	*electrostatic_energy_direct_computation += (charges[0] * charges[1] * charge_bjerrum_length) / distance;
}

int Charges::do_charges= 0;