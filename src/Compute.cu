// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include <sstream>
#include "globals.h"
#include "timing.h"
#include <algorithm>

using namespace std;

void write_kspace_data(const char*, complex<float>*);
__global__ void d_multiplyComplex(cufftComplex*, cufftComplex*,
    cufftComplex*, int);
__global__ void d_prepareDensity(int, float*, cufftComplex*, int);



Compute::Compute(std::istringstream &iss) {
    this->input_command = iss.str();
    this->compute_id = this->total_computes++;

    // Set defaults for optional arguments
    this->compute_wait = 0;
    this->compute_freq = 100;
};

Compute::~Compute() {};


__global__ void d_removeMolecFromFields(
    float* d_t_rho,             // [M*ntypes] All density fields
    const int removedMoleculeID,// index of molecule being removed
    const int* mID,             // [ns] molecule IDs
    const float* d_grid_W,      // [ns*grid_per_partic], weights for each grid 
    const int* d_grid_inds,     // [ns*grid_per_partic], indices of the grids
    const int* d_tp,            // [ns] particle types
    const float gvol,           // Volume of grid elements
    const int grid_per_partic,  // Number of grid points each particle interacts with
    const int ns,               // Total number of sites
    const int M,                // Total number of grid points
    const int Dim)              // System dimensionality
    {

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    // If this isn't the molecule to remove, do nothing.
    if ( mID[id] != removedMoleculeID )
        return;

    int id_typ = d_tp[id];
    int gind, typ_ind;
    float W3;

    for (int m = 0; m < grid_per_partic; m++) {
        gind = d_grid_inds[id * grid_per_partic + m];

        W3 = d_grid_W[id * grid_per_partic + m];

        typ_ind = id_typ * M + gind;

        atomicAdd(&d_t_rho[typ_ind], -W3);

    }
}

__global__ void d_restoreMolecToFields(
    float* d_t_rho,             // [M*ntypes] All density fields
    const int removedMoleculeID,// index of molecule being removed
    const int* mID,             // [ns] molecule IDs
    const float* d_grid_W,      // [ns*grid_per_partic], weights for each grid 
    const int* d_grid_inds,     // [ns*grid_per_partic], indices of the grids
    const int* d_tp,            // [ns] particle types
    const float gvol,           // Volume of grid elements
    const int grid_per_partic,  // Number of grid points each particle interacts with
    const int ns,               // Total number of sites
    const int M,                // Total number of grid points
    const int Dim)              // System dimensionality
    {

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    // If this isn't the molecule to remove, do nothing.
    if ( mID[id] != removedMoleculeID )
        return;

    int id_typ = d_tp[id];
    int gind, typ_ind;
    float W3;

    for (int m = 0; m < grid_per_partic; m++) {
        gind = d_grid_inds[id * grid_per_partic + m];

        W3 = d_grid_W[id * grid_per_partic + m];

        typ_ind = id_typ * M + gind;

        atomicAdd(&d_t_rho[typ_ind], W3);

    }

}

__global__ void d_removeMolecule(
    float* x_removed,         // [ns*Dim] array to contain positions after molecule is removed
    const float* x_all,       // [(ns+num_removed)*Dim] array containing all original positions
    const int first_ind,      // index of first site to be removed
    const int num_removed,    // number of sites to be removed
    const int Dim,            // system dimensionality
    const int ns) {           // number of sites *AFTER num_removed SUBTRACTED*

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    // id < first_ind are just straight copies
    if ( id < first_ind ) {
        for ( int j=0 ; j<Dim ; j++ ) 
            x_removed[id*Dim + j] = x_all[id*Dim + j];
    }

    // Otherwise, id is shifted by num_removed
    else {
        int new_id = id + num_removed;
        for ( int j=0 ; j<Dim ; j++ )
            x_removed[id * Dim + j] = x_all[new_id * Dim + j];
    }

}

void Compute::set_optional_args(std::istringstream& iss){
    while (!iss.eof()) {
      string word;
      iss >> word;
      if (word == "freq") {
        iss >> compute_freq;
      } else if (word == "wait") {
        iss >> compute_wait;
      }
    }
}

int Compute::total_computes = 0;

Compute* ComputeFactory(istringstream &iss){
	string s1;
	iss >> s1;
    std::transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
	if (s1 == "avg_s_k"){
		return new Avg_sk(iss);
	}
    else if (s1 == "avg_rho") {
        return new Avg_rho(iss);
    }
	else if (s1 == "s_k"){
		return new Sk(iss);
	}
	else if (s1 == "widom"){
		return new Widom(iss);
	}
	else if (s1 == "chemical_potential"){
		return new ChemPot(iss);
	}
	
	die(s1 + " is not a valid Compute");
	return 0;
}
