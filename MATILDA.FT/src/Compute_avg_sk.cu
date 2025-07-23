// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Compute_avg_sk.h"
#include "globals.h"
#include <algorithm>
#include "global_templated_functions.h"

using namespace std;

Avg_sk::~Avg_sk(){}

Avg_sk::Avg_sk(istringstream& iss) : Compute(iss)
{
    style = "avg_s_k";
    readRequiredParameter(iss, particle_type);

    num_data_pts = 0;

    set_optional_args(iss);

    cout << "  Calculating <S(k)> for component " << particle_type + 1 << " every " << this->compute_freq << " steps after " << this->compute_wait << " steps have passed." << endl;
}

// Generate AllocStorage for the Avg_sk class

void Avg_sk::allocStorage()
{
    this->cpx.resize(M);
    fill(this->cpx.begin(), this->cpx.end(), 0.0f);

    cout << " this->cpx has initial size " << this->cpx.size() << " and capacity " << this->cpx.capacity() << endl;

}

void Avg_sk::writeResults(){

  for (int i = 0; i < M; i++) {
    if ( num_data_pts > 0)
      k_tmp[i] = this->cpx[i] / float(num_data_pts);
    else
      k_tmp[i] = 0.0f;
  }

  char nm[50];

  // compute_id is used in the name instead of "type" in case multiple
  // computes operate on the same type
  sprintf(nm, "avg_sk_%d.dat", compute_id);
  write_kspace_data(nm, k_tmp);
}



void Avg_sk::doCompute(){

    // Extract the density of the relevant type
    d_prepareDensity<<<M_Grid, M_Block>>>(particle_type, d_all_rho, d_cpx1, M);
    check_cudaError("Compute->doCompute.style = avg_sk prepare density");

    // fourier from d_cpx1 to d_cpx2 forward
    cufftExecC2C(fftplan, d_cpx1, d_cpx2, CUFFT_FORWARD);
    check_cudaError("Compute->doCompute.style = avg_sk cufftExec");


    // Multiply by the complex conjugate and scale by 1/M
    // Store it in d_cpx1 as the values inside are not needed at this point
    d_multiplyComplex<<<M_Grid, M_Block>>> (d_cpx2, d_cpx2, d_cpx1, M);
    check_cudaError("Compute->doCompute.style = avg_sk multiplyComplex");


    // Copy data to host and store
    // NOTE: this should probably be stored on the device and only 
    // communicated when writing, but may be OK for now.
    cudaMemcpy(cpx1, d_cpx1, M * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++)
        this->cpx.at(i) += cpx1[i].x + I * cpx1[i].y;


    num_data_pts++;
}