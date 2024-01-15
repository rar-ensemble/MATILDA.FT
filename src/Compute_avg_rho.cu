// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Compute_avg_rho.h"
#include "globals.h"
#include <algorithm>
#include "global_templated_functions.h"

using namespace std;

Avg_rho::~Avg_rho(){}

Avg_rho::Avg_rho(istringstream& iss) : Compute(iss)
{
    style = "avg_rho";
    readRequiredParameter(iss, particle_type);

    num_data_pts = 0;

    set_optional_args(iss);

    cout << "  Calculating <rho(r)> for component " << particle_type + 1 << " every " << this->compute_freq << " steps after " << this->compute_wait << " steps have passed." << endl;
}

// Generate AllocStorage for the Avg_sk class

void Avg_rho::allocStorage()
{
    fstore1 = (float*) malloc( M * sizeof(float) );
    for ( int i=0 ; i<M; i++ ) {
      fstore1[i] = 0.0f;
    }

}

void Avg_rho::writeResults(){

  for (int i = 0; i < M; i++) {
    if ( num_data_pts > 0)
      tmp[i] = fstore1[i] / float(num_data_pts);
    else
      tmp[i] = 0.0f;
  }

  char nm[50];

  // compute_id is used in the name instead of "type" in case multiple
  // computes operate on the same type
  sprintf(nm, "avg_rho%d.dat", compute_id);
  write_grid_data(nm, tmp);
}



void Avg_rho::doCompute(){

    // Extract the density of the relevant type
    // Uses complex data type to utilize this routine
    d_prepareDensity<<<M_Grid, M_Block>>>(particle_type, d_all_rho, d_cpx1, M);
    check_cudaError("Compute->doCompute.style = avg_sk prepare density");

    // Copy data to host
    cudaMemcpy(cpx1, d_cpx1, M * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // Accumulate the real part of the density field
    for (int i = 0; i < M; i++)
        fstore1[i] += cpx1[i].x;

    num_data_pts++;
}