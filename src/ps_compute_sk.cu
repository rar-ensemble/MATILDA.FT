// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_compute_sk.h"
#include "PS_Box.h"

PS_ComputeSK::PS_ComputeSK(std::istringstream &iss, PS_Box* box) : PS_Compute(iss, box) {
    this->compute_id = this->total_computes++;



    std::cout << "  Computing s(k) for group " << group << std::endl;

};


void PS_ComputeSK::do_compute(int step) {

    if ( step > this->compute_wait && step%this->compute_freq == 0 ) {

        float *d_rho, *d_ftp;
        cuComplex *d_tp1, *d_tp2;

        d_ftp = mybox->d_Alex;
        d_rho = mybox->psGroup[group_int].d_rho;
        d_tp1 = mybox->d_cpxAlex;
        d_tp2 = mybox->d_cpxGabe;

        // Pointer to density field 
        d_rho = mybox->psGroup[group_int].d_rho;

        // d_tp1 = d_rho
        d_floatToCpx<<<GRID, BLOCK>>>(d_tp1, d_rho, mybox->M);

        // d_tp2 = FT(d_tp1) = FT(rho)
        mybox->cufftWrapperSingle(d_tp1, d_tp2, 1);


        // d_ftp = rho(k) * rho(-k)
        d_multiplyCpxByCpxConj<<<GRID, BLOCK>>>(d_ftp, d_tp2, d_tp2, mybox->M);

        // Accumulate values in dev array
        d_floatPlusEqFloat<<<GRID, BLOCK>>>(d_sk, d_ftp, mybox->M);

        // Accumulate denominator for averaging
        num_data_pts += 1;

    }// step > wait && step%freq == 0
}



void PS_ComputeSK::write_output() {

    cudaMemcpy(sk_real, d_sk, mybox->M*sizeof(float), cudaMemcpyDeviceToHost);
    for ( int i=0 ; i<mybox->M ; i++ ) {
        sk_real[i] *= 1.0 / float(num_data_pts);
    }

    mybox->writeKFieldFloat(output_name.c_str(), sk_real);

}



void PS_ComputeSK::initialize_compute() {
    // Set the group index
    group_int = mybox->findGroupInteger(group);

    // Real part of structure factor
    sk_real = (float*) malloc( mybox->M * sizeof(float));


    cudaMalloc(&d_sk, mybox->M*sizeof(float));

    GRID = mybox->M_Grid;
    BLOCK = mybox->M_Block;

    d_assignFloatVal<<<GRID, BLOCK>>>(d_sk, 0.0, mybox->M);

}



PS_ComputeSK::~PS_ComputeSK() {}