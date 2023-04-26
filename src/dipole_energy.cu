// want local copies of potentials
// want local copy of charge density - atomic copy

void Charges::CalcCharges() {

    //zero d_cpx1 and d_cpx2
    d_resetComplexes<<<M_Grid, M_Block>>>(d_cpx1, d_cpx2, M);

    __global__ void d_resetComplexes(cufftComplex* one, cufftComplex* two,
        const int M) {

        const int ind = blockIdx.x * blockDim.x + threadIdx.x;

        if (ind >= M)
            return;

        one[ind].x = 0.f;
        one[ind].y = 0.f;

        two[ind].x = 0.f;
        two[ind].y = 0.f;
    }

    //fft charge density
    d_prepareChargeDensity<<<M_Grid, M_Block>>>(d_charge_density_field, d_cpx1, M);

    
    __global__ void d_prepareChargeDensity(float* d_t_charge_density,
        cufftComplex* d_tc, int M) {

        const int ind = blockIdx.x * blockDim.x + threadIdx.x;

        if (ind >= M)
            return;

        d_tc[ind].x = d_t_charge_density[ind];
        d_tc[ind].y = 0.f;
}



    cufftExecC2C(fftplan, d_cpx1, d_cpx2, CUFFT_FORWARD);//now fourier transformed density data in cpx2

    d_divideByDimension<<<M_Grid, M_Block>>>(d_cpx2, M);//normalizes the charge density field


    __global__ void d_divideByDimension(cufftComplex* in, const int M) {
        const int ind = blockIdx.x * blockDim.x + threadIdx.x;
        if (ind >= M)
            return;

        in[ind].x *= 1.0f / float(M);
        in[ind].y *= 1.0f / float(M); 
    }

    //electric potential in cpx1
    d_prepareElectrostaticPotential<<<M_Grid, M_Block>>>(d_cpx2, d_cpx1, charge_bjerrum_length, charge_smearing_length,
        M, Dim, d_L, d_Nx); 


        //calculates electrostatic potential in Fourier space
        // d_tc:         [M] Fourier transform of density field
        // d_ep:         [M] variable to store electrostatic potential
        // bjerrum:      Bjerrum length
        // length_scale: charge smearing length
        __global__ void d_prepareElectrostaticPotential(cufftComplex* d_tc, cufftComplex* d_ep, 
            float bjerrum, float length_scale, const int M, const int Dim, const float* L,
            const int* Nx) {
            
            const int ind = blockIdx.x * blockDim.x + threadIdx.x;

            if (ind >= M)
                return;

            float kv[3], k2;
            k2 = d_get_k(ind, kv, L, Nx, Dim);

            if (k2 != 0) {
                //d_ep[ind].x = ((d_tc[ind].x * 4 * PI * bjerrum) / k2) * exp(-1 * k2 / (2 * length_scale * length_scale));
                d_ep[ind].x = ((d_tc[ind].x * 4 * PI * bjerrum) / k2) * exp( -k2 * length_scale * length_scale / 2.0);
                d_ep[ind].y = ((d_tc[ind].y * 4 * PI * bjerrum) / k2) * exp( -k2 * length_scale * length_scale / 2.0);
            }
            else {
                d_ep[ind].x = 0.f;
                d_ep[ind].y = 0.f;
            }
        }



    // for (int j = 0; j < Dim; j++) {
    //     d_prepareElectricField<<<M_Grid, M_Block>>>(d_cpx2, d_cpx1, charge_smearing_length, M, Dim, d_L, d_Nx, j);//new data for electric field in cpx2

    //     check_cudaError("d_prepareElectrostaticField");

    //     cufftExecC2C(fftplan, d_cpx2, d_cpx2, CUFFT_INVERSE); //d_cpx2 now holds the electric field, in place transform

    //     check_cudaError("cufftExec2");

    //     if (j == 0)
    //         d_accumulateGridForceWithCharges<<<M_Grid, M_Block>>>(d_cpx2,
    //             d_charge_density_field, d_all_fx_charges, M);
    //     if (j == 1)
    //         d_accumulateGridForceWithCharges<<<M_Grid, M_Block>>>(d_cpx2,
    //             d_charge_density_field, d_all_fy_charges, M);
    //     if (j == 2)
    //         d_accumulateGridForceWithCharges<<<M_Grid, M_Block>>>(d_cpx2,
    //             d_charge_density_field, d_all_fz_charges, M);

    //     check_cudaError("d_accumulateGridForceWithCharges");

    //     d_setElectricField<<<M_Grid, M_Block>>>(d_cpx2, d_electric_field, j, M);
    // }

    //prepares d_electrostatic_potential to be copied onto host

    cufftExecC2C(fftplan, d_cpx1, d_cpx1, CUFFT_INVERSE);

    d_setElectrostaticPotential<<<M_Grid, M_Block>>>(d_cpx1, d_electrostatic_potential, M);
    
    __global__ void d_setElectrostaticPotential(cufftComplex* d_ep,
        float* d_electrostatic_potential, const int M) {

        const int ind = blockIdx.x * blockDim.x + threadIdx.x;

        if (ind >= M)
            return;

        d_electrostatic_potential[ind] = d_ep[ind].x;
}

}


__global__ void d_charge_grid_charges(float* d_x, float* d_grid_W, int* d_grid_inds,
    int* d_tp, float* d_rho, // d_rho, Has dimensions ntypes*M
    const int* d_Nx, const float* d_dx, const float V,
    const int ns, const int pmeorder, const int M, const int Dim,
    float* d_charge_density, float* charges, int charge_flag) {// float* d_charges_over_M) {

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    float local_charge_density[M];
    for (int i = 0; i < M; ++i){
        local_charge_density[i] = atomicAdd(d_charge_density[i],0.0)
    }

    int grid_ct = 0;
    int id_typ = d_tp[id];

    if (Dim == 2) {
        for (int ix = 0; ix < pmeorder + 1; ix++) {
            for (int iy = 0; iy < pmeorder + 1; iy++) {
                Mindex = d_grid_inds[id * grid_per_partic + grid_ct]
                W3_charges = d_grid_W[id * grid_per_partic + grid_ct]
                atomicAdd(&local_charge_density[Mindex], W3_charges * charges[id]);
                grid_ct++;

            }// iy = 0:pmeorder+1
        }// ix=0:pmeorder+1
    }// if Dim==2

    else if (Dim == 3) {
        for (int ix = 0; ix < pmeorder + 1; ix++) {
            for (int iy = 0; iy < pmeorder + 1; iy++) {
                for (int iz = 0; iz < pmeorder + 1; iz++) {
                    Mindex = d_grid_inds[id * grid_per_partic + grid_ct]
                    W3_charges = d_grid_W[id * grid_per_partic + grid_ct]
                    atomicAdd(&local_charge_density[Mindex], W3_charges * charges[id]);
                    grid_ct++;
                }
            }
        }// for ix
    }// if Dim == 3

    float dU = 0.0;
    for (int j = 0; j < grid_per_partic; j++) {
    ind1 = d_grid_inds[my_id * grid_per_partic + j]
    ind2 = d_grid_inds[my_id * grid_per_partic + j]
    dU += d_electrostatic_potential[ind1] * d_grid_W[ind1] * donor_charge
    dU += d_electrostatic_potential[ind2] * d_grid_W[ind2] * acceptor_charge
    }


    // if yes then update charges


    d_charges[ind] = donor_charge
    d_charges[nid] = acceptor_charge

        if (Dim == 2) {
        for (int ix = 0; ix < pmeorder + 1; ix++) {
            for (int iy = 0; iy < pmeorder + 1; iy++) {
                Mindex = d_grid_inds[id * grid_per_partic + grid_ct]
                W3_charges = d_grid_W[id * grid_per_partic + grid_ct]
                atomicAdd(&d_charge_density[Mindex], W3_charges * charges[id]);
                grid_ct++;

            }// iy = 0:pmeorder+1
        }// ix=0:pmeorder+1
    }// if Dim==2

    else if (Dim == 3) {
        for (int ix = 0; ix < pmeorder + 1; ix++) {
            for (int iy = 0; iy < pmeorder + 1; iy++) {
                for (int iz = 0; iz < pmeorder + 1; iz++) {
                    Mindex = d_grid_inds[id * grid_per_partic + grid_ct]
                    W3_charges = d_grid_W[id * grid_per_partic + grid_ct]
                    atomicAdd(&d_charge_density[Mindex], W3_charges * charges[id]);
                    grid_ct++;
                }
            }
        }// for ix
    }// if Dim == 3






}