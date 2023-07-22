// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"

using namespace std;

void get_r(int, float*);
void unstack(int, int*);
void init_binary_output(void);

void write_grid_data(const char* lbl, float* dat) {

    int i, j, * nn;
    nn = new int[Dim];
    FILE* otp;
    float* r = new float [Dim];


    otp = fopen(lbl, "w");

    for (i = 0; i < M; i++) {
        get_r(i, r);
        unstack(i, nn);

        for (j = 0; j < Dim; j++)
            fprintf(otp, "%f ", r[j]);

        fprintf(otp, "%1.8e \n", dat[i]);

        if (Dim == 2 && nn[0] == Nx[0] - 1)
            fprintf(otp, "\n");
    }

    fclose(otp);

}

void write_kspace_cudaComplex(const char* lbl, cufftComplex* kdt) {
    int i, j, nn[3];
    FILE* otp;
    float kv[3], k2;

    otp = fopen(lbl, "w");

    for (i = 1; i < M; i++) {
        unstack(i, nn);

        k2 = get_k(i, kv, Dim);

        for (j = 0; j < Dim; j++)
            fprintf(otp, "%f ", kv[j]);

        float cpx_abs = sqrtf(kdt[i].x * kdt[i].x + kdt[i].y * kdt[i].y);
        fprintf(otp, "%1.5e %1.5e %1.5e %1.5e\n", cpx_abs, sqrtf(k2),
            kdt[i].x, kdt[i].y);

        if (Dim == 2 && nn[0] == Nx[0] - 1)
            fprintf(otp, "\n");
    }

    fclose(otp);
}

void write_kspace_data(const char* lbl, complex<float> * kdt) {
    int i, j, nn[3];
    FILE* otp;
    float kv[3], k2;

    otp = fopen(lbl, "w");

    for (i = 1; i < M; i++) {
        unstack(i, nn);

        k2 = get_k(i, kv, Dim);

        for (j = 0; j < Dim; j++)
            fprintf(otp, "%f ", kv[j]);

        fprintf(otp, "%1.5e %1.5e %1.5e %1.5e\n", abs(kdt[i]), sqrt(k2),
            real(kdt[i]), imag(kdt[i]));

        if (Dim == 2 && nn[0] == Nx[0] - 1)
            fprintf(otp, "\n");
    }

    fclose(otp);
}

void init_binary_output() {

    // soutput_id = std::to_string(output_id[mpi_rank]);

    FILE* otp;

    otp = fopen(("grid_densities_" + srank +  ".bin").c_str(), "wb");
    if (otp == NULL){
        die("failed to open grid_densities!");
    }
    
    fwrite(&Dim, sizeof(int), 1, otp);
    fwrite(Nx, sizeof(int), Dim, otp);
    fwrite(L, sizeof(float), Dim, otp);
    fwrite(&ntypes, sizeof(int), 1, otp);

    otp = fopen(("grid_densities_eid_" + soutput_id +  ".bin").c_str(), "wb");
    if (otp == NULL){
        die("failed to open grid_densities!");
    }


    fwrite(&Dim, sizeof(int), 1, otp);
    fwrite(Nx, sizeof(int), Dim, otp);
    fwrite(L, sizeof(float), Dim, otp);
    fwrite(&ntypes, sizeof(int), 1, otp);

    fclose(otp);

    otp = fopen(("positions_" + srank + ".bin").c_str(), "wb");
    if (otp == NULL){
        die("Failed to open positions file!");
    }

    fwrite(&ns, sizeof(int), 1, otp);
    fwrite(&Dim, sizeof(int), 1, otp);
    fwrite(L, sizeof(float), 3, otp);
    fwrite(tp, sizeof(int), ns, otp);
    fwrite(molecID, sizeof(int), ns, otp);
    if ( Charges::do_charges == 1 )
        fwrite(charges, sizeof(float), ns, otp);

    fclose(otp);
    otp = fopen(("positions_eid_" + soutput_id  + ".bin").c_str(), "wb");
    if (otp == NULL){
        die("Failed to open positions file!");
    }

    fwrite(&ns, sizeof(int), 1, otp);
    fwrite(&Dim, sizeof(int), 1, otp);
    fwrite(L, sizeof(float), 3, otp);
    fwrite(tp, sizeof(int), ns, otp);
    fwrite(molecID, sizeof(int), ns, otp);
    if ( Charges::do_charges == 1 )
        fwrite(charges, sizeof(float), ns, otp);

    fclose(otp);
}

void write_binary() {
    FILE* otp;

    otp = fopen(("grid_densities_" + srank +  ".bin").c_str(), "ab");
    if (otp == NULL){
        die("Failed to append to grid_densities.bin");
    }
    fwrite(all_rho, sizeof(float), M * ntypes, otp);
    fclose(otp);

    otp = fopen(("grid_densities_eid_" +  soutput_id +  ".bin").c_str(), "ab");
    if (otp == NULL){
        die("Failed to append to grid_densities _eid.bin");
    }
    fwrite(all_rho, sizeof(float), M * ntypes, otp);
    fclose(otp);

    // Positions

    otp = fopen(("positions_" + srank +  ".bin").c_str(), "ab");
    if (otp == NULL){
        die("Failed to append to positions!");
    }
    fwrite(h_ns_float, sizeof(float), ns*Dim, otp);
    fclose(otp);

    otp = fopen(("positions_eid_" + soutput_id +  ".bin").c_str(), "ab");
    if (otp == NULL){
        die("Failed to append to positions!");
    }
    fwrite(h_ns_float, sizeof(float), ns*Dim, otp);
    fclose(otp);

}

void write_struc_fac() {
    // Declare Local Variables
    FILE* otp;
    int i, j, k, nn[3];
    float kv[3], k2;
    double temp;
    char label [30];
    
    for (i = 0; i < ntypes; i++) {
        // Open output file

        sprintf(label, "sk%d.dat", i);
        //sprintf(label, "sk%d_%d.dat", i,step);
        otp = fopen(label, "w");
        if (otp == NULL)
            die("Failed to write to sk.dat"); //Check to see that output file actually opened


    
        //fprintf(otp, "Printing type: %d\n", i);
        // SKIP j=o to skip 0 0 0 point
        for (j = 1; j < M; j++) {
            /// Get kspace coordinates
            unstack(j, nn);
            k2 = get_k(j, kv, Dim);
            /// Print kspace coordinates
            for (k = 0; k < Dim; k++) {
                fprintf(otp, "%f ", kv[k]);
            }
            /// calculate the avg (sum) over total calcuations
            temp = avg_sk[i][j] / n_avg_calc;
            /// Write data points. Calculations based on binToSk script on git
            fprintf(otp, "%1.5e %1.5e %1.5e %1.5e\n", abs(temp), sqrt(k2), real(temp), imag(temp));
        }
        fclose(otp);
    }
    
}