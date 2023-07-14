// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "potential.h"
#include "global_templated_functions.h"
#include <algorithm>

using namespace std;

__global__ void d_prepareDensity(int, float*, cufftComplex*, int);
__global__ void d_prepareChargeDensity(float*, cufftComplex*, int);
__global__ void d_prepareForceKSpace(cufftComplex*, cufftComplex*,
    cufftComplex*, const int, const int, const int);
__global__ void d_accumulateGridForce(cufftComplex*, float*, float*,
    const int, const int);
__global__ void d_multiplyComplex(cufftComplex*, cufftComplex*, 
    cufftComplex*, int);
__global__ void d_prepareIntegrand(cufftComplex*, int, float*, float*, int);
__global__ void d_extractForceComp(cufftComplex*, cufftComplex*, const int, const int, const int);
__global__ void d_initVirial(float*, const float*, const float*, const float*, 
    const float*, const int, const int, const int*, const int);
__global__ void d_extractVirialCompR2C(cufftComplex*, const float*, const int,
    const int, const int);
__global__ void d_insertVirialCompC2C(cufftComplex*, const cufftComplex*, const int,
    const int, const int);
__global__ void d_prepareVirial(const cufftComplex*, const cufftComplex*,
    cufftComplex*, const int, const int, const int, const int);
__global__ void d_divideByDimension(cufftComplex*, const int);
__global__ void d_multiply_cufftCpx_scalar(cufftComplex*, float, int);
__global__ void d_multiply_cufftCpx_scalar(cufftComplex*, float, cufftComplex*, int);

float reduce_device_float(float*, const int, const int);
//bool checknan_cpx(cufftComplex*, int);
void write_lammps_traj(void);
void cuda_collect_x(void);
void cuda_collect_rho(void);
void write_kspace_cudaComplex(const char*, cufftComplex*);


// Constructor, allocates memory for:
// potential, forces, virial contribution
// in both r- and k-space
void Potential::Initialize_Potential() {
    allocated = true;
    int n_off_diag = Dim + (Dim * Dim - Dim) / 2;
        
    size = M ;

    this->u = (float*)calloc(M, sizeof(float));
    this->f = (float**)calloc(Dim, sizeof(float*));
    this->vir = (float**)calloc(n_off_diag, sizeof(float*));

    this->u_k = (complex<float>*) calloc(M, sizeof(complex<float>));
    this->f_k = (complex<float>**) calloc(Dim, sizeof(complex<float>*));
    this->vir_k = (complex<float>**) calloc(n_off_diag, sizeof(complex<float>*));

    for (int i = 0; i < Dim; i++) {
        this->f[i] = (float*)calloc(M, sizeof(float));
        this->f_k[i] = (complex<float>*) calloc(M, sizeof(complex<float>));
    }

    total_vir = (float*)calloc(n_off_diag, sizeof(float));

    // Vir stores the diagonal plus the off-diagonal terms
    // The term in parenthesis (Dim*Dim-Dim) will always be even
    for (int i = 0; i < n_off_diag; i++) {
        this->vir[i] = (float*)calloc(M, sizeof(float));
        this->vir_k[i] = (complex<float>*) calloc(M, sizeof(complex<float>));
    }

    cudaMalloc(&this->d_u, M * sizeof(float));
    cudaMalloc(&this->d_f, Dim * M * sizeof(float));
    cudaMalloc(&this->d_vir, n_P_comps * M * sizeof(float));
    device_mem_use += M * (Dim + 1 + n_P_comps) * sizeof(float);

    cudaMalloc(&this->d_u_k, M * sizeof(cufftComplex));
    cudaMalloc(&this->d_f_k, Dim * M * sizeof(cufftComplex));
    cudaMalloc(&this->d_vir_k, n_P_comps * M * sizeof(cufftComplex));
    cudaMalloc(&this->d_master_u_k, M * sizeof(cufftComplex));
    cudaMalloc(&this->d_master_f_k, Dim * M * sizeof(cufftComplex));
    cudaMalloc(&this->d_master_vir_k, n_P_comps * M * sizeof(cufftComplex));
    device_mem_use += 2 * M * (Dim + 1 + n_P_comps) * sizeof(cufftComplex);


}

__global__ void add(float* df_global, float* df_local, int Dim, int M){

    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id > M) return;

    for (int i = 0; i < Dim; i++) {
        df_global[i * M + id] += df_local[i * M + id];
    }

}

void Potential::AddForces(){
    add<<<M_Grid, M_Block>>>(::d_f, this->d_f, Dim, ns);
}


// Calculates forces on rho1, rho2 for this pairstyle
void Potential::CalcForces() {

    /////////////////////////
    // rho2 acting on rho1 //
    /////////////////////////
 //cudaDeviceSynchronize();
    // fft rho2
    d_prepareDensity<<<M_Grid, M_Block>>>(type2, d_all_rho, d_cpx1, M);

    check_cudaError("d_prepareDensity");

 //cudaDeviceSynchronize();
    cufftExecC2C(fftplan, d_cpx1, d_cpx2, CUFFT_FORWARD);
    check_cudaError("cufftExec1");

     //cudaDeviceSynchronize();
    for (int j = 0; j < Dim; j++) {
        // d_cpx1 = d_cpx2 * d_f_k
        d_prepareForceKSpace<<<M_Grid, M_Block>>>(this->d_f_k, 
            d_cpx2, d_cpx1, j, Dim, M);

        check_cudaError("d_prepareForceKSpace");
         //cudaDeviceSynchronize();
        // back to real space, in-place transform
        cufftExecC2C(fftplan, d_cpx1, d_cpx1, CUFFT_INVERSE);
        //cudaDeviceSynchronize();

        check_cudaError("cufftExec1");
        
        
        // Accumulate the forces on type 1
        if (j == 0)
            d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpx1, 
                d_all_rho, d_all_fx, type1, M);
        if (j == 1)
            d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpx1,
                d_all_rho, d_all_fy, type1, M);
        if (j == 2)
            d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpx1,
                d_all_rho, d_all_fz, type1, M);
     	 	
        check_cudaError("d_accumulateGridForce");
        
    }


    // fft rho1
    d_prepareDensity<<<M_Grid, M_Block>>> (type1, d_all_rho, d_cpx1, M);
    cufftExecC2C(fftplan, d_cpx1, d_cpx2, CUFFT_FORWARD);

    check_cudaError("cufftExec");

    
    for (int j = 0; j < Dim; j++) {
        // d_cpx1 = d_cpx2 * d_f_k
        d_prepareForceKSpace<<<M_Grid, M_Block>>>(this->d_f_k,
            d_cpx2, d_cpx1, j, Dim, M);

        // back to real space, in-place transform
        cufftExecC2C(fftplan, d_cpx1, d_cpx1, CUFFT_INVERSE);


        // Accumulate the forces on type 2
        if (j == 0)
            d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpx1,
                d_all_rho, d_all_fx, type2, M);
        if (j == 1)
            d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpx1,
                d_all_rho, d_all_fy, type2, M);
        if (j == 2)
            d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpx1,
                d_all_rho, d_all_fz, type2, M);
        
    }// j=0:Dim
    
    /*cudaMemcpy(tmp, d_all_fx, M * sizeof(float), cudaMemcpyDeviceToHost);
    write_grid_data("fx0.dat", tmp);
    cudaMemcpy(tmp, d_all_fy, M * sizeof(float), cudaMemcpyDeviceToHost);
    write_grid_data("fy0.dat", tmp);
    cuda_collect_rho();
    for (int i = 0; i < ntypes; i++) {
        char nm[30];
        sprintf(nm, "rho%d.dat", i);
        write_grid_data(nm, Components[i].rho);
    }
    if ( type1 == 0 && type2 == 1 ) 
        exit(1);*/
}



// Calculates the energy involved in this potential as
// energy = \int dr rho1(r) \int dr' u(r-r') rho2(r')
// The convolution theorem is used to efficiently evaluate
// the integral over r'
float Potential::CalcEnergy() {

    // fft rho2
    d_prepareDensity<<<M_Grid, M_Block>>>(type2, d_all_rho, d_cpx1, M);
    cufftExecC2C(fftplan, d_cpx1, d_cpx2, CUFFT_FORWARD);

    // Multiply by d_u_k
    d_multiplyComplex<<<M_Grid, M_Block>>>(this->d_u_k, d_cpx2,
        d_cpx1, M);

    // Back to real space
    cufftExecC2C(fftplan, d_cpx1, d_cpx1, CUFFT_INVERSE);

    d_prepareIntegrand<<<M_Grid, M_Block>>>(d_cpx1, type1, d_all_rho,
        d_tmp, M);

    // Copy the integrand back to host for integration
    // This should be replaced with on-device integration at some point
    cudaMemcpy(tmp, d_tmp, M * sizeof(float), cudaMemcpyDeviceToHost);
    this->energy = integ_trapPBC(tmp);

    //float temp = reduce_device_float(d_tmp, threads, M);
    //cout << "host: " << this->energy << " device: " << temp << endl;
    //die("here!");



    return this->energy;
    
}

void Potential::CalcVirial()
{
    // fft rho2
    d_prepareDensity<<<M_Grid, M_Block>>>(this->type2, d_all_rho, d_cpx1, M);
    cufftExecC2C(fftplan, d_cpx1, d_cpx1, CUFFT_FORWARD);
    d_divideByDimension<<<M_Grid, M_Block>>>(d_cpx1, M);


    for (int j = 0; j < n_P_comps; j++) {
        d_prepareVirial<<<M_Grid, M_Block>>>(this->d_vir_k, d_cpx1, d_cpx2,
            j, Dim, n_P_comps, M);

        // back to real space, in-place transform
        cufftExecC2C(fftplan, d_cpx2, d_cpx2, CUFFT_INVERSE);

        d_prepareIntegrand<<<M_Grid, M_Block>>>(d_cpx2, this->type1, d_all_rho,
            d_tmp, M);

        // Copy the integrand back to host for integration
        // This should be replaced with on-device integration at some point
        cudaMemcpy(tmp, d_tmp, M * sizeof(float), cudaMemcpyDeviceToHost);

        this->total_vir[j] = integ_trapPBC(tmp) / V / float(Dim);
    }

}

void Potential::InitializeVirial()
{
    
    d_initVirial<<<M_Grid, M_Block>>>(this->d_vir, this->d_f,
        d_L, d_Lh, d_dx, Dim, n_P_comps, d_Nx, M);


    for (int j = 0; j < n_P_comps; j++) {
        d_extractVirialCompR2C<<<M_Grid, M_Block>>>(d_cpx1, this->d_vir, j, n_P_comps, M);
        cufftExecC2C(fftplan, d_cpx1, d_cpx1, CUFFT_FORWARD);
        d_divideByDimension<<<M_Grid, M_Block>>>(d_cpx1, M);
        d_insertVirialCompC2C<<<M_Grid, M_Block>>>(this->d_vir_k, d_cpx1, j, n_P_comps, M);
    }

}



Potential::~Potential() {
    //printf("PS here for some reason!\n"); fflush(stdout);
    if (allocated){
        int n_off_diag = Dim + (Dim * Dim - Dim) / 2;
        free(this->u);
        free(this->u_k);
    
        for (int i = 0; i < Dim; i++) {
            free(this->f[i]);
            free(this->f_k[i]);
        }

        for (int i = 0; i < n_off_diag; i++) {
            free(this->vir[i]);
            free(this->vir_k[i]);
	    }
        free(this->f);
        free(this->f_k);
        free(this->vir);
        free(this->vir_k);
    }

}

Potential::Potential() {
    return;
}

Potential::Potential(istringstream &iss) {
    input_command = iss.str();
    return;
}

void Potential::Update() {
    if (!ramp) return;
    if (equil) return;

	double Ao = initial_prefactor + (final_prefactor-initial_prefactor)/double(prod_steps) * double(step+1);

    d_multiply_cufftCpx_scalar << <M_Grid, M_Block >> > (this->d_master_u_k, Ao, this->d_u_k , M);
    d_multiply_cufftCpx_scalar << <M_Grid * Dim, M_Block >> > (this->d_master_f_k, Ao, this->d_f_k , M * Dim);

}

void Potential::ramp_check_input(istringstream& iss){

    if (iss.fail()){
        die("Error during input script; failed to properly read:\n" + iss.str());
    }

    string convert;
    iss >> convert;

    if (!iss.fail()){
        if (convert == "ramp") {
            ramp = true;
            iss >> final_prefactor;
            if(iss.fail()) die("no final prefactor specified");
            cout << "Ramping prefactor of " <<potential_type<< " style, between types " << type1+1 << " and " \
                << type2 + 1 << " from " << initial_prefactor \
                << " to " << final_prefactor << endl;

            cout << "Estimated per time step change: " << \
                (final_prefactor - initial_prefactor) / (prod_steps)
                << endl;

        }
        else 
            die("Invalid keyword: " + convert);
    }


}

void Potential::check_types(){
	if (type1 >= ntypes ||
		type2 >= ntypes)
		die("Invalid particle type in " + potential_type + "  potential input!");
}

// This function will keep track of which types are involved in the
// potential and will Fourrier transform them before each integration step 
// to save some computation time.
void Potential::determine_types_to_fft(){
    static std::vector<int> types_to_fft(0);
        for (auto& Iter: Potentials){
            if (Iter->potential_type != "Charges" || Iter->potential_type != "MaierSaupe"){
                if (std::find(types_to_fft.begin(), types_to_fft.end(), Iter->type1) == types_to_fft.end()) 
                    types_to_fft.push_back(Iter->type1);
                if (std::find(types_to_fft.begin(), types_to_fft.end(), Iter->type2) == types_to_fft.end())
                    types_to_fft.push_back(Iter->type2);
        }
    }
    
}

Potential* PotentialFactory(istringstream &iss){
	string s1;
	iss >> s1;
    transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
	if (s1 == "erf"){
		return new Erf(iss);
	}
	if (s1 == "gaussian"){
		return new Gaussian(iss);
	}
	if (s1 == "gaussian_erf"){
		return new GaussianErf(iss);
	}
	if (s1 == "fieldphase"){
		return new FieldPhase(iss);
	}
	if (s1 == "maiersaupe"){
		return new MaierSaupe(iss);
	}
	if (s1 == "charges"){
		return new Charges(iss);
	}
	
	die("Unsupported potential");
	return 0;
}




// // Calculates forces on rho1, rho2 for this pairstyle
// void Potential::CalcForces() {

//     /////////////////////////
//     // rho2 acting on rho1 //
//     /////////////////////////
//  //cudaDeviceSynchronize();
//     // fft rho2
//     d_prepareDensity<<<M_Grid, M_Block>>>(type2, d_all_rho, d_cpx1, M);

//     check_cudaError("d_prepareDensity");

//  //cudaDeviceSynchronize();
//     cufftExecC2C(fftplan, d_cpx1, d_cpx2, CUFFT_FORWARD);
//     check_cudaError("cufftExec1");

//      //cudaDeviceSynchronize();
//     // for (int j = 0; j < Dim; j++) {
//     //     // d_cpx1 = d_cpx2 * d_f_k
//     //     d_prepareForceKSpace<<<M_Grid, M_Block>>>(this->d_f_k, 
//     //         d_cpx2, d_cpx1, j, Dim, M);

//     //     check_cudaError("d_prepareForceKSpace");
//     //      //cudaDeviceSynchronize();
//     //     // back to real space, in-place transform
//     //     cufftExecC2C(fftplan, d_cpx1, d_cpx1, CUFFT_INVERSE);
//     //     //cudaDeviceSynchronize();

//     //     check_cudaError("cufftExec1");
        
        
//     //     // Accumulate the forces on type 1
//     //     if (j == 0)
//     //         d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpx1, 
//     //             d_all_rho, d_all_fx, type1, M);
//     //     if (j == 1)
//     //         d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpx1,
//     //             d_all_rho, d_all_fy, type1, M);
//     //     if (j == 2)
//     //         d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpx1,
//     //             d_all_rho, d_all_fz, type1, M);
     	 	
//     //     check_cudaError("d_accumulateGridForce");
        
//     // }


//     d_prepareForceKSpace<<<M_Grid, M_Block>>>(this->d_f_k, d_cpx2, d_cpxx, d_cpxy, d_cpxz, Dim, M);

//     check_cudaError("d_prepareForceKSpace");
//         //cudaDeviceSynchronize();
//     // back to real space, in-place transform
//     cufftExecC2C(fftplan, d_cpxx, d_cpxx, CUFFT_INVERSE);
//     cufftExecC2C(fftplan, d_cpxy, d_cpxy, CUFFT_INVERSE);
//     cufftExecC2C(fftplan, d_cpxz, d_cpxz, CUFFT_INVERSE);
//     //cudaDeviceSynchronize();

//     check_cudaError("cufftExec1");
    
    
//     // Accumulate the forces on type 1

//     d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpxx, 
//         d_all_rho, d_all_fx, type1, M);

//     d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpxy,
//         d_all_rho, d_all_fy, type1, M);
//     if (Dim == 3){
//         d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpxz,
//             d_all_rho, d_all_fz, type1, M);
//     }    	 	
//     check_cudaError("d_accumulateGridForce");


//     // fft rho1
//     d_prepareDensity<<<M_Grid, M_Block>>> (type1, d_all_rho, d_cpx1, M);
//     cufftExecC2C(fftplan, d_cpx1, d_cpx2, CUFFT_FORWARD);

//     check_cudaError("cufftExec");

    

//     d_prepareForceKSpace<<<M_Grid, M_Block>>>(this->d_f_k, d_cpx2, d_cpxx, d_cpxy, d_cpxz, Dim, M);

//     check_cudaError("d_prepareForceKSpace");
//         //cudaDeviceSynchronize();
//     // back to real space, in-place transform
//     cufftExecC2C(fftplan, d_cpxx, d_cpxx, CUFFT_INVERSE);
//     cufftExecC2C(fftplan, d_cpxy, d_cpxy, CUFFT_INVERSE);
//     cufftExecC2C(fftplan, d_cpxz, d_cpxz, CUFFT_INVERSE);
//     //cudaDeviceSynchronize();

//     check_cudaError("cufftExec1");
    
    
//     // Accumulate the forces on type 1

//     d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpxx, 
//         d_all_rho, d_all_fx, type1, M);

//     d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpxy,
//         d_all_rho, d_all_fy, type1, M);
//     if (Dim == 3){
//         d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpxz,
//             d_all_rho, d_all_fz, type1, M);
//     }    	 	
//     check_cudaError("d_accumulateGridForce");


// }

