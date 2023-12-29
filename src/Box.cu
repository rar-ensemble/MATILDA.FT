// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Box.h"
#include "FTS_Box.h"

void die(const char*);

__global__ void init_devCuRand(unsigned int, curandState*, int);

Box::Box() {}
Box::~Box() {}
Box::Box(std::istringstream& iss) {
    input_command = iss.str();
}


// Sets the dimensionality of the box and resizes Nx, L, Lh vectors
void Box::setDimension(int d) { 
    Dim = d;

    Nx.resize(Dim);
    d_Nx.resize(Dim);

    dx.resize(Dim);

    L.resize(Dim);
    d_L.resize(Dim);

    Lh.resize(Dim);
    d_Lh.resize(Dim);
}

int Box::returnDimension() {
    return Dim;
}

// This routine assumes all received data is device thrust vectors
// Typical use case for this is to convolve density fields and
// potential fields with smearing functions, so it is expected that
// the smear function will be stored in k-space.
void Box::convolveTComplexDouble(
    thrust::device_vector<thrust::complex<double>> input_r,     // Input data, assumed in real-space to convolved
    thrust::device_vector<thrust::complex<double>> &dest_r,     // destiny array to store result
    thrust::device_vector<thrust::complex<double>> convFunc_k)  // Convolution function, assumed stored in k-space
    {

        // Take input to k-space
        cufftWrapperDouble(input_r, dest_r, 1);

        // Affect convolution in k-space
        thrust::transform(dest_r.begin(), dest_r.end(), convFunc_k.begin(),
            dest_r.begin(), thrust::multiplies<thrust::complex<double>>());        

        // Back to real space
        cufftWrapperDouble(dest_r, dest_r, -1);

}

void Box::cufftWrapperDouble(
    thrust::device_vector<thrust::complex<double>> in,
    thrust::device_vector<thrust::complex<double>> &out,
    const int fftDir)      // fftDir = 1 for forward, -1 for backwards FFT
    {

    int startTime = time(0);
	
    cuDoubleComplex* _in = (cuDoubleComplex*)thrust::raw_pointer_cast(in.data());
    cuDoubleComplex* _out = (cuDoubleComplex*)thrust::raw_pointer_cast(out.data());

    if ( fftDir == 1 ) {
        cufftExecZ2Z(fftplan, _in, _out, CUFFT_FORWARD);

        // Normalize the FT
        thrust::device_vector<thrust::complex<double>> norm(M);
        thrust::fill(norm.begin(), norm.end(), 1.0/double(M));
        
        thrust::transform(out.begin(), out.end(), norm.begin(), out.begin(), 
            thrust::multiplies<thrust::complex<double>>());
    }

    else if ( fftDir == -1 ) {
        cufftExecZ2Z(fftplan, _in, _out, CUFFT_INVERSE);        
    }


    ftTimer += time(0) - startTime;
}

// Receives index id in [0 , M ) and makes array
// nn[Dim] in [ 0 , Nx[Dim] )
void Box::unstack2(int id, int* nn) {

    if (Dim == 1) {
        nn[0] = id;
        return;
    }
    else if (Dim == 2) {
        nn[1] = id / Nx[0];
        nn[0] = (id - nn[1] * Nx[0]);
        return;
    }
    else if (Dim == 3) {
        nn[2] = id / Nx[1] / Nx[0];
        nn[1] = id / Nx[0] - nn[2] * Nx[1];
        nn[0] = id - (nn[1] + nn[2] * Nx[1]) * Nx[0];
        return;
    }
    else {
        die("Bad dimension in unstack2");
        return;
    }
}

// For a given id \in [0,M), defines Fourier vector k
// Returns |k|^2
double Box::get_kD(int id, double* k) {
    // id between 0 and M-1 (i value in loop), float k kx,ky,ky (that vector), Dim is dimensionality
    // declare a vector for this
    double kmag = 0.0f;
    int i, *n;
    n = new int[Dim];

    this->unstack2(id, n);

    for (i = 0; i < Dim; i++) {
        if (float(n[i]) < float(Nx[i]) / 2.)
            k[i] = PI2 * float(n[i]) / L[i];

        else
            k[i] = PI2 * float(n[i] - Nx[i]) / L[i];

        kmag += k[i] * k[i];
    }
    delete n;
    return kmag;

}

// For given id \in [0,M), defines position vector for that grid point
void Box::get_r(int id, double *r) {
    
    int i, *n;
    n = new int[Dim];

    this->unstack2(id, n);
   
    for (i = 0; i < Dim; i++) {
        r[i] = double(n[i]) * dx[i];
    }

    delete n;
}

// Initialize the CUDA RNG routine
void Box::initCuRand() { 
    if ( boxType == "fts" ) {
        cudaMalloc(&d_states, M * sizeof(curandState));
        init_devCuRand<<<M_Grid, M_Block>>>(RAND_SEED, d_states, M);
    }
}


std::string Box::printCommand() {
    return input_command;
}


__global__ void init_devCuRand(unsigned int seed, curandState* d_states, int MAX) {

    //check index for >= MAX
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= MAX)
		return;

	curand_init(seed, idx, 0, &d_states[idx]);

}


Box* BoxFactory(std::istringstream &iss) {
	std::string s1;
	iss >> s1;
	if ( s1 == "fts" ) {
        return new FTS_Box(iss);
    }

    else {
        std::string s2 = s1 + " is not a valid box style yet";
        die(s2.c_str());
    }
    return 0;
}

