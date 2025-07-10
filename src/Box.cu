// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Box.h"
#include "FTS_Box.h"
#include "PS_Box.h"

void die(const char*);
__global__ void d_multiplyCuComplexByFloat(cuComplex*, const float, const int);
__global__ void sumCpxDoubleArrayKernel(cuDoubleComplex*, cuDoubleComplex*, int);

Box::Box() {}
Box::~Box() {}
Box::Box(std::istringstream& iss) {
    input_command = iss.str();
    ftTimer = ioTimer = 0;
}


// Sets the dimensionality of the box and resizes Nx, L, Lh vectors
void Box::setDimension(int d) { 
    Dim = d;

    Nx.resize(Dim);
    d_Nx.resize(Dim);
    _d_Nx = (int*) thrust::raw_pointer_cast(d_Nx.data());

    dx.resize(Dim);
    d_dx.resize(Dim);
    _d_dx = (double*) thrust::raw_pointer_cast(d_dx.data());


    L = (float*) malloc(Dim*sizeof(float));
    Lh = (float*)malloc(Dim*sizeof(float));

    cudaMalloc(&d_L, Dim*sizeof(float));
    cudaMalloc(&d_Lh, Dim*sizeof(float));

}

int Box::returnDimension() {
    return Dim;
}

// Returns the simulation box style
// Currently only either "fts" or "ps"
std::string Box::returnBoxStyle() {
    return boxStyle;
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

// Takes cuComplex data structure and either FFT or inverse FFTs it
void Box::cufftWrapperSingle(
    cuComplex *in,          // [M] input data
    cuComplex *out,         // [M] output data
    const int fftDir)       // fftDir = 1 for forward, -1 for backwards FFT
    {

    int startTime = time(0);
	

    if ( fftDir == 1 ) {
        cufftExecC2C(fftplanSingle, in, out, CUFFT_FORWARD);
        d_multiplyCuComplexByFloat<<<M_Grid, M_Block>>>(out, 1.0/float(M), M);
    }

    else if ( fftDir == -1 ) {
        cufftExecC2C(fftplanSingle, in, out, CUFFT_INVERSE); 
        check_cudaError("inverse FFT");       
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
float Box::get_kD(int id, float* k) {
    // id between 0 and M-1 (i value in loop), float k kx,ky,ky (that vector), Dim is dimensionality
    // declare a vector for this
    float kmag = 0.0f;
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

// For given id \in [0,M), defines position vector for that grid point
void Box::get_rf(
    int id,     // grid index \in [0,M)
    float *r    // position array to be filled
    ) {
    
    int i, *n;
    n = new int[Dim];

    this->unstack2(id, n);
   
    for (i = 0; i < Dim; i++) {
        r[i] = float(n[i]) * dx[i];
    }

    delete n;
}


double Box::pbc_dr2(double* dr, const double* ri, const double* rj) {
    double mdr2 = 0.0;
    for ( int j=0 ; j<Dim ; j++ ) {
        dr[j] = ri[j] - rj[j];

        if ( dr[j] > Lh[j] ) dr[j] -= L[j];
        else if ( dr[j] < -Lh[j] ) dr[j] += L[j];

        mdr2 += dr[j] * dr[j];
    }

    return mdr2;
}

float Box::pbc_dr2(float* dr, const float* ri, const float* rj) {
    float mdr2 = 0.0;
    for ( int j=0 ; j<Dim ; j++ ) {
        dr[j] = ri[j] - rj[j];

        if ( dr[j] > Lh[j] ) dr[j] -= L[j];
        else if ( dr[j] < -Lh[j] ) dr[j] += L[j];

        mdr2 += dr[j] * dr[j];
    }

    return mdr2;
}



std::string Box::printCommand() {
    return input_command;
}

Box* BoxFactory(std::istringstream &iss) {
	std::string s1;
	iss >> s1;
	if ( s1 == "fts" ) {
        return new FTS_Box(iss);
    }

    else if ( s1 == "ps" ) {
        return new PS_Box(iss);
    }

    else {
        std::string s2 = s1 + " is not a valid box style yet";
        die(s2.c_str());
    }
    return 0;
}


std::complex<double> Box::sumCpxDoubleDeviceArray(
    cuDoubleComplex *d_dat,   // [N] array to be summed
    int blockSize,  // blockSize for CUDA calls
    int N           // array size
    ) {

    die("sumCpxDoubleDeviceArray needs further debugging");

    cuDoubleComplex *d_output;
    int numBlocks = (N + blockSize - 1) / blockSize;
    cuDoubleComplex *h_output;// = new float[numBlocks];
    h_output = (cuDoubleComplex*) malloc(numBlocks * sizeof(cuDoubleComplex));

    // Allocate device memory
    cudaMalloc(&d_output, numBlocks * sizeof(cuDoubleComplex));
    
    // Launch kernel
    sumCpxDoubleArrayKernel<<<numBlocks, blockSize, blockSize * sizeof(cuDoubleComplex)>>>(
        d_dat, d_output, N);

   
    check_cudaError("sumDevArray");
    cudaDeviceSynchronize();

    // Copy partial results back to host
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    
    // Sum up partial results on CPU
    std::complex<double> totalSum(0.0, 0.0);
    
    for(int i = 0; i < numBlocks; i++) {
        totalSum += std::complex<double>(h_output[i].x, h_output[i].y);
    }
    
    // Cleanup
    cudaFree(d_output);
    free(h_output);
    
    return totalSum;
}