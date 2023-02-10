// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Box.h"
#include "FTS_Box.h"

void die(const char*);

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


std::string Box::printCommand() {
    return input_command;
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
