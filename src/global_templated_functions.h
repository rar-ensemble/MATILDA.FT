// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include <vector>
#include <sstream>
#include <thrust/host_vector.h>
#ifndef HERE
#define HERE
template<typename T>
static void die(T msg){
	std::cout << msg << std::endl;
    exit(1);
};

/// Macro to read required parameters and report values or errors

#define readRequiredParameter(ss,parameter) readParameter(ss,parameter); //std::cout <<"Read parameter: " << #parameter; readParameter(ss, parameter);

template <typename T> 
static void readParameter(std::istringstream& ss, T& parameter){
    if(ss >> parameter){
        //std::cout <<": " << parameter << std::endl;
    }
    else{
        std::cout << " Invalid Value! Exiting..." << std::endl;
        exit(1);
    }
};


// Sends host vector h with 'size' elements to device array d
template <typename T>
static void sendThrustVectorToDeviceArray(thrust::host_vector<T> h, T* d, int size) {
    T* temp;
    temp = (T*) malloc( size * sizeof(T) );
    for ( int i=0 ; i<size ; i++ ) {
        temp[i] = h[i];
    }
    cudaMemcpy(d, temp, size * sizeof(T), cudaMemcpyHostToDevice);
    free(temp);
};

#endif