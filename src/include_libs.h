// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#ifndef _INCLUDE_LIBS
#define _INCLUDE_LIBS

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <complex>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/complex.h>
#include "global_templated_functions.h"

void check_cudaError(const char*);
__global__ void d_floatToCpx(cuComplex*, const float*, const int);
__global__ void d_cpxToFloat(float*, const cuComplex*, const int);
__global__ void d_extractCpxDirToCpx(cuComplex*, const cuComplex*, const int, const int, const int);
__global__ void d_multiplyCpxDirByCpx(cuComplex*, const cuComplex*, const cuComplex*, 
        const int, const int, const int);
__global__ void d_cpxToFloatVecComponent(float*, const cuComplex*, const int, const int, const int);
__global__ void d_assignFloatVal(float*, const float, const int);
__global__ void d_floatPlusEqFloat(float*, const float*, const int);
__global__ void d_floatVecPlusEqFloatComp(float*, const float*, const int, const int, const int);
double ran2(void);

#endif
