// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#define EIGEN_NO_CUDA
#include "globals.h"
#include "tensor_potential_MaierSaupe.h"
#include <iostream>
#include <sstream>
#include "device_utils.cuh"
#include "timing.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;

float MaierSaupe::CalculateMaxEigenValue(float* dim_dim_tensor)
{
    MatrixXf q_tensor(Dim, Dim);

    for ( int i=0 ; i<Dim ; i++ ) {
        for ( int j=0 ; j<Dim ; j++ ) {
            q_tensor(i,j) = dim_dim_tensor[i*Dim + j];
        }
    }

    // Calculate the eigenvalue
    EigenSolver<MatrixXf> es;

   
   return es.compute(q_tensor).eigenvalues().array().real().maxCoeff() * 1.5;
    
}