// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "tensor_potential.h"

void TensorPotential::Initialize_TensorPotential() {

    this->Tfield1 = (float*) calloc(Dim*Dim*M, sizeof(float));
    this->Tfield2 = (float*) calloc(Dim*Dim*M, sizeof(float));

    cudaMalloc(&this->d_Tfield1, Dim*Dim*M * sizeof(float));
    cudaMalloc(&this->d_Tfield2, Dim*Dim*M * sizeof(float));
    
    device_mem_use += M * (2*Dim*Dim) * sizeof(float);
    
}

TensorPotential::TensorPotential() {}

TensorPotential::~TensorPotential(){}
