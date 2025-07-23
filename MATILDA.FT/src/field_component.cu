// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "field_component.h"
#include "device_launch_parameters.h"

void FieldComponent::ZeroGradient() {

    int i, j;
    for (j = 0; j < Dim; j++)
        for (i = 0; i < M; i++)
            this->force[j][i] = 0.0;
}



FieldComponent::FieldComponent() {
    int alloc_size = M;
    this->rho = (float*)calloc(alloc_size, sizeof(float));
    this->force = (float**)calloc(Dim, sizeof(float*));

    cudaMalloc(&this->d_rho, alloc_size * sizeof(float));
    cudaMalloc(&this->d_force, Dim*alloc_size * sizeof(float));
    
    device_mem_use += alloc_size * (Dim + 1) * sizeof(float);

    for (int j = 0; j < Dim; j++) {
        this->force[j] = (float*)calloc(alloc_size, sizeof(float));
    }

    this->ZeroGradient();
}

FieldComponent::~FieldComponent() {
}