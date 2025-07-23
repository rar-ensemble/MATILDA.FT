// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


__global__ void d_copyPositions(float*, float*, int, int);
void update_device_positions(float** out, float* d_target);
void send_3n_to_device(float** out, float *d_target);
void cuda_collect_x();
void cuda_collect_rho();
void cuda_collect_charge_density_field();
void cuda_collect_electric_field();
void cuda_collect_electrostatic_potential();
void cuda_collect_f();
void send_box_params_to_device();
