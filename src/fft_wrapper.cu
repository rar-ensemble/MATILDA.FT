// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).



#include "fft_wrapper.h"
#include "Box.h"



void Box::init_fft_plan(cufftType type ){

#if defined(FFT_SINGLE)
  if (this->Dim == 2)
    cufftPlan2d(&fftplan, Nx[1], Nx[0], CUFFT_C2C);
  if (this->Dim == 3)
    cufftPlan3d(&fftplan, Nx[2], Nx[1], Nx[0], CUFFT_C2C);
#else
  if (this->Dim == 2)
    cufftPlan2d(&fftplan, Nx[1], Nx[0], CUFFT_Z2Z);
  if (this->Dim == 3)
    cufftPlan3d(&fftplan, Nx[2], Nx[1], Nx[0], CUFFT_Z2Z);
#endif
}

void Box::execute_fft(FFT_COMPLEX* fft_in, FFT_COMPLEX* fft_out, int direction){
#if defined(FFT_SINGLE)
  if (this->Dim == 2)
    cufftExecC2C(fftplan, fft_in, fft_out, direction);
  if (this->Dim == 3)
    cufftExecC2C(fftplan, fft_in, fft_out, direction);
#else
  if (this->Dim == 2)
    cufftExecZ2Z(fftplan, fft_in, fft_out, direction);
  if (this->Dim == 3)
    cufftExecZ2Z(fftplan, fft_in, fft_out, direction);

#endif
}