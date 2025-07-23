// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).



#include <cufft.h>
#include <cufftXt.h>

#if defined(FFT_SINGLE)
typedef cufftComplex FFT_COMPLEX;
#else 
typedef cufftDoubleComplex FFT_COMPLEX;
#endif
