#include <complex>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include "mpi.h"
#include "fftw3-mpi.h"

using namespace std ;

#define PI   3.141592653589793238462643383

#define KDelta(i,j) ( i==j ? 1.0 : 0.0 ) 
#define pow2(x) ((x)*(x))
#define pow3(x) ((x)*(x)*(x))
#define pow4(x) ((x)*(x)*(x)*(x))
#define min(A,B) ((A)<(B) ? (A) : (B) )

#ifndef MAIN
extern
#endif
float dx[3], V, L[3], *all_rho, **rho ;

#ifndef MAIN
extern
#endif
int Nx[3], M, ntypes, Dim, NxL[3], ML, size, zstart ;


#ifndef MAIN
extern
#endif
complex<double> *ktmp2, *ktmp, I, **avg_sk ;




#ifndef MAIN
extern
#endif
fftw_complex *fmin0, *fmot0 ;

#ifndef MAIN
extern
#endif
fftw_plan fwd0, fbk0 ;

