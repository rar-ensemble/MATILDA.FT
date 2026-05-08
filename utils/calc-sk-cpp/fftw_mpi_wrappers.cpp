#include "globals.h"

// Forward transform 
void fftw_fwd(float* in, complex<double>* out) {

  int i;

  // Store fft input (in the type that it deals with)- col 0 real, col 1 immaginary
  for (i=0; i<ML; i++) {
    fmin0[i][0] = double(in[i]);
    fmin0[i][1] = 0.0 ; // no immaginary part for what we have
  }
  // fmin0 (fm in 0), fmot0 (fm out 0) input and output variables (do not need to specify here)
  fftw_execute(fwd0);       //runs the fourier transform

  double norm = 1.0 / double(M);
  // Store fft output
  for (i=0; i<ML; i++)
      // real part + immarginary I * complex part normalized to number of grid points
    out[i] =( fmot0[i][0] + I * fmot0[i][1] ) * norm ;

}



// Backwards transform and normalization
void fftw_back(complex<double>* in, double* out ) {

  int i;
  // Store input
  for (i=0; i<ML; i++) {
    fmin0[i][0] = real(in[i]);
    fmin0[i][1] = imag(in[i]);
  }


  // Perform fft
  fftw_execute(fbk0);
 

  // Store output
  for (i=0; i<ML; i++)
    out[i] = fmot0[i][0] ;
  
}

  
int fft_init( ) {

  int i, b , total_alloced = 0 ;

  ptrdiff_t Dm = Dim, Nfp[Dim], NxLtp, ztp;
  for (i=0; i<Dim; i++)
    Nfp[i] = Nx[Dim-1-i];
 
  size = fftw_mpi_local_size_many(Dm, Nfp, 1, 0, MPI_COMM_WORLD,
            &NxLtp, &ztp );
 
  NxL[Dim-1] = NxLtp;
  for (i=0; i<Dim-1; i++)
    NxL[i] = Nx[i];
 
  zstart = ztp;
  fmin0 = (fftw_complex*) fftw_malloc( size * sizeof(fftw_complex) );
  fmot0 = (fftw_complex*) fftw_malloc( size * sizeof(fftw_complex) );
  
  if ( fmin0 == NULL ) { cout << "ERROR ALLOCATING fmin0!" << endl; }

  fwd0 = fftw_mpi_plan_dft(Dim, Nfp, fmin0, fmot0,
      MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE );
  fbk0 = fftw_mpi_plan_dft(Dim, Nfp, fmin0, fmot0,
      MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE );

  ML = 1;
  for (i=0; i<Dim; i++) {
    ML *= NxL[i];
  }
  
  total_alloced += size*sizeof(fftw_complex)*2 ;

  return total_alloced ;
}
