#define MAIN
#include "globals.h"
void unstack(int, int*, int*, int);
void write_header( FILE *, char* , int , int* , int );
int fft_init(void);
void fftw_fwd(float*, complex<double>*);
void write_kspace_data(const char*, complex<double>*);
double get_k(int, double*);


int main( int argc, char** argv ) {

  if ( argc < 2 ) {
    cout << "Usage: dump-grid-dens [input.bin] [optional: n skipped frames]" << endl;
    exit(1);
  }
  int rt, skip, myrank, nprocs;

  MPI_Init( &argc, &argv);
  MPI_Comm_rank( MPI_COMM_WORLD, &myrank);
  MPI_Comm_size( MPI_COMM_WORLD, &nprocs);
  fftw_mpi_init();


  skip = -1 ;
  if ( argc == 3 )
    skip = atoi( argv[2] );

  FILE *inp, *otp;

  inp = fopen(argv[1], "rb");
  if ( inp == NULL ) {
    cout << "Failed to open " << argv[1] << endl;
    exit(1);
  }



  /////////////////
  // READ HEADER //
  /////////////////
  // Dim = dimensions
  rt = fread(&Dim, sizeof(int), 1, inp);
  // Nx- nodes in each dim
  rt = fread(Nx, sizeof(int), Dim, inp);
  // L Size of the box in each dim
  rt = fread(L, sizeof(float), Dim, inp);
  // Number of molecule types
  rt = fread(&ntypes, sizeof(int), 1,inp);

  // Print variables
  cout << "Dim = " << Dim << endl;
  cout << "Nx[0]: " << Nx[0] << " Nx[1]: " << Nx[1] ;
  if ( Dim == 3 ) cout << " Nx[2]: " << Nx[2];
  cout << endl;

  cout << "ntypes: " << ntypes << endl;

  // Check to see if initialized correctly
  if ( fft_init() ) {
    cout << "fft initialized!" << endl;
  }
  else {
    cout << "Failed to initialize fft!" << endl;
    exit(1);
  }

  // Get number of grid points
  M = 1;
  for ( int i=0 ; i<Dim ; i++ ) {
    M *= Nx[i];                     // Multiply grid points by number in each direction
    dx[i] = L[i] / float(Nx[i]);    // Space between each grid point (total length/ number of grid points)
  }
  if ( Dim == 2 )
    dx[2] = 1.0;
  cout << "M: " << M << " ML: " << ML << endl;

  all_rho = new float[ntypes*M];
  
  rho = new float*[ntypes];
  
  complex<double> **sk, **avg_sk ;
  sk = new complex<double>*[ntypes];
  avg_sk = new complex<double>*[ntypes];
  
  for ( int i=0 ; i<ntypes ; i++ ) { //Loop through each type of molecule
    rho[i] = new float[M];              // Density of molecule type i at all grid points M
    sk[i] = new complex<double>[M];     // Structure factor ...
    avg_sk[i] = new complex<double>[M];
    for ( int j=0 ; j<M ; j++ )
      avg_sk[i][j] = 0.0;
  }



  int nframes = 0, ncalc = 0;


  while ( !feof(inp) ) {
      // get density
    rt = fread( all_rho, sizeof(float), M*ntypes, inp ) ;

    if ( rt != M*ntypes ) {
      cout << "Successfully read " << nframes << " frames" << endl;
      break;
    }


    if ( nframes >= skip ) {
      for ( int i=0 ; i<ntypes ; i++ ) {
        for ( int j=0 ; j<M ; j++ )
          rho[i][j] = all_rho[i*M+j]; // pull out the denity from all rho for just molecule i
        
        fftw_fwd( rho[i], sk[i] ) ;         // fft the density to get strucutre factor

        for ( int j=0 ; j<M ; j++ ) 
          avg_sk[i][j] = sk[i][j];     ///// Add it to average s(k), as the not average gets overwritten each frame
      
        char nm[30];
        sprintf(nm, "sk%d_%d.sk", i ,ncalc);
        write_kspace_data(nm, avg_sk[i]);

      }
      ++ncalc;               // write the results                          // keep track of the number of s(k) that goes into the average
    }
    nframes++;
  }
  fclose(inp);

  return 0;

}



void write_kspace_data( const char *lbl , complex<double> *kdt ) {
  int i, j , nn[Dim] ;
  FILE *otp ;
  double kv[Dim], k2 ;

  otp = fopen( lbl , "w" ) ;

  for ( i=1 ; i<M ; i++ ) {
    unstack( i , nn, Nx, Dim ) ;

    k2 = get_k( i , kv ) ;

    for ( j=0 ; j<Dim ; j++ )
      fprintf( otp , "%lf " , kv[j] ) ;         // I think this is x,y,z coordinates of wave vector k in fourier space

    fprintf( otp , "%1.5e %1.5e %1.5e %1.5e\n" , abs(kdt[i]), sqrt(k2),     // structure factor at the drid point
        real(kdt[i]) , imag(kdt[i]) ) ;

    if ( Dim == 2 && nn[0] == Nx[0]-1 )
      fprintf( otp , "\n" ) ;
  }

  fclose( otp ) ;
}




void write_header( FILE *ot, char* nm, int ntypes, int* Nx, int Dim) {

  fprintf(ot, "TITLE = \"%s\"\n", nm );
  fprintf(ot, "VARIABLES = \"X\", \"Y\", \"Z\"");
  for ( int i=0 ; i<ntypes ; i++ ) 
    fprintf(ot, " \"rho%d\"", i);
  fprintf(ot,"\n");
  fprintf(ot, "ZONE I=%d, J=%d, K=%d, F=POINT\n", Nx[0], Nx[1], (Dim==3 ? Nx[2] : 0 ) );

}




void unstack( int id , int *nn , int *Nx , int Dim ) {

    if (Dim==1) {
        nn[0] = id;
        return;
    }
    else if (Dim==2) {
        nn[1] = id/Nx[0];
        nn[0] = (id - nn[1]*Nx[0]);
        return;
    }
    else if (Dim==3) {
        nn[2] = id/Nx[1]/Nx[0];
        nn[1] = id/Nx[0] - nn[2]*Nx[1];
        nn[0] = id - (nn[1] + nn[2]*Nx[1])*Nx[0];
    }
    else {
        cout << "Dim is goofy!" << endl;
        return;
    }
}
           
