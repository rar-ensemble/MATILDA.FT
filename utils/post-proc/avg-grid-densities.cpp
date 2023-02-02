#include <complex>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

using namespace std ;

#define PI   3.141592653589793238462643383
void unstack(int, int*, int*, int);
void write_header( FILE *, char* , int , int* , int );

int main( int argc, char** argv ) {

  if ( argc < 3 ) {
    cout << "Usage: dump-grid-dens [input.bin] [output\%d.tec] [optional: n skipped frames]" << endl;
    exit(1);
  }
  int Dim, Nx[3], ntypes, M, rt, skip ;
  float L[3], *all_rho, dx[3];

  skip = -1 ;
  if ( argc == 4 )
    skip = atoi( argv[3] );

  FILE *inp, *otp;

  inp = fopen(argv[1], "rb");
  if ( inp == NULL ) {
    cout << "Failed to open " << argv[1] << endl;
    exit(1);
  }



  /////////////////
  // READ HEADER //
  /////////////////
  rt = fread(&Dim, sizeof(int), 1, inp);
  rt = fread(Nx, sizeof(int), Dim, inp);
  rt = fread(L, sizeof(float), Dim, inp);
  rt = fread(&ntypes, sizeof(int), 1,inp);

  cout << "Dim = " << Dim << endl;
  cout << "Nx[0]: " << Nx[0] << endl;
  cout << "ntypes: " << ntypes << endl;

  M = 1;
  for ( int i=0 ; i<Dim ; i++ ) {
    M *= Nx[i];
    dx[i] = L[i] / float(Nx[i]);
  }
  if ( Dim == 2 )
    dx[2] = 1.0;

  all_rho = new float[ntypes*M];
  


  int nframes = 0;

  while ( !feof(inp) ) {

    rt = fread( all_rho, sizeof(float), M*ntypes, inp ) ;

    if ( rt != M*ntypes ) {
      cout << "Successfully read " << nframes << " frames" << endl;
      break;
    }

    char nm[50];
    sprintf(nm, "%s%05d.tec", argv[2], nframes);
    otp = fopen(nm, "w");
    if ( otp == NULL ) {
      cout << "Failed to open " << nm << endl;
      exit(1);
    }

    if ( nframes >= skip ) {
      write_header(otp, argv[1], ntypes, Nx, Dim);
 
      for ( int i=0 ; i<M ; i++ ) {
        int nn[3];
        unstack(i, nn, Nx, Dim);
        for ( int j=0 ; j<Dim ; j++ )
          fprintf( otp, "%1.3e ", float( nn[j] ) * dx[j] );
  
        for ( int j=0 ; j<ntypes ; j++ )
          fprintf( otp , "%1.3e ", all_rho[j*M + i] );
 
        fprintf(otp, "\n");
 
        if ( Dim == 2 && nn[0] == Nx[0]-1 )
          fprintf(otp, "\n");
      }
    }

    nframes++;
    fclose(otp);
  }
  fclose(inp);

  return 0;

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
           
