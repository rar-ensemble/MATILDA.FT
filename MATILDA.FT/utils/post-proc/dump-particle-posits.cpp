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
void write_frame(FILE*, float**, int*, int*, float*, float*, int, int, int, int);

int main( int argc, char** argv ) {

  if ( argc < 3 ) {
    cout << "Usage: dump-grid-dens [input.bin] [output.lammpstrj] [optional: n skipped frames]" << endl;
    exit(1);
  }
  int Dim, *mID, *tp, ns, skip, rt, do_charges ;
  float L[3], **x, *xstack, *q;

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
  rt = fread(&ns, sizeof(int),1,inp);
  rt = fread(&Dim, sizeof(int), 1, inp);
  rt = fread(L, sizeof(float), 3, inp);

  mID = new int[ns];
  tp = new int[ns];

  x = (float**) calloc(ns, sizeof(float*));
  for ( int i=0 ; i<ns ; i++ ) 
    x[i] = (float*) calloc(Dim, sizeof(float));

  xstack = (float*) calloc(ns*Dim,sizeof(float));

  cout << "ns = " << ns << endl;
  cout << "Dim = " << Dim << endl;
  cout << "L: [" << L[0] << ", " << L[1] << ", " << L[2] << "]" << endl;

  rt = fread( tp, sizeof(int),ns,inp);
  rt = fread( mID, sizeof(int), ns, inp);


  // Check whether to read charges //
  rt = fread( &do_charges, sizeof(int), 1, inp);
  if ( do_charges ) {
    q = new float[ns];
    rt = fread(q, sizeof(float), ns, inp);
    cout << "Charges read!" << endl;
  }


  int nframes = 0;

  otp = fopen(argv[2], "w");
  if ( otp == NULL ) {
    cout << "Failed to open " << argv[2] << endl;
    exit(1);
  }


  while ( !feof(inp) ) {

    int nread = fread(xstack, sizeof(float), ns*Dim, inp);
    for ( int i=0 ; i<ns ; i++ ) {
      for ( int j=0 ; j<Dim ; j++ ) {
        x[i][j] = xstack[i*Dim+j] ;
        //float temp ;
        //rt = fread(&temp, sizeof(float),1,inp);
        //x[i][j] = temp;
        //nread++;
      }
    }


    if ( nread != ns*Dim ) {
      cout << "Successfully read " << nframes << " frames" << endl;
      break;
    }

    // cout << "here! " << nframes << " " << skip << "\n";
    if ( nframes >= skip ) {
      write_frame(otp, x, tp, mID, q, L, Dim, do_charges, ns, nframes);
    }

    nframes++;
  }
  fclose(inp);

  return 0;

}

void write_frame(FILE* otp, float** x, int *tp, int *molecID, float *q,
    float* L, int Dim, int do_charges, int ns, int fr) {

	int i, j;


	fprintf(otp, "ITEM: TIMESTEP\n%d\nITEM: NUMBER OF ATOMS\n%d\n", fr, ns);
	fprintf(otp, "ITEM: BOX BOUNDS pp pp pp\n");
	fprintf(otp, "%f %f\n%f %f\n%f %f\n", 0.f, L[0],
      0.f, L[1], 
      (Dim == 3 ? 0.f : 1.f), (Dim == 3 ? L[2] : 1.f));

  if ( do_charges )
  	fprintf(otp, "ITEM: ATOMS id type mol x y z q\n");
  else 
  	fprintf(otp, "ITEM: ATOMS id type mol x y z\n");

	for (i = 0; i < ns; i++) {
		fprintf(otp, "%d %d %d  ", i + 1, tp[i] + 1, molecID[i] + 1);
		for (j = 0; j < Dim; j++) {
			fprintf(otp, "%f ", x[i][j]);
    }

		for (j = Dim; j < 3; j++)
			fprintf(otp, "%f", 0.f);

    if ( do_charges )
      fprintf(otp, " %f", q[i] );

		fprintf(otp, "\n");

	}

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
           
