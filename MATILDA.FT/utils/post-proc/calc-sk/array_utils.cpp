#include "globals.h"
void add_segment( int ) ;
int unstack_stack( int ) ;
void unstack_local( int, int* ) ;
void unstack(int, int*);
int stack(int*) ;


int unstack_stack(int id ) {
  int n[3];
  unstack_local(id,n);
  n[Dim-1] += zstart ;
  return (stack(n));
}

// Receives index id in [ 0 , ML ] and turns it into
// // array nn[Dim]
 void unstack_local(int id, int nn[3] ) {

   if (Dim==1) {
     nn[0] = id;
     return;
   }
   else if (Dim==2) {
     nn[1] = id/NxL[0];
     nn[0] = (id - nn[1]*NxL[0]);
     return;
   }
   else if (Dim==3) {
     nn[2] = id/NxL[1]/NxL[0];
     nn[1] = id/NxL[0] - nn[2]*NxL[1];
     nn[0] = id - (nn[1] + nn[2]*NxL[1])*NxL[0];
   }
   else {
     cout << "Dim is goofy!" << endl;
     return;
   }
}

void field_gradient_cdif( double *inp , double *out , int dir ) {

  
  int i, j, nx2, nx, px, px2, nn[3], ntmp[3] ;
  
  for ( i=0 ; i<ML ; i++ ) {

    unstack( i , nn ) ;
    
    for ( j=0 ; j<Dim ; j++ )
      ntmp[j] = nn[j] ;

    
    ntmp[ dir ] = nn[ dir ] - 2 ;
    if ( ntmp[ dir ] < 0 ) ntmp[ dir ] += Nx[ dir ] ;
    nx2 = stack( ntmp ) ;

    ntmp[ dir ] = nn[ dir ] - 1 ;
    if ( ntmp[ dir ] < 0 ) ntmp[ dir ] += Nx[ dir ] ;
    nx = stack( ntmp ) ;


    ntmp[ dir ] = nn[ dir ] + 1 ;
    if ( ntmp[ dir ] >= Nx[ dir ] ) ntmp[ dir ] -= Nx[ dir ] ;
    px = stack( ntmp ) ;

    ntmp[ dir ] = nn[ dir ] + 2 ;
    if ( ntmp[ dir ] >= Nx[ dir ] ) ntmp[ dir ] -= Nx[ dir ] ;
    px2 = stack( ntmp ) ;

    out[i] =  ( inp[ nx2 ] - 8.0 * inp[ nx ] + 8.0 * inp[ px ] - inp[ px2 ] ) 
      / ( 12.0 * dx[ dir ] ) ;

  }
 

}



int stack_input(int x[3], int Nxx[3]) {
  if (Dim==1)
    return x[0];
  else if (Dim==2)
    return (x[0] + x[1]*Nxx[0]);
  else
    return  (x[0] + (x[1] + x[2]*Nxx[1])*Nxx[0] );
}





// Stacks vector x into 1D array index in [ 0, M ]
int stack( int x[3] ) { 
  if (Dim==1)
    return x[0];
  else if (Dim==2)
    return (x[0] + x[1]*Nx[0]);
  else
    return  (x[0] + (x[1] + x[2]*Nx[1])*Nx[0] );
}


void unstack(int id, int nn[3] ) {

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

void unstack_input(int id, int nn[3], int Nxx[3]) {

  if (Dim==1) {
    nn[0] = id;
    return;
  }
  else if (Dim==2) {
    nn[1] = id/Nxx[0];
    nn[0] = (id - nn[1]*Nxx[0]);
    return;
  }
  else if (Dim==3) {
    nn[2] = id/Nxx[1]/Nxx[0];
    nn[1] = id/Nxx[0] - nn[2]*Nxx[1];
    nn[0] = id - (nn[1] + nn[2]*Nxx[1])*Nxx[0];
  }
  else {
    cout << "Dim is goofy!" << endl;
    return;
  }
}



void get_r( int id , double r[3] ) {
  int i, id2, n[3];

  id2 = unstack_stack( id ) ;

  unstack(id2, n);


  for ( i=0; i<Dim; i++) 
    r[i] = dx[i] * double( n[i] );
}

double get_k_alias( int id , double k[3] ) {

  double kmag = 0.0;
  int i, id2, n[3] , j , has_nyquist = 0;
  for ( i=0 ; i<Dim ; i++ )
    if ( Nx[i] % 2 == 0 )
      has_nyquist = 1;

  id2 = unstack_stack( id ) ;

  unstack(id2, n);

  if ( Nx[0] % 2 == 0 && n[0] == Nx[0] / 2 )
    k[0] = 0.0 ;
  else if ( double(n[0]) < double(Nx[0]) / 2.)
   k[0] = 2*PI*double(n[0])/L[0];
  else
   k[0] = 2*PI*double(n[0]-Nx[0])/L[0];

  if (Dim>1) {
    if ( Nx[1] % 2 == 0 && n[1] == Nx[1] / 2 )
      k[1] = 0.0 ;
    else if ( double(n[1]) < double(Nx[1]) / 2.)
      k[1] = 2*PI*double(n[1])/L[1];
    else
      k[1] = 2*PI*double(n[1]-Nx[1])/L[1];
  }

  if (Dim==3) {
    if ( Nx[2] % 2 == 0 && n[2] == Nx[2] / 2 )
      k[2] = 0.0 ;
    else if ( double(n[2]) < double(Nx[2]) / 2.)
      k[2] = 2*PI*double(n[2])/L[2];
    else
      k[2] = 2*PI*double(n[2]-Nx[2])/L[2];
  }

  // Kills off the Nyquist modes
  if ( id2 != 0 && has_nyquist ) {
    for ( i=0 ; i<Dim ; i++ ) {
      if ( k[i] == 0.0 ) {
        for ( j=0 ; j<Dim ; j++ )
          k[j] = 0.0 ;
        kmag = 0.0;
        break;
      }
    }
  }
  
  for (i=0; i<Dim; i++)
    kmag += k[i]*k[i];

  return kmag;

}



// Receives index id in [ 0 , ML ] and returns 
// proper k-value, whether running in parallel or not
double get_k(int id, double k[3]) {

  double kmag = 0.0;
  int i, id2, n[3];

  id2 = unstack_stack( id ) ;

  unstack(id2, n);

  for ( i=0 ; i<Dim ; i++ ) {
    if ( double( n[i] ) < double( Nx[i] ) / 2. )
      k[i] = 2 * PI * double( n[i] ) / L[i] ;
    else
      k[i] = 2 * PI * double( n[i] - Nx[i] ) / L[i] ;

    kmag += k[i] * k[i] ;
  }

  return kmag;

}




int malloc2ddouble( double ***array, int n, int m ) {

  double *p = ( double* ) malloc( n*m*sizeof(double) ) ;
  if ( !p ) return -1 ;

  (*array) = (double**) malloc(n*sizeof(double*)) ;
  if ( !(*array) ) {
    free(p);
    return -1 ;
  }

  for ( int i=0 ; i<n ; i++ )
    (*array)[i] = &(p[i*m]) ;
 
  return 0 ;

}

int malloc2dint( int ***array, int n, int m ) {

  int *p = ( int* ) malloc( n*m*sizeof(int) ) ;
  if ( !p ) return -1 ;

  (*array) = (int**) malloc(n*sizeof(int*)) ;
  if ( !(*array) ) {
    free(p);
    return -1 ;
  }

  for ( int i=0 ; i<n ; i++ )
    (*array)[i] = &(p[i*m]) ;
 
  return 0 ;

}

