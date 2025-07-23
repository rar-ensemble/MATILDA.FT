// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


/* ======================================================================== */
/* random.c                                                                 */
/*            Random number generator from uniform distribution [0,1]       */
/*             from Numerical Recipes                                       */
/* ======================================================================== */
#include <cmath>

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0 / IM1)
#define IMM1 (IM1 - 1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1 + IMM1 / NTAB)
#define EPS 1.2e-7
#define RNMX (1.0 - EPS)


double ran2 (void)
{
  int j;
  long int k;
  static long idum2 = 123456789;
  static long iy = 0;
  static long iv[NTAB];
  double temp;
  extern long idum;
  
  if (idum <= 0) {
    if (-(idum) < 1) idum = 1;
    else idum = -(idum);
    idum2 = (idum);
    for (j = NTAB + 7; j >=0; j--) {
      k = (idum) / IQ1;
      idum = IA1 * (idum - k * IQ1) - k * IR1;
      if (idum < 0) idum += IM1;
      if (j < NTAB) iv[j] = idum;
    }
    iy = iv[0];
  }
  
  k = (idum) / IQ1;
  idum = IA1 * (idum - k * IQ1) - k * IR1;
  if (idum < 0) idum += IM1;
  k = idum2 / IQ2;
  idum2 = IA2 * (idum2 - k * IQ2) - k * IR2;
  if (idum2 < 0) idum2 += IM2;
  j = iy / NDIV;
  iy = iv[j] - idum2;
  iv[j] = idum;
  if (iy < 1) iy += IMM1;
  if ((temp = AM * iy) > RNMX) return RNMX;
  else return temp;
}


void random_unit_vec( double *v ) {
  double xi[3], mxi, mxi2;
  int i;

  mxi2 = 23432.;

  while ( mxi2 > 1.0 ) {
    mxi2 = 0.0 ;
    for ( i=0 ; i<3 ; i++ ) {
      xi[i] = 1.0 - 2.0 * ran2() ;
      mxi2 += xi[i] * xi[i] ;
    }

  }

  mxi = sqrt( mxi2 );

  for ( i=0 ; i<3 ; i++ )
    v[i] = xi[i] / mxi ;
} 


double gasdev2() {
  static int iset = 0;
  static double gset;
  double fac, rsq, v1, v2;
  if(iset == 0) {
    do {
      v1=2.0*ran2()-1.0;
      v2=2.0*ran2()-1.0;
      rsq=v1*v1+v2*v2;
    } while (rsq >= 1.0 || rsq == 0.0);

    fac=sqrt(-2.0*log(rsq)/rsq);
    gset=v1*fac;
    iset=1;
    return v2*fac;

  } 
  else {
    iset=0;
    return gset;
  }
}

