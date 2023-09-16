// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"


// Calculates the bonded energy for molecule ``molec''
float moleculeBondEnergy(int molec) {


  int i, j, id1, id2, btype;
  float mdr2, mdr, dr[3], delr;

  float mBE = 0.0f; // accumulator for molecule's bonded energy

  for (i = 0; i < ns; i++) {
    
    if ( molecID[i] != molec )
      continue;
    

    id1 = i;

    for (j = 0; j < n_bonds[id1]; j++) {
      id2 = bonded_to[id1][j];

      if (id2 < id1)
        continue;

      btype = bond_type[id1][j];
      
      mdr2 = pbc_mdr2(x[id1], x[id2], dr);
      mdr = sqrt(mdr2);
      delr = mdr - bond_req[btype];

      mBE += delr * delr * bond_k[btype] ;


    }//for ( j=0 ; j<n_bonds ;
  }// for ( i=0 ; i<ns_loc

  return mBE;
}

void host_bonds() {


  int i, j, k, id1, id2, btype;
  float mdr2, mdr, dr[3], delr, mf;

  Ubond = 0.0f;
  for (i = 0; i < n_P_comps; i++)
    bondVir[i] = 0.f;

  for (i = 0; i < ns; i++) {
    id1 = i;

    for (j = 0; j < n_bonds[id1]; j++) {
      id2 = bonded_to[id1][j];

      if (id2 < id1)
        continue;

      btype = bond_type[id1][j];
      
      mdr2 = pbc_mdr2(x[id1], x[id2], dr);
      mdr = sqrt(mdr2);
      delr = mdr - bond_req[btype];

      float local_Ubond = delr * delr * bond_k[btype];

      Ubond += local_Ubond;

      molec_Ubond[ molecID[id1] ] += local_Ubond;

      mf = 2.0f * bond_k[btype] * delr / mdr;
      for (k = 0; k < Dim; k++) {
        f[id1][k] -= mf * dr[k];
        f[id2][k] += mf * dr[k];
      }

      bondVir[0] += -mf * dr[0] * dr[0];
      bondVir[1] += -mf * dr[1] * dr[1];
      if ( Dim == 2 )
        bondVir[2] += -mf * dr[0] * dr[1];
      else if (Dim == 3)
      {
        bondVir[2] += -mf * dr[2] * dr[2];
        bondVir[3] += -mf * dr[0] * dr[1];
        bondVir[4] += -mf * dr[0] * dr[2];
        bondVir[5] += -mf * dr[1] * dr[2];
      }
    }//for ( j=0 ; j<n_bonds ;
  }// for ( i=0 ; i<ns_loc

  Udynamicbond = 0.0f;

  for (auto Iter: DynamicBonds){
    Iter->UpdateVirial();
  }

  for (i = 0; i < n_P_comps; i++)
    bondVir[i] *= 1.0 / V / float(Dim);


  // dynamic bonds
}
