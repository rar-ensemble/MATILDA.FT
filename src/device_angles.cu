// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "device_utils.cuh"

__global__ void d_angles(
    const float* x,                 // [ns*Dim] particle positions
    float* f,                       // [ns*Dim] particle forces
    const float* angle_k,           // [nangle_types] force constants
    const float* angle_theta_eq,    // [nangle_types] equilibrium angle
    const int* angle_style,         // [nangle_types] angle style flag (0=WLC, 1=Harmonic)
    const int* n_angles,            // [ns] Number of angles for each particle
    const int* angle_type,          // [ns * MAX_ANGLES] angle potential style
    const int* angle_first,         // [ns * MAX_ANGLES] first index of particles involved
    const int* angle_mid,           // [ns * MAX_ANGLES] second index 
    const int* angle_end,           // [ns * MAX_ANGLES] final index
    const float* L, const float* Lh,// [Dim]
    const int ns, const int MAX_ANGLES, const int Dim ) {

        const int ind = blockIdx.x * blockDim.x + threadIdx.x;
        if ( ind >= ns )
            return;

        float lforce[3];
        for ( int j=0; j<Dim ; j++ ) 
            lforce[j] = 0.0f;

        for ( int i=0 ; i < n_angles[ind] ; i++ ) {
            int aind = ind * MAX_ANGLES + i;        // stacked index to access ns*MAX_ANGLES arrays

            int atp = angle_type[aind];
            int ai = angle_first[aind];
            int aj = angle_mid[aind];
            int ak = angle_end[aind];

            float fi[3], fk[3];
            for ( int j=0 ; j<Dim ; j++ ) 
              fi[j] = fk[j] = 0.f;

            // Define vector connecting first to mid
            float rij[3];
            float mrij2 = 0.0f;
            for ( int j=0 ; j<Dim ; j++ ) {
                rij[j] = x[ai * Dim + j] - x[aj * Dim + j];
                if ( rij[j] > Lh[j] ) rij[j] -= L[j];
                else if ( rij[j] < -Lh[j] )  rij[j] += L[j];

                mrij2 += rij[j] * rij[j];
            }
            if ( mrij2 < 1.0E-4f)
                continue;
            float mrij = sqrtf(mrij2);

            // Vector connecting end to mid
            float rkj[3];
            float mrkj2 = 0.0f;
            for ( int j=0 ; j<Dim ; j++ ) {
                rkj[j] = x[ak * Dim + j] - x[aj * Dim + j];
                if ( rkj[j] > Lh[j] ) rkj[j] -= L[j];
                else if ( rkj[j] < -Lh[j] )  rkj[j] += L[j];

                mrkj2 += rkj[j] * rkj[j];
            }
            if ( mrkj2 < 1.0E-4f)
                continue;
            float mrkj = sqrtf(mrkj2);

            // Define angle between bonds
            float dot = 0.0f;
            for ( int k=0 ; k < Dim ; k++ )
                dot += rij[k] * rkj[k];
            float cos_th = dot / mrij / mrkj ;



            // WLC potential derivative and forces //
            if ( angle_style[atp] == 0 ) {
              // Derivative of cos(\theta_ijk) w.r.t. r_i 
              float DcosDri[3];
              for ( int j=0 ; j<Dim ; j++ )
                  DcosDri[j] = rkj[j] / mrij / mrkj - cos_th * rij[j] / mrij2;
              
              // Derivative of cos(\theta_ijk) w.r.t. r_k 
              float DcosDrk[3];
              for ( int j=0 ; j<Dim ; j++ )
                  DcosDrk[j] = rij[j] / mrkj / mrij - cos_th * rkj[j] / mrkj2;

              float kang = angle_k[atp];

              // Define forces on i, k
              // In the future, this will be angle-style dependent
              for ( int j=0 ; j<Dim ; j++ ) {
                  fi[j] = -kang * DcosDri[j];
                  fk[j] = -kang * DcosDrk[j];
              }
            }// angle_style[0] == 0 , WLC




            // Harmonic angle potential forces
            else if ( angle_style[atp] == 1 ) {
              if ( cos_th < -1.f || cos_th > 1.f )
                continue;

              float theta = acosf(cos_th);
              float denom = sqrtf(1.f - cos_th * cos_th);
              float iSinTheta = 0.f;
              if ( denom > 1.0E-4)
                iSinTheta = 1.0 / denom;

              float dtheta = theta - angle_theta_eq[atp];
              float du_mag = 2.f * angle_k[atp] * dtheta;

              for ( int j=0 ; j<Dim ; j++ ) {
                  fi[j] = du_mag * iSinTheta * ( rkj[j] / mrij / mrkj - cos_th * rij[j] / mrij2 );
                  fk[j] = du_mag * iSinTheta * ( rij[j] / mrij / mrkj - cos_th * rkj[j] / mrkj2 );
              }

            }// Harmonic angle potential forces





            // Accumulate the force based on which particle
            // has index "ind"
            for ( int j=0 ; j<Dim ; j++ ) {
                if ( ind == ai )
                    lforce[j] += fi[j];
                
                else if ( ind == aj ) 
                    lforce[j] -= ( fi[j] + fk[j] );

                else if ( ind == ak ) 
                    lforce[j] += fk[j];
            }

        }// i=0:n_angles[ind]
    

    // Finally accumulate the force on the 
    // global force array
    for ( int j=0 ; j<Dim ; j++ ) 
        f[ind*Dim+j] += lforce[j];
}
