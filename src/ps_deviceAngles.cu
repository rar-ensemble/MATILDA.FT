// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


// Routine to calculate the forces due to angle potentials. 
// Routine is parallelized over particles, and only accumulates forces
// on particle "ind" 
__global__ void d_angles(
    float* f,                       // [ns*Dim] particle forces
    const float* x,                 // [ns*Dim] particle positions
    const float* angle_k,           // [nangle_types] force constants
    const float* angle_theta_eq,    // [nangle_types] equilibrium angle
    const int* angle_style,         // [nangle_types] angle style flag (0=WLC, 1=Harmonic)
    const int* n_angles,            // [ns] Number of angles for each particle
    const int* angle_type,          // [ns * MAX_ANGLES] angle potential style
    const int* angleGroup,          // [ns * 3*MAX_ANGLES] first index of particles involved
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

            aind = ind * MAX_ANGLES * 3 + i * 3;
            int ai = angleGroup[aind+0];
            int aj = angleGroup[aind+1];
            int ak = angleGroup[aind+2];

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




// Routine to calculate the enrgy and virial due to angle potentials. 
// Routine is parallelized over particles. 
__global__ void d_anglesStressEnergy(
    float* d_e,                     // [ns] energy of each particle
    float* d_vir,                   // [ns*n_P_comps] pressure tensor terms
    const float* x,                 // [ns*Dim] particle positions
    const float* angle_k,           // [nangle_types] force constants
    const float* angle_theta_eq,    // [nangle_types] equilibrium angle
    const int* angle_style,         // [nangle_types] angle style flag (0=WLC, 1=Harmonic)
    const int* n_angles,            // [ns] Number of angles for each particle
    const int* angle_type,          // [ns * MAX_ANGLES] angle potential style
    const int* angleGroup,          // [ns * 3*MAX_ANGLES] first index of particles involved
    const float* L, const float* Lh,// [Dim]
    const int ns, const int MAX_ANGLES, const int n_P_comps, const int Dim ) {

        const int ind = blockIdx.x * blockDim.x + threadIdx.x;
        if ( ind >= ns )
            return;

        d_e[ind] = 0.0f;
        for ( int i=0 ; i<n_P_comps ; i++ ) 
            d_vir[ind * n_P_comps + i] = 0.0f;

        for ( int i=0 ; i < n_angles[ind] ; i++ ) {
            int aind = ind * MAX_ANGLES + i;        // stacked index to access ns*MAX_ANGLES arrays

            int atp = angle_type[aind];

            aind = ind * MAX_ANGLES * 3 + i * 3;
            int ai = angleGroup[aind+0];
            int aj = angleGroup[aind+1];
            int ak = angleGroup[aind+2];

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

              d_e[ind] += kang * ( 1.0 + cos_th );

              // Define forces on i, k
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


              d_e[ind] += angle_k[atp] * dtheta * dtheta;
              

              for ( int j=0 ; j<Dim ; j++ ) {
                  fi[j] = du_mag * iSinTheta * ( rkj[j] / mrij / mrkj - cos_th * rij[j] / mrij2 );
                  fk[j] = du_mag * iSinTheta * ( rij[j] / mrij / mrkj - cos_th * rkj[j] / mrkj2 );  
              }

            }// Harmonic angle potential forces


            d_vir[ind * n_P_comps + 0] += fi[0] * rij[0] + fk[0] * rkj[0];
            d_vir[ind * n_P_comps + 1] += fi[1] * rij[1] + fk[1] * rkj[1];
            if (Dim == 2)
                d_vir[ind * n_P_comps + 2] += fi[0] * rij[1] + fk[0] * rkj[1];
            else if (Dim == 3) {
                d_vir[ind * n_P_comps + 2] += fi[2] * rij[2] + fk[2] * rkj[2];
                d_vir[ind * n_P_comps + 3] += fi[0] * rij[1] + fk[0] * rkj[1];
                d_vir[ind * n_P_comps + 4] += fi[0] * rij[2] + fk[0] * rkj[2];
                d_vir[ind * n_P_comps + 5] += fi[1] * rij[2] + fk[1] * rkj[2];
            }


        }// i=0:n_angles[ind]
    

}
