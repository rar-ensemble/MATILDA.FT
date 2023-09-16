// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"

// Calcuates and returns total energy of angles of molecule ``molec''
float moleculeAngleEnergy(int molec) {

    float mAE = 0.0f;

    for (int i = 0; i < ns; i++) {
        if ( molecID[i] != molec )
            continue;

        for (int j = 0; j < n_angles[i]; j++) {

            if (i != angle_first[i][j])
                continue;

            int atype = angle_type[i][j];

            int im = angle_mid[i][j];
            int ie = angle_end[i][j];

            float rij[3];
            float mrij2 = pbc_mdr2(x[i], x[im], rij);
            float mrij = sqrt(mrij2);

            float rkj[3];
            float mrkj2 = pbc_mdr2(x[ie], x[im], rkj);
            float mrkj = sqrt(mrkj2);

            float dot = 0.0f;
            for (int k = 0; k < Dim; k++)
                dot += rij[k] * rkj[k];

            float cos_th = dot / mrij / mrkj;

            // Accumulate WLC angle energy
            if ( angleIntStyle[atype] == 0 )
                mAE += angle_k[atype] * (1.0f + cos_th);

            else if ( angleIntStyle[atype] == 1 ) {
                float theta = acosf(cos_th);
                float dtheta = ( theta - angle_theta_eq[atype] );
                mAE += angle_k[atype] * dtheta * dtheta;
            }
        }// j=0 ; j<n_angles[i]
    }// i=0 ; i<ns

    return mAE;
}


void host_angles() {

    Uangle = 0.0f;
    for (int i = 0; i < n_P_comps; i++)
        angleVir[i] = 0.0;

    for (int i = 0; i < ns; i++) {
        for (int j = 0; j < n_angles[i]; j++) {

            if (i != angle_first[i][j])
                continue;

            int atype = angle_type[i][j];

            int im = angle_mid[i][j];
            int ie = angle_end[i][j];

            float rij[3];
            float mrij2 = pbc_mdr2(x[i], x[im], rij);
            float mrij = sqrt(mrij2);

            float rkj[3];
            float mrkj2 = pbc_mdr2(x[ie], x[im], rkj);
            float mrkj = sqrt(mrkj2);

            float dot = 0.0f;
            for (int k = 0; k < Dim; k++)
                dot += rij[k] * rkj[k];

            float cos_th = dot / mrij / mrkj;

            float local_Uangle = 0.f;

            // Accumulate WLC angle energy
            if ( angleIntStyle[atype] == 0 )
                local_Uangle = angle_k[atype] * (1.0f + cos_th);

            else if ( angleIntStyle[atype] == 1 ) {
                float theta = acosf(cos_th);
                float dtheta = ( theta - angle_theta_eq[atype] );
                local_Uangle = angle_k[atype] * dtheta * dtheta;
            }

            Uangle += local_Uangle;

            molec_Ubond[ molecID[i] ] += local_Uangle;

            float fi[3], fk[3];
            for (int k = 0; k < Dim; k++) {
                fi[k] = 2.0 * angle_k[atype] * (cos_th * rij[k] / mrij2
                    - rkj[k] / mrij / mrkj);
                fk[k] = 2.0 * angle_k[atype] * (cos_th * rkj[k] / mrkj2
                    - rij[k] / mrij / mrkj);
            }

            angleVir[0] += fi[0] * rij[0] + fk[0] * rkj[0];
            angleVir[1] += fi[1] * rij[1] + fk[1] * rkj[1];
            if (Dim == 2)
                angleVir[2] += fi[0] * rij[1] + fk[0] * rkj[1];
            else if (Dim == 3) {
                angleVir[2] += fi[2] * rij[2] + fk[2] * rkj[2];
                angleVir[3] += fi[0] * rij[1] + fk[0] * rkj[1];
                angleVir[4] += fi[0] * rij[2] + fk[0] * rkj[2];
                angleVir[5] += fi[1] * rij[2] + fk[1] * rkj[2];
            }
        }
    } // i=0:ns

    for (int i = 0; i < n_P_comps; i++)
        angleVir[i] *= 1.0 / V / float(Dim);
}