// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"

using namespace std;

void unstack(int, int*);

void get_r(int id, float* r) {
    int i, * n = new int[Dim];

    unstack(id, n);

    for (i = 0; i < Dim; i++)
        r[i] = dx[i] * float(n[i]);
}




float get_k(int id, float* k, int Dim) {
    // id between 0 and M-1 (i value in loop), float k kx,ky,ky (that vecotor), Dim is dimensionality
    // declare a vector for this
    float kmag = 0.0f;
    int i, *n;
    n = new int[Dim];

    unstack(id, n);

    for (i = 0; i < Dim; i++) {
        if (float(n[i]) < float(Nx[i]) / 2.)
            k[i] = PI2 * float(n[i]) / L[i];

        else
            k[i] = PI2 * float(n[i] - Nx[i]) / L[i];

        kmag += k[i] * k[i];
    }
    delete [] n;
    return kmag;

}


// Receives index id in [0 , M ) and makes array
// nn[Dim] in [ 0 , Nx[Dim] )
void unstack(int id, int* nn) {

    if (Dim == 1) {
        nn[0] = id;
        return;
    }
    else if (Dim == 2) {
        nn[1] = id / Nx[0];
        nn[0] = (id - nn[1] * Nx[0]);
        return;
    }
    else if (Dim == 3) {
        nn[2] = id / Nx[1] / Nx[0];
        nn[1] = id / Nx[0] - nn[2] * Nx[1];
        nn[0] = id - (nn[1] + nn[2] * Nx[1]) * Nx[0];
        return;
    }
    else {
        cout << "Dim is goofy!" << endl;
        return;
    }
}


void unstack_like_device(int id, int* nn) {

    if (Dim == 1) {
        nn[0] = id;
        return;
    }
    else if (Dim == 2) {
        nn[1] = id / Nx[0];
        nn[0] = (id - nn[1] * Nx[0]);
        return;
    }
    else if (Dim == 3) {
        int idx = id;
        nn[2] = idx / (Nx[1] * Nx[0]);
        idx -= (nn[2] * Nx[0] * Nx[1]);
        nn[1] = idx / Nx[0];
        nn[0] = idx % Nx[0];
        return;
    }
    else {
        cout << "Dim is goofy!" << endl;
        return;
    }
}




