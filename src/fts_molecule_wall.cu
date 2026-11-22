// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "fts_molecule_wall.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <thrust/copy.h>
#include "FTS_Box.h"
#include "fts_species.h"
#include "fts_potential.h"
#include <sstream>
#include <string>
#include <iostream>
void die(const char*);

WallMolec::~WallMolec(){}

WallMolec::WallMolec(std::istringstream& iss, FTS_Box* p_box) : FTS_Molec(iss, p_box) {
 // iss comes into this routine have already passed "molecule" and "wall"

    int dim = mybox->returnDimension();
    int M = mybox->M;

    // Set default values
    xi = 0.1;
    size = 1.0;
    dir = 0;

    R.resize(dim, 0.0);
    d_R.resize(dim, 0.0);


    // Find species of the wall
    iss >> species;
    int_species = -1;
    for (int i = 0; i<mybox->Species.size(); i++ ) {
        if ( species == mybox->Species[i].fts_species ) {
            int_species = i;
        }
    }
    if ( int_species < 0 ) {
        die("wall species not found!");
    }



    // Read in wall style (planar, cylinder, etc)
    iss >> wall_style;
    if ( wall_style != "planar" ) {
        std::string last_words = wall_style + " is not a valid molecule wall style";
        die(last_words);
    }
    

    // Parse optoinal arguments    
    if ( iss.tellg() != -1 ) {
        std::string s1;
        iss >> s1;

        if ( s1 == "center" ) {
            iss >> R[dir] ;
            // if ( wall_style == "planar" )
        }
        else if ( s1 == "dir" ) {
            iss >> dir;
        }
        else if ( s1 == "xi" ) {
            iss >> xi;
        }
        else if ( s1 == "width") {
            iss >> size;
        }
        else {
            std::string last_words = s1 + " is not a valid wall molecule argument";
            die(last_words);
        }
    }// optional arguments

   
    //resize density arrays
    density.resize(M);
    d_density.resize(M);

    V_walltot = 0.0;

    double mdr2, r[dim], dr[dim], r0[dim], mdr;
    for ( int j=0 ; j<dim ; j++ ) r0[j] = R[j];
    double prefactor = mybox->Nr * mybox->C * 0.5;

    for ( int i=0 ; i<M ; i++ ) {
        mybox->get_r(i, r);

        // dr = r - r0
        mdr2 = mybox->pbc_dr2(dr, r, r0);

        if ( wall_style == "planar" ) {
            // Planar walls only care about | r - r0 | in [dir]-direction
            mdr = fabs(dr[dir]);
        }
        density[i] = prefactor * (1.0 - tanh( ( mdr - size ) / xi )  );
    }




    double gvol = 1.0;
    for ( int j=0 ; j<mybox->returnDimension() ; j++ ) {
        gvol *= mybox->L[j] / double(mybox->Nx[j]);
    }
    
    double psum = 0.0;
    for ( int i=0 ; i<mybox->M ; i++ ) {
        psum += density[i].real();
    }

    V_walltot = psum * gvol / mybox->Nr / mybox->C;
    

    mybox->Vfree = mybox->Vfree - V_walltot;
    
    if ( mybox->Vfree < 0.0 ) {
        die("moleculeHParticle made free volume < 0.0");
    }


    //transfer density to device
    d_density = density;

    // Also need to accumulate density onto the relevant species fields
    int is = int_species; // short species index
    thrust::transform(d_density.begin(), d_density.end(),  mybox->Species[is].d_density.begin(), mybox->Species[is].d_density.begin(), thrust::plus<thrust::complex<double>>());
}


void WallMolec::calcDensity() {
                
    int is = int_species; 
    
    // species[is].d_density += d_density
    thrust::transform(d_density.begin(), d_density.end(),  mybox->Species[is].d_density.begin(), mybox->Species[is].d_density.begin(), thrust::plus<thrust::complex<double>>());
    
} 


// here we will calculate the Hamiltonian term which incorporates the NP density
// - I * C * int (wpl(r) * ( - rhoNP(r)))

std::complex<double> WallMolec::calcHTerm() {

    
    int ip = -1;
    // find Helfand potential to get wpl
    for (int i = 0; i < mybox->Potentials.size(); i++ ) {
        std::string pstyle = mybox->Potentials[i]->printStyle();
        if ( pstyle == "Helfand" || pstyle == "Incompress" ) {
            ip = i;
        }
    }    
    if ( ip < 0 ) {
        std::string LW = "HParticle did not find a potential of type Helfand or Incompress";
        die(LW);
    }

    // Convert relevant fields to cuDoubleCpx
    cuDoubleComplex* _d_wpl = 
        (cuDoubleComplex*) thrust::raw_pointer_cast(mybox->Potentials[ip]->d_wpl.data());

    cuDoubleComplex* _d_nprho = (cuDoubleComplex*) thrust::raw_pointer_cast(d_density.data());
    
    cuDoubleComplex negativeI;
    negativeI.x = 0.0;
    negativeI.y = -1.0;

    // Gabe = -I * wpl * rho
    d_multiplyDoubleCpxByCpxByCpxScalar<<<mybox->M_Grid, mybox->M_Block>>>(mybox->d_cpxGabe,
        _d_wpl, _d_nprho, negativeI, mybox->M);

    cuDoubleComplex *htp;
    htp = (cuDoubleComplex*) malloc(mybox->M * sizeof(cuDoubleComplex));

    cudaMemcpy(htp, mybox->d_cpxGabe, mybox->M*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    
    Hterm = 0.0;
    for ( int i=0 ; i<mybox->M ; i++ ) {
        Hterm += std::complex<double>(htp[i].x, htp[i].y);
    }

    Hterm *= mybox->gvol;

    return Hterm;

} 


void WallMolec::modifyMolecule(std::istringstream& iss) {}

void WallMolec::computeLinearTerms() { }

void WallMolec::recomputeNmolecs() { }