// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "fts_moleculeHParticle.h"
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

ParticleMolec::~ParticleMolec(){}

ParticleMolec::ParticleMolec(std::istringstream& iss, FTS_Box* p_box) : FTS_Molec(iss, p_box) {
 // iss comes into this routine have already passed "molecule" and "particle"
 // structure of iss should be: 
 // particleNum, int, total number of particles 
 // particleSpecies, char, should be a species defined earlier in input file
 // Rp, float, for particle 1
 // xi, float, for particle 1
 // center x, y, z positions, Dim floats, for particle 1. Should be a number between 0 and 1 which will be scaled to the box length. 
 // Repeat for number of particles

 //we are having the user optionally put in a file which has 
 // column 1 = particle number
 // column 2 = NP radius 
 // column 3 = NP xi 
 //column 4 = x center position 
 // column 5 = y center position 
 // column 6 = z center position 
  
 
 //Later we can implement a particleType for field-based or nanorods
 // Right now I'm just goin to do explicit NPs

    int dim = mybox->returnDimension();

    
    particleNum = nmolecs;
    nSites = particleNum;
    
    iss >> particleSpecies;
    // determine integer species
    intSpecies.resize(1);
    d_intSpecies.resize(1);
    for (int i = 0; i<mybox->Species.size(); i++ ) {
        if ( particleSpecies == mybox->Species[i].fts_species ) {
            intSpecies[0] = i;
        }
    }
    d_intSpecies = intSpecies;

    //resize density arrays
    density.resize(mybox->M);
    d_density.resize(mybox->M);
     
    //need to resize R, xi, center arrays based on particleNum
    Rp.resize(particleNum);
    d_Rp.resize(particleNum);

    xi.resize(particleNum);
    d_xi.resize(particleNum);

    center.resize(particleNum*3);
    
    d_center.resize(particleNum*3);
    
    Vnptot = 0; //zero total NP volume 
    //read input file 
    std::string s1;
    iss >> s1;
    if (s1 == "file") {
        iss >> s1;
    
        std::ifstream in2(s1);
    
        if (not in2.is_open()){
            std::cout << "File" << s1 << " does not exist."<<std::endl;
            die("");
        }
        // Store the contents into a vector of strings
        int npCounter = 0;
            
        std::string line;
        for ( int j=0 ; j<particleNum; j++ ) {
            std::getline(in2, line);
            std::istringstream iss(line);
            std::string word;
                std::vector<std::string> outputs;
            while (iss >> word) {
                outputs.push_back(word);
            }
            Rp[npCounter] = stof(outputs[1]);
            xi[npCounter] = stof(outputs[2]);
            center[3*npCounter] = stof(outputs[3]);
            center[(3*npCounter)+1] = stof(outputs[4]);
            center[(3*npCounter)+2] = stof(outputs[5]);
            npCounter += 1;
        }
    }

    Vnptot = 0.0;
    //loop to compute volume 
    for (int j=0; j<particleNum; j++ ) {
        // std::cout << "Particle Number " << j << std::endl;
        // std::cout << "R = " << Rp[j] << std::endl;
        // std::cout << "xi = " << xi[j] << std::endl;
        // std::cout << "x center = " << center[(3*j)] << std::endl;
        // std::cout << "y center = " << center[(3*j)+1] << std::endl;
        // std::cout << "z center = " << center[(3*j)+2] << std::endl;
        float Vnp = PI * Rp[j] * Rp[j];
        if ( dim == 3 ) {
            Vnp *= 4.0 * Rp[j] / 3.0;
        }
    
        Vnptot += Vnp; // sum volume of all particles        
    }
    // copy center positions, radii, xi to device (necessary?)
    d_center = center;
    d_Rp = Rp;
    d_xi = xi;
    
    //calculate particle volume fraction
    phiNP = Vnptot / mybox->V;
    
    
    thrust::fill(d_NPdensity.begin(), d_NPdensity.end(), 0.0);
    thrust::fill(density.begin(), density.end(), 0.0);
    // Loop over particles
    for (int j = 0; j < particleNum; j++ ) {

        // Loop over grid points
        for (int i=0; i < mybox->M; i++) {
            double r[mybox->returnDimension()];
            mybox->get_r(i, r); // gives position in each direction based on grid point
            double mdr2 = 0;
            double dr_abs;
            double dr[mybox->returnDimension()];
            // Loop over dimensions
            for (int k = 0; k < mybox->returnDimension(); k++) {
                //calculate distance from NP center
                dr[k] = center[(3*j)+k] - r[k];
                //take into account periodic boundaries
                if (dr[k] >= 0.5 * mybox-> L[k]) dr[k] -= mybox->L[k];
                else if (dr[k] < -0.5 * mybox -> L[k]) dr[k] += mybox->L[k];
                mdr2 += dr[k] * dr[k]; 
            }
            dr_abs = sqrt(mdr2);
            // density[i] += mybox->Nr * mybox->rho0 * 0.5 * erfc( ( dr_abs-Rp[j] ) / xi[j] );    
			density[i] += mybox->Nr * mybox->C * 0.5 * erfc( ( dr_abs-Rp[j] ) / xi[j] );    
            
        }
    }
    double gvol = 1.0;
    for ( int j=0 ; j<mybox->returnDimension() ; j++ ) {
        gvol *= mybox->L[j] / double(mybox->Nx[j]);
    }
    
    double psum = 0.0;
    for ( int i=0 ; i<mybox->M ; i++ ) {
        psum += density[i].real();
    }

    double NPvol = psum * gvol / mybox->Nr / mybox->C;
    
    // std::cout << "  Integrated NP density/Nr/C: " << NPvol << std::endl;
    // std::cout << "  HParticle updated mybox->Vfree from " << mybox->Vfree ;

    mybox->Vfree = mybox->Vfree - NPvol;
    // std::cout << " to " << mybox->Vfree << std::endl;

    if ( mybox->Vfree < 0.0 ) {
        die("moleculeHParticle made free volume < 0.0");
    }


    //transfer density to device
    d_NPdensity = density;
    int is = intSpecies[0];

    // Also need to accumulate density onto the relevant species fields
    thrust::transform(d_NPdensity.begin(), d_NPdensity.end(),  mybox->Species[is].d_density.begin(), mybox->Species[is].d_density.begin(), thrust::plus<thrust::complex<double>>());
}


void ParticleMolec::calcDensity() {
    d_density=d_NPdensity;            
     int is = intSpecies[0]; 
    thrust::transform(d_density.begin(), d_density.end(),  mybox->Species[is].d_density.begin(), mybox->Species[is].d_density.begin(), thrust::plus<thrust::complex<double>>());
    // calcHamiltonian();
} 


// here we will calculate the Hamiltonian term which incorporates the NP density
// - I * C * int (wpl(r) * ( - rhoNP(r)))

std::complex<double> ParticleMolec::calcHTerm() {
    // thrust::device_vector<thrust::complex<double>> dtmp(mybox->M);
    // thrust::complex<double> I(0.0, 1.0);
    // thrust::device_vector<thrust::complex<double>> dtmp2(mybox->M);

    // std::cout << "calcH 1" << std::endl;
    // //dtmp(r) = wpl(r) * (- rhoNP(r))

    // // first, create  - rhoNP(r)....

    // //filling vector with 1
    // thrust::device_vector<float> V1(mybox->M);
    // thrust::fill(V1.begin(), V1.end(), -1.0);

    // //multiplying -1*rhoNP, storing in dtmp2
    // thrust::transform(V1.begin(), V1.end(), d_NPdensity.begin(), dtmp2.begin(), thrust::multiplies<thrust::complex<double>>()); 
    // //then multiply wpl(r) * (- rhoNP(r)) = d_wpl * dtmp2, storing in dtmp
    // int ip;
    // // find Helfand potential to get wpl
    // for (int i = 0; i < mybox->Potentials.size(); i++ ) {
    //     if ( mybox->Potentials[i]->printStyle() == "Helfand" ) {
    //         ip = i;
    //     }
    // }        
    // std::cout << "calcH 2" << std::endl;
    
    // thrust::transform(mybox ->Potentials[ip]->d_wpl.begin(),mybox -> Potentials[ip]->d_wpl.end(), dtmp2.begin(), dtmp.begin(), thrust::multiplies<thrust::complex<double>>());
    
    
    // std::cout << "new version" << std::endl;
    
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

    cuDoubleComplex* _d_nprho = (cuDoubleComplex*) thrust::raw_pointer_cast(d_NPdensity.data());
    
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

    //Hterm = mybox->sumCpxDoubleDeviceArray(mybox->d_cpxGabe, mybox->blockSize, mybox->M );

    Hterm *= mybox->gvol;

    // // integrate int (wpl(r) * (rhoNP(r)))
    // thrust::complex<double> integral = thrust::reduce(dtmp.begin(), dtmp.end()) * mybox->gvol;

    // std::cout << "calcH 3" << std::endl;
    // // -i*C*int
    // Hterm = -I * mybox->C * integral;

    return Hterm;

} 


void ParticleMolec::modifyMolecule(std::istringstream& iss) {}


void ParticleMolec::computeLinearTerms() { }

void ParticleMolec::recomputeNmolecs() { }