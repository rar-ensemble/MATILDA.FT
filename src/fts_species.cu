// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "fts_species.h"

void die(const char*);


FTS_Species::FTS_Species(std::istringstream &iss, FTS_Box* p_box) : box(p_box) {
    this->input_command = iss.str();
    
    iss >> this->fts_species; 

    int M = box->M;

    // Resize the arrays to be of the grid dimension size
    this->density.resize(M);
    this->d_density.resize(M);
    this->d_Ak.resize(M);
    this->d_w.resize(M);

    return;
}

FTS_Species::~FTS_Species() {
    return;
}
FTS_Species::FTS_Species() {
    return;
}


// Loops over all of the molecules and zeroes the species densities
void FTS_Species::zeroDensity() {
    thrust::fill(d_density.begin(), d_density.end(), 0.0);
}


// Allows one step to directly add I * wpl field to species field ws
struct plusITimes {
    __host__ __device__ 
        thrust::complex<double> operator()(const thrust::complex<double> &ws, const thrust::complex<double> &wpl) {
            thrust::complex<double> I = thrust::complex<double>(0.0,1.0);
            return ws + I * wpl;
        }
};



// Loops over potentials and constructs the fields for each species
void FTS_Species::buildPotentialField() {

    // Zero the fields
    thrust::fill(d_w.begin(), d_w.end(), thrust::complex<double>(0.0,0.0));

    for ( int i=0 ; i<box->Potentials.size() ; i++ ) {

        if ( box->Potentials[i]->printStyle() == "Helfand" ) {

            // d_w += I * wpl from Helfand
            thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wpl.begin(),
                d_w.begin(), plusITimes());

        }

        else if ( box->Potentials[i]->printStyle() == "Flory" ) {
            if ( fts_species == box->Potentials[i]->actsOn[0] ) {
                
                // wIJ- field is *subtracted* from the first listed species
                thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wmi.begin(),
                    d_w.begin(), thrust::minus<thrust::complex<double>>());

                // d_w += I * wpl for both listed species
                thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wpl.begin(),
                     d_w.begin(), plusITimes());
            }

            if ( fts_species == box->Potentials[i]->actsOn[1] ) {
                
                // wIJ- field is *added* to the second listed species
                thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wmi.begin(), 
                    d_w.begin(), thrust::plus<thrust::complex<double>>());

                // d_w += I * wpl for both listed species
                thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wpl.begin(), 
                    d_w.begin(), plusITimes());                    
            }
        }
    }// i=0:potentials.size()

    // Normalize the field by Nr
    thrust::device_vector<thrust::complex<double>> invNr(box->M);
    thrust::fill(invNr.begin(), invNr.end(), 1.0/double(box->Nr));
    thrust::transform(d_w.begin(), d_w.end(), invNr.begin(), d_w.begin(), thrust::multiplies<thrust::complex<double>>());


    // thrust::host_vector<thrust::complex<double>> htmp(box->M);
    // htmp = d_w;
    // if ( fts_species == "A" )
    //     box->writeTComplexGridData("from_speciesA.dat", htmp);
    // else {
    //     box->writeTComplexGridData("from_speciesB.dat", htmp);
    // //    die("done");
    // }
}


// Writes the density field to a text file
void FTS_Species::writeDensity(const int id) {
    char nm[40];

    sprintf(nm, "rhoSpecies%d.dat", id);
    
    // Transfer data to host
    density = d_density;
    box->writeTComplexGridData(std::string(nm), density);
}

// Writes the total field acting on a particular species
void FTS_Species::writeSpeciesFields(const int id) {
    char nm[40];
    sprintf(nm, "fieldSpecies%d.dat", id);

    thrust::host_vector<thrust::complex<double>> htmp = d_w;
    box->writeTComplexGridData(std::string(nm), htmp);

}