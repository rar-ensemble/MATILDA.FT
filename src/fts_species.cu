// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "fts_species.h"
#include "fts_potential_maiersaupe.h"

void die(const char*);


FTS_Species::FTS_Species(std::istringstream &iss, FTS_Box* p_box) : box(p_box) {
    this->input_command = iss.str();
    
    iss >> this->fts_species;

    // Set default value of charge
    charge = 0.0;
    tensor_field = 0;

    // Check for optional arguments 
    std::string test;
    while ( iss >> test ) {
        if ( test == "charge" ) {
            iss >> charge;
        }
        else if ( test == "orientation" || test == "tensor" ) {
            tensor_field = 1;
        }
    }


    // Finish initialization //
    int M = box->M;

    // Resize the arrays to be of the grid dimension size
    this->density.resize(M);
    this->d_density.resize(M);
    this->d_density_avg.resize(M);
    this->d_Ak.resize(M);
    this->d_w.resize(M);

    // Define tensor fields as needed
    if ( tensor_field ) {
        int Nu_tot = box->Nu_tot;
        if ( Nu_tot == 0 ) {
            die("u_grid must be defined before a species with a tensor field");
        }
        int dim = box->returnDimension();
        int Ntensor = dim*(dim+1)/2 - 1;
        this->d_Mw.resize(M*Ntensor);
        this->d_S.resize(M*Ntensor);
    }

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
    if ( tensor_field ) {
        thrust::fill(d_S.begin(), d_S.end(), 0.0);
    }
}

// Allows one step to directly add I * q * wpl field to species field for 
// charge potentials
struct plus_Iq_times {
    double q;

    plus_Iq_times(double q_) : q(q_) {}

    __host__ __device__ 
        thrust::complex<double> operator()(const thrust::complex<double> &ws, const thrust::complex<double> &wpl) {
            thrust::complex<double> I = thrust::complex<double>(0.0,1.0);
            return ws + q * I * wpl;
        }
};

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
    if ( tensor_field ) {
        thrust::fill(d_Mw.begin(), d_Mw.end(), thrust::complex<double>(0.0,0.0));
    }

    for ( int i=0 ; i<box->Potentials.size() ; i++ ) {

        if ( box->Potentials[i]->printStyle() == "Charges" && fabs(this->charge) > 1.0E-7 ) {

            // d_w += I * wpl from Edwards
            thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wpl.begin(),
                d_w.begin(), plus_Iq_times(this->charge));

        }

        else if ( box->Potentials[i]->printStyle() == "Edwards" ) {

            // d_w += I * wpl from Edwards
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

        else if ( box->Potentials[i]->printStyle() == "Helfand" ) {

            // d_w += I * wpl from Helfand
            thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wpl.begin(),
                d_w.begin(), plusITimes());

        }

        else if ( box->Potentials[i]->printStyle() == "Incompress" ) {

            // d_w += I * wpl from Helfand
            thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wpl.begin(),
                d_w.begin(), plusITimes());

        }

        else if ( box->Potentials[i]->printStyle() == "MaierSaupe" ) {
            if ( tensor_field && fts_species == box->Potentials[i]->actsOn[0] ) {
                PotentialMaierSaupe* ms = static_cast<PotentialMaierSaupe*>(box->Potentials[i]);

                // d_Mw += M for each independent tensor component
                thrust::transform(d_Mw.begin(), d_Mw.end(), ms->d_M.begin(),
                    d_Mw.begin(), thrust::plus<thrust::complex<double>>());
            }
        }

        else if (box->Potentials[i]->printStyle() == "Particle" ) {
            if (fts_species == box->Potentials[i]->actsOn[0] ) {
                // AN: add C* chi* rho_exp_Nr to field...this is wpl from particle pot.
                thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wpl.begin(), d_w.begin(), thrust::plus<thrust::complex<double>>()); 
            }
        }

        else if (box->Potentials[i]->printStyle() == "Wall" ) {
            if (fts_species == box->Potentials[i]->actsOn[0] ) {
                // AN: add C* chi* rho_exp_Nr to field...this is wpl from particle pot.
                thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wpl.begin(), d_w.begin(), thrust::plus<thrust::complex<double>>()); 
            }
        }
        
    }// i=0:potentials.size()

    // Normalize the field by Nr
    thrust::device_vector<thrust::complex<double>> invNr(box->M);
    thrust::fill(invNr.begin(), invNr.end(), 1.0/double(box->Nr));
    thrust::transform(d_w.begin(), d_w.end(), invNr.begin(), d_w.begin(),   
                        thrust::multiplies<thrust::complex<double>>());

    // Tensor field carries the same 1/Nr scaling
    if ( tensor_field ) {
        const double inv = 1.0 / double(box->Nr);
        thrust::transform(d_Mw.begin(), d_Mw.end(), d_Mw.begin(),
            [=] __device__ (thrust::complex<double> x) -> thrust::complex<double> { return x * inv; });
    }


}


// Writes the density field to a text file
void FTS_Species::writeDensity(const int id) {
    char nm[40];

    sprintf(nm, "rhoSpecies%d.dat", id);

    // Transfer data to host
    density = d_density;
    box->writeTComplexGridData(std::string(nm), density);

    // Orientation tensor components and scalar order parameter
    if ( tensor_field ) {
        const int M = box->M;
        const int Ntensor = d_S.size() / M;
        const char* comp2[2] = {"xx", "xy"};
        const char* comp3[5] = {"xx", "yy", "xy", "xz", "yz"};

        thrust::host_vector<thrust::complex<double>> hS = d_S;
        thrust::host_vector<thrust::complex<double>> htmp(M);

        for ( int nt=0 ; nt<Ntensor ; nt++ ) {
            sprintf(nm, "S%sSpecies%d.dat", (Ntensor == 2 ? comp2[nt] : comp3[nt]), id);
            thrust::copy(hS.begin() + nt*M, hS.begin() + (nt+1)*M, htmp.begin());
            box->writeTComplexGridData(std::string(nm), htmp);
        }

        // Scalar order parameter per site of this species from the largest
        // eigenvalue lambda of the real part of S. A perfectly aligned site 
        // has lambda = (d-1)/d, so the order parameter is d/(d-1) * lambda / rho.
        // 2D: real part is the order parameter, imag part the director angle (radians).
        // 3D: real part is the order parameter, imag part is zero.
        for ( int i=0 ; i<M ; i++ ) {
            double rho = density[i].real();
            double lam = 0.0, angle = 0.0;

            if ( Ntensor == 2 ) {
                double sxx = hS[i].real();
                double sxy = hS[M + i].real();
                lam = 2.0 * sqrt(sxx*sxx + sxy*sxy);
                angle = 0.5 * atan2(sxy, sxx);
            }

            else if ( Ntensor == 5 ) {
                double sxx = hS[i].real(),       syy = hS[M + i].real();
                double sxy = hS[2*M + i].real(), sxz = hS[3*M + i].real(), syz = hS[4*M + i].real();
                double szz = -sxx - syy;

                // Largest eigenvalue of a traceless symmetric 3x3 matrix:
                // lambda = 2 sqrt(p/3) cos( acos( 3q/(2p) sqrt(3/p) ) / 3 ),
                // p = (S:S)/2, q = det(S)
                double pp = 0.5 * ( sxx*sxx + syy*syy + szz*szz ) + sxy*sxy + sxz*sxz + syz*syz;
                if ( pp > 1.0E-300 ) {
                    double q = sxx*(syy*szz - syz*syz) - sxy*(sxy*szz - syz*sxz) + sxz*(sxy*syz - syy*sxz);
                    double arg = 1.5 * q / pp * sqrt(3.0/pp);
                    if ( arg > 1.0 ) arg = 1.0;
                    if ( arg < -1.0 ) arg = -1.0;
                    lam = 1.5 * 2.0 * sqrt(pp/3.0) * cos( acos(arg) / 3.0 );
                }
            }

            double op = ( fabs(rho) > 1.0E-12 ? lam / rho : 0.0 );
            htmp[i] = thrust::complex<double>(op, angle);
        }
        sprintf(nm, "orderParamSpecies%d.dat", id);
        box->writeTComplexGridData(std::string(nm), htmp);
    }
}

// Adds the current density field into the running CL sample sum
void FTS_Species::accumulateDensity() {
    thrust::transform(d_density.begin(), d_density.end(), d_density_avg.begin(),
        d_density_avg.begin(), thrust::plus<thrust::complex<double>>());
}

// Writes the running-average density field (CL sampling) to a text file
void FTS_Species::writeAvgDensity(const int id, const int nSamples) {
    char nm[40];

    sprintf(nm, "rhoSpeciesAvg%d.dat", id);

    thrust::host_vector<thrust::complex<double>> avg = d_density_avg;
    for ( int i=0 ; i<avg.size() ; i++ ) {
        avg[i] /= double(nSamples);
    }
    box->writeTComplexGridData(std::string(nm), avg);
}

// Writes the total field acting on a particular species
void FTS_Species::writeSpeciesFields(const int id) {
    char nm[40];
    sprintf(nm, "fieldSpecies%d.dat", id);

    thrust::host_vector<thrust::complex<double>> htmp = d_w;
    box->writeTComplexGridData(std::string(nm), htmp);

}