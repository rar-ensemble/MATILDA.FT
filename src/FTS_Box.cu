// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "FTS_Box.h"

#include "fts_molecule.h"
#include "fts_species.h"
#include "fts_potential.h"
#include "include_libs.h"

void die(const char*);
FTS_Molec* FTS_MolecFactory(std::istringstream&, FTS_Box*);
FTS_Potential* FTS_PotentialFactory(std::istringstream&, FTS_Box*);

// Executes the commands for a given time step
// Updates all fields, then recomputes all molecule densities
// then populating species densities
void FTS_Box::doTimeStep(int step) {

    // Update the potential fields
    for ( int i=0 ; i<Potentials.size(); i++ ) {
        int ti = time(0);
        Potentials[i]->updateFields();
        fieldUpdateTimer += time(0) - ti;
    }
    
    // Zero the species densities and rebuilt the fields
    for ( int i=0 ; i<Species.size(); i++ ) {
        int ti = time(0);
        Species[i].zeroDensity();
        Species[i].buildPotentialField();
        speciesTimer += time(0) - ti;
    }

    // Recalculate all density fields, including populating species densities
    for ( int i=0 ; i<Molecs.size(); i++ ) {
        int ti = time(0);
        Molecs[i]->calcDensity();
        moleculeTimer += time(0) - ti;
    }


    // If using predictor-corrector scheme, repeat above with predicted
    // densities and forces
    if ( PCflag == 1 ) {
        // Update the potential fields with corrector step
        for ( int i=0 ; i<Potentials.size(); i++ ) {
            int ti = time(0);
            Potentials[i]->correctFields();
            fieldUpdateTimer += time(0) - ti;
        }
        
        // Zero the species densities and rebuilt the fields
        for ( int i=0 ; i<Species.size(); i++ ) {
            int ti = time(0);
            Species[i].zeroDensity();
            Species[i].buildPotentialField();
            speciesTimer += time(0) - ti;
        }

        // Recalculate all density fields, including populating species densities
        for ( int i=0 ; i<Molecs.size(); i++ ) {
            int ti = time(0);
            Molecs[i]->calcDensity();
            moleculeTimer += time(0) - ti;
        }        
    }


    /////////////////
    // I/O section //
    /////////////////

    // Write the species densities
    if ( densityFieldFreq > 0 && step % densityFieldFreq == 0 ) {
        for ( int i=0 ; i<Species.size(); i++ ) {
            Species[i].writeDensity(i);
        }
    }
    
    // Write the species densities
    if ( chemFieldFreq > 0 && step % chemFieldFreq == 0 ) {
        for ( int i=0 ; i<Potentials.size(); i++ ) {
            Potentials[i]->writeFields(i);
        }
        for ( int i=0 ; i<Species.size(); i++ ) {
            Species[i].writeSpeciesFields(i);
        }
    }


    // Write log data
    if ( step % logFreq == 0 ) {
        writeData(step);
    }

}


// Write Hamiltonian terms to output file
void FTS_Box::writeData(int step) {

    computeHamiltonian();

    OTP.open("fts_data.dat", std::ios_base::app);
    std::string outline;

    OTP << step << " " << Heff.real() << " " ;
    std::cout << step << " " << Heff.real() << " " ;
    if ( ftsStyle == "cl" ) {
        OTP << Heff.imag() << " " ;
    }
    for ( int i=0 ; i<Potentials.size() ; i++ ) {
        OTP << Potentials[i]->Hterm << " " ;
        std::cout << Potentials[i]->Hterm << " " ;
    }

    for ( int i=0 ; i<Molecs.size(); i++ ) {
        OTP << -Molecs[i]->nmolecs*log(Molecs[i]->Q) << " " ;
        std::cout << -Molecs[i]->nmolecs*log(Molecs[i]->Q) << " " ;
    }

    OTP << std::endl;
    std::cout << std::endl;

    OTP.close();
}


void FTS_Box::initializeSim() {
    
    Potentials[0]->wpl = Potentials[0]->d_wpl;

    
    // Zero the species densities
    for ( int i=0 ; i<Species.size(); i++ ) {
        Species[i].zeroDensity();
        Species[i].buildPotentialField();
    }


    thrust::host_vector<thrust::complex<double>> htmp(M);
    // Calculate all density fields, including populating species densities
    for ( int i=0 ; i<Molecs.size(); i++ ) {
        Molecs[i]->calcDensity();
    }

    // Initialize the output stream
    OTP.open("fts_data.dat");
    OTP.close();


}


// Writes field data from std::vector to a text file. Both real and imaginary components are written
void FTS_Box::writeComplexGridData(std::string name, std::vector<std::complex<double>> field) {

    FILE *otp;
    otp = fopen(name.c_str(), "w");

    double *r = new double [Dim];
    int *nn = new int [Dim];
    for ( int i=0 ; i<M; i++ ) {
        get_r(i,r);
        unstack2(i,nn);
        
        for ( int j=0 ; j<Dim ; j++ ) fprintf(otp, "%lf ", r[j]);

        fprintf(otp, "%1.4e %1.4e\n", field[i].real(), field[i].imag());

        if ( Dim == 2 && nn[0] == (Nx[0]-1) ) fprintf(otp,"\n");
    }

    delete r;
    delete nn;
    fclose(otp);
}

// Writes field data from thrust::vector to a text file. Both real and imaginary components are written
void FTS_Box::writeTComplexGridData(std::string name, thrust::host_vector<thrust::complex<double>> field) {

    FILE *otp;
    otp = fopen(name.c_str(), "w");

    double *r = new double [Dim];
    int *nn = new int [Dim];
    for ( int i=0 ; i<M; i++ ) {
        get_r(i,r);
        unstack2(i,nn);
        
        for ( int j=0 ; j<Dim ; j++ ) fprintf(otp, "%lf ", r[j]);

        fprintf(otp, "%1.4e %1.4e\n", field[i].real(), field[i].imag());

        if ( Dim == 2 && nn[0] == (Nx[0]-1) ) fprintf(otp,"\n");
    }

    delete r;
    delete nn;
    fclose(otp);
}


void FTS_Box::readInput(std::ifstream& inp) {

    // Set some preliminary variables
    M = 1;
    V = 1.0;
    rho0 = C = -1.0;

    // Some default values
    logFreq = 100;
    maxSteps = 10000;
    chemFieldFreq = 0;
    densityFieldFreq = 0;
    Hold = 1.0E8;       // Arbitrary large value for old Hamiltonian
    tolerance = 1.0E-5; // Arbitrary small value for convergance tolerance
    PCflag = 0;

    std::string line, firstWord;

    bool readDimension = false;

    while (!inp.eof()) {
        getline(inp, line);

        if ( line.length() == 0 || line.at(0) == '#')
            continue;

        std::istringstream iss(line);
        
        while ( iss >> firstWord ) {

            if ( firstWord == "endBox" ) {
                std::cout << "endBox caught" << std::endl;
                break;
            }

            // Commands are alphabetical from here on
            else if ( firstWord == "boxLengths" ) {
                if (!readDimension) 
                    std::cout << "\nWARNING! WARNING!\nboxLengths read before dimension specified, assuming default value of 2"<<std::endl;

                for ( int j=0 ; j<Dim ; j++ ) {
                    iss >> L[j];
                    V *= L[j];
                    if ( Nx[0] > 0 ) { dx[j] = L[j] / double(Nx[j]); }
                }
            }

            else if ( firstWord == "chemFieldFreq" || firstWord == "chemfieldfreq" ) { iss >> chemFieldFreq; }

            else if ( firstWord == "densityFieldFreq" || firstWord == "densityfieldfreq" ) { iss >> densityFieldFreq; }

            else if ( firstWord == "Dim" ) {
                iss >> Dim;
                setDimension(Dim);
                readDimension = true;
            }

            else if ( firstWord == "grid" ) {
                if ( !readDimension ) { die("Dim must be defined before grid!" );}
                for ( int j=0 ; j<Dim ; j++ ) {
                    iss >> Nx[j];
                    M *= Nx[j];
                    if ( L[0] > 0.0 ) { dx[j] = L[j] / double(Nx[j]); }
                }
            }

            else if ( firstWord == "logFreq" || firstWord == "logfreq" ) { iss >> logFreq; }

            else if ( firstWord == "maxSteps" || firstWord == "maxSteps" ) { iss >> maxSteps; }

            else if ( firstWord == "molecule" ) {
                Molecs.push_back(FTS_MolecFactory(iss, this));
            }

            else if ( firstWord == "Nr" ) {
                iss >> Nr;
            }

            else if ( firstWord == "Nx") {
                iss >> Nx[0];
                M *= Nx[0];
            }

            else if (firstWord == "Ny") {
                iss >> Nx[1];
                M *= Nx[1];
            }

            else if (firstWord == "Nz") {
                if ( Dim != 3 ) die("Nz supplied in a non-3D simulation!");
                iss >> Nx[2];
                M *= Nx[2];
            }

            else if (firstWord == "potential") {
                Potentials.push_back(FTS_PotentialFactory(iss, this));
            }

            else if (firstWord == "randSeed" || firstWord == "RAND_SEED") {
                std::cout << idum << " Before " << std::endl;
                fflush(stdout);
                iss >> idum;
                std::cout << idum << " after " << std::endl;
            }

            else if (firstWord == "rho0") {
                iss >> rho0;
            }

            else if ( firstWord == "species" ) {
                Species.push_back(FTS_Species(iss, this));
            }

            else if ( firstWord == "tolerance" ) {
                iss >> tolerance;
            }

            else {
                std::string s1 = "Invalid keyword " + firstWord + " in FTS_Box::readInput()";
                die(s1.c_str());
            }
            std::cout << line << std::endl;

        }// while ( iss >> firstWord && firstWord != "endBox" ) 
        
        
        if ( firstWord == "endBox" ) {
            std::cout << "endBox caught" << std::endl;
            break;
        }

    }// while (!inp.eof())

    // After input read, make the FFT plan
    // This currently assumes complex-double to complex-double transforms
    // change Z2Z to C2C to switch to cpx-float
    if ( this->Dim == 2 ) 
        cufftPlan2d(&fftplan, Nx[1], Nx[0], CUFFT_Z2Z);
    if ( this->Dim == 3 ) 
        cufftPlan3d(&fftplan, Nx[2], Nx[1], Nx[0], CUFFT_Z2Z);

    // Define gvol, dx
    gvol = 1.0;
    for ( int j=0 ; j<Dim ; j++ ) {
        dx[j] = L[j] / double(Nx[j]);
        gvol *= dx[j];
    }

    M_Block = 512;
    M_Grid = (int)ceil((double)(M) / M_Block);

    if ( rho0 > 0 && C > 0 ) { die("Cannot define both C and rho0!"); }

    double Rg = pow( (Nr-1.0)/6.0, 0.5);
    if ( rho0 > 0 ) {
        C = rho0/Nr;
        for ( int j=0; j<Dim ; j++ ) C *= Rg;
        std::cout << "Using Nr = " << Nr << ", computed C = " << C << std::endl;
    }
    else if ( C > 0 ) {
        rho0 = C * Rg * Rg * Rg / Nr;
        std::cout << "Using Nr = " << Nr << ", computed rho0 = " << rho0 << " [b^-3]" << std::endl;
    }

    simTime = time(0);
    ftTimer = speciesTimer = moleculeTimer = fieldUpdateTimer = 0;

    // Initialize linear coeffs
    for ( int i=0 ; i<Potentials.size(); i++ ) {
        Potentials[i]->initLinearCoeffs();
    }

    // Compute the linear coefficients
    for ( int i=0 ; i<Molecs.size() ; i++ ) {
        Molecs[i]->computeLinearTerms();
    }
}

void FTS_Box::writeTime() {

    int dt = time(0) - simTime;
    std::cout << "Total simulation time: " << dt / 60 << "m" << dt % 60 << "sec" << std::endl;
    
    dt = ftTimer;
    std::cout << "Total FT time: " << dt / 60 << "m" << dt % 60 << "sec" << std::endl;

    dt = fieldUpdateTimer;
    std::cout << "Total Field Update time: " << dt / 60 << "m" << dt % 60 << "sec" << std::endl;

    dt = speciesTimer;
    std::cout << "Total species class time: " << dt / 60 << "m" << dt % 60 << "sec" << std::endl;

    dt = moleculeTimer;
    std::cout << "Total molecule class time: " << dt / 60 << "m" << dt % 60 << "sec" << std::endl;
}

void FTS_Box::computeHomopolyDebye(
    thrust::host_vector<thrust::complex<double>> &g,// Debye function
    const double alpha                                    // N/Nr
    ) {

    g[0] = 1.0;

    for ( int i=1; i<M ; i++ ) {
        double kv[3];
        double k2 = get_kD(i, kv);

        g[i] = 2.0 * ( exp(-k2*alpha) + k2 * alpha - 1.0 ) / (k2 * k2) ;
    }
}

void FTS_Box::computeIntRABlockDebye(
    thrust::host_vector<thrust::complex<double>> &gaa,// Debye function
    const double f,   // f = Nblock / N
    const double alpha                                    // N/Nr
    ) {

    gaa[0] = 0.0;

    for ( int i=1; i<M ; i++ ) {
        double kv[3];
        double k2 = get_kD(i, kv);

        gaa[i] = 2.0 * ( exp(-k2*alpha*f) + k2 * alpha * f - 1.0 ) / (k2 * k2) ;
    }
}


// The cross correlation term between blocks A and C
// for an A-B-C triblock. Compatible with fB = 0.0
// Includes the prefactor of 2.0
void FTS_Box::computeIntERBlockDebye(
    thrust::host_vector<thrust::complex<double>> &gac,// Debye function
    const double fA,   // f = Na / N
    const double fB,   // f = Nb / N
    const double fC,   // f = Nc / N
    const double alpha                                    // N/Nr
    ) {

    gac[0] = 0.0;
    for ( int i=1; i<M ; i++ ) {
        double kv[3];
        double k2 = get_kD(i, kv);

        gac[i] = 2. * (1.0 - exp(-k2*alpha*fA)) * exp(-k2*alpha*fB) * (1.0 - exp(-k2*alpha*fC)) / (k2 * k2) ;
    }
}


void FTS_Box::computeHamiltonian() {
    Heff = 0.0;

    // Compute field-dependent terms
    for ( int i=0 ; i<Potentials.size() ; i++ ) {
        Heff += Potentials[i]->calcHamiltonian();
    }

    // accumulate the -n*log(Q) terms
    for ( int i=0 ; i<Molecs.size(); i++ ) {
        Heff += (std::complex<double>)( -Molecs[i]->nmolecs * log( Molecs[i]->Q));
    }
}

void FTS_Box::initSmearGaussian(
    thrust::host_vector<thrust::complex<double>> &smear, // Stores the smearing function
    const double amplitude, // prefactor of smearing function, generally 1.0
    const double sigma) {   // std dev of the Gaussian 

    double kv[3], k2;

    // Define smearing Gaussian in k-space
    for ( int i=0 ; i<M; i++ ) {
        k2 = get_kD(i, kv);
        smear[i] = exp(-k2 * sigma * sigma / 2.0 ) * amplitude;
    }
}


// Write the species densities
void FTS_Box::writeSpeciesDensityFields() {
    for ( int i=0 ; i<Species.size(); i++ ) {
        Species[i].writeDensity(i);
    }
}

void FTS_Box::writeFields() {
    writeSpeciesDensityFields();
    Potentials[0]->wpl = Potentials[0]->d_wpl;
    writeTComplexGridData("wpl.dat", Potentials[0]->wpl);
}




FTS_Box::~FTS_Box() {}

FTS_Box::FTS_Box(std::istringstream& iss ) : Box(iss) {
    std::string s1;
    iss >> ftsStyle;

    std::cout << "Made FTS_Box with style " << ftsStyle << std::endl;
}

// Integrate the given field using trapezoid rule (this is spectral accurate
// when using PBCs)
thrust::complex<double> FTS_Box::integTComplexD(thrust::host_vector<thrust::complex<double>> dat) {
    thrust::complex<double> sum = 0.0;
    for ( int i=0 ; i<this->M; i++ ) {
        sum += dat[i];
    }

    sum *= this->gvol;
    return sum;
}

// Integrate the given field using trapezoid rule (this is spectral accurate
// when using PBCs)
std::complex<double> FTS_Box::integComplexD(std::complex<double> *dat) {
    std::complex<double> sum = 0.0;
    for ( int i=0 ; i<this->M; i++ ) {
        sum += dat[i];
    }

    sum *= this->gvol;
    return sum;
}


// Check convergence of SCFT simulation
int FTS_Box::converged(int step) {

    // Immediately return if not an SCFT calculation
    if ( ftsStyle != "scft" ) return 0;
    
    // Check if H has been updated
    if ( step % logFreq != 0 ) return 0;
    
    // Check if Hamiltonian change is less than the tolerance
    if ( fabs(Hold - real(Heff)) < tolerance ) {
        std::cout << "SCFT converged on step " << step << " deltaH: " << fabs(Hold - real(Heff)) << std::endl;
        std::cout << Hold << " " << real(Heff) << std::endl;
        return 1;
    }

    else {
        Hold = real(Heff);
        return 0;
    }
}


std::string FTS_Box::returnFTSstyle() {
    return ftsStyle;
}
