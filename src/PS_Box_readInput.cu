#include "PS_Box.h"
#include "random.h"
#include "include_libs.h"
#include "gsd.h"
#include <algorithm>
#include <map>


void die(const char*);
double ran2(void);


Integrator* IntegratorFactory(std::istringstream&, PS_Box*);

// Reads all of the commands from the input file from when this
// box is created (using the 'box' command) until the 'endBox' 
// command is caught. 
// This routine also serves to initialize the simulation box
// in preparation for a run.
void PS_Box::readInput(std::ifstream& inp) {
    std::cout << "In read input!" << std::endl;

    // Set some preliminary/default variables
    M = 1;
    V = 1.0;
    blockSize = 512;
    MAXBONDS = 2;
    MAXANGLES = 3;
    Nr = 100;
    rho0 = C = -1.0;
    nstot = nBondsTot = nAnglesTot = nBondTypes = nAngleTypes = 0;

    // Some default values
    logFreq = 100;
    gsdFreq = 0;
    fieldFreq = 0;
    gsd_name = "traj.gsd";
    doCharges = 0;

    std::string line, firstWord;

    bool readDimension = false;

    while (!inp.eof()) {
        getline(inp, line);

        if ( line.length() == 0 || line.at(0) == '#')
            continue;

        std::istringstream iss(line);
        
        while ( iss >> firstWord ) {
            // std::cout << "Line read: " << line << std::endl;

            if ( firstWord == "endBox" ) {
                break;
            }

            if ( firstWord == "blocksize" || firstWord == "blockSize" ) { iss >> blockSize ; }


            else if ( firstWord == "bond" ) {
                int btype;
                iss >> btype;
                if ( btype > bondK.size() ) {
                    bondK.resize(btype);
                    bondReq.resize(btype);
                    bondStyle.resize(btype);
                }

                std::string style;
                iss >> style;

                iss >> bondK[btype-1];
                iss >> bondReq[btype-1];

                if ( style == "harmonic" ) {
                    bondStyle[btype-1] = 0;
                }
                else if ( style == "fene" || style == "FENE" ) {
                    bondStyle[btype-1] = 1;
                }
                nBondTypes = bondK.size();
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


            else if ( firstWord == "Dim" ) {
                iss >> Dim;
                setDimension(Dim);
                readDimension = true;
                if ( Dim == 2 ) n_P_comps = 3;
                else n_P_comps = 6;
            }

            else if ( firstWord == "grid" ) {
                if ( !readDimension ) { die("Dim must be defined before grid!" );}
                for ( int j=0 ; j<Dim ; j++ ) {
                    iss >> Nx[j];
                    M *= Nx[j];
                    if ( L[0] > 0.0 ) { dx[j] = L[j] / double(Nx[j]); }
                }
            }

            else if ( firstWord == "gsdFreq" || firstWord == "gsd_freq" ) {
                iss >> gsdFreq;
            }

            else if ( firstWord == "gsdName" || firstWord == "gsd_name" ) {
                iss >> gsd_name;
            }

            else if ( firstWord == "integrator" ) {
                integrators.push_back( IntegratorFactory(iss, this) );
            }

            else if ( firstWord == "logFreq" || firstWord == "logfreq" ) { 
                iss >> logFreq; 
            }

            else if ( firstWord == "MAXANGLES" ) {
                if ( nstot > 0 ) die("MAXANGLES must be set before defining any molecules");
                iss >> MAXANGLES;
            }

            else if ( firstWord == "MAXBONDS" ) {
                if ( nstot > 0 ) die("MAXBONDS must be set before defining any molecules");
                iss >> MAXBONDS;
            }

            else if ( firstWord == "molecule" ) {
                std::string nextWord;
                iss >> nextWord;
                if ( nextWord == "linear" ) {
                    makeLinear(iss);
                }
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

            else if ( firstWord == "pmeorder" ) {
                iss >> pmeorder;
            }


            else if (firstWord == "randSeed" || firstWord == "RAND_SEED" || firstWord == "RANDSEED") {
                iss >> idum;        // Set CPU RNG to have seed = RANDSEED
                RANDSEED = idum;    // Set GPU RNG to have seed = RANDSEED
            }

            else if (firstWord == "rho0") {
                iss >> rho0;
            }

            else if ( firstWord == "species" ) {
                species.push_back(PS_Species(iss, this));
            }

            else {
                std::string s1 = "Invalid keyword " + firstWord + " in FTS_Box::readInput()";
                die(s1.c_str());
            }
            std::cout << "Finished input line: " << line << std::endl;

        }// while ( iss >> firstWord && firstWord != "endBox" ) 
        
        
        if ( firstWord == "endBox" ) {
            std::cout << "endBox caught" << std::endl;
            break;
        }

    }// while (!inp.eof()), finished reading up to 'endBox' or end of file


    finishInitialization();
    simTime = time(0);

}// End of readInput()

