// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "fts_potential.h"
#include "fts_potential_helfand.h"
#include "fts_potential_incompress.h"
#include "fts_potential_edwards.h"
#include "fts_potential_flory.h"
#include "fts_potentialParticle.h"
#include "include_libs.h"
#include <istream>
void die(const char*);

FTS_Potential::FTS_Potential(std::istringstream &iss, FTS_Box* p_box) : mybox(p_box){

    input_command = iss.str();

    mybox = p_box;
    wplAlloc_flag = wmiAlloc_flag = 0;
}

FTS_Potential::~FTS_Potential() {}

FTS_Potential* FTS_PotentialFactory(std::istringstream &iss, FTS_Box* box) {

    std::string s1;
	iss >> s1 ;

    if ( s1 == "Edwards" || s1 == "edwards" ) {
        return new PotentialEdwards(iss, box);
    }

    else if (s1 == "Helfand" || s1 == "helfand"){
		return new PotentialHelfand(iss, box);
    }

    else if ( s1 == "incompress" || s1 == "Incompress" || s1 == "incompressible" ) {
        return new PotentialIncompress(iss, box);
    }

    else if ( s1 == "Flory" || s1 == "flory" ) {
        return new PotentialFlory(iss, box);
    }

    else if ( s1 == "particle" || s1 == "Particle" ) {
        return new PotentialParticle(iss, box);
    }
	
	else {
        std::string s2 = s1 + " is not a valid FTS_Potential"; 
        die(s2.c_str());
    }
	return 0;
}


// initializes the fields according to one of several options.
// Receives the field because different potentials may init
// either wpl or wmi (e.g., Flory initializes wmi, Helfand wpl)
// 'value' sets a uniform constant
// 'random' sets random values with amplitude \in [0, Amp]
// 'sin' sets to a sine wave (REAL PART ONLY)
void FTS_Potential::initializeField(
    std::istringstream &iss,                        // Input file command stream
    thrust::host_vector<thrust::complex<double>> &w // Field to be initialized (may be wpl or wmi)
    ) {

    std::string s1;
    iss >> s1;
    if ( s1 == "value" ) {
        double rVal, iVal;
        iss >> rVal;
        iss >> iVal;
        thrust::fill(w.begin(), w.end(), std::complex<double>(rVal, iVal));
    }
    // Two floats expected: amplitude of noise on real part and imag part
    else if ( s1 == "random" ) {
        double rAmp, iAmp;
        iss >> rAmp;
        iss >> iAmp;
        // Fill host field with random noise
        for ( int i=0 ; i<mybox->M ; i++ ) {
            w[i] = std::complex<double>(rAmp * ran2(), iAmp * ran2() );
        }
        
    }

    // Expects an int and two doubles [int dir] [double amplitude] [double period]
    else if ( s1 == "sin" || s1 == "sine" ) {
        double real_amp, imag_amp, period;
        int dir;
        iss >> dir;
        iss >> real_amp;
        iss >> imag_amp;
        iss >> period; 

        std::complex<double> I(0.0,1.0);
        for ( int i=0 ; i<mybox->M ; i++ ) {
            double r[3];
            mybox->get_r(i, r);
            w[i] = (real_amp + I * imag_amp) * sin(2.0 * PI * r[dir] * period / mybox->L[dir]);
        }

    }

    else if ( s1 == "readDatFile" ) {
        std::string s2;
        iss >> s2;
        mybox->readDatFile(s2, wpl);
    }

    else {
        die("Invalid initialize option on potential edwards");
    }    
}

std::string FTS_Potential::printCommand() {
    return input_command;
}

std::string FTS_Potential::printStyle() {
    return potentialStyle;
}

int FTS_Potential::wplAllocated() {
    if ( wplAlloc_flag ) return 1;
    else return 0;
}

int FTS_Potential::wmiAllocated() {
    if ( wmiAlloc_flag ) return 1;
    else return 0;
}