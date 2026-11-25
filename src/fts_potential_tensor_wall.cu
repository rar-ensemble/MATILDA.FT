// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "include_libs.h"
#include "fts_potential_tensor_wall.h"
#include "FTS_Box.h"
#include "fts_species.h"

#include "fts_molecule.h"


PotentialTWall::PotentialTWall(std::istringstream& iss, FTS_Box* p_box) : FTS_Potential(iss, p_box) {
	iss.seekg(0);
	std::string s1;
	iss >> s1;
	iss >> s1;

	int dim = mybox->returnDimension();
	for ( int j=0 ; j<dim ; j++ ) {
		rmin[j] = -1.0;
		rmax[j] = mybox->L[j]+1.0;
	}
	


	potentialStyle = "TWall";
	
	iss >> typeI;
	iss >> typeJ;
	std::cout << "   TWall Flory-like potential acting on species " << typeI << std::endl;
	std::cout << "   Assumes wall is species " << typeJ << std::endl;

	iss >> chi;


	
	while ( iss.tellg() != -1 ) {
		std::string s1;
		iss >> s1;
		if ( s1 == "xrange" ) {
			iss >> rmin[0];
			iss >> rmax[0];
		}
		else if ( s1 == "yrange" ) {
			iss >> rmin[1];
			iss >> rmax[1];
		}
		else if ( s1 == "zrange" ) {
			iss >> rmin[2];
			iss >> rmax[2];
		}
		else {
			std::string last_words = s1 + " is not a valid option for potential wall";
			die(last_words);
		}
	}


	actsOn.push_back(typeI);
	actsOn.push_back(typeJ);

	
	//Assign species I and J
	intSpecies.resize(2); 
	intSpecies[0] = intSpecies[1] = -1;
	for (int i=0 ; i<mybox->Species.size() ; i++ ) {
		if ( actsOn[0] == mybox->Species[i].fts_species ) {
			intSpecies[0] = i;
		}
		else if (actsOn[1] == mybox->Species[i].fts_species ) {
			intSpecies[1] = i;
		}
	}
	if ( intSpecies[0] < 0 || intSpecies[1] < 0 ) {
		die("Failed to find species in PotentialTWall");
	}
	wpl.resize(mybox->M);
	d_wpl.resize(mybox->M);

	//update w[X] to incorporate wall interaction term
	int is1 = intSpecies[0] ;
	int is2 = intSpecies[1] ;


	thrust::host_vector<thrust::complex<double>> htmp(mybox->M);
	htmp = mybox->Molecs[2]->d_density;

	for ( int i=0 ; i<mybox->M ; i++ ) {
		double heavi = 1.0;
		double r[3];

		mybox->get_r(i, r);

		for ( int j=0 ; j<dim ; j++ ) {
			if ( r[j] < rmin[j] || r[j] > rmax[j] ) {
				heavi = 0.0;
			}
		}

		wpl[i] = heavi * htmp[i] * chi / mybox->C ;
	}

	// Send the field to the device
	d_wpl = wpl;	

}

void PotentialTWall::initLinearCoeffs() {
}

void PotentialTWall::writeFields(int potInd) {
	char nm[30];
	sprintf(nm, "wpl_TWall%d.dat", potInd);
	
	wpl = d_wpl;
	mybox->writeTComplexGridData(nm, wpl);

}

void PotentialTWall::updateFields() {
}

void PotentialTWall::correctFields() {
}

void PotentialTWall::storePredictorData() {
}


std::complex<double> PotentialTWall::calcHamiltonian() {
	Hterm = 0 ;
	return Hterm ;
}

PotentialTWall::~PotentialTWall() {}
