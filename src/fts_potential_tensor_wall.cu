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
	int M = mybox->M;
	int Ntens = mybox->Ntens;
	for ( int j=0 ; j<dim ; j++ ) {
		rmin[j] = -1.0;
		rmax[j] = mybox->L[j]+1.0;
	}
	


	potentialStyle = "TWall";
	
	iss >> typeI;
	iss >> typeJ;
	std::cout << "   tensor wall Flory-like potential acting on species " << typeI << std::endl;
	std::cout << "   Assumes wall is species " << typeJ << std::endl;



	iss >> Ao;



	// Initialize orientation vector
	double u[3];
	u[0] = u[1] = u[2] = 0.0;
	u[0] = 1.0;

	wpl.resize(M);

	while ( iss.tellg() != -1 ) {
		std::string s1;
		iss >> s1;
		if ( s1 == "u" || s1 == "orientation" ) {
			double mg2 = 0.0;
			for ( int j=0 ; j<dim ; j++ ) {
				iss >> u[j];
				mg2 += u[j] * u[j];
			}
			double mg = sqrt(mg2);
			for ( int j=0 ; j<dim ; j++ ) u[j] *= 1.0 / mg;
		}
		else if ( s1 == "xrange" ) {
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
			std::string last_words = s1 + " is not a valid option for potential tensor wall";
			die(last_words);
		}
	}// while( iss.tellg() != 1 )


	double Stp[5];
	if ( dim == 2 ) { 
		Stp[0] = u[0] * u[0] - 0.5;
		Stp[1] = u[0] * u[1]; 
	}
	else if ( dim == 3 ) { 
		Stp[0] = u[0] * u[0] - 1.0/3.0;
		Stp[1] = u[1] * u[1] - 1.0/3.0;
		Stp[2] = u[0] * u[1]; 
		Stp[3] = u[0] * u[2]; 
		Stp[4] = u[1] * u[2]; 
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

	//update w[X] to incorporate wall interaction term
	int is1 = intSpecies[0] ;
	int is2 = intSpecies[1] ;
	
	if ( mybox->Species[is2].tensor_field == 0 ) {
		std::string last_words = "Species " + actsOn[1] + " not initialized with a tensor field when setting up potential tensor wall";
		die(last_words);
	}


	d_M.resize(Ntens * M, 0.0);


	thrust::host_vector<thrust::complex<double>> Mtmp(Ntens * M);
	thrust::host_vector<thrust::complex<double>> htmp(M);
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

		wpl[i] = heavi * htmp[i] * Ao / mybox->C ;

		
		if ( dim == 2 ) {
			Mtmp[0 * M + i] = Stp[0] * wpl[i];
			Mtmp[1 * M + i] = Stp[1] * wpl[i];
		}
		if ( dim == 3 ) {
			Mtmp[0 * M + i] = Stp[0] * wpl[i];
			Mtmp[1 * M + i] = Stp[1] * wpl[i];
			Mtmp[2 * M + i] = Stp[2] * wpl[i];
			Mtmp[3 * M + i] = Stp[3] * wpl[i];
			Mtmp[4 * M + i] = Stp[4] * wpl[i];
		}
	}//i=0:M

	// Send the field to the device
	d_M = Mtmp;	

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
