// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

//////////////////////////////////////////////////////
// fts_potentialParticle.cu             7/11/2025   //
// Rob Riggleman, Anastasia Neumann                 //
// Incorporates an explicit nanoparticle into FTS.  //
// Primarily written by A. Neumann, adapted to v2   //
// version of the code by Riggleman.                //
//////////////////////////////////////////////////////

#ifndef _FTS_POTEN_PARTICLE
#define _FTS_POTEN_PARTICLE 


#include "fts_potential.h"


class FTS_Box;

class PotentialParticle : public FTS_Potential {

	private:

	public:
		PotentialParticle();
		PotentialParticle(std::istringstream &iss, FTS_Box*);
		~PotentialParticle();
		
		void updateFields() override;
		void correctFields() override;
		std::complex<double> calcHamiltonian() override;
		void writeFields(int) override;
		void initLinearCoeffs() override;
		void storePredictorData() override;
		

		double chi; //Strength of the potential
		std::string typeI, typeJ; //Types involved in the potential
		int intTypeI, intTypeJ; //species integer for types I, J 
		thrust::host_vector<int> intSpecies;
};

#endif
