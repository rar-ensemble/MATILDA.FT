// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

//////////////////////////////////////////////////////
// fts_potential_tensor_wall            11/25/2026  //
// Rob Riggleman									//
//////////////////////////////////////////////////////

#ifndef _FTS_POTEN_TWALL
#define _FTS_POTEN_TWALL 


#include "fts_potential.h"


class FTS_Box;

class PotentialTWall : public FTS_Potential {

	private:

	public:
		PotentialTWall();
		PotentialTWall(std::istringstream &iss, FTS_Box*);
		~PotentialTWall();
		
		void updateFields() override;
		void correctFields() override;
		std::complex<double> calcHamiltonian() override;
		void writeFields(int) override;
		void initLinearCoeffs() override;
		void storePredictorData() override;
		

		double Ao; //Strength of the potential
		std::string typeI, typeJ; //Types involved in the potential
		double rmin[3], rmax[3];
		int intTypeI, intTypeJ; //species integer for types I, J 
		thrust::host_vector<int> intSpecies;
};

#endif
