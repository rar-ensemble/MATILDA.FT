// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

//////////////////////////////////////////////////////
// fts_moleculeHParticle.cu             7/20/2025   //
// Rob Riggleman, Anastasia Neumann                 //
// Incorporates an explicit nanoparticle into FTS.  //
// Primarily written by A. Neumann, adapted to v2   //
// version of the code by Riggleman.                //
//////////////////////////////////////////////////////

#ifndef _FTS_MOLEC_WALL
#define _FTS_MOLEC_WALL
#include "fts_molecule.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>

class FTS_Box;

class WallMolec : public FTS_Molec {
	protected:

	public:
		
		float V_walltot;
		
		std::string wall_style;		// planar, cylinder, etc
		std::string species;
		int int_species;
		
		double xi;		// Width of interface
		double size;	// Wall size (width for plane, radius for cylinder)
		int dir;		// Defines orientation (normal to plan, along cylinder)

		thrust::device_vector<double> d_R;  // Wall position on device
		thrust::host_vector<double> R; 		// wall position on host
		
		~WallMolec();
		WallMolec(std::istringstream& iss, FTS_Box*);

		void calcDensity() override;
		void computeLinearTerms() override;
		std::complex<double> calcHTerm() override;
		void modifyMolecule(std::istringstream&) override;
		void recomputeNmolecs(void) override;

};

#endif
