// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "potential.h"
#include <vector>
extern std::vector<Potential*> Potentials;
void update_potentials() {
 
	for (auto Iter: Potentials){
		Iter->Update();
	}

}
