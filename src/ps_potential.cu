// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_potential.h"




// Allocates memory for:
// potential, forces, virial contribution
// in both r- and k-space
void PS_Potential::Initialize_Potential() {
    


}



// Calculates forces on rho1, rho2 for this pairstyle
void PS_Potential::CalcForces() {

}



// Calculates the energy involved in this potential as
// energy = \int dr rho1(r) \int dr' u(r-r') rho2(r')
// The convolution theorem is used to efficiently evaluate
// the integral over r'
float PS_Potential::CalcEnergy() {
    energy = 0.0f;


    return this->energy;
    
}






PS_Potential::~PS_Potential() {


}

PS_Potential::PS_Potential() {
    return;
}

PS_Potential::PS_Potential(std::istringstream &iss, PS_Box* box) : mybox(box) {
    input_command = iss.str();
    return;
}


void PS_Potential::ramp_check_input(std::istringstream& iss){

    // if (iss.fail()){
    //     die("Error during input script; failed to properly read:\n" + iss.str());
    // }

    // string convert;
    // iss >> convert;

    // if (!iss.fail()){
    //     if (convert == "ramp") {
    //         ramp = true;
    //         iss >> final_prefactor;
    //         if(iss.fail()) die("no final prefactor specified");
    //         cout << "Ramping prefactor of " <<potential_type<< " style, between types " << type1+1 << " and " \
    //             << type2 + 1 << " from " << initial_prefactor \
    //             << " to " << final_prefactor << endl;

    //         cout << "Estimated per time step change: " << \
    //             (final_prefactor - initial_prefactor) / (prod_steps)
    //             << endl;

    //     }
    //     else 
    //         die("Invalid keyword: " + convert);
    // }


}


// Potential* PotentialFactory(istringstream &iss){
// 	string s1;
// 	iss >> s1;
//     transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
// 	if (s1 == "erf"){
// 		return new Erf(iss);
// 	}
// 	if (s1 == "gaussian"){
// 		return new Gaussian(iss);
// 	}
// 	if (s1 == "gaussian_erf"){
// 		return new GaussianErf(iss);
// 	}
// 	if (s1 == "fieldphase" || s1 == "biasfield"){
// 		return new BiasField(iss);
// 	}
// 	if (s1 == "maiersaupe"){
// 		return new MaierSaupe(iss);
// 	}
// 	if (s1 == "charges"){
// 		return new Charges(iss);
// 	}
	
// 	die("Unsupported potential");
// 	return 0;
// }
