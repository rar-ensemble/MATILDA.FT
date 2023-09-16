// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "global_templated_functions.h"

using namespace std;

Widom::Widom(std::istringstream& iss) : Compute(iss) {
    style = "Widom";


    // The Widom compute will estimate the excess chemical potential by 
    // growing a copy of a specified molecule at random and calculating the
    // energy of the system. 

    readRequiredParameter(iss, mole_id); mole_id--;
    readRequiredParameter(iss, num_configs);


    // Check that sufficient memory is allocated
    first_site = -1;
    int n_molec_sites = 0;

    for (int i = 0; i < ns; i++) {
      if (molecID[i] == mole_id) {
        n_molec_sites++;

        // Assign the first index
        if (first_site < 0) first_site = i;

      }
    }

    if (n_molec_sites > extra_ns_memory)
      die("Insufficient memory allocated with extraSiteMemory for Widom insertions!");

    if (first_site)
      die("Molecule not found in Compute Widom!");

    std::string message = 
    "\nCalculating Widom insertion estimate of chemical potential using molecule " + std::to_string(mole_id + 1) + " as the template.";

    message += "NOTICE NOTICE NOTICE\n";
    message +=  "1. The chemical potential calculation assumes the molecules are in order in by site ID.\n";
    message +=   " I.e., the first site of molecule 3 will NOT have a larger particle index than molecule 5.\n";
    message +=   "2. To use on charged molecules, the electrostatic potential needs to be recomputed.\n" ;
    message +=   "  This is not currently implemented.\n";
    message +=   "3. Only linear polymer architectures are supported.\n";

    cout << message << endl;

    int new_molec_id = molecID[ns - 1] + 1;
    // Store the static information (bond types, connectivity, etc)
    for (int i = 0; i < n_molec_sites; i++) {
      int s1 = first_site;

      molecID[ns + i] = new_molec_id;
      tp[ns + i] = tp[s1 + i];

      if (Charges::do_charges == 1)
        charges[ns + i] = charges[s1 + i];

      n_bonds[ns + i] = n_bonds[s1 + i];
      for (int j = 0; j < n_bonds[ns + i]; j++) {
        int delta_ind = s1 + i - bonded_to[s1 + i][j];
        bonded_to[ns + i][j] = ns + i - delta_ind;
        bond_type[ns + i][j] = bond_type[s1 + i][j];
      }

      n_angles[ns + i] = n_angles[s1 + i];
      for (int j = 0; j < n_angles[ns + i]; j++) {
        int delta_ind = s1 + i - angle_first[s1 + i][j];
        angle_first[ns + i][j] = ns + i - delta_ind;

        delta_ind = s1 + i - angle_end[s1 + i][j];
        angle_end[ns + i][j] = ns + i - delta_ind;

        delta_ind = s1 + i - angle_mid[s1 + i][j];
        angle_mid[ns + i][j] = ns + i - delta_ind;

        angle_type[ns + i][j] = angle_type[s1 + i][j];
      }
    }

    set_optional_args(iss);

}


Widom::~Widom(){

}


void Widom::allocStorage() {

    ////////////////////////////////////////////
    // Set the intermittant storage variables //
    ////////////////////////////////////////////
    
    // This value needs to be updated to account for the number of molecules
    int number_stored = log_freq * (num_configs + 1) / compute_freq ; // The +1 is just a buffer
    this->fstore1 = (float*) malloc(number_stored * sizeof(float));
    this->fstore2 = (float*) malloc(number_stored * sizeof(float));

    num_sites = 0;

    // Initialize output file
    FILE *otp;
    char nm[50];
    sprintf(nm, "Widom%d.dat", compute_id);
    otp = fopen(nm, "w");
    fclose(otp);
}

void Widom::doCompute() {
  int ns_per_molec = num_sites;

  prepareDensityFields();
  calc_properties(0);

  float Unbi = Unb;

  // Loop over the number of trial configurations
  for (int t = 0; t < num_configs; t++) {
    // Place the first monomer at random
    for (int j = 0; j < Dim; j++)
      x[ns][j] = ran2() * L[j];

    // Loop over the remaining sites in the molecule
    for (int i = ns + 1; i < ns + ns_per_molec; i++) {
      // Check if bonded to something already placed
      int bonded_to_ind = -1;
      int btp = -1;
      for (int k = 0; k < n_bonds[i]; k++) {
        if (bonded_to[i][k] < i) {
          bonded_to_ind = bonded_to[i][k];
          btp = bond_type[i][k];
          break;
        }
      }

      // Site not bonded to anything (could be a counterion), place at random
      if (bonded_to_ind < 0) {
        for (int j = 0; j < Dim; j++)
          x[i][j] = float(ran2()) * L[j];
      }

      // Site is bonded to something, use Boltzmann inversion of bond potential
      // to find the new bond length and place the new particle
      else {
        if (btp < 0)
          die("Error in bond type in Compute Widom!");

        float bk = bond_k[btp];
        float r0 = bond_req[btp];

        float dr[3], mdr = 0.f;

        // Expected fluctuations around equil length
        float stdev = sqrtf(1. / 2.0 / bk);

        for (int j = 0; j < Dim; j++) {
          dr[j] = float(gasdev2()) * stdev;
          mdr += dr[j] * dr[j];
        }
        mdr = sqrtf(mdr);

        // Scale the Gaussian distributed components by the expected std dev
        for (int j = 0; j < Dim; j++) {
          x[i][j] = x[bonded_to_ind][j] + dr[j] * (r0 + 1.0f);
          if (x[i][j] >= L[j])
            x[i][j] -= L[j];
          else if (x[i][j] < 0.f)
            x[i][j] += L[j];
        }

      }  // Place bonded monomers

    }  // i=ns+1 : ns+ns_per_molec;

    // Back up the global variables that will change
    int ns_Grid_bkp = ns_Grid;

    // Update ns, ns_Grid
    ns = ns + ns_per_molec;
    ns_Grid = (int)ceil((float)(ns) / ns_Block);

    // Send positions to GPU
    update_device_positions(x, d_x);

    prepareDensityFields();
    float Unb_new = calc_nbEnergy();

    float Ubonded_molec = calc_moleculeBondedEnergy(molecID[ns - 1]);  // Uses ns-1 bc ns updated above

    float deltaU = Unb_new - Unbi + Ubonded_molec;

    // Store the dU
    int ind = num_data_pts++;
    this->fstore1[ind] = deltaU;


    // Remove the extra molecule from the density fields
    d_removeMolecFromFields<<<ns_Grid, ns_Block>>>(d_all_rho, molecID[ns - 1],
                                                   d_molecID, d_grid_W, d_grid_inds, d_typ, gvol, grid_per_partic, ns, M, Dim);

    // Restore ns, ns_Grid
    ns = ns - ns_per_molec;
    ns_Grid = ns_Grid_bkp;

  }  // t=0:n_trial_configs
}

void Widom::writeResults(){

    FILE *otp;
    char nm[50];
    sprintf(nm, "Widom%d.dat", compute_id);
    otp = fopen(nm, "a");
    for ( int i=0 ; i<num_data_pts; i++ ) 
        fprintf(otp, "%lf \n", this->fstore1[i]);

    fclose(otp);

    num_data_pts = 0;
}