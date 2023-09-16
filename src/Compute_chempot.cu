// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "global_templated_functions.h"
#include "Compute_chempot.h"

using namespace std; 


ChemPot::~ChemPot(){ }

ChemPot::ChemPot(std::istringstream& iss) : Compute(iss){
    style="chemical_potential";

    readRequiredParameter(iss, first_mol_id); first_mol_id--;
    readRequiredParameter(iss, last_mol_id); last_mol_id--;
    readRequiredParameter(iss, fraction_to_sample);

    // Number of candidate molecules
    nmolecules = (last_mol_id - first_mol_id) + 1;  // +1 bc IDs meant to be inclusive

    // Number of molecules to remove per call to Compute();
    ntries = (int)(float(nmolecules) * fraction_to_sample);
    if ( first_mol_id < 0 ) 
        die("Invalid first molecule ID in compute chemical_potential");

    if ( last_mol_id > molecID[ns-1] ) {
        cout << "Failing to compute chemical potential. This may occur if the last atom isn't the last molecule";
        die("Invalid second molecule ID in compute chemical_potential");
    }
    
    
    cout << "\nCalculating chemical potential for molecules " << first_mol_id+1 << " through " << last_mol_id+1 << 
    "\nNOTICE NOTICE NOTICE\n" 
    "1. The chemical potential calculation assumes the molecules are in order in by site ID."
    " I.e., the first site of molecule 3 will NOT have a larger index than molecule 5.\n"
    "2. To use on charged molecules, the electrostatic potential needs to be recomputed.\n" 
    " This is not currently implemented.\n" <<
    "Chemical potential calculation will remove " << ntries << " molecules each call." << endl;

    // Check that the molecules are charge neutral if charges are on
    if (Charges::do_charges) {
      float net_charge = 0.0f;
      for (int i = 0; i < ns; i++) {
        if (molecID[i] == first_mol_id)
          net_charge += charges[i];
      }

      if (fabs(net_charge) > 1.0E-5)
        die("chemical potential molecule not charge neutral!");
    }

    // Check that the first and last molecule have the same number of sites
    ns_per_molec = 0;
    int last_molec = 0;
    for ( int i=0 ; i<ns ; i++ ) {
        if ( molecID[i] == first_mol_id ) ns_per_molec++;
        if ( molecID[i] == last_mol_id ) last_molec++;
    }
    if ( last_molec != ns_per_molec ) 
        die("Mismatch in the number of sites in the first, last molecule in compute chemical_potential" + to_string(first_mol_id) + " " + to_string(last_mol_id));

    set_optional_args(iss);

}

void ChemPot::doCompute(){

    d_copyPositions<<<ns_Grid, ns_Block>>>(this->d_fdata, d_x, Dim, ns);

    // Regenerate density fields
    prepareDensityFields();

    // Calculate initial potential energy
    calc_properties(0);
    
    // Initial energy value
    float Uinit = Upe;
    float Unbi = Unb;
    
    
    // Back up the global variables that will change
    int ns_bkp = ns;
    int ns_Grid_bkp = ns_Grid;

    // Update ns, ns_Grid
    ns = ns - ns_per_molec;
    ns_Grid = (int)ceil((float)(ns) / ns_Block);
    

    // This will loop over the number of tries. For each attempt, a random 
    // molecule in the range specified will be randomly removed, and the energy 
    // recalculated. 
    for ( int i=0 ; i<ntries ; i++ ) {

        // Select a number on [0, nmolecules). Upper limit is open bc nmolecules already shifted by 1 above
        int molecID_shift = (int)((float)ran2() * (float)nmolecules);

        int chosenMolec = molecID_shift + first_mol_id;

        // cout << "Selected moddlec's bonded energy: " << molec_Ubond[ chosenMolec ] << endl;

        d_removeMolecFromFields<<<ns_Grid, ns_Block>>>(d_all_rho, chosenMolec,
            d_molecID, d_grid_W, d_grid_inds, d_typ, gvol, grid_per_partic, ns, M, Dim);

        float new_Unb = calc_nbEnergy();

        d_restoreMolecToFields<<<ns_Grid, ns_Block>>>(d_all_rho, chosenMolec,
            d_molecID, d_grid_W, d_grid_inds, d_typ, gvol, grid_per_partic, ns, M, Dim);

        float orig_Unb = calc_nbEnergy();

        float Unew = Uinit - molec_Ubond[ chosenMolec] - Unbi + new_Unb;

        float deltaU = Uinit - Unew;

        // cout << "Unbi: " << Unbi << " removed: " << new_Unb << " orig_Unb: " << orig_Unb << endl;

        // Index of the first 
        // int first_site_index = first_ind + molecID_shift * ns_per_molec;

        // Remove the chosen molecule
        //d_removeMolecule<<<ns_Grid, ns_Block>>>(d_x, this->d_fdata, first_site_index, ns_per_molec, Dim, ns);

        // Regenerate density fields
        //prepareDensityFields();

        //calc_properties(0);

        //cout << "Initial bond energy: " << Ubondi << " delta: " << Ubondi-Ubond << endl;
        //cout << "Brute forced Unb: " << Unb << endl;
        //exit(1);

        // float deltaU = Uinit - Upe;
        
        // Store the dU
        this->fstore1[num_data_pts++] = deltaU;

        d_copyPositions<<<ns_Grid_bkp, ns_Block>>>(d_x, this->d_fdata, Dim, ns_bkp);
        ns = ns_bkp;
        ns_Grid = ns_Grid_bkp;
    
        ns = ns - ns_per_molec;
        ns_Grid = (int)ceil((float)(ns) / ns_Block);

    }


    ///////////////////////////
    // Restore original data //
    ///////////////////////////
    ns = ns_bkp;
    ns_Grid = ns_Grid_bkp;
    d_copyPositions<<<ns_Grid, ns_Block>>>(d_x, this->d_fdata, Dim, ns);
    
    // Regenerate density fields
    prepareDensityFields();

}

void ChemPot::writeResults(){

    // Append fstore1 to file chemPotential.dat  
    ofstream outfile;
    outfile.open("chemPotential" + to_string(compute_id) +".dat", ios::app);
    for ( int i=0 ; i<num_data_pts ; i++ ) {
        outfile << fstore1[i] << "\n";
    }
    outfile.close();

    num_data_pts = 0;
}

void ChemPot::allocStorage(){

    ////////////////////////////////////////////
    // Set the intermittant storage variables //
    ////////////////////////////////////////////

    // This value needs to be updated to account for the number of molecules
    number_stored = log_freq * (ntries + 1) / this->compute_freq;  // The +1 is just a buffer

    this->fstore1 = (float *)malloc(number_stored * sizeof(float));
    this->fstore2 = (float *)malloc(number_stored * sizeof(float));

    cudaMalloc(&this->d_fdata, ns * Dim * sizeof(float));

    check_cudaError("Allocating d_fdata in compute");

    num_data_pts = 0;

    // Initialize output file
    ofstream outfile;
    outfile.open("chemPotential" + to_string(compute_id) +".dat", ios::out);
    outfile.close();
}