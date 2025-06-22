#include "PS_Box.h"
#include "random.h"
#include "include_libs.h"
#include "gsd.h"
#include <algorithm>
#include <map>

void die(const char*);





void PS_Box::writeGSDtraj() {

    int i;

    gsd_handle gsd_file; 

    if (totSteps == 0){
        std::vector<unsigned int> types(nstot), molecule_ids(nstot);

        for (i = 0; i < nstot; i++) {
            types[i] = intSpecies[i] + 1;
            molecule_ids[i] = mID[i] + 1;
        }

        std::vector<float> masses(nstot);

        for (i = 0; i < nstot; i++) {
            masses[i] = speciesMass[intSpecies[i]];
        }


        auto version = gsd_make_version(1, 4);
        gsd_create_and_open(&gsd_file, gsd_name.c_str(), "gpu-tild", "hoomd", version, gsd_open_flag::GSD_OPEN_APPEND, 0);

        unsigned int frame = totSteps;
        gsd_write_chunk(
            &gsd_file, "configuration/step", gsd_type::GSD_TYPE_UINT64,
            1, 1, 0, &frame
        );
        
        unsigned int gsdDim = 3;
        gsd_write_chunk(
                &gsd_file, "configuration/dimensions", gsd_type::GSD_TYPE_UINT8,
                1, 1, 0, &gsdDim
                );


        std::vector<float> box(6,0);
        for ( int j=0 ; j<Dim ; j++ ) {
            box[j] = L[j];
        }
        if ( Dim == 2 ) box[2] = 5.0;
        
        gsd_write_chunk(
                &gsd_file, "configuration/box", gsd_type::GSD_TYPE_FLOAT,
                6, 1, 0, box.data() );

        {
            unsigned int ntypes = nstot;
            gsd_write_chunk(&gsd_file, "particles/N", gsd_type::GSD_TYPE_UINT32, 1, 1, 0, &ntypes);
        }
        gsd_write_chunk(&gsd_file, "particles/mass", gsd_type::GSD_TYPE_FLOAT, masses.size(), 1, 0, masses.data());
        
        // Write the particle types 
        gsd_write_chunk(&gsd_file, "particles/typeid", gsd_type::GSD_TYPE_UINT32, nstot, 1, 0, types.data());

        // Write the particle molecule ids
        gsd_write_chunk(&gsd_file, "log/particles/moleculeid", gsd_type::GSD_TYPE_UINT32, nstot, 1, 0, molecule_ids.data());



        int max_len = 10;
        char* names = (char*) calloc((nTypes+1) * 13,  sizeof(char));

        std::string str("type");
        for (i = 0; i < (nTypes+1); i++) {
            strcpy(names + i * max_len, (str + std::to_string(i)).c_str());
        }


        gsd_write_chunk(&gsd_file, "particles/types", gsd_type::GSD_TYPE_INT8, (nTypes + 1), max_len, 0, names);

        unsigned int N_bonds = nBondsTot;
        // Write the number of bonds
        gsd_write_chunk(&gsd_file, "bonds/N", gsd_type::GSD_TYPE_UINT32, 1, 1, 0, &N_bonds);

        // // Write the number of bond types
        // gsd_write_chunk(&gsd_file, "bond/types", gsd_type::GSD_TYPE_UINT32, 1, 1, 0, &nbond_types);

        // Write the bondids
        gsd_write_chunk(&gsd_file, "bonds/typeid", gsd_type::GSD_TYPE_UINT32, nBondsTot, 1, 0, list_of_bond_type.data());

        // Write the bonds/group
        gsd_write_chunk(&gsd_file, "bonds/group", gsd_type::GSD_TYPE_UINT32, nBondsTot, 2, 0, list_of_bond_partners.data());

        // // Write the number of angle types
        // gsd_write_chunk(&gsd_file, "angle/types", gsd_type::GSD_TYPE_UINT32, 1, 1, 0, &nangle_types);

        // Write the number of angles
        gsd_write_chunk(&gsd_file, "angles/N", gsd_type::GSD_TYPE_UINT32, 1, 1, 0, &nAnglesTot);

        // Write the angleids
        gsd_write_chunk(&gsd_file, "angles/typeid", gsd_type::GSD_TYPE_UINT32, nAnglesTot, 1, 0, list_of_angle_type.data());

        // Write the angles/group
        gsd_write_chunk(&gsd_file, "angles/group", gsd_type::GSD_TYPE_UINT32, nAnglesTot, 3, 0, list_of_angle_partners.data());

    }
    else{
        gsd_open(&gsd_file, gsd_name.c_str(), gsd_open_flag::GSD_OPEN_APPEND);
        unsigned int frame = totSteps;
        gsd_write_chunk(
            &gsd_file, "configuration/step", gsd_type::GSD_TYPE_UINT64,
            1, 1, 0, &frame
        );
        
    }

    // Transfer coordinates from device to host
    //x = d_x;


    // Make a copy of positions that can be shifted by Lh
    // std::cout << "allocating..." ; fflush(stdout);
    float* h_ns_float;
    h_ns_float = (float*) malloc(nstot*3*sizeof(float));
    if ( h_ns_float == NULL ) die("failed to allocate h_ns_float");

    float* xtmp;
    xtmp = (float*) malloc( nstot*Dim*sizeof(float));

    // std::cout << "copying..." ; fflush(stdout);
    cudaMemcpy(xtmp, d_x, nstot*Dim*sizeof(float), cudaMemcpyDeviceToHost);

    for (i = 0; i < nstot; i++) {
        for (int j = 0; j < Dim; j++) {
            // h_ns_float[i * 3 + j] = x[i * Dim + j] - Lh[j];
            h_ns_float[i * 3 + j] = xtmp[i * Dim + j] - Lh[j];
        }
        if ( Dim == 2 ) h_ns_float[i*3+2] = 0.0;

    }

    // std::cout << "writing..." ; fflush(stdout);
    gsd_write_chunk(&gsd_file, "particles/position", gsd_type::GSD_TYPE_FLOAT, nstot, 3, 0, h_ns_float);

    if ( doCharges ) die("Charges not set up yet in write gsd routine");
        // gsd_write_chunk(&gsd_file, "particles/charge", gsd_type::GSD_TYPE_FLOAT, ns, 1, 0, charges);

    // std::cout << "closing..." ; fflush(stdout);
    gsd_end_frame(&gsd_file);
    gsd_close(&gsd_file);

    // std::cout << "freeing..." ; fflush(stdout);
    free(h_ns_float);
    free(xtmp);


}



// Reads a GSD frame as an input configuration
// This routine assumes the following have been defined:
// all "species" keyword/classes
// any non-default values of MAXBONDS, MAXANGLES
void PS_Box::readGSDtraj(const char* file_name, int frame_num, int process){

    // If process == 0, then we are doing a read_resume
    // If process == 1, then we are doing a read_restart

    int base_index = 0;

    gsd_handle gsd_file;
    int f =	gsd_open(&gsd_file, file_name, gsd_open_flag::GSD_OPEN_READONLY);
    if (f){
        std::cout << "Error opening gsd file" << std::endl;
        exit(1);
    }

    int tmp_frame = gsd_get_nframes(&gsd_file);
    if (tmp_frame < 0){
        die("No frames in the gsd file");
    }

    if (frame_num < 0){
        frame_num = tmp_frame - 1;
    }

    if (frame_num > tmp_frame){
        std::cout << "Frame number is too large" << std::endl;
        std::string str = "Frame number is too large. Max frame number is " + std::to_string(tmp_frame);
        die(str);
    }

    std::cout << frame_num << std::endl;

    // Get the box dimensions
    const gsd_index_entry* chunk_index;
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "configuration/dimensions");
    if (chunk_index == NULL) {

        chunk_index = gsd_find_chunk(&gsd_file, base_index, "configuration/dimensions");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'configuration/dimensions' in the GSD file.";
            die(me);
        }
    }
    gsd_read_chunk(&gsd_file, &Dim, chunk_index);
    n_P_comps = int(Dim*(Dim+ 1))/2;

    
    // Read in the number of atoms in the box
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "particles/N");
    int tmp_ns = 0;
    if (chunk_index == NULL) {
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "particles/N");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'particles/N' in the GSD file.";
            die(me);
        }
    }

    gsd_read_chunk(&gsd_file, &tmp_ns, chunk_index);
    if (process == 0 && tmp_ns != nstot){
        std::string me = "Error: The number of atoms in the GSD file does not match the number of atoms in the simulation.";
        die(me);
    }
    else {
        nstot = tmp_ns;
    }

    
    // read the charges of the particles
    std::vector<float> charges_tmp(nstot);
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "particles/charge");
    if (chunk_index != NULL){
        doCharges = true;
        gsd_read_chunk(&gsd_file, charges_tmp.data(), chunk_index);
    }


    // Read in the box size
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "configuration/box");
    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "configuration/box");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'configuration/box' in the GSD file.";
            die(me);
        }
    }
    gsd_read_chunk(&gsd_file, &L, chunk_index);

    V = 1;
    for (int i = 0; i < Dim; i++) {
        Lh[i] = L[i] / 2.0;
        V *= L[i];
    }


    // Read in the number of bonds
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "bonds/N");
    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "bonds/N");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'bonds/N' in the GSD file.";
            die(me);
        }
    }
    gsd_read_chunk(&gsd_file, &nBondsTot, chunk_index);

    std::cout << "  from GSD, nBondsTot = " << nBondsTot << std::endl;

    // Read in the number of angles
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "angles/N");
    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "angles/N");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'angles/N' in the GSD file.";
            die(me);
        }
    }
    gsd_read_chunk(&gsd_file, &nAnglesTot, chunk_index);

    std::cout << "n_total_angles = " << nAnglesTot << std::endl;

    // Read in all the particle types to determine the number of types
    // Instead of sorting, we could just allocate the highest amount that there would be in the box
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "particles/typeid");
    std::vector<int> typeids(nstot), typ_id(nstot);
    typeids.resize(nstot);
    typ_id.resize(nstot);

    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "particles/typeid");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'particles/typeid' in the GSD file.";
            die(me);
        }
    }

    gsd_read_chunk(&gsd_file, typeids.data(), chunk_index);

    typ_id = typeids;
    {
        std::sort(typ_id.begin(), typ_id.end());
        int max_val = *max_element(typ_id.begin(), typ_id.end());
        int max_valu2 = std::unique(typ_id.begin(), typ_id.end()) - typ_id.begin();
        nTypes = max(max_valu2, max_val - 1 );
    }

    std::cout << "ntypes = " << nTypes << std::endl;

    // Read in all the particle bonds to determine the number of bonds
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "bonds/typeid");
    std::vector<unsigned int> bonds(nBondsTot);

    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "bonds/typeid");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'bonds/typeid' in the GSD file.";
            die(me);
        }
    }

    gsd_read_chunk(&gsd_file, bonds.data(), chunk_index);
    list_of_bond_type = bonds;

    if (nBondsTot > 0) {
        std::sort(bonds.begin(), bonds.end());
        int max_val = *max_element(bonds.begin(), bonds.end());
        int max_valu2 = std::unique(bonds.begin(), bonds.end()) - bonds.begin();
        nBondTypes = max(max_valu2, max_val - 1 );
    }

    std::cout << "nbond_types = " << nBondTypes << std::endl;


    // Read in all the particle angles to determine the number of angles
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "angles/typeid");
    std::vector<unsigned int> angles(nAnglesTot);

    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "angles/typeid");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'angles/typeid' in the GSD file.";
            die(me);
        }
    }

    gsd_read_chunk(&gsd_file, angles.data(), chunk_index);
    list_of_angle_type = angles;
    if (nAnglesTot > 0) {
        std::sort(angles.begin(), angles.end());
        int max_val = *max_element(angles.begin(), angles.end());
        int max_valu2 = std::unique(angles.begin(), angles.end()) - angles.begin();
        nAngleTypes = max(max_valu2, max_val - 1 );
    }
    else {
        nAngleTypes = 0;
    }

    std::cout << "nangle_types = " << nAngleTypes << std::endl;

    list_of_angle_type.resize(nAnglesTot);

    // Read in the atoms participating in each bond
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "bonds/group");
    std::vector<unsigned int> bond_partners(nBondsTot*2);

    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "bonds/group");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'bonds/group' in the GSD file.";
            die(me);
        }
    }

    gsd_read_chunk(&gsd_file, bond_partners.data(), chunk_index);
    list_of_bond_partners = bond_partners;


    // Read in the atoms participating in each angle
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "angles/group");
    std::vector<unsigned int> angle_partners(nAnglesTot*3);

    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "angles/group");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'angles/group' in the GSD file.";
            die(me);
        }
    }

    gsd_read_chunk(&gsd_file, angle_partners.data(), chunk_index);
    list_of_angle_partners = angle_partners;


    // Read in the molecule ids
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "log/particles/moleculeid");
    std::vector<unsigned int> molecule_id(nstot);

    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "log/particles/moleculeid");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'log/particles/moleculeid' in the GSD file.";
            die(me);
        }
    }

    // Read in the particle moleculeids
    gsd_read_chunk(&gsd_file, molecule_id.data(), chunk_index);
    auto local = molecule_id;

    {
        std::sort(local.begin(), local.end());
        int max_val = *max_element(local.begin(), local.end());
        int max_valu2 = std::unique(local.begin(), local.end()) - local.begin();
        nMolecules  = max(max_valu2, max_val - 1 );
    }

    std::cout << "n_molecules = " << nMolecules << std::endl;

    // Read in particle masses
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "particles/mass");
    std::vector<float> masses(nstot);

    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "particles/mass");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'particles/mass' in the GSD file.";
            die(me);
        }
    }

    gsd_read_chunk(&gsd_file, masses.data(), chunk_index);

    // This commented out by RAR
    // Not clear its needed with masses associated with particle species
    // std::map<int, int> map_of_particle_id_mass;

    // for (int i = 0; i < ns; i++) {
    //     if (map_of_particle_id_mass.find(typeids.at(i)) == map_of_particle_id_mass.end()) {
    //         map_of_particle_id_mass.insert(std::pair<int, int>(typeids.at(i), masses.at(i)));
    //     }
    //     else {
    //         // Check if the masses are the same
    //         if (map_of_particle_id_mass.at(typeids.at(i)) != masses.at(i)) {
    //             std::string me = "Error: The masses of the particles with the same type are not the same.";
    //             die(me);
    //         }
    //     }
    // }



    // This is for reading in configuration file data
    if (process != 0) {
        allocHostParticleArrays(nstot);

        printf("Particle memory allocated on host via GSD!\n");


        // Store the types
        for (int i = 0; i < nstot; i++){
            intSpecies[i] = typeids.at(i) - 1;
        }

        // Store the molecule idsids
        for (int i = 0; i < nstot; i++){
            mID[i] = molecule_id.at(i) - 1;
        }

        // As above, commented out bc maybe not necessary?
        // Assign the masses using the map
        // for (auto& x: map_of_particle_id_mass) {
        //     mass[x.first - 1] = x.second;
        // }


        // Zero out the counters
        for ( int i=0 ; i<nstot; i++ ) {
            nBonds[i] = 0;
            nAngles[i] = 0;
        }

        // store the bonds
        for (unsigned int i = 0;  i<list_of_bond_type.size(); i++){
            int i1 = list_of_bond_partners.at(i*2);
            int i2 = list_of_bond_partners.at(i*2+1);
            int b_type = list_of_bond_type.at(i);

            bondedTo[i1 * MAXBONDS + nBonds[i1]] = i2;
            bondType[i1 * MAXBONDS + nBonds[i1]] = b_type;
            nBonds[i1]++;

            bondedTo[i2 * MAXBONDS + nBonds[i2]] = i1;
            bondType[i2 * MAXBONDS + nBonds[i2]] = b_type;
            nBonds[i2]++;
        }


        // store the angles
        for (unsigned int i = 0; i < list_of_angle_type.size(); i++){
            int i1 = list_of_angle_partners.at(i*3);
            int i2 = list_of_angle_partners.at(i*3+1);
            int i3 = list_of_angle_partners.at(i*3+2);

            int a_type = list_of_angle_type.at(i);

            int na = nAngles[i1];
            angleGroup[i1*MAXANGLES*3 + na*3 + 0] = i1;
            angleGroup[i1*MAXANGLES*3 + na*3 + 1] = i2;
            angleGroup[i1*MAXANGLES*3 + na*3 + 2] = i3;
            angleType[i1*MAXANGLES + na] = a_type;
            nAngles[i1] += 1;

            na = nAngles[i2];
            angleGroup[i2*MAXANGLES*3 + na*3 + 0] = i1;
            angleGroup[i2*MAXANGLES*3 + na*3 + 1] = i2;
            angleGroup[i2*MAXANGLES*3 + na*3 + 2] = i3;
            angleType[i2*MAXANGLES + na] = a_type;
            nAngles[i2] += 1;

            na = nAngles[i3];
            angleGroup[i3*MAXANGLES*3 + na*3 + 0] = i1;
            angleGroup[i3*MAXANGLES*3 + na*3 + 1] = i2;
            angleGroup[i3*MAXANGLES*3 + na*3 + 2] = i3;
            angleType[i3*MAXANGLES + na] = a_type;
            nAngles[i3] += 1;
 
        }
    }


    // Make a copy of positions that can be shifted by Lh
    float* h_ns_float;
    h_ns_float = (float*) malloc(nstot*Dim*sizeof(float));
    if ( h_ns_float == NULL ) die("failed to allocate h_ns_float");

    // read in the position of the particles
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "particles/position");
    gsd_read_chunk(&gsd_file, h_ns_float, chunk_index);
    if (chunk_index == NULL) {
        std::string me = "error: could not find the chunk 'particles/position' in the gsd file.";
        die(me);
    }

    // Store the positions
    for (int i = 0; i < nstot; i++) {
        for (int j = 0; j < Dim; j++) {
            x[ i * Dim + j] = h_ns_float[i * Dim + j] + Lh[j];
        }
    }

    // if (Charges::do_charges){
    //     for (int idx = 0; idx < charges_tmp.size(); idx++)
    //         charges[idx] = charges_tmp.at(idx);
    // }


    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "configuration/step");

    if (chunk_index == NULL) {
        std::string me = "error: could not find the chunk 'configuration/step' in the gsd file.";
        die(me);
    }

    int tmp_step;
    gsd_read_chunk(&gsd_file, &tmp_step, chunk_index);
    totSteps = tmp_step;

    gsd_close(&gsd_file);
    free(h_ns_float);
}




void PS_Box::GSDinit() {
    // For dynamic bonds, may make sense to "reserve" for MAXBONDS*nstot instead of
    // using resize, which initializes values. Another routine that updates bond 
    // lists could be called if bond number is dynamic.
    list_of_bond_type.resize(nBondsTot,0);
    list_of_bond_partners.resize(nBondsTot*2,0);

    int bcount = 0;
    for ( int i=0 ; i<nstot; i++ ) {
        for ( int j=0 ; j<nBonds[i]; j++ ) {
            
            if ( bondedTo[i*MAXBONDS + j] > i ) {
                list_of_bond_partners[bcount*2 + 0] = i;
                list_of_bond_partners[bcount*2 + 1] = bondedTo[i*MAXBONDS + j];
                list_of_bond_type[bcount] = bondType[i*MAXBONDS + j];

                bcount++;
            }
        }
    }

    list_of_angle_type.resize(nAnglesTot,0);
    list_of_angle_partners.resize(nAnglesTot*3,0);
}