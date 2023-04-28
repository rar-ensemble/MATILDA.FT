// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "gsd.h"
#include <algorithm>
#include <map>

using namespace std;
#define max(a,b) ((a)>(b)?(a):(b))

void allocate_host_particles(void);

// Must know Dim before read_config() is called! //
void read_config(const char *nm) {

	int i, di, ind, ltp;
	double dx, dy;
	char tt[120];

	FILE* inp;
	inp = fopen(nm, "r");
	if (inp == NULL) {
		char death[50];
		sprintf(death, "Failed to open %s!\n", nm);
		die(death);
	}

	(void)!fgets(tt, 120, inp);
	(void)!fgets(tt, 120, inp);

	(void)!fscanf(inp, "%d", &ns);      (void)!fgets(tt, 120, inp);
	(void)!fscanf(inp, "%d", &n_total_bonds);  (void)!fgets(tt, 120, inp);
	(void)!fscanf(inp, "%d", &n_total_angles);  (void)!fgets(tt, 120, inp);

	(void)!fgets(tt, 120, inp);

	(void)!fscanf(inp, "%d", &ntypes);  (void)!fgets(tt, 120, inp);
	(void)!fscanf(inp, "%d", &nbond_types);  (void)!fgets(tt, 120, inp);
	(void)!fscanf(inp, "%d", &nangle_types);  (void)!fgets(tt, 120, inp);

	// Read in box shape
	(void)!fgets(tt, 120, inp);
	(void)!fscanf(inp, "%lf %lf", &dx, &dy);   (void)!fgets(tt, 120, inp);
	L[0] = (float)(dy - dx);
	(void)!fscanf(inp, "%lf %lf", &dx, &dy);   (void)!fgets(tt, 120, inp);
	L[1] = (float)(dy - dx);
	(void)!fscanf(inp, "%lf %lf", &dx, &dy);   (void)!fgets(tt, 120, inp);
	if (Dim > 2)
		L[2] = (float)(dy - dx);
	else
		L[2] = 1.0f;

	V = 1.0f;
	for (i = 0; i < Dim; i++) {
		Lh[i] = 0.5f * L[i];
		V *= L[i];
	}


	// Allocate memory for positions //
	allocate_host_particles();
	
	printf("Particle memory allocated on host!\n");

	(void)!fgets(tt, 120, inp);

	// Read in particle masses
	(void)!fgets(tt, 120, inp);
	(void)!fgets(tt, 120, inp);
	for (i = 0; i < ntypes; i++) {
		(void)!fscanf(inp, "%d %lf", &di, &dx); (void)!fgets(tt, 120, inp);
		mass[di - 1] = (float) dx;
	}
	(void)!fgets(tt, 120, inp);

	
	// Read in atomic positions
	(void)!fgets(tt, 120, inp);
	(void)!fgets(tt, 120, inp);

    n_molecules = -5 ;
	for (i = 0; i < ns; i++) {
		if (feof(inp)) die("Premature end of input.conf!");

		(void)!fscanf(inp, "%d %d %d", &ind, &di, &ltp);
		ind -= 1;

        if ( di > n_molecules ) 
          n_molecules = di;

		molecID[ind] = di - 1;
		tp[ind] = ltp - 1;

		for (int j = 0; j < Dim; j++) {
			(void)!fscanf(inp, "%lf", &dx);
			x[ind][j] = (float) dx;
		}

		(void)!fgets(tt, 120, inp);
	}
	(void)!fgets(tt, 120, inp);


	// Read in bond information
	(void)!fgets(tt, 120, inp);
	(void)!fgets(tt, 120, inp);
	
	for (i = 0; i < ns; i++)
		n_bonds[i] = 0;

	list_of_bond_partners.reserve(n_total_bonds*2);
	list_of_bond_type.reserve(n_total_bonds);

	for (i = 0; i < n_total_bonds; i++) {
		if (feof(inp)) die("Premature end of input.conf!");
		(void)!fscanf(inp, "%d", &di);
		(void)!fscanf(inp, "%d", &di);
		int b_type = di - 1;

		(void)!fscanf(inp, "%d", &di);
		int i1 = di - 1;

		(void)!fscanf(inp, "%d", &di);
		int i2 = di - 1;

		if (i2 < i1) {
			di = i2;
			i2 = i1;
			i1 = di;
		}

		if ( n_bonds[i1] >= MAX_BONDS ) {
			string emsg = "Too many bonds on particle " + i1;
			std::cout << "Increase MAX_BONDS in input file and run again " << std::endl;
			die(emsg);
		}

		bonded_to[i1][n_bonds[i1]] = i2;
		bond_type[i1][n_bonds[i1]] = b_type;
		n_bonds[i1]++;

		if ( n_bonds[i2] >= MAX_BONDS ) {
			string emsg = "Too many bonds on particle " + i2;
			std::cout << "Increase MAX_BONDS in input file and run again " << std::endl;
			die(emsg);
		}

		bonded_to[i2][n_bonds[i2]] = i1;
		bond_type[i2][n_bonds[i2]] = b_type;
		n_bonds[i2]++;

		list_of_bond_type.push_back(b_type);
		list_of_bond_partners.push_back(i1);
		list_of_bond_partners.push_back(i2);

	}
	(void)!fgets(tt, 120, inp);



	list_of_angle_partners.reserve(n_total_angles*3);
	list_of_angle_type.reserve(n_total_angles);

	// Read in angle information
	for (i = 0; i < ns; i++)
		n_angles[i] = 0;

	(void)!fgets(tt, 120, inp);
	(void)!fgets(tt, 120, inp);
	for (i = 0; i < n_total_angles; i++) {

		(void)!fscanf(inp, "%d", &di);
		(void)!fscanf(inp, "%d", &di);

		int a_type = di - 1;

		(void)!fscanf(inp, "%d", &di);
		int i1 = di - 1;

		(void)!fscanf(inp, "%d", &di);
		int i2 = di - 1;

		(void)!fscanf(inp, "%d", &di);
		int i3 = di - 1;

		if (i3 < i1) {
			di = i3;
			i3 = i1;
			i1 = di;
		}

		if ( n_angles[i1] >= MAX_ANGLES ) {
			string emsg = "Too many angles on particle " + i1;
			std::cout << "Increase MAX_ANGLES in input file and run again " << std::endl;
			die(emsg.c_str());
		}

		int na = n_angles[i1];
		angle_first[i1][na] = i1;
		angle_mid[i1][na] = i2;
		angle_end[i1][na] = i3;
		angle_type[i1][na] = a_type;
		n_angles[i1] += 1;


		if ( n_angles[i2] >= MAX_ANGLES ) {
			string emsg = "Too many angles on particle " + i2;
			std::cout << "Increase MAX_ANGLES in input file and run again " << std::endl;
			die(emsg.c_str());
		}

		na = n_angles[i2];
		angle_first[i2][na] = i1;
		angle_mid[i2][na] = i2;
		angle_end[i2][na] = i3;
		angle_type[i2][na] = a_type;
		n_angles[i2] += 1;

		if ( n_angles[i3] >= MAX_ANGLES ) {
			string emsg = "Too many angles on particle " + i3;
			std::cout << "Increase MAX_ANGLES in input file and run again " << std::endl;
			die(emsg.c_str());
		}

		na = n_angles[i3];
		angle_first[i3][na] = i1;
		angle_mid[i3][na] = i2;
		angle_end[i3][na] = i3;
		angle_type[i3][na] = a_type;
		n_angles[i3] += 1;

		(void)!fgets(tt, 120, inp);
		
		list_of_angle_type.push_back(a_type);
		list_of_angle_partners.push_back(i1);
		list_of_angle_partners.push_back(i2);
		list_of_angle_partners.push_back(i3);

	}
    
    // Allocate memory to store molecular bonded energies
    molec_Ubond = (float*) calloc(n_molecules, sizeof(float));

	fclose(inp);
}

void read_charge_config(const char* nm) {

	int i, di, ind, ltp;
	float dx, dy, dcharge;
	char tt[120];

	FILE* inp;
	inp = fopen(nm, "r");
	if (inp == NULL) {
		char death[50];
		sprintf(death, "Failed to open %s!\n", nm);
		die(death);
	}

	(void)!fgets(tt, 120, inp);
	(void)!fgets(tt, 120, inp);

	(void)!fscanf(inp, "%d", &ns);      (void)!fgets(tt, 120, inp);
	(void)!fscanf(inp, "%d", &n_total_bonds);  (void)!fgets(tt, 120, inp);
	(void)!fscanf(inp, "%d", &n_total_angles);  (void)!fgets(tt, 120, inp);

	(void)!fgets(tt, 120, inp);

	(void)!fscanf(inp, "%d", &ntypes);  (void)!fgets(tt, 120, inp);
	(void)!fscanf(inp, "%d", &nbond_types);  (void)!fgets(tt, 120, inp);
	(void)!fscanf(inp, "%d", &nangle_types);  (void)!fgets(tt, 120, inp);

	// Read in box shape
	(void)!fgets(tt, 120, inp);
	(void)!fscanf(inp, "%f %f", &dx, &dy);   (void)!fgets(tt, 120, inp);
	L[0] = (float)(dy - dx);
	(void)!fscanf(inp, "%f %f", &dx, &dy);   (void)!fgets(tt, 120, inp);
	L[1] = (float)(dy - dx);
	(void)!fscanf(inp, "%f %f", &dx, &dy);   (void)!fgets(tt, 120, inp);
	if (Dim > 2)
		L[2] = (float)(dy - dx);
	else
		L[2] = 1.0f;

	V = 1.0f;
	for (i = 0; i < Dim; i++) {
		Lh[i] = 0.5f * L[i];
		V *= L[i];
	}


	// Allocate memory for positions //
	allocate_host_particles();

	printf("Particle memory allocated on host!\n");

	(void)!fgets(tt, 120, inp);

	// Read in particle masses
	(void)!fgets(tt, 120, inp);
	(void)!fgets(tt, 120, inp);
	for (i = 0; i < ntypes; i++) {
		(void)!fscanf(inp, "%d %f", &di, &dx); (void)!fgets(tt, 120, inp);
		mass[di - 1] = float(dx);
	}
	(void)!fgets(tt, 120, inp);


	// Read in atomic positions
	(void)!fgets(tt, 120, inp);
	(void)!fgets(tt, 120, inp);

    n_molecules = -5;

	for (i = 0; i < ns; i++) {
		if (feof(inp)) die("Premature end of input.conf!");

		(void)!fscanf(inp, "%d %d %d", &ind, &di, &ltp);
		ind -= 1;

        if ( di > n_molecules ) 
            n_molecules = di;

		molecID[ind] = di - 1;
		tp[ind] = ltp - 1;

		(void)!fscanf(inp, "%f", &dcharge);
		charges[ind] = dcharge;

		for (int j = 0; j < Dim; j++) {
			(void)!fscanf(inp, "%f", &dx);
			x[ind][j] = dx;
		}

		(void)!fgets(tt, 120, inp);
	}
	(void)!fgets(tt, 120, inp);

	// Read in bond information
	(void)!fgets(tt, 120, inp);
	(void)!fgets(tt, 120, inp);

	for (i = 0; i < ns; i++)
		n_bonds[i] = 0;

	list_of_bond_partners.reserve(n_total_bonds*2);
	list_of_bond_type.reserve(n_total_bonds);

	for (i = 0; i < n_total_bonds; i++) {
		(void)!fscanf(inp, "%d", &di);
		(void)!fscanf(inp, "%d", &di);
		int b_type = di - 1;

		(void)!fscanf(inp, "%d", &di);
		int i1 = di - 1;

		(void)!fscanf(inp, "%d", &di);
		int i2 = di - 1;

		if (i2 < i1) {
			di = i2;
			i2 = i1;
			i1 = di;
		}

		bonded_to[i1][n_bonds[i1]] = i2;
		bond_type[i1][n_bonds[i1]] = b_type;
		n_bonds[i1]++;

		bonded_to[i2][n_bonds[i2]] = i1;
		bond_type[i2][n_bonds[i2]] = b_type;
		n_bonds[i2]++;

		list_of_bond_type.push_back(b_type);
		list_of_bond_partners.push_back(i1);
		list_of_bond_partners.push_back(i2);

	}
	(void)!fgets(tt, 120, inp);


	list_of_angle_partners.reserve(n_total_angles*3);
	list_of_angle_type.reserve(n_total_angles);

	// Read in angle information
	for (i = 0; i < ns; i++)
		n_angles[i] = 0;

	(void)!fgets(tt, 120, inp);
	(void)!fgets(tt, 120, inp);
	for (i = 0; i < n_total_angles; i++) {

		(void)!fscanf(inp, "%d", &di);
		(void)!fscanf(inp, "%d", &di);

		int a_type = di - 1;

		(void)!fscanf(inp, "%d", &di);
		int i1 = di - 1;

		(void)!fscanf(inp, "%d", &di);
		int i2 = di - 1;

		(void)!fscanf(inp, "%d", &di);
		int i3 = di - 1;

		if (i3 < i1) {
			di = i3;
			i3 = i1;
			i1 = di;
		}

		int na = n_angles[i1];
		angle_first[i1][na] = i1;
		angle_mid[i1][na] = i2;
		angle_end[i1][na] = i3;
		angle_type[i1][na] = a_type;
		n_angles[i1] += 1;

		na = n_angles[i2];
		angle_first[i2][na] = i1;
		angle_mid[i2][na] = i2;
		angle_end[i2][na] = i3;
		angle_type[i2][na] = a_type;
		n_angles[i2] += 1;

		na = n_angles[i3];
		angle_first[i3][na] = i1;
		angle_mid[i3][na] = i2;
		angle_end[i3][na] = i3;
		angle_type[i3][na] = a_type;
		n_angles[i3] += 1;

		(void)!fgets(tt, 120, inp);

		list_of_angle_type.push_back(a_type);
		list_of_angle_partners.push_back(i1);
		list_of_angle_partners.push_back(i2);
		list_of_angle_partners.push_back(i3);

	}
	fclose(inp);

    molec_Ubond = (float*) calloc(n_molecules, sizeof(float));
}

void write_lammps_traj() {

	FILE* otp;
	int i, j;
	if (step == 0){
		if (equil && equilData){
			otp = fopen(dump_name.c_str(), "w");
			fclose(otp);
			otp = fopen(equil_name.c_str(), "w");
		}
		else{
			otp = fopen(dump_name.c_str(), "w");
		}
		
	}
	else{
		if (equil){
			if (!equilData)
				return;
			otp = fopen(equil_name.c_str(), "a");
		}
		else
			otp = fopen(dump_name.c_str(), "a");
	}

	fprintf(otp, "ITEM: TIMESTEP\n%d\nITEM: NUMBER OF ATOMS\n%d\n", global_step, ns);
	fprintf(otp, "ITEM: BOX BOUNDS pp pp pp\n");
	fprintf(otp, "%f %f\n%f %f\n%f %f\n", 0.f, L[0],
		0.f, L[1],
		(Dim == 3 ? 0.f : 1.f), (Dim == 3 ? L[2] : 1.f));

	if ( Charges::do_charges )
		fprintf(otp, "ITEM: ATOMS id type mol x y z q\n");
	else
		fprintf(otp, "ITEM: ATOMS id type mol x y z\n");

	for (i = 0; i < ns; i++) {
		fprintf(otp, "%d %d %d  ", i + 1, tp[i] + 1, molecID[i] + 1);
		for (j = 0; j < Dim; j++)
			fprintf(otp, "%f ", x[i][j]);

		for (j = Dim; j < 3; j++)
			fprintf(otp, "%f", 0.f);

		if ( Charges::do_charges )
			fprintf(otp, " %f", charges[i]);

		fprintf(otp, "\n");
	}
	fclose(otp);
}


void write_gsd_traj() {

	int i;

	gsd_handle gsd_file; 

	if (step == 0){
		vector<unsigned int> types(ns), molecule_ids(ns);

		for (i = 0; i < ns; i++) {
			types[i] = tp[i] + 1;
			molecule_ids[i] = molecID[i] + 1;
		}

		vector<float> masses(ns);

		for (i = 0; i < ns; i++) {
			masses[i] = mass[tp[i]];
		}


		auto version = gsd_make_version(1, 4);
		gsd_create_and_open(&gsd_file, gsd_name.c_str(), "gpu-tild", "hoomd", version, gsd_open_flag::GSD_OPEN_APPEND, 0);

		unsigned int frame = global_step;
		gsd_write_chunk(
			&gsd_file, "configuration/step", gsd_type::GSD_TYPE_UINT64,
			1, 1, 0, &frame
		);
		
		gsd_write_chunk(
				&gsd_file, "configuration/dimensions", gsd_type::GSD_TYPE_UINT8,
				1, 1, 0, &Dim
				);


		std::vector<float> box = {L[0], L[1], L[2], 0, 0, 0};
		gsd_write_chunk(
				&gsd_file, "configuration/box", gsd_type::GSD_TYPE_FLOAT,
				6, 1, 0, box.data() );

		{
			unsigned int ntypes = ns;
			gsd_write_chunk(&gsd_file, "particles/N", gsd_type::GSD_TYPE_UINT32, 1, 1, 0, &ntypes);
		}
        gsd_write_chunk(&gsd_file, "particles/mass", gsd_type::GSD_TYPE_FLOAT, masses.size(), 1, 0, masses.data());
		
		// Write the particle types 
		gsd_write_chunk(&gsd_file, "particles/typeid", gsd_type::GSD_TYPE_UINT32, ns, 1, 0, types.data());

		// Write the particle molecule ids
		gsd_write_chunk(&gsd_file, "log/particles/moleculeid", gsd_type::GSD_TYPE_UINT32, ns, 1, 0, molecule_ids.data());



		int max_len = 10;
		char* names = (char*) calloc((ntypes+1) * 13,  sizeof(char));

		std::string str("type");
		for (i = 0; i < (ntypes+1); i++) {
			strcpy(names + i * max_len, (str + std::to_string(i)).c_str());
		}


		gsd_write_chunk(&gsd_file, "particles/types", gsd_type::GSD_TYPE_INT8, (ntypes + 1), max_len, 0, names);

		unsigned int N_bonds = n_total_bonds;
		// Write the number of bonds
		gsd_write_chunk(&gsd_file, "bonds/N", gsd_type::GSD_TYPE_UINT32, 1, 1, 0, &N_bonds);

		// // Write the number of bond types
		// gsd_write_chunk(&gsd_file, "bond/types", gsd_type::GSD_TYPE_UINT32, 1, 1, 0, &nbond_types);

		// Write the bondids
		gsd_write_chunk(&gsd_file, "bonds/typeid", gsd_type::GSD_TYPE_UINT32, n_total_bonds, 1, 0, list_of_bond_type.data());

		// Write the bonds/group
		gsd_write_chunk(&gsd_file, "bonds/group", gsd_type::GSD_TYPE_UINT32, n_total_bonds, 2, 0, list_of_bond_partners.data());

		// // Write the number of angle types
		// gsd_write_chunk(&gsd_file, "angle/types", gsd_type::GSD_TYPE_UINT32, 1, 1, 0, &nangle_types);

		// Write the number of angles
		gsd_write_chunk(&gsd_file, "angles/N", gsd_type::GSD_TYPE_UINT32, 1, 1, 0, &n_total_angles);

		// Write the angleids
		gsd_write_chunk(&gsd_file, "angles/typeid", gsd_type::GSD_TYPE_UINT32, n_total_angles, 1, 0, list_of_angle_type.data());

		// Write the angles/group
		gsd_write_chunk(&gsd_file, "angles/group", gsd_type::GSD_TYPE_UINT32, n_total_angles, 3, 0, list_of_angle_partners.data());

	}
	else{
		gsd_open(&gsd_file, gsd_name.c_str(), gsd_open_flag::GSD_OPEN_APPEND);
		unsigned int frame = global_step;
		gsd_write_chunk(
			&gsd_file, "configuration/step", gsd_type::GSD_TYPE_UINT64,
			1, 1, 0, &frame
		);
		
	}

	for (i = 0; i < ns; i++) {
		for (int j = 0; j < Dim; j++) {
			h_ns_float[i * Dim + j] -= Lh[j];
		}
	}

	gsd_write_chunk(&gsd_file, "particles/position", gsd_type::GSD_TYPE_FLOAT, ns, 3, 0, h_ns_float);

	if ( Charges::do_charges )
		gsd_write_chunk(&gsd_file, "particles/charge", gsd_type::GSD_TYPE_FLOAT, ns, 1, 0, charges);


	gsd_end_frame(&gsd_file);
	gsd_close(&gsd_file);

	
}

void gsd_read_conf(const char* file_name, int frame_num, int process){

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

	std::cout << frame_num << endl;

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

	std::cout << "Dim = " << Dim << endl;
	std::cout << "n_P_comps = " << n_P_comps << endl;

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
	if (process == 0 && tmp_ns != ns){
		std::string me = "Error: The number of atoms in the GSD file does not match the number of atoms in the simulation.";
		die(me);
	}
	else {
		ns = tmp_ns;
	}

	cout << "tmp_ns = " << tmp_ns << " ns = " << ns << endl;

 	// read the charges of the particles
	std::vector<float> charges_tmp(ns);
	chunk_index = gsd_find_chunk(&gsd_file, frame_num, "particles/charge");
	if (chunk_index != NULL){
		Charges::do_charges = true;
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

	cout << "L = " << L[0] << " " << L[1] << " " << L[2] << endl;

	// Read in the number of bonds
	chunk_index = gsd_find_chunk(&gsd_file, frame_num, "bonds/N");
	if (chunk_index == NULL){
		chunk_index = gsd_find_chunk(&gsd_file, base_index, "bonds/N");
		if (chunk_index == NULL) {
			std::string me = "Error: Could not find the chunk 'bonds/N' in the GSD file.";
			die(me);
		}
	}
	gsd_read_chunk(&gsd_file, &n_total_bonds, chunk_index);

	cout << "n_total_bonds = " << n_total_bonds << endl;

	// Read in the number of angles
	chunk_index = gsd_find_chunk(&gsd_file, frame_num, "angles/N");
	if (chunk_index == NULL){
		chunk_index = gsd_find_chunk(&gsd_file, base_index, "angles/N");
		if (chunk_index == NULL) {
			std::string me = "Error: Could not find the chunk 'angles/N' in the GSD file.";
			die(me);
		}
	}
	gsd_read_chunk(&gsd_file, &n_total_angles, chunk_index);

	cout << "n_total_angles = " << n_total_angles << endl;
	
	// Read in all the particle types to determine the number of types
	// Instead of sorting, we could just allocate the highest amount that there would be in the box
	chunk_index = gsd_find_chunk(&gsd_file, frame_num, "particles/typeid");
	std::vector<int> typeids(ns), typ_id(ns);
	typeids.resize(ns);
	typ_id.resize(ns);

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
		ntypes = max(max_valu2, max_val - 1 );
	}

	cout << "ntypes = " << ntypes << endl;

	// Read in all the particle bonds to determine the number of bonds
	chunk_index = gsd_find_chunk(&gsd_file, frame_num, "bonds/typeid");
	std::vector<unsigned int> bonds(n_total_bonds);

	if (chunk_index == NULL){
		chunk_index = gsd_find_chunk(&gsd_file, base_index, "bonds/typeid");
		if (chunk_index == NULL) {
			std::string me = "Error: Could not find the chunk 'bonds/typeid' in the GSD file.";
			die(me);
		}
	}

	gsd_read_chunk(&gsd_file, bonds.data(), chunk_index);
	list_of_bond_type = bonds;
	
	if (n_total_bonds > 0) {
		std::sort(bonds.begin(), bonds.end());
		int max_val = *max_element(bonds.begin(), bonds.end());
		int max_valu2 = std::unique(bonds.begin(), bonds.end()) - bonds.begin();
		nbond_types = max(max_valu2, max_val - 1 );
	}

	std::cout << "nbond_types = " << nbond_types << endl;


	// Read in all the particle angles to determine the number of angles
	chunk_index = gsd_find_chunk(&gsd_file, frame_num, "angles/typeid");
	std::vector<unsigned int> angles(n_total_angles);

	if (chunk_index == NULL){
		chunk_index = gsd_find_chunk(&gsd_file, base_index, "angles/typeid");
		if (chunk_index == NULL) {
			std::string me = "Error: Could not find the chunk 'angles/typeid' in the GSD file.";
			die(me);
		}
	}

	gsd_read_chunk(&gsd_file, angles.data(), chunk_index);
	list_of_angle_type = angles;
	if (n_total_angles > 0) {
		std::sort(angles.begin(), angles.end());
		int max_val = *max_element(angles.begin(), angles.end());
		int max_valu2 = std::unique(angles.begin(), angles.end()) - angles.begin();
		nangle_types = max(max_valu2, max_val - 1 );
	}
	else {
		nangle_types = 0;
	}

	cout << "nangle_types = " << nangle_types << endl;

	list_of_angle_type.resize(n_total_angles);
	
	// Read in the atoms participating in each bond
	chunk_index = gsd_find_chunk(&gsd_file, frame_num, "bonds/group");
	std::vector<unsigned int> bond_partners(n_total_bonds*2);
	
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
	std::vector<unsigned int> angle_partners(n_total_angles*3);

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
	std::vector<unsigned int> molecule_id(ns);

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
		n_molecules  = max(max_valu2, max_val - 1 );
	}

	cout << "n_molecules = " << n_molecules << endl;
	
	// Read in particle masses
	chunk_index = gsd_find_chunk(&gsd_file, frame_num, "particles/mass");
	std::vector<float> masses(ns);

	if (chunk_index == NULL){
		chunk_index = gsd_find_chunk(&gsd_file, base_index, "particles/mass");
		if (chunk_index == NULL) {
			std::string me = "Error: Could not find the chunk 'particles/mass' in the GSD file.";
			die(me);
		}
	}

	gsd_read_chunk(&gsd_file, masses.data(), chunk_index);
	
	std::map<int, int> map_of_particle_id_mass;

	for (int i = 0; i < ns; i++) {
		if (map_of_particle_id_mass.find(typeids.at(i)) == map_of_particle_id_mass.end()) {
			map_of_particle_id_mass.insert(std::pair<int, int>(typeids.at(i), masses.at(i)));
		}
		else {
			// Check if the masses are the same
			if (map_of_particle_id_mass.at(typeids.at(i)) != masses.at(i)) {
				std::string me = "Error: The masses of the particles with the same type are not the same.";
				die(me);
			}
		}
	}



	// This is for reading in configuration file data
	if (process != 0) {
		allocate_host_particles();

		printf("Particle memory allocated on host via GSD!\n");


		// Store the types
		for (int i = 0; i < ns; i++){
			tp[i] = typeids.at(i) - 1;
		}

		// Store the molecule idsids
		for (int i = 0; i < ns; i++){
			molecID[i] = molecule_id.at(i) - 1;
		}

		// Assign the masses using the map
		for (auto& x: map_of_particle_id_mass) {
			mass[x.first - 1] = x.second;
		}

		// store the bonds
		for (unsigned int i = 0;  i<list_of_bond_type.size(); i++){
			int i1 = list_of_bond_partners.at(i*2);
			int i2 = list_of_bond_partners.at(i*2+1);
			int b_type = list_of_bond_type.at(i);

			bonded_to[i1][n_bonds[i1]] = i2;
			bond_type[i1][n_bonds[i1]] = b_type;
			n_bonds[i1]++;

			bonded_to[i2][n_bonds[i2]] = i1;
			bond_type[i2][n_bonds[i2]] = b_type;
			n_bonds[i2]++;
		}

		molec_Ubond = (float*) calloc(n_molecules, sizeof(float));

		// store the angles
		for (unsigned int i = 0; i < list_of_angle_type.size(); i++){
			int i1 = list_of_angle_partners.at(i*3);
			int i2 = list_of_angle_partners.at(i*3+1);
			int i3 = list_of_angle_partners.at(i*3+2);

			int a_type = list_of_angle_type.at(i);

			int na = n_angles[i1];
			angle_first[i1][na] = i1;
			angle_mid[i1][na] = i2;
			angle_end[i1][na] = i3;
			angle_type[i1][na] = a_type;
			n_angles[i1] += 1;

			na = n_angles[i2];
			angle_first[i2][na] = i1;
			angle_mid[i2][na] = i2;
			angle_end[i2][na] = i3;
			angle_type[i2][na] = a_type;
			n_angles[i2] += 1;

			na = n_angles[i3];
			angle_first[i3][na] = i1;
			angle_mid[i3][na] = i2;
			angle_end[i3][na] = i3;
			angle_type[i3][na] = a_type;
			n_angles[i3] += 1;
			
		}
	}


	// read in the position of the particles
	chunk_index = gsd_find_chunk(&gsd_file, frame_num, "particles/position");
	gsd_read_chunk(&gsd_file, h_ns_float, chunk_index);
	if (chunk_index == NULL) {
		std::string me = "error: could not find the chunk 'particles/position' in the gsd file.";
		die(me);
	}

	// Store the positions
	for (int i = 0; i < ns; i++) {
		for (int j = 0; j < Dim; j++) {
			x[i][j] = h_ns_float[i * Dim + j] + Lh[j];
		}
	}

	if (Charges::do_charges){
		for (int idx = 0; idx < charges_tmp.size(); idx++)
			charges[idx] = charges_tmp.at(idx);
	}


	chunk_index = gsd_find_chunk(&gsd_file, frame_num, "configuration/step");

	if (chunk_index == NULL) {
		std::string me = "error: could not find the chunk 'configuration/step' in the gsd file.";
		die(me);
	}

	int tmp_step;
	gsd_read_chunk(&gsd_file, &tmp_step, chunk_index);
	global_step = tmp_step;

	gsd_close(&gsd_file);
}


/*
void write_conf_file() {
    FILE* otp;
    int i, j;
    otp = fopen(final_conf_name.c_str(), "w");

    fprintf(otp, "GPU TILD\n\n");
    
    fprintf(otp, "%d atoms\n", ns);
    fprintf(otp, "%d bonds\n", n_total_bonds);
    fprintf(otp, "%d angles\n", n_total_angles);
    fprintf(otp, "\n");

    fprintf(otp, "%d atom types\n", ntypes);
    fprintf(otp, "%d bond types\n", nbond_types);
    fprintf(otp, "%d angle types\n", nangle_types);
    fprintf(otp, "\n");


    fprintf(otp, "0 %f xlo xhi\n",L[0]);
    fprintf(otp, "0 %f ylo yhi\n",L[1]);
    fprintf(otp, "0 %f zlo zhi\n",L[2]);

    fprintf(otp, "\n");

    fprintf(otp, "Masses\n\n");

	for (i = 0; i < ntypes; i++) {
        fprintf(otp, "%d %f\n");
    }
    fprintf(otp, "\n");

    fprintf(otp, "Atoms \n\n");

	for (i = 0; i < ns; i++) {
		fprintf(otp, "%d %d %d  ", i + 1, tp[i] + 1, molecID[i] + 1);
		for (j = 0; j < Dim; j++)
			fprintf(otp, "%f ", x[i][j]);

		if ( Charges::do_charges )
			fprintf(otp, "%f", charges[i]);

		for (j = Dim; j < 3; j++)
			fprintf(otp, " %f", 0.f);

		fprintf(otp, "\n");
	}
    fprintf(otp, "\n");

    if (n_total_bonds){
        fprintf(otp, "Bonds \n\n");
        int i = 0;
        for (int i1 = 0; 
        for (i = 0; i < n_total_bonds; i++) {
            fprintf(otp, "%d %d %d %d", i, 
            
        }
    }
}

*/

void read_resume( const char *nm ) {

	FILE* inp;
	inp = fopen(nm, "r");
	if (inp == NULL) {
		char death[50];
		sprintf(death, "Failed to open %s!\n", nm);
		die(death);
	}

  char tt[120];

  (void)!fgets(tt, 100, inp);
  (void)!fscanf(inp, "%d", &global_step);
  (void)!fgets(tt, 100, inp);
  (void)!fgets(tt, 100, inp);
  int ns_tmp;
  (void)!fscanf(inp, "%d", &ns_tmp);
  if (ns != ns_tmp)
	  die("number of sites %d in resume file doens't match input!");

  (void)!fgets(tt, 100, inp);
  (void)!fgets(tt, 100, inp);
  (void)!fgets(tt, 100, inp);
  (void)!fgets(tt, 100, inp);
  (void)!fgets(tt, 100, inp);
  (void)!fgets(tt, 100, inp);


  for (int i = 0; i < ns; i++) {
	  int t1, t2, t3;
	  (void)!fscanf(inp, "%d %d %d  ", &t1, &t2, &t3);

	  for (int j = 0; j < Dim; j++)
		  (void)!fscanf(inp, "%f ", &x[i][j]);

	  for (int j = Dim; j < 3; j++) {
		  float f1;
		  (void)!fscanf(inp, "%f", &f1);
	  }
	  if (Charges::do_charges) {
		  float f1;
		  (void)!fscanf(inp, "%f", &f1);
	  }
	  
  }

  fclose(inp);
  cout << "SUCCESSFULLY READ RESUME FILE" << endl;
}
