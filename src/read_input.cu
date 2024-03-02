// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include <sstream>
// #include "git-version.h"

using namespace std;

void set_defaults(void);
void read_config(const char*);
void read_charge_config(const char*);
void read_resume(const char*);
void make_group_type(int, int);
void make_group_all(void);
void write_runtime_parameters(int, string);
void gsd_read_conf(const char* file_name, int frame_num, int process);

Potential* PotentialFactory(istringstream &iss);
Measure* MeasureFactory(istringstream &iss);
ExtraForce* ExtraForceFactory(istringstream &iss);
NList* NListFactory(istringstream &iss);
Compute* ComputeFactory(istringstream &iss);
Group* GroupFactory(istringstream &iss);
Integrator* IntegratorFactory(istringstream& iss);

int read_charge_flag, config_read_flag;

void read_input() {

	set_defaults();

	if (input_file.length() == 0) input_file = "input";
	ifstream in2(input_file);
	if (not in2.is_open()){
		cout << "File " << input_file << " does not exist."<<endl;
		die("");
	}

	string word, line, rname;
	int config_read_flag = 0, read_resume_flag = 0;
	bool maxRead = false, prodRead =false,equilRead=false; 
	read_charge_flag = 0;
	

	while (!in2.eof()) {
		getline(in2, line);

		// Blank or commented line
		if (line.length() == 0 || line.at(0) == '#')
			continue;

		cout << line << endl;
		istringstream iss(line);

		// Loop over words in line
		while (iss >> word) {

        if ( word == "angle" ) {
            if ( !config_read_flag ) 
                die("Error in input file, angle keyword before config read!");
            else if (nangle_types == 0 )
                die("angle keyword found, but no angle types in config file!");
            
            string toint, tofloat, aform;
            iss >> toint ;
            int atype = stoi(toint);
            if ( atype > nangle_types ) 
                die("Invalid angle type in input file!");
            
            iss >> aform;

    				if ( aform == "wlc" ) {
    					iss >> tofloat;
    					angle_k[atype-1] = stof(tofloat);
    					angleIntStyle[atype-1] = 0;
    					angleStyle[atype-1] = "wlc";
    				}
    
    				else if ( aform == "harmonic" ) {
    					iss >> tofloat;
    					angle_k[atype-1] = stof(tofloat);
    
    					iss >> tofloat;
    					angle_theta_eq[atype-1] = stof(tofloat) * PI / 180.0f;
    					angleIntStyle[atype-1] = 1;
    					angleStyle[atype-1] = "harmonic";
    				}
				
            else
              die("You fool, that angle type is not supported!");
			}

			else if (word == "binary_freq") {
				iss >> bin_freq;
			}


			else if (word == "pos_skip") {
				iss >> pos_skip;
			}

			else if (word == "grid_skip") {
				iss >> grid_skip;
			}



			else if (word == "equil_binary_freq"){
				iss >> equil_bin_freq;
			}

			else if (word == "bond") {
				if (!config_read_flag)
					die("Error in input file, bond keyword before config read");
				string toint, tofloat;
				int btype;
				iss >> btype;

				string style;
				iss >> style;

				if (style == "harmonic") {
					bond_style[btype - 1] = 0;
				}

				else if (style == "FENE"){
					bond_style[btype - 1] = 1;
				}

				else(die("Bond type not supported!"));

				iss >> bond_k[btype - 1];
				iss >> bond_req[btype - 1];	
			}

			else if (word == "bond_log_freq") {
				iss >> bond_log_freq;
			}

			else if (word == "compute") {
				if (!config_read_flag)
					die("Error in input file, compute defined before config read");
				Computes.push_back(ComputeFactory(iss));
				while (!iss.eof())
					iss >> word;
			}


			else if (word == "delt") {
				string tofloat;
				iss >> tofloat;
				delt = stof(tofloat);
		        noise_mag = sqrtf(2.f * delt);
			}


			else if (word == "diffusivity") {
				if (!config_read_flag)
					die("Error in input file, diffusivities defined before config read");
				string toint, tofloat;
				iss >> toint;
				int atype = stoi(toint);
				if (atype > ntypes) {
					die("Invalid type for diffusivity!");
				}

				iss >> tofloat;
				float Df = stof(tofloat);

				Diff[atype - 1] = Df;
			}

			else if (word == "Dim") {
				iss >> Dim;
			}
			
            else if (word == "extraforce") {
				ExtraForces.push_back( ExtraForceFactory(iss));

                        while (iss >> word) {};// Loop over rest of the line
            }
			
			else if (word == "nlist") {
				NLists.push_back(NListFactory(iss));
            }

			else if (word == "measure") {
				MeasureFactory(iss);
				while (iss >> word) {};// Loop over rest of the line
            }

            else if (word == "extraSiteMemory") {
                if ( config_read_flag )
                    die("extraSiteMemory command must come before read_data in input file!");
                iss >> extra_ns_memory;
            }
            
      
			else if (word == "equil_grid_freq")
				iss >> equil_grid_freq;

			else if (word == "set_timestep"){
				// if (config_read_flag)
				// 	die("Error in input file, set_timestep defined after config read");
				iss >> global_step;
			}

			else if (word == "grid_freq") {
				iss >> grid_freq;
			}

			else if (word == "grid_update_freq"){
				iss >> GRID_UPDATE_FREQ;
			}



      else if (word == "gsd_freq"){
				iss >> gsd_freq;
			}

			else if (word == "gsd_name"){
				iss >> gsd_name;
			}

			else if (word == "group") {
				if (config_read_flag == 0)
					die("Must read the configuration file before defining a group!");
				Groups.push_back(GroupFactory(iss));

			}

			else if (word == "integrator") {
				if (!config_read_flag)
					die("Integrator must be defined after read_data command!");
				Integrators.push_back(IntegratorFactory(iss));
			}

			else if (word == "log_freq") {
				iss >> log_freq;
			}

			else if (word == "equil_log_freq"){
				iss >> equil_log_freq;
			}
			
			else if (word == "MAX_ANGLES") {
				iss >> MAX_ANGLES;
			}

			else if (word == "MAX_BONDS") {
				iss >> MAX_BONDS;
			}

			else if (word == "max_steps"){
				iss >> max_steps;
				if (equilRead && prodRead)
					die("You cannot use all 3 of max_steps, prod_steps, and equil_steps!");
				maxRead = true;
				if (!prodRead);
					prod_steps = max_steps;
			}
			
			else if (word == "prod_steps") {
				iss >> prod_steps;
				if (maxRead && equilRead)
					die("You cannot use all 3 of max_steps, prod_steps, and equil_steps!");
				prodRead = true;
				if (maxRead){
					equil_steps = max_steps - prod_steps;
					if (equil_steps < 0)
						die("Equilibration steps must be positive!");
					else if (equil_steps == 0)
						equil = false;
					else
						equil = true;
				}
				else
					max_steps = prod_steps + equil_steps;				
			}
			
			else if (word == "equil_steps"){
				iss >> equil_steps;
				if (maxRead && prodRead)
					die("You cannot use all 3 of max_steps, prod_steps, and equil_steps!");
				equilRead = true;
				if (equil_steps <=0  ) { 
					equil_steps = 0; equil = false;
				} else{
				equil = true;
				}
				if (maxRead){
					prod_steps = max_steps - equil_steps;
					if (prod_steps < 0)
						die("Production steps must be positive!");
				}
				else if (prodRead){
					max_steps = prod_steps + equil_steps;
				}
				
			}

			else if (word == "equil_data"){
				iss >> equilData;
			}


			else if (word == "potential" || word == "tensor_potential") {
				Potentials.push_back(PotentialFactory(iss));
				}

			// Included for backwards capatability; do not use 
			else if (word == "n_erfs" || word == "n_fieldphases" ||
					 word == "n_fieldphases" || word == "n_gaussians" ){
                        while (iss >> word) {};// Loop over rest of the line
					}

			// Included for backwards capatability; please include potential in frot
			else if (word == "erf" || word == "fieldphase" ||
					 word == "fieldphase" || word == "gaussian" || word == "charges"){
						iss.seekg(0);
					std::cout<< "Warning: " << word << " is deprecated. Please include \"potential\"  before " << word << " moving forward." << std::endl;
					Potentials.push_back(PotentialFactory(iss));
			}

			else if (word == "Nx") {
				iss >> Nx[0];
			}

			else if (word == "Ny") {
				iss >> Nx[1];
			}

			else if (word == "Nz") {
				if (Dim > 2)
					iss >> Nx[2];
				else{
					std::cout << "Warning: Nz is not defined for 2D simulations!" << std::endl;
					string s1; 
					iss >> s1; 
				}
			}

			else if (word == "pmeorder") {
				iss >> pmeorder;
				if (pmeorder > 4)
					die("PMEORDER greater than 4 not supported!");
			}

			else if (word == "rand_seed") {
				iss >> RAND_SEED;
			}

			else if (word == "read_gsd"){
				string file_name, tmp; 
				iss >> file_name;
				int start_step = -1;
				int process = 0;

				while (iss >> tmp){
					if (tmp == "frame"){
						iss >> start_step;

					}
					if (tmp == "resume"){
						process = 0;
					}
					if (tmp == "read_config"){
						process = 1;
					}
				}

				gsd_read_conf(file_name.c_str(), start_step, process);

				if (process == 1)
				{
					config_read_flag = 1;
					Groups.push_back(new Group());
				}
			}			
			
			else if (word == "read_data") {
				string to_char;
				iss >> to_char;
				if (read_charge_flag == 1) {
					cout << "READING CHARGE CONFIG" << endl;
					read_charge_config(to_char.c_str());
				}
				else {
					read_config(to_char.c_str());
				}
				config_read_flag = 1;

				Groups.push_back(new Group());


			}

			else if (word == "read_resume") {
				string to_char;
				iss >> to_char;
				read_resume(to_char.c_str());
				read_resume_flag = 1;
				rname = to_char;
			}

			else if (word == "skip_steps") {
				iss >> skip_steps;
			}

			else if (word == "struc_freq") {
				iss >> struc_freq;
			}

			else if (word == "equil_struc_freq") {
				iss >> equil_struc_freq;
			}

            // else if ( word == "tensorPotential" ) {
            //     string style;
            //     iss >> style;
            //     if ( style == "MaierSaupe" ) {
            //         n_MaierSaupe += 1;
            //         MS.resize(n_MaierSaupe);
            //         MS[n_MaierSaupe-1].Initialize(line);
            //     }
            //     else 
            //         die("Invalid tensorPotential!");
            //     while (iss >> word ) {} // read the rest of the line
            // }

			else if (word == "threads") {
				iss >> threads;
			}

			else if (word == "traj_freq") {
				iss >> traj_freq;
			}

			else if (word == "equil_traj_freq"){
				iss >> equil_traj_freq;
			}

			else if (word == "traj_name") {
				dump_name.clear();
				iss >> dump_name;
			}



			else {
				cout << "Invalid keyword: " << word << endl;
				die("Invalid keyword error");
			}
		}// while (iss >> word)
	}// while (!in2.eof())

  if ( Integrator::using_GJF && delt < 0.002f ) {
    cout << "WARNING!!!!! WARNING!!!! WARNING!!!!" << endl;
    cout << "Previous tests have seen problems with GJF algorithm and time steps < 0.002" << endl;
  }


	write_runtime_parameters(read_resume_flag, rname);
	
}


void set_defaults() {
	
	Dim = 2;
	delt = 0.005f;
	pmeorder = 1;
	for (int j = 0; j < Dim; j++) {
		Nx[j] = 35;
	}

	threads = 512;
	noise_mag = sqrtf(2.0f * delt);
	MAX_BONDS = 3;
	MAX_ANGLES = 3;

	max_steps = 100000;
	RAND_SEED = int(time(0));
	log_freq = 2500;
	grid_freq = 0;
	traj_freq = 0;
	gsd_freq = 0;
	struc_freq = 0;
	skip_steps = 20000;
	bin_freq = 5000;
	dump_name = "traj.lammpstrj";
	equil_name = "equil.lammpstrj";
	gsd_name = "traj.gsd";
	equil = false;
	equilData = 1;
	equil_steps = 0;
	prod_steps = 0;
	// MAX_DISP = 0.0f;
	prod_bin_freq = 0;
	prod_log_freq = 0;
	prod_traj_freq = 0;
	prod_struc_freq = 0;
	prod_grid_freq = 0;
	global_step = 0;
	LOW_DENS_FLAG = 0;
	GRID_UPDATE_FREQ = 1;
}


void write_runtime_parameters(int rflag, string rname) {
		ofstream otp;
		otp.open("MATILDA.FT.input");
		// Git hash info
		// otp << "# Git hash of the generating executable: " << MY_GIT_VERSION << endl;

		otp << "Dim " << Dim << endl;
		otp << "Nx " << Nx[0] << endl;
		otp << "Ny " << Nx[1] << endl;
		if (Dim == 3) {
			otp << "Nz " << Nx[2] << endl;
		}
		cout << "extraSiteMemory " << extra_ns_memory << endl;
		otp << "delt " << delt << endl;

		for (auto Iter: Groups){
			otp << Iter->printCommand() << endl;
		}

		for (auto Iter: Integrators){
			otp << Iter->printCommand() << endl;
		}

        otp << "rand_seed " << RAND_SEED << endl;
		if (rflag)
			otp << "read_resume " << rname << endl;
		otp << "max_steps " << max_steps << endl;
		if (equil){
			otp << "equil " << equil_steps << endl;
			if (equilData==1){
				otp << "WriteEquilData over " << equil_steps << endl;
				if (equil_traj_freq > 0)
					otp << "equil_traj_freq " << equil_traj_freq << endl;
				if (equil_struc_freq > 0)	
					otp << "equil_struc_freq " << equil_struc_freq << endl;
				if (equil_bin_freq > 0)
					otp << "equil_bin_freq " << equil_bin_freq << endl;
				if (equil_log_freq > 0)
					otp << "equil_log_freq " << equil_log_freq << endl;
				if (equil_grid_freq > 0)
					otp << "equil_grid_freq " << equil_grid_freq << endl;
			}
			else
				otp << "Equilibrium data not written" << endl;
		}
		otp << "MAX_BONDS " << MAX_BONDS << endl;
		otp << "MAX_ANGLES " << MAX_ANGLES << endl;
		otp << "log_freq " << log_freq << endl;
		otp << "threads " << threads << endl;
		otp << "grid_freq " << grid_freq << endl;
		otp << "traj_freq " << traj_freq << endl;
		otp << "gsd_freq " << gsd_freq << endl;
		otp << "binary_freq " << bin_freq << endl;
		otp << "struc_freq " << struc_freq << endl;
		otp << "skip_steps " << skip_steps << endl;
		otp << "pmeorder " << pmeorder << endl;
		if (Charges::do_charges == 1)
			otp << "charges " << charge_bjerrum_length << " " << charge_smearing_length << endl;

		for (int i = 0; i < ntypes; i++) {
			otp << "diffusivity " << i + 1 << " " << Diff[i] << endl;
		}
		
		for (auto Iter: Potentials){
			otp << Iter->printCommand() << endl;
			}

		for (auto Iter: ExtraForces){
			otp << Iter->printCommand() << endl;
		}

		for (auto Iter: Integrators){
			otp << Iter->printCommand() << endl;
		};


		for (auto Iter: ExtraForces){
			otp << Iter->printCommand() << endl;
        }

		for (auto Iter: Computes)
			otp << Iter->printCommand() << endl;

		for (int i = 0; i < nbond_types; i++)
			otp << "bond " << i + 1 << " harmonic " << bond_k[i] \
			<< " " << bond_req[i] << endl;

    for ( int i=0 ; i<nangle_types ; i++ ) { 
      otp << "angle " << i+1 << " " << angleStyle[i] << " " << angle_k[i] ;
      if ( angleStyle[i] == "harmonic" )
        otp << " " << angle_theta_eq[i] ;
      otp << endl;
    }
}
