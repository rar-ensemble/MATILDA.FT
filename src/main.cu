// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#define MAIN
#include <string.h>
#include "globals.h"
#include "timing.h"
#include "random.h"
#include <vector>
#include <fstream>
#include "git-version.h"
#include "Box.h"
#include <mpi.h>
#include <algorithm>
#include <random>



using namespace std;

void forces(void);
void update_potentials(void);
void calc_properties(int);
void initialize(void);
void write_lammps_traj(void);
void write_gsd_traj(void);
void cuda_collect_x(void);
void cuda_collect_f(void);
void cuda_collect_rho(void);
void write_binary(void);
void write_data_header(std::string);
void set_write_status(void);
void init_binary_output(void);
void write_struc_fac(void);
void write_grid_data(const char*, float*);
void write_kspace_data(const char*, complex<float>*);
void write_kspace_cudaComplex(const char*, cufftComplex*);
__global__ void d_prepareDensity(int, float*, cufftComplex*, int);
int print_timestep();
ofstream dout;
void unstack_like_device(int id, int* nn);
void run_computes();
void run_frame_printing();

void run_fts_sim(void);
void set_ft_config(void);



using namespace std;

Box* BoxFactory(istringstream&);

__global__ void cu_random_posits(float*, float*, int, int, curandState*);


__global__ void d_real2complex(float*, cufftComplex*, int);
__global__ void d_complex2real(cufftComplex*, float*, float*, int);
__global__ void d_make_step(cufftComplex* , float*, float*, int*, int, int);
__global__ void d_multiplyComplex(cufftComplex*, cufftComplex*,
	cufftComplex*, int);

int main(int argc, char** argv)
{


/////// Initialize MPI ////////

    MPI_Init(&argc, &argv);
    MPI_Comm communicator = MPI_COMM_WORLD;

    int size, rank;
    MPI_Comm_size(communicator, &size);
    MPI_Comm_rank(communicator, &rank);
	srank = std::to_string(rank);

    const char* nl_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int node_local_rank = atoi(nl_rank);

    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);

    int device_id = node_local_rank % num_devices;
    cudaSetDevice(device_id);

	int my_device_id;
	cudaGetDevice(&my_device_id);

	char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

	int replica_exchange_flag = 0;
	std::vector<int> replica_id_vec(size);
	std::vector<float> replica_E_vec;
	std::iota (std::begin(replica_id_vec), std::end(replica_id_vec), 0);

	old_E_arr = (float*)calloc(size, sizeof(float));
	current_E_arr = (float*)calloc(size, sizeof(float));

	// auto rng = std::default_random_engine {};
	std::default_random_engine rng;
	std::uniform_int_distribution<int> pick_me_please(0,size);
	int msg_tag;


	if ( argc < 2 ) {
		std::cout << "ERROR: simulation style not specified!" << std::endl;
		std::cout << "Execute matilda.ft as either\nmatilda.ft -particle\nfor a particle-based simulation or"<< std::endl;
		std::cout << "matilda.ft -ft\nfor a field-theoretic simulation." << std::endl;
		die("Insufficient arguments");
	}

	printf("\n\n\t\t##### MPI INFO #####\nName: %s\nGlobal Rank: %2d of %2d, Local Rank: %2d, GPU: %2d (%2d) of %2d\n\n",
		processor_name,rank, size, node_local_rank, device_id, my_device_id, num_devices);


	// cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
	
	// printf("Git Version hash: %s\n", MY_GIT_VERSION);
	main_t_in = int(time(0));
	init_t_in = main_t_in;
	std::vector<std::string> string_vec;
	std::cout << std::flush;

	for (int i = 0; i < argc; i++)
	{
		std::string arg = argv[i];
		string_vec.push_back(arg);
	}

	if(string_vec[1] == "-ft"){
		std::cout << "Set simulation style to: FT" << std::endl;
		set_ft_config();
		input_file = "input";
	    ifstream in2(input_file);

	    string word, line, rname;
		while (!in2.eof()) {
			getline(in2, line);

			// Blank or commented line
			if (line.length() == 0 || line.at(0) == '#')
				continue;

			istringstream iss(line);
			// Loop over words in line
			while (iss >> word) {
				if( word == "box" ) {
				box.push_back(BoxFactory(iss));
				box.back()->readInput(in2);
				field_sim = 1;
				particle_sim = 0;
				}
			}
		}
		run_fts_sim();
		return 0;
	}

	else if (string_vec[1] == "-particle"){
		std::cout << "Set simulation style to: TILD" << std::endl;

		for (int i = 2; i < string_vec.size(); ++i) {
			if (string_vec[i] == "-in") {
				input_file = string_vec[++i];
			}
			if (string_vec[i] == "-replica"){
				replica_exchange_flag = 1;
				replica_freq = std::stoi(string_vec[i+1]);
				replica_file = string_vec[i+2];
			}
		}

		if (replica_exchange_flag == 1){
			std::fstream file;
			std::string dummy_wrd;
			file.open(replica_file.c_str());

			while(file >> dummy_wrd) {
				replica_E_vec.push_back(std::stof(dummy_wrd));
				}
			file.close();

			std::cout << "Replica IDs: " << std::endl;
			for (auto& i : replica_id_vec)
				std::cout << ' ' << i;
			std::cout << endl;

			std::cout << "Replica bond energies: " << std::endl;
			for (auto& i : replica_E_vec)
				std::cout << ' ' << i;
			std::cout << endl;
			current_E = replica_E_vec[rank];
			std::cout <<"My energy: " << current_E << std::endl;
			for (int j = 0; j < replica_E_vec.size(); ++j){
				old_E_arr[j] = replica_E_vec[j];
				current_E_arr[j] = replica_E_vec[j];
			}			
		}



		initialize();

		set_write_status();


		// Write initial positions to lammpstrj file
		cuda_collect_x();
		
		forces();
		//cudaDeviceSynchronize();

		cuda_collect_rho();
		cuda_collect_x();

		run_frame_printing();

		calc_properties(1);  // 1 indicates to calculate virial pressure

		if (grid_freq > 0) {
			print_t_in = int(time(0));
			//cudaDeviceSynchronize();

			for (int i = 0; i < ntypes; i++) {
				char nm[30];
				sprintf(nm, "rho%d.dat", i);
				write_grid_data(nm, Components[i].rho);
			}
				
			print_t_out = int(time(0));
			print_tot_time += print_t_out - print_t_in;
		}

		int die_flag = 0;

		init_t_out = int(time(0));


		cout << "ENTERING MAIN LOOP!!" << endl;
		///////////////////////////////////////
		// BEGINNING OF MAIN SIMULATION LOOP //
		///////////////////////////////////////

		for (step = 1, global_step = global_step + 1; step <= max_steps; step++, global_step++) {
			if (equil  && step >= equil_steps) {
				dout.close();
				equil = false;
				set_write_status();
			}

			if (replica_exchange_flag == 1 && step%replica_freq == 0){
				if (rank == 0){
					std::shuffle(replica_id_vec.begin(), replica_id_vec.end(), rng);
					std::cout << "Step " << step << " | Replica IDs: " << std::endl;
					for (auto& i : replica_id_vec)
						std::cout << ' ' << i;
					std::cout << endl;

					for(int j = 0; j < size; j = j + 2){

						int rid = replica_id_vec[j];
						int n_rid = replica_id_vec[j+1];

						current_E_arr[n_rid] = old_E_arr[rid];
						current_E_arr[rid] = old_E_arr[n_rid];

						}
					std::cout << "Energies: " << std::endl;
					for(int j = 0; j < size; j++){
						old_E_arr[j] = current_E_arr[j];
						std::cout << current_E_arr[j] << " ";
						}
					std::cout << std::endl;
					
				} // if rank == 0

				MPI_Bcast(current_E_arr,size,MPI_FLOAT,0,communicator);
				MPI_Barrier(communicator);

				current_E = current_E_arr[rank];
				std::cout << current_E << std::endl;
			} // if replice_freq % step == 0



				// if (step%100 == 0 && step > 0){
				// 	if (rank == 0){
				// 		oldT = T;
				// 		MPI_Send(&oldT,1,MPI_INT,1, 0,communicator);
				// 		MPI_Recv(&T, 1, MPI_INT, 1, 0, communicator, MPI_STATUS_IGNORE);

				// 	}
				// 	else if(rank == 1){
				// 		oldT = T;
				// 		MPI_Recv(&T, 1, MPI_INT, 0, 0, communicator, MPI_STATUS_IGNORE);
				// 		MPI_Send(&oldT,1,MPI_INT,0, 0,communicator);
				// 	}
				// }


			for (auto Iter: Groups){
				Iter->CheckGroupMembers();
			}

			for (auto Iter: Integrators){
				Iter->Integrate_1();
			}

			check_cudaError("Integrator step 1");	
			
			if ( NLists.size() > 0 ) {
				nList_t_in = time(0);
				for (auto Iter: NLists)
					Iter->MakeNList();
				check_cudaError("Error in N-lists");
				nList_tot_time += time(0) - nList_t_in;
			}		

			forces();
			
			if ( ExtraForces.size() > 0 ) {
				extraForce_t_in = time(0);
				for (auto Iter: ExtraForces)
					Iter->AddExtraForce();
				check_cudaError("extraForces");
				extraForce_tot_time += time(0) - extraForce_t_in;
			}

			for (auto Iter: Integrators){
				Iter->Integrate_2();
			}

			check_cudaError("Integrator step 2");
			
			// Run computes
			run_computes();	

			// Write frames if applicable
			run_frame_printing();

			// Write to log file, write compute results
			if (step % log_freq == 0) {
			print_t_in = int(time(0));
			//cudaDeviceSynchronize();

			calc_properties(1);

			die_flag = print_timestep();

			for (auto Iter : Computes) {
				if (step > Iter->compute_wait) {
				Iter->writeResults();
				}
			}

			print_t_out = int(time(0));
			print_tot_time += print_t_out - print_t_in;
			if (die_flag) {
				break;
			}
			}

					// Finalize time step //
			update_potentials();

		}// main loop over steps



		// Write resume frame and finish //
		if (max_steps % log_freq != 0) {
			cuda_collect_x();
			write_lammps_traj();
			write_gsd_traj();
		}
	}// if -particle

    else {
        die("Invalid simulation style, argument must be '-ft' or '-particle'\n");
    }

	main_t_out = int(time(0));
	int dt = main_t_out - main_t_in;
	cout << "Total run time: " << dt / 60 << "m" << dt % 60 << "sec" << endl;
	
	dt = init_t_out - init_t_in;
	cout << "Total init time: " << dt / 60 << "m" << dt % 60 << "sec" << endl;
	
	dt = bond_tot_time;
	cout << "Bond E, P props on host: " << dt / 60 << "m" << dt % 60 << "sec" << endl;

	dt = print_tot_time;
	cout << "I/O + Comm time: " << dt / 60 << "m" << dt % 60 << "sec" << endl;

	dt = compute_tot_time;
	cout << "Computes time: " << dt / 60 << "m" << dt % 60 << "sec" << endl;

    dt = extraForce_tot_time;
	cout << "ExtraForces time: " << dt / 60 << "m" << dt % 60 << "sec" << endl;

    if ( DPD_time > 0 ) {
        dt = DPD_time;
        cout << "DPD Forces time: " << dt / 60 << "m" << dt % 60 << "sec" << endl;
    }
    
    if ( nList_tot_time > 0 ) {
        dt = nList_tot_time;
        cout << "NList time: " << dt / 60 << "m" << dt % 60 << "sec" << endl;
    }

	// cudaStreamDestroy(stream1);
    MPI_Finalize();
	return 0;
}


int print_timestep() {
	//if (!equilData)
		//return
	int die_flag = 0;
	cout << "Step " << step << " of " << max_steps << " ";
	cout << "Global step " << global_step;


	cout << " U/V: " << Upe / V << \
		" Ubond: " << Ubond ;
	if ( n_total_angles > 0 )
	    cout << " Uangle: " << Uangle ;
	cout << " Pdiags: " << Ptens[0] << " " << Ptens[1] << " ";

	if (Dim == 3)
		cout << Ptens[2] << " ";

	dout << step << " " <<  global_step << " " << Upe << " " << Ubond << " ";
	if ( n_total_angles > 0 ) 
	    dout << Uangle << " " ;
	for (int i = 0; i < n_P_comps; i++)
		dout << Ptens[i] << " ";

	
	for (auto& Iter: Potentials)
	{
		Iter->ReportEnergies(die_flag);
	}
	cout << " UDBond: " << Udynamicbond;
	dout << " " << Udynamicbond;

	// cout << " T: " << T;
	// dout << " " << T;

    dout << endl;
	cout<<endl;
	return die_flag;

}


void run_computes(){
  compute_t_in = int(time(0));
  for (auto Iter : Computes) {
    if (step > Iter->compute_wait && step % Iter->compute_freq == 0) {
      Iter->doCompute();
    }
    check_cudaError("Compute");
  }
  compute_t_out = int(time(0));
  compute_tot_time += compute_t_out - compute_t_in;
}

void run_frame_printing() {
  // I/O blocks //
  if (traj_freq > 0 && step % traj_freq == 0) {
    print_t_in = int(time(0));
    //cudaDeviceSynchronize();

    cuda_collect_x();
    write_lammps_traj();
    print_t_out = int(time(0));
    print_tot_time += print_t_out - print_t_in;
  }

  if (gsd_freq > 0 && step % gsd_freq == 0) {
    print_t_in = int(time(0));
    //cudaDeviceSynchronize();

    cuda_collect_x();
    write_gsd_traj();
    print_t_out = int(time(0));
    print_tot_time += print_t_out - print_t_in;
  }

  if (grid_freq > 0 && step % grid_freq == 0) {
    print_t_in = int(time(0));
    //cudaDeviceSynchronize();

    cuda_collect_rho();
    for (int i = 0; i < ntypes; i++) {
      char nm[30];
      sprintf(nm, "rho%d.dat", i);
      write_grid_data(nm, Components[i].rho);
    }

    print_t_out = int(time(0));
    print_tot_time += print_t_out - print_t_in;
  }

  if (bin_freq > 0 && step % bin_freq == 0) {
    print_t_in = int(time(0));
    //cudaDeviceSynchronize();

    cuda_collect_rho();
    cuda_collect_x();

    write_binary();
    print_t_out = int(time(0));
    print_tot_time += print_t_out - print_t_in;

  }
}

void write_data_header(std::string lbl){
    dout.open(lbl);
	dout << "# step global_step Upe Ubond ";
	if ( n_total_angles > 0 )
	    dout << "Angles " ;
	if (Dim == 2)
		dout << "Pxx Pyy Pxy";
	else if (Dim == 3)
		dout << " Pxx Pyy Pzz Pxy Pxz Pyz";
	
	for (auto Iter: Potentials){
		dout << " " + Iter->potential_type;
		if (Iter->potential_type != "Charges")
			dout << Iter->type_specific_id ;
		if (Iter->potential_type == "MaierSaupe")
			dout << " Lambda" + Iter->potential_type << Iter->type_specific_id;
	}
	dout << " UDBond";
	dout << " T";

	dout << endl;

}

void set_write_status(){
	cout << "Setting frequencies" << endl;
	if (equil ){
		cout << "Setting frequencies to equilibration values" << endl;
		cout << "Equil bin freq: " << equil_bin_freq << endl;
		cout << "Equil traj freq: " << equil_traj_freq << endl;
		cout << "Equil grid freq: " << equil_grid_freq << endl;
		cout << "Equil log freq: " << equil_log_freq << endl;
		if (equilData)
			write_data_header("equil_data.dat");
		prod_bin_freq = bin_freq;
		prod_traj_freq = traj_freq;
		prod_grid_freq = grid_freq;
		prod_log_freq = log_freq;
		prod_struc_freq = struc_freq;
		
		if (equil_bin_freq > 0){
			bin_freq = equil_bin_freq;
			cout << "Equil Binary output frequency: " << equil_bin_freq << endl;
			cout << "Binary output frequency: " << bin_freq << endl;
		}
		if (equil_traj_freq > 0)
			traj_freq = equil_traj_freq;
		if (equil_grid_freq > 0){
			grid_freq = equil_grid_freq;
			cout << "Equil Grid output frequency: " << equil_grid_freq << endl;
			cout << "Grid output frequency: " << grid_freq << endl;
		}
		if (equil_log_freq > 0)	{
			cout << "Equil Log output frequency: " << equil_log_freq << endl;
			log_freq = equil_log_freq;
			cout << "Log output frequency: " << log_freq << endl;
		}
		if (equil_struc_freq > 0)
			struc_freq = equil_struc_freq;
	}
	else{
		write_data_header("data" + srank + ".dat");
		if (prod_traj_freq > 0)
			traj_freq = prod_traj_freq;
		if (prod_grid_freq > 0)
			grid_freq = prod_grid_freq;
		if (prod_bin_freq > 0)
			bin_freq = prod_bin_freq;
		if (prod_log_freq > 0)	
			log_freq = prod_log_freq;
		if (prod_struc_freq > 0)	
			struc_freq = prod_struc_freq;
	}
	if (bin_freq != 0)
		init_binary_output();
}


void set_ft_config(){
	step = 0;
	mem_use = 0;
	extra_ns_memory = 0;
	particle_sim = 0;
	field_sim = 0;

	// Global unit complex value
	I = complex<float>(0.0f, 1.0f);
	
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
	MAX_DISP = 0.0f;
	prod_bin_freq = 0;
	prod_log_freq = 0;
	prod_traj_freq = 0;
	prod_struc_freq = 0;
	prod_grid_freq = 0;
	global_step = 0;
	LOW_DENS_FLAG = 0;

}
