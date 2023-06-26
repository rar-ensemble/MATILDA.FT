// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "nlist.h"
#include "nlist_distance.h"
#include "nlist_half_distance.h"
#include "nlist_bonding.h"

using namespace std;


int NList::total_num_nlists = 0;
int NList::total_num_triggers = 0;


NList::NList(istringstream& iss) {

    id = total_num_nlists++;

    command_line = iss.str();

    readRequiredParameter(iss, group_name);
    readRequiredParameter(iss, style);
    readRequiredParameter(iss, name);

    group_index = get_group_id(group_name);
    group = Groups.at(group_index);


    int tmp;
    xyz = 1;

    readRequiredParameter(iss, r_n);
    readRequiredParameter(iss, r_skin);
    readRequiredParameter(iss, ad_hoc_density);
    readRequiredParameter(iss, nlist_freq);
    
    out_file_name = name;

    delta_r = r_skin - r_n;
    d_trigger = delta_r/2.0;

    // sanity check

    // if (delta_r <= 0){die("delta_r cannot be negative or 0!");}

    std::cout << "Group name: " << group_name << ", id: " << id << endl;
    std::cout << "Style: " << style << endl;
    std::cout << "r_n: " << r_n << ", r_skin: " << r_skin <<  ", delta_r: " << delta_r << ", nlist_freq: " << nlist_freq << endl;

    for (int i = 0; i < Dim; i++)
    {
        tmp = floor(L[i]/r_skin);
        Nxx.push_back(tmp);
        xyz *= int(tmp);
    }

    for (int i = 0; i < Dim; i++)
    {
        Lg.push_back(L[i] / float(Nxx[i]));
    }
    for (int i = 0; i < Dim; i++)
    {
        std::cout << "Nxx[" << i << "]: " << Nxx[i] << " |L:" << L[i] << " |dL: " << Lg[i] << endl;
    }

    std::cout << "Output file name: " << out_file_name << endl;

    if (Dim == 2){Nxx.push_back(1); Lg[2] = 1.0;}

    d_Nxx.resize(3);
    d_Nxx = Nxx;
    d_Lg.resize(3);
    d_Lg = Lg;

    if (Dim == 3){nncells = 25;}
    if (Dim == 2){nncells = 9;}          

    d_MASTER_GRID.resize(xyz * ad_hoc_density);                 
    d_MASTER_GRID_counter.resize(xyz);

    d_RN_ARRAY.resize(group->nsites * ad_hoc_density * nncells);
    d_RN_ARRAY_COUNTER.resize(group->nsites);

    d_LOW_DENS_FLAG.resize(group->nsites);

    thrust::fill(d_MASTER_GRID.begin(),d_MASTER_GRID.end(),-1);
    thrust::fill(d_MASTER_GRID_counter.begin(),d_MASTER_GRID_counter.end(),0);

    thrust::fill(d_RN_ARRAY.begin(),d_RN_ARRAY.end(),-1);
    thrust::fill(d_RN_ARRAY_COUNTER.begin(),d_RN_ARRAY_COUNTER.end(),0);

    thrust::fill(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 0);

    std::cout << "Distance parameters || xyz: " << xyz << ", ad_hoc_density: " << ad_hoc_density << ", Dim: " << Dim << ", n_cells: " << nncells << endl;
    std::cout << "Size: " << d_RN_ARRAY.size() << endl;

    NList::KillingMeSoftly();
}

NList::~NList(){}

NList* NListFactory(istringstream &iss){

	std::stringstream::pos_type pos = iss.tellg(); // get the current position - beginning of the string
	string s1;
	iss >> s1 >> s1; // dynamicgroup <name> type
	iss.seekg(pos); // reset the stringf to the initial position and pass to the specific contructor

    // Specialized constructors

	// stores a "half" neighbour list - particle only stores a neioghbour with an index lower than its own
	if (s1 == "half_distance"){
		return new NListHalfDistance(iss);
	}

	if (s1 == "bonding"){
		return new NListBonding(iss);
	}	

	die(s1 + " is not a valid neighbour list style");
	return 0;
}


int NList::CheckTrigger(){

	if (nlist_freq < 0 && MAX_DISP>= d_trigger){
		MAX_DISP = -1.0;
		return 1;
	}
	else if(nlist_freq > 0 && step%nlist_freq == 0){
		return 1;
	}
	else {
		return 0;
	}
}

void NList::WriteNList(void)
{
    const char* fname = (out_file_name + "_grid").c_str();
    if (step == 0){remove(fname);}

    ofstream nlist_file;
    nlist_file.open(out_file_name + "_grid", ios::out | ios::app);
    nlist_file << "TIMESTEP: " << step << endl;
    for (int j = 0; j < xyz; ++j){
        nlist_file << j << "|" << d_MASTER_GRID_counter[j] << ": ";
        for (int i = 0; i < ad_hoc_density; i++){
            if (d_MASTER_GRID[j*ad_hoc_density + i] != -1)
            nlist_file << group->index[d_MASTER_GRID[j*ad_hoc_density + i]] << " ";
            else nlist_file << "* ";
        }
        nlist_file << endl;
    }

    const char* pfname = (out_file_name + "_pp").c_str();
    if (step == 0){remove(pfname);}

    ofstream pnlist_file;
    pnlist_file.open(out_file_name + "_pp", ios::out | ios::app);
    pnlist_file << "TIMESTEP: " << step << endl;
    for (int j = 0; j < group->nsites; ++j){
        pnlist_file << group->index[j] << "|" << d_RN_ARRAY_COUNTER[j] << ": ";
        for (int i = 0; i < nncells * ad_hoc_density; i++){
            if (d_RN_ARRAY[j * nncells * ad_hoc_density + i] != -1)
            pnlist_file << group->index[d_RN_ARRAY[j * nncells * ad_hoc_density + i]] <<" ";
            else pnlist_file <<"* ";
        }
        pnlist_file << endl;
    }
}

void NList::ResetArrays(){

    // increase the capacity of the density array if it is not enough to hold all the particles

    int sum = thrust::count(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 1);
    float ldc = float(sum)/float(d_MASTER_GRID_counter.size());

    if (ldc > 0){
        cout << "Input density was: " << ad_hoc_density <<" but at least "<< ad_hoc_density + ldc <<" is required"<<endl;
        ad_hoc_density += ceil(ldc * 1.2);
        cout << "Increasing the density to " <<  ad_hoc_density <<  " at step " << step << endl;

        d_MASTER_GRID.resize(xyz * ad_hoc_density);              
        d_RN_ARRAY.resize(group->nsites * ad_hoc_density * nncells);

        ldc = 0;
    }


    thrust::fill(d_MASTER_GRID.begin(),d_MASTER_GRID.end(),-1);
    thrust::fill(d_MASTER_GRID_counter.begin(),d_MASTER_GRID_counter.end(),0);

    thrust::fill(d_RN_ARRAY.begin(),d_RN_ARRAY.end(),-1);
    thrust::fill(d_RN_ARRAY_COUNTER.begin(),d_RN_ARRAY_COUNTER.end(),0);

    thrust::fill(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 0);
}


void NList::KillingMeSoftly(){
	if (Dim == 3)
        if(Nxx[0] < 5 || Nxx[1] < 5 || Nxx[2] < 5)
        {
		    cout << "Must have at least 5 cells in each direction!" << endl; exit(1);
        }
}