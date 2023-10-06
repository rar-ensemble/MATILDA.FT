// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "group.h"
#include <string>
#include <sstream>
#include <sstream>
#include <string>
#include <curand_kernel.h>
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifndef _NLIST
#define _NLIST

class NList {
protected:

    static int total_num_nlists;    // # of nlists
    static int total_num_triggers;  // # of triggers for automatically updating nlists

                   // index of the n-list
    int group_index;                // index of group on which this acts
    std::string group_name;         // Name of the group on which this acts
    Group* group;                   // pointer to the group

    std::string command_line;       // Full line of the command from input file

public:


    // Basics

    NList(std::istringstream&);
    ~NList();

    // Class Memebers

    int id;      

    std::string name;
    std::string out_file_name;
    std::string style;    // Style of the NList

    int ad_hoc_density;
    int nncells;
    int nlist_freq; //, n_donors, n_acceptors;

    float r_n;
    float r_skin;
    float delta_r;
    float d_trigger;  

    // grid variables

    int xyz;

    thrust::device_vector<int> d_Nxx; // grid spacing for the n-list [Dim]
    thrust::device_vector<float> d_Lg; // grid cell length [Dim]

    thrust::host_vector<int> Nxx; // grid spacing for the n-list [Dim]
    thrust::host_vector<float> Lg; // grid cell length [Dim]

    thrust::device_vector<int> d_LOW_DENS_FLAG;

    thrust::device_vector<int> d_MASTER_GRID;
    thrust::device_vector<int> d_MASTER_GRID_counter;
    thrust::device_vector<int> d_RN_ARRAY;
    thrust::device_vector<int> d_RN_ARRAY_COUNTER;

    thrust::host_vector<int> RN_ARRAY;
    thrust::host_vector<int> RN_ARRAY_COUNTER;


    
    // Additional functions

    void KillingMeSoftly(); // Checks if the grid division is correct
    int CheckTrigger();
    void WriteNList();
    void ResetArrays();


    std::string printCommand(){return command_line;}

    // Override functions

    virtual void MakeNList(void){return;};
};



#endif