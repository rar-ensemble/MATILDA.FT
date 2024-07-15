// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

//////////////////////////////////////////////////
// ps_groups.h                  Rob Riggleman   //
//                              July 15, 2024   //
// Class that contains group information for    //
// particle-based simulations (e.g., TILD).     //
//////////////////////////////////////////////////

#ifndef _PS_GROUPS
#define _PS_GROUPS

#include "include_libs.h"

class PS_Box;

class PS_Group {
    private:
        std::string inputCommand;   // Command used to create this group
        std::string name;           // Text name of this group

    public:
        int nsites;                         // Number of particles in this group
        thrust::host_vector<int> siteList;  // List of sites in this group
        thrust::device_vector<int> d_siteList;  // Device list of sites in this group

        thrust::host_vector<float> rho;     // [M] density field for this group
        thrust::device_vector<float> d_rho; // [M] device density field for this group

        PS_Box* mybox;
        PS_Group();
        PS_Group(std::istringstream&, PS_Box*);
        PS_Group(std::string, int, PS_Box*);
        std::string returnName();
        virtual ~PS_Group();
};

#endif