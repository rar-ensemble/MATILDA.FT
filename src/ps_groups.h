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
__global__ void d_assignFloatVal(float*, const float, const int);
__global__ void d_fillDensityGrid(float*, const int*, const int*, const float*,
const int, const int);

class PS_Box;

class PS_Group {
    private:
        std::string inputCommand;   // Command used to create this group
        std::string name;           // Text name of this group

    public:
        int nsites;                         // Number of particles in this group

        int *siteList, *d_siteList;         // [nsites] list of sites in this group
        float *rho, *d_rho;                 // [M] density field for this group

        int Grid, Block;    // GPU config variables for this group

        PS_Box* mybox;
        PS_Group();
        PS_Group(std::istringstream&, PS_Box*);
        PS_Group(std::string, int, PS_Box*);

        void zeroFields();
        void allocateGroupMemory(int);
        void makeDensityField();
        void writeDensityField();
        int isGroup(std::string);
        std::string returnName();
        virtual ~PS_Group();
};

#endif