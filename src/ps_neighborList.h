// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include <sstream>
#include <string>

#ifndef _PS_NEIGHBORLIST
#define _PS_NEIGHBORLIST

class PS_Box;

class PS_NeighborList {
public:
    PS_NeighborList(std::istringstream& iss, PS_Box* box);
    ~PS_NeighborList();
    void build();

    int   groupInd;
    float rcut, rcut2;
    int   maxNeighbors;

    int*  d_neighborList;   // [nsites * maxNeighbors] global particle indices
    int*  d_nNeighbors;     // [nsites] actual neighbor count per particle
    int   nsites;

private:
    PS_Box* mybox;
    int   nCellsX, nCellsY, nCellsZ, nCells;
    float cellWidthX, cellWidthY, cellWidthZ;

    int*  d_cellID;         // [nsites] cell index for each (sorted) particle
    int*  d_particleID;     // [nsites] global particle IDs (sorted by cellID)
    int*  d_cellStart;      // [nCells]
    int*  d_cellEnd;        // [nCells]
};

#endif
