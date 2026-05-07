// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include <sstream>
#include <string>

#ifndef _PS_NEIGHBORLIST
#define _PS_NEIGHBORLIST

class PS_Box;

class PS_NeighborList {
public:
    void initializeNList();
    PS_NeighborList(std::istringstream& iss, PS_Box* box);
    ~PS_NeighborList();
    void build();

    std::string grpName;
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

    int*  d_cellID;         // [nsites] unsorted cell indices (CUB sort input)
    int*  d_particleID;     // [nsites] unsorted global particle IDs (CUB sort input)
    int*  d_cellID_sorted;      // [nsites] sorted cell indices (CUB sort output)
    int*  d_particleID_sorted;  // [nsites] global IDs sorted by cellID (CUB sort output)
    int*  d_cellStart;      // [nCells]
    int*  d_cellEnd;        // [nCells]
};

#endif
