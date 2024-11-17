// Copyright (c) 2024 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_groups.h"
#include "PS_Box.h"

void die(const char*);

PS_Group::PS_Group() {}
PS_Group::~PS_Group() {}

PS_Group::PS_Group(std::istringstream& iss, PS_Box* box) : mybox(box) {
    inputCommand = iss.str();
}

PS_Group::PS_Group(std::string inp, int typ, PS_Box* box) : mybox(box) {
    inputCommand = std::string("group ") + inp;
    forceFlag = 0;
    
    // Make the group for "all" particles
    if ( inp == "all" || inp == "All" ) {
        name = "all";
        nsites = mybox->nstot;

        this->allocateGroupMemory(nsites);

        for ( int i=0 ; i<mybox->nstot; i++ ) {
            siteList[i] = i;
        }

    }// group "all"


    // Make the group for particles of integer type 'typ'
    if ( inp == "type" || inp == "Type" ) {
        name = mybox->species[typ].returnSpecies();

        inputCommand = inputCommand + std::string(" ") + std::to_string(typ+1);
        nsites = 0;

        // Count the number of this type
        for ( int i=0 ; i<mybox->nstot; i++ ) {
            if ( mybox->intSpecies[i] == typ ) nsites++;
        }

        this->allocateGroupMemory(nsites);

        // Store the particle list
        int listInd = 0;
        for ( int i=0 ; i<mybox->nstot; i++ ) {
            if ( mybox->intSpecies[i] == typ ) {
                siteList[listInd] = i;
                listInd++;
            }
        }

        //std::cout << "Group based on type " << typ+1 << " has " << nsites << " members, is named " << name << "." << std::endl;

    }// type-based group


    // std::cout << "Group constructor for typ = " << typ << ", command: " << inputCommand << std::endl;

    // Copy site list to device
    cudaMemcpy(d_siteList, siteList, nsites*sizeof(int), cudaMemcpyHostToDevice);

    // Set group grid, block size
    Block = mybox->blockSize;
    Grid = (int)ceil((double)(nsites) / Block);
}


// Fills the density field for this group
void PS_Group::makeDensityField() {
    
    d_fillDensityGrid<<<Grid, Block>>>(d_rho, d_siteList, mybox->d_gridInds, 
        mybox->d_gridW, mybox->gridPerPartic, nsites);
    check_cudaError("Group ps_group::makeDensityField()");
}


// Takes a grid-based force, generally passed from a potential,
// and adds it to this group's grid force
void PS_Group::accumulateGridForces(
    const float* d_fg   // [Dim*M] grid-based force
) {

    int Grid = mybox->DMGrid;
    int Block = mybox->M_Block;
    int ndof = mybox->returnDimension() * mybox->M;

    d_floatPlusEqFloat<<<Grid, Block>>>(d_gridForce, d_fg, ndof);
}


// Sets the field variables to zero to start each time step
void PS_Group::zeroFields() {

    // this routine is in device_utils.cu
    d_assignFloatVal<<<mybox->M_Grid, mybox->M_Block>>>(d_rho, 0.0, mybox->M);

    if ( forceFlag ) { 
        d_assignFloatVal<<<mybox->DMGrid, mybox->M_Block>>>(d_gridForce, 0.0, 
            mybox->returnDimension() * mybox->M);
    }
}


// Allocates memory for arrays in this group.
void PS_Group::allocateGroupMemory(int ns) {
    // Allocate needed memory for lists
    siteList = (int*) calloc(ns, sizeof(int));
    cudaMalloc(&d_siteList, ns * sizeof(int));

    // std::cout << "  in group allocating density fields for M grid points: " << mybox->M << std::endl;
    
    // Allocate memory for fields
    rho = (float*) malloc(mybox->M * sizeof(float));
    cudaMalloc(&d_rho, mybox->M * sizeof(float));

    check_cudaError("group allocation for d_rho");
    
}


// Allocates storage for the grid-based forces for this group
// Not on by default bc group scan be used for other things
void PS_Group::enableForce() {
    if ( mybox->verbose ) { std::cout << "group labeled " << name << " allocating forces " << std::endl; }

    int nalloc = mybox->M * mybox->returnDimension() * sizeof(float);
    gridForce = (float*) malloc( nalloc );
    cudaMalloc(&d_gridForce, nalloc);
    
    forceFlag = 1;
}



// Copies density field to host, calls
// subroutine to write thrust float vector
void PS_Group::writeDensityField() {
    std::string fileName = std::string("density-") + name + std::string(".dat");
    
    // std::cout << "  file name: " << fileName << std::endl;
    // std::cout << "  DEBUGGING: M=" << mybox->M << std::endl;


    // rho = d_rho;
    cudaMemcpy(rho, d_rho, mybox->M*sizeof(float), cudaMemcpyDeviceToHost);
    check_cudaError("PS_Group::writeDensityFields memory copy");

    // std::cout << "  attempting to write field " << name << std::endl;

    mybox->writeFieldFloat(fileName.c_str(), rho);

}



// Return the name of this group
std::string PS_Group::returnName() {
    return  name;
}

int PS_Group::hasForce() {
    if ( forceFlag ) return 1;
    else return 0;
}

int PS_Group::isGroup(std::string testName) {
    if ( name == testName ) 
        return 1;
    else 
        return 0;
}