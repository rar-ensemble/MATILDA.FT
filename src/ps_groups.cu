// Copyright (c) 2024 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_groups.h"
#include "PS_Box.h"

void die(const char*);

PS_Group::PS_Group() {}
PS_Group::~PS_Group() {}

PS_Group::PS_Group(std::istringstream& iss, PS_Box* box) : mybox(box) {
    inputCommand = iss.str();

    iss >> name;
    nsites = 0;
    forceFlag = 0;

    std::string s1 ;
    
    iss >> s1;
    if ( s1 == "types" ) {
        std::vector<int> typ_ints;

        while ( iss.tellg() != -1 ) {
            iss >> s1;
            typ_ints.push_back( mybox->findSpeciesInteger(s1) );            
        }

        for ( int i=0 ; i<mybox->nstot; i++ ) {
            for ( int j=0 ; j<typ_ints.size() ; j++ ) {
                if ( mybox->intSpecies[i] == typ_ints[j] ) {
                    nsites++;
                }
            }
        }// i=0:mybox->nstot

        this->allocateGroupMemory(nsites);

        // Populate the list
        int listInd = 0;
        for ( int i=0 ; i<mybox->nstot; i++ ) {
            for ( int j=0 ; j<typ_ints.size() ; j++ ) {
                if ( mybox->intSpecies[i] == typ_ints[j] ) {
                    siteList[listInd] = i;
                    listInd++;
                    break;
                }
            }
        }

        std::cout << "    making group with " << typ_ints.size() << " types with " << nsites << " total sites." << std::endl;
    }

    // Copy site list to device
    cudaMemcpy(d_siteList, siteList, nsites*sizeof(int), cudaMemcpyHostToDevice);

    // Set group grid, block size
    Block = mybox->blockSize;
    Grid = (int)ceil((double)(nsites) / Block);


    std::cout << "    Created group '" << name << "' with " << nsites << " sites" << std::endl;
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

    }// type-based group


    // make the group for non-zero charges
    if ( inp == "charges" ) {
        name = "charges";

        nsites = 0;
        for ( int i=0 ; i<mybox->nstot; i++ ) {
            if ( mybox->charges[i] != 0.0 ) nsites++;
        }


        this->allocateGroupMemory(nsites);

        int listInd = 0;
        for ( int i=0; i<mybox->nstot; i++ ) {
            if ( mybox->charges[i] != 0.0 ) {
                siteList[listInd] = i;
                listInd++;
            }
        }
        this->enableForce();
        std::cout << "Charge group generated with " << nsites << " particles." << std::endl;
        
    }


    // Copy site list to device
    cudaMemcpy(d_siteList, siteList, nsites*sizeof(int), cudaMemcpyHostToDevice);

    // Set group grid, block size
    Block = mybox->blockSize;
    Grid = (int)ceil((double)(nsites) / Block);
}


// Fills the density field for this group
void PS_Group::makeDensityField() {
    
    if ( name == "charges" ) {
        d_fillChargeDensityGrid<<<Grid, Block>>>(d_rho, d_rhoq, d_siteList, mybox->d_charges,
            mybox->d_gridInds, mybox->d_gridW, mybox->gridPerPartic, nsites);
    }

    else {
        d_fillDensityGrid<<<Grid, Block>>>(d_rho, d_siteList, mybox->d_gridInds, 
            mybox->d_gridW, mybox->gridPerPartic, nsites);
    }
    check_cudaError("Group ps_group::makeDensityField()");
}


// Takes a grid-based force, generally passed from a potential,
// and adds it to this group's grid force
void PS_Group::accumulateGridForceComp(
    const float* d_fx,  // [M] grid-based force comopnent
    const int dir       // Component of the vector to accumulate into
) {

    int MGrid = mybox->M_Grid;
    int MBlock = mybox->M_Block;
    int dim = mybox->returnDimension();
    int M = mybox->M;
    // Guard must be M (not dim*M): MGrid*MBlock ≈ M threads are launched, each
    // handling one grid point.  Passing dim*M let the few extra padding threads
    // (id ∈ [M, MGrid*MBlock)) slip past the guard and write past d_gridForce.
    d_floatVecPlusEqFloatComp<<<MGrid, MBlock>>>(d_gridForce, d_fx, dir, dim, M);

}// accumulateGridForces()


// Maps the forces on this group from the grid to the particles
void PS_Group::mapForces() {

    if ( forceFlag ) {
        if ( name == "charges" ) {
            d_mapGridChargeForcesToPartics<<<Grid, Block>>>(mybox->d_f, mybox->d_charges, d_gridForce, 
                d_siteList, mybox->d_gridInds, mybox->d_gridW, mybox->gvol, mybox->gridPerPartic, 
                mybox->returnDimension(), nsites);

        }
        else {
            d_mapGridForcesToPartics<<<Grid, Block>>>(mybox->d_f, d_gridForce, d_siteList, mybox->d_gridInds,
                mybox->d_gridW, mybox->gvol, mybox->gridPerPartic, mybox->returnDimension(), nsites);
        }
        check_cudaError("Group ps_group::mapForces()");

    }
    else {
        if ( mybox->verbose ) std::cout << "NO FORCE HERE, group name = " << name << std::endl;
    }
}

// Sets the field variables to zero to start each time step
void PS_Group::zeroFields() {

    // this routine is in device_utils.cu
    d_assignFloatVal<<<mybox->M_Grid, mybox->M_Block>>>(d_rho, 0.0, mybox->M);
    if ( this->name == "charges" ) {
        d_assignFloatVal<<<mybox->M_Grid, mybox->M_Block>>>(d_rhoq, 0.0, mybox->M);
    }

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

    
    // Allocate memory for fields
    rho = (float*) malloc(mybox->M * sizeof(float));
    cudaMalloc(&d_rho, mybox->M * sizeof(float));

    check_cudaError("group allocation for d_rho");


    if ( this->name == "charges" ) {
        rhoq = (float*) malloc(mybox->M * sizeof(float));
        cudaMalloc(&d_rhoq, mybox->M * sizeof(float));
    }

    check_cudaError("group allocation for d_rhoq");
    
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


// Writes density field in a binary format
void PS_Group::writeDensityFieldBinary() {
    std::string nm = std::string("density-") + name + std::string(".bin");
    cudaMemcpy(rho, d_rho, mybox->M*sizeof(float), cudaMemcpyDeviceToHost);

    mybox->writeBinaryData(nm, rho);
}


// Copies density field to host, calls
// subroutine to write thrust float vector
void PS_Group::writeDensityField() {

    std::string fileName = std::string("density-") + name + std::string(".dat");

    // rho = d_rho;
    cudaMemcpy(rho, d_rho, mybox->M*sizeof(float), cudaMemcpyDeviceToHost);
    check_cudaError("PS_Group::writeDensityFields memory copy");

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