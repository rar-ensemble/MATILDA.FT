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
    inputCommand = inp + char(typ);
    
    // Make the group for "all" particles
    if ( inp == "all" || inp == "All" ) {
        name = "all";
        nsites = mybox->nstot;
        siteList.resize(nsites);
        d_siteList.resize(nsites);

        for ( int i=0 ; i<mybox->nstot; i++ ) {
            siteList[i] = i;
        }

    }// group "all"


    // Make the group for particles of integer type 'typ'
    if ( inp == "type" || inp == "Type" ) {
        nsites = 0;

        // Count the number of this type
        for ( int i=0 ; i<mybox->nstot; i++ ) {
            if ( mybox->partic[i].intSpecies == typ ) nsites++;
        }

        // Allocate needed memory
        siteList.resize(nsites);
        d_siteList.resize(nsites);

        // Store the particle list
        int listInd = 0;
        for ( int i=0 ; i<mybox->nstot; i++ ) {
            if ( mybox->partic[i].intSpecies == typ ) {
                siteList[listInd] = i;
                listInd++;
            }
        }

    }// type-based group


    // Copy site list to device
    d_siteList = siteList;

}


// Return the name of this group
std::string PS_Group::returnName() {
    return  name;
}