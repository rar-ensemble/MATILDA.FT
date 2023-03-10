// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


//////////////////////////////////////////////
// Rob Riggleman                7/26/2021   //
// Class for adding additional forces.      //
// Will initially be designed to add the DPD//
// dissipation and friction forces, but     //
// should be extensible to make groups      //
// interact with walls and the like.        //
//////////////////////////////////////////////

#include <sstream>
#include <string>
#include <curand_kernel.h>
#include <curand.h>
#include "group.h"
#include "nlist.h"

#ifndef _EXTRAFORCE
#define _EXTRAFORCE

class ExtraForce {
protected:

    int group_index;    // index of group on which this acts
    std::string group_name;  // Name of the group on which this acts
    std::string style;    // Style of the extraforce
    std::string command_line;// Full line of the command from input file
    Group* group;     // pointer to the group to be ExtraForce'd

    int nlist_index;
    std::string nlist_name;  // Name of the group on which this acts
    NList* nlist;     // pointer to the group to be ExtraForce'd

public:

    std::string name;            // Name of the ExtraForce
    int id;
    virtual void AddExtraForce(void) = 0;
    std::string printCommand(){
        return command_line;
    }
    ExtraForce(std::istringstream&);
    virtual ~ExtraForce();
    virtual void UpdateVirial(void) = 0;
};
#endif
