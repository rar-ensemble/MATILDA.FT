// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


//////////////////////////////////////////////
// Rob Riggleman                7/21/2021   //
// Defining integrator class that will call //
// particle integration routines based on   //
// groups.                                  //
//////////////////////////////////////////////

#include <string>
#include "group.h"
#include "include_libs.h"

#ifndef _INTEGRATOR
#define _INTEGRATOR

class Integrator {
protected:
    int group_index;  // index of group to be integrated
    std::string group_name;// name of the group to be integrated
    Group* group;     // pointer to the group to be integrated
    std::string command_line;

public:
    std::string name;      // name of the integrator to be used
    Integrator(std::istringstream& iss);
    virtual ~Integrator();    // destructor
    virtual void Integrate_1(void); // Calls the pre-force integrator
    virtual void Integrate_2(void); // Calls the pre-force integrator
    std::string printCommand(){return command_line;}
    static int using_GJF;
};

#endif