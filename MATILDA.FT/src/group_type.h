// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).



#include <string>
#include <sstream>
#include <vector>
#include "group.h"

#ifndef _GROUP_TYPE
#define _GROUP_TYPE


class GroupType : public Group {
private:

    thrust::host_vector<int> atom_types;

public:

    GroupType(std::istringstream& iss);
    ~GroupType();

    // Override functions

    void CheckGroupMembers(void) override;

};

#endif