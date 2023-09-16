// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).



#include <string>
#include <sstream>
#include <vector>
#include "group.h"

#ifndef _GROUPS_ID
#define _GROUPS_ID


class GroupID : public Group {
private:

    std::string input_file;
    std::vector<int> atom_id;

public:
    GroupID(std::istringstream& iss); // Constructor for parsing input file
    ~GroupID();
    void CheckGroupMembers(void) override;
};


#endif