// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include <vector>
#include <sstream>

#ifndef HERE
#define HERE
template<typename T>
static void die(T msg){
	std::cout << msg << std::endl;
    exit(1);
};

/// Macro to read required parameters and report values or errors

#define readRequiredParameter(ss,parameter) std::cout <<"Read parameter: " << #parameter; readParameter(ss, parameter);

template <typename T> 
static void readParameter(std::istringstream& ss, T& parameter){
    if(ss >> parameter){
        std::cout <<": " << parameter << std::endl;
    }
    else{
        std::cout << " Invalid Value! Exiting..." << std::endl;
        exit(1);
    }
};


extern std::vector<Group*> Groups;
template<typename T>
int get_group_id(T group_name){
    for (auto group: Groups){
        if (group->name == group_name){
            return group->id;
        }
    }
    die("Group"+group_name+"not found!");
    return -1;
};


extern std::vector<NList*> NLists;
template<typename T>
int get_nlist_id(T nlist_name){
    for (auto nlist: NLists){
        if (nlist->name == nlist_name){
            return nlist->id;
        }
    }
    die("NList"+nlist_name+"not found!");
    return -1;
};

#endif