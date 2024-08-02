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

#define readRequiredParameter(ss,parameter) readParameter(ss,parameter); //std::cout <<"Read parameter: " << #parameter; readParameter(ss, parameter);

template <typename T> 
static void readParameter(std::istringstream& ss, T& parameter){
    if(ss >> parameter){
        //std::cout <<": " << parameter << std::endl;
    }
    else{
        std::cout << " Invalid Value! Exiting..." << std::endl;
        exit(1);
    }
};



#endif