// Copyright (c) 2024 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_species.h"

void die(const char*);

PS_Species::PS_Species() {}
PS_Species::~PS_Species() {}


// Constructor with command, box label passed
PS_Species::PS_Species(std::istringstream& iss, PS_Box* box) : mybox(box) {
    this->inputCommand = iss.str();
    
    // Set default values
    mass = 1.0;
    mobility = 1.0;

    if ( iss.tellg() == -1 ) die("Missing label for species");

    // Store the label
    iss >> speciesLabel;

    while ( iss.tellg() != -1 ) {
        std::string word;
        iss >> word;
        
        if ( word == "mass" || word == "Mass" ) {
            iss >> mass;
        }

        else if ( word == "mobility" || word == "Mobility" || word == "Diffusivity" 
                  || word == "diffusivity" ) {
            iss >> mobility;
        }
    }
}

// Return integer label for the species
int PS_Species::returnIntSpecies() {
    return intSpecies;
}

void PS_Species::setIntSpecies(int speciesIntVal) {
    intSpecies = speciesIntVal;
}

int PS_Species::isSpecies(std::string test) {
    if ( speciesLabel == test ) {
        return 1;
    }
    else { return 0; }
}