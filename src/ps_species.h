#ifndef _PS_SPECIES
#define _PS_SPECIES

#include "include_libs.h"

class PS_Box;
class PS_Species {
    protected:
        int intSpecies;
        std::string speciesLabel;
        std::string inputCommand;
        
        PS_Box* mybox;

    public:
        PS_Species();
        virtual ~PS_Species();
        PS_Species(std::istringstream&, PS_Box*); // constructor to parse input command

        float mass;     // particle mass
        float mobility; // particle mobility coefficient (default 1)


        int returnIntSpecies();     // Returns integer species number
        void setIntSpecies(int);    // Sets integer species number

        int isSpecies(std::string); // Compares text string to species text label
};

#endif