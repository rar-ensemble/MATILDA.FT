#ifndef _PS_SPECIES
#define _PS_SPECIES

#include "include_libs.h"

class PS_Box;
class PS_Species {
    protected:
        int intSpecies;
        std::string speciesLabel;
        std::string inputCommand;
        int groupID;
        
        PS_Box* mybox;

    public:

        PS_Species();
        virtual ~PS_Species();
        PS_Species(std::istringstream&, PS_Box*); // constructor to parse input command

        float mass;     // particle mass
        float mobility; // particle mobility coefficient (default 1)


        int returnIntSpecies();     // Returns integer species number
        std::string returnSpecies();// Returns text species label
        void setIntSpecies(int);    // Sets integer species number
        
        void setGroupID(int);       // Sets integer group ID
        int returnGroupID(void);    // Returns integer group ID

        int isSpecies(std::string); // Compares text string to species text label
};

#endif