#include "PS_Box.h"
#include "random.h"
#include "include_libs.h"
#include "gsd.h"
#include <algorithm>
#include <map>

void die(const char*);
double ran2(void);
void random_unit_vec(double*, int);
void make_lc_file(std::string, int, std::vector<int>, std::vector<int>);


// Generate a new linear polymer of arbitrary blockiness and add it to the box
void PS_Box::makeLinear(std::istringstream& iss ) {
    if ( rho0 < 0.0 ) die("rho0 must be defined before molecules created!");

    int numBlocks, Ntot = 0;

    // Both set to negative values to determine which keyword given
    double phi = -1.0;  
    int nmolecs = -1; 
    int angleFlag = 0;

    // Flag for dealing with charges on this molec type
    int doMolecCharges = 0;

    std::string s1;

    // read either phi or nmolecs keyword
    iss >> s1;

    if ( s1 == "phi" ) {
        iss >> phi;
    }
    else if ( s1 == "nmolecs" ) {
        iss >> nmolecs;
    }
    else { 
        std::cout << "invalid keyword: " << s1 << std::endl;
        die("invalid keyword after molecule linear. must be 'phi' or 'nmolecs'"); 
    }

    // Read in the number of blocks
    iss >> numBlocks; 
    
    std::vector<int> Nblocks(numBlocks);
    std::vector<int> blockBondType(numBlocks,1);
    std::vector<int> blockAngleType(numBlocks,0);
    std::vector<int> drudeFlag(numBlocks,0);
    std::vector<int> chargeFlag(numBlocks,0);
    std::vector<std::string> speciesBlocks(numBlocks);
    std::vector<int> intSpeciesBlocks(numBlocks);
    std::vector<float> blockCharge(numBlocks, 0.0);
    
    
    std::vector<float> Rmin(Dim,0.0);
    std::vector<float> Rmax(Dim);
    for ( int j=0 ; j<Dim ; j++ ) Rmax[j] = L[j];

    // Store the basic block info
    for (int j=0 ; j<numBlocks; j++ ) {
        iss >> Nblocks[j];
        iss >> speciesBlocks[j];

        Ntot += Nblocks[j];
    }

    // Check for optional arguments
    while ( iss.tellg() != -1 ) {

        iss >> s1;
        if ( s1 == "drude" ) {
            die("drude oscillators not implemented!");
        }

        else if ( s1 == "charge" ) {
            if ( this->doCharges == 0 ) {
                this->enableCharges();
            }

            doMolecCharges = 1;
            
            for ( int j=0 ; j<numBlocks ; j++ ) {
                float q1;
                iss >> q1;
                blockCharge[j] = q1;
            }
        }// s1==charge

        else if ( s1 == "bondType" || s1 == "bondtype" ) {
            for ( int j=0 ; j<numBlocks ; j++ ) {
                int t1; 
                iss >> t1;
                blockBondType[j] = t1;
            }
            // make sure to decide on how to handle junction cases, document it
        }
        else if ( s1 == "angleType" || s1 == "angletype" ) {
            for ( int j=0 ; j<numBlocks ; j++ ) {
                int t1;
                iss >> t1;
                blockAngleType[j] = t1;
                std::cout << "  block ang type: " << j << " " << blockAngleType[j] << std::endl;
                angleFlag = 1;
            }
        }
        else if ( s1 == "xrange" ) {
            iss >> Rmin[0];
            iss >> Rmax[0];
        }
        else if ( s1 == "yrange" ) {
            iss >> Rmin[1];
            iss >> Rmax[1];
        }
        else if ( s1 == "zrange" ) {
            if ( Dim != 3 ) { die("z-range defined for a non-3D simulation!"); }
            iss >> Rmin[2];
            iss >> Rmax[2];
        }
    }

    // Compute number of molecules of this type to add
    // if volume fraction was read
    if ( phi > 0.0 ) { nmolecs = int( rho0 * V * phi / float(Ntot) ); }

    std::cout << "Generating " << nmolecs << " molecules each with " << Ntot << " sites" << std::endl;

    // particle index to be incremented as particles added
    int ind = nstot;


    // Update number of sites in the box
    nstot += nmolecs * Ntot;
    allocHostParticleArrays(nstot);
    std::cout << "nstot changed values to: " << nstot << ", starting index: " << ind << std::endl;



    // Find starting molecule ID
    int molecInd = -1;
    for ( int i=0 ; i<nstot ; i++ ) {
        if (mID[i] > molecInd) molecInd = mID[i];
    }
    if ( molecInd < 0 ) molecInd = 0;


    // Initializes angle counter to zero
    int aind = ind;
    for ( int i=0 ; i<nmolecs ; i++ ) {
        for ( int j=0 ; j<numBlocks ; j++ ) {
            for (int s=0; s<Nblocks[j] ; s++ ) {
                nAngles[aind] = 0;
                aind++;
            }
        }
    }
    std::cout << "   MAX ANGLE INDEX: " << 3 * nstot * MAXANGLES << std::endl;






    ///////////////////////////////////////////////////////////
    // Main loop over molecules, blocks, sites on each block //
    ///////////////////////////////////////////////////////////
    for ( int i=0 ; i<nmolecs ; i++ ) {

        // backbone monomer index for molecule i
        int bbIndex = 0;
        double ru[3];
        for ( int j=0 ; j<numBlocks; j++ ) {
            int speciesVal = findSpeciesInteger(speciesBlocks[j]);
            
            for ( int s=0 ; s<Nblocks[j]; s++ ) {

                // Track species info
                intSpecies[ind] = speciesVal;
                
                
                // Is this a chain end? 
                // If so, place randomly in the box
                if ( j==0 && s==0 ) {
                    for ( int a=0 ; a<Dim ; a++ ) {
                        x[ind*Dim+a] = ran2() * (Rmax[a] - Rmin[a]) + Rmin[a];
                    }
                }

                // Not a chain end: place monomer a unit vector away
                // from previous monomer
                else {
                    
                    // if first monomer past the end or there are no angles,
                    // re-generate the bond orientation
                    if ( ( s==1 && j==0 ) || (!angleFlag) )
                        random_unit_vec(ru, Dim);

                    for ( int a=0 ; a<Dim ; a++ ) {
                        int prevXInd = (ind-1)*Dim+a;
                        int Xind = ind*Dim+a;

                        x[Xind] = x[prevXInd] + ru[a];

                        if ( x[Xind] > L[a] ) x[Xind] -= L[a];
                        else if ( x[Xind] < 0.0 ) x[Xind] += L[a];
                    }
                }

                // Initialize velocities, forces to 0.0
                for ( int a=0 ; a<Dim ; a++ ) {
                    v[ind*Dim+a] = f[ind*Dim+a] = 0.0;
                }


                if ( doMolecCharges ) {
                    charges[ind] = blockCharge[j];
                }

                nBonds[ind] = 0;
                
                // Initialize bonds
                // If not the first monomer on a chain, make a bond to previous monomer
                if ( j != 0 || s != 0 ) {
                    int nb = nBonds[ind];
                    
                    bondedTo[ind*MAXBONDS+nb] = ind-1;
                    bondType[ind*MAXBONDS+nb] = blockBondType[j] - 1;
                    nBonds[ind]++;
                }

                // if not the last monomer on a chain, make a bond to next monomer
                if ( j != (numBlocks-1) || s != (Nblocks[j]-1 ) ) {
                    int nb = nBonds[ind];
                    
                    bondedTo[ind*MAXBONDS+nb] = ind+1;
                    bondType[ind*MAXBONDS+nb] = blockBondType[j] - 1;
                    nBonds[ind]++;
                }


                // Initialize angles
                // logic is different from bonds. If *central* particle is
                // in an angle, then i1 and i3 angle lists are both updated.
                // This ensures angles accounted for when one site is on a
                // different block that may have different angleTypes (or none)
                if ( ( blockAngleType[j] != 0 ) &&                  // this block has angles
                     ( j > 0 || s > 0 ) &&                          // not the first monomer
                     ( j < numBlocks-1 || s < Nblocks[j]-1 ) ) {    // not the final monomer

                    // std::cout << "j: " << j << " s: " << s << " ind: " << ind << std::endl;
                    int i1 = ind-1;
                    int i2 = ind;
                    int i3 = ind+1;

                    // std::cout << "  ivals: " << i1 << " " << i2 << " " << i3 << std::endl;

                    int n1 = nAngles[i1];
                    int n2 = nAngles[i2];
                    int n3 = nAngles[i3];

                    // std::cout << "  n vals: " << n1 << " " << n2 << " " << n3 << std::endl;

                    int index1 = i1*MAXANGLES*3+3*n1;
                    int index2 = i2*MAXANGLES*3+3*n2;
                    int index3 = i3*MAXANGLES*3+3*n3;

                    // std::cout << "  index: " << index1 << " " << index2 << " " << index3 << std::endl;

                    
                    angleGroup[index1+0] = i1;
                    angleGroup[index1+1] = i2;
                    angleGroup[index1+2] = i3;

                    angleGroup[index2+0] = i1;
                    angleGroup[index2+1] = i2;
                    angleGroup[index2+2] = i3;

                    angleGroup[index3+0] = i1;
                    angleGroup[index3+1] = i2;
                    angleGroup[index3+2] = i3;

                    angleType[i1*MAXANGLES+n1] = blockAngleType[j] - 1;
                    angleType[i2*MAXANGLES+n2] = blockAngleType[j] - 1;
                    angleType[i3*MAXANGLES+n3] = blockAngleType[j] - 1;
                    
                    nAngles[i1]++;
                    nAngles[i2]++;
                    nAngles[i3]++;

                    
                }// blockAngleType[j]



                mID[ind] = molecInd;

                // Increment the particle index
                ind++;
                bbIndex++;
            }// s=0:N[j]

        }// j=0:numBlocks; 

        // Increment molecule index
        molecInd++;
    }// i=0:nmolecs


    std::cout << "nstot is " << nstot << " after molecule creation" << std::endl;
}


// Generate grafted (comb/bottlebrush) polymers of arbitrary blockiness and add it to the box
void PS_Box::makeGrafted(std::istringstream& iss ) {
    if ( rho0 < 0.0 ) die("rho0 must be defined before molecules created!");

    int numBlocks, Ntot = 0;

    // Both set to negative values to determine which keyword given
    double phi = -1.0;  
    int nmolecs = -1; 

    // Flag for dealing with charges on this molec type
    int doMolecCharges = 0;

    std::string s1;

    // read either phi or nmolecs keyword
    iss >> s1;

    if ( s1 == "phi" ) {
        iss >> phi;
    }
    else if ( s1 == "nmolecs" ) {
        iss >> nmolecs;
    }
    else { 
        std::cout << "invalid keyword: " << s1 << std::endl;
        die("invalid keyword after molecule linear. must be 'phi' or 'nmolecs'"); 
    }

    // Read in the number of blocks
    iss >> numBlocks; 
    
    std::vector<int> Nblocks(numBlocks);
    std::vector<int> blockBondType(numBlocks,1);
    std::vector<int> blockAngleType(numBlocks,0);
    std::vector<int> drudeFlag(numBlocks,0);
    std::vector<int> chargeFlag(numBlocks,0);
    std::vector<std::string> speciesBlocks(numBlocks);
    std::vector<int> intSpeciesBlocks(numBlocks);
    std::vector<float> blockCharge(numBlocks, 0.0);

    
    // Storage for side chain graft polymers
    std::vector<int> SC_numBlocks(numBlocks);
    std::vector<std::string> SC_style(numBlocks);       // "linear" or (eventually) "sclc"
    std::vector<std::vector<int>> SC_Nblocks;
    std::vector<std::vector<int>> SC_blockBondTypes;
    std::vector<std::vector<int>> SC_blockAngleType;
    std::vector<std::vector<std::string>> SC_speciesBlocks;
    std::vector<std::vector<int>> SC_intSpeciesBlocks;
    

    
    std::vector<float> Rmin(Dim,0.0);
    std::vector<float> Rmax(Dim);
    for ( int j=0 ; j<Dim ; j++ ) Rmax[j] = L[j];

    // Store the basic block info
    for (int j=0 ; j<numBlocks; j++ ) {
        iss >> Nblocks[j];
        iss >> speciesBlocks[j];

        Ntot += Nblocks[j];
    }

    // Check for optional arguments
    while ( iss.tellg() != -1 ) {

        iss >> s1;
        if ( s1 == "drude" ) {
            die("drude oscillators not implemented!");
        }

        else if ( s1 == "charge" ) {
            if ( this->doCharges == 0 ) {
                this->enableCharges();
            }

            doMolecCharges = 1;
            
            for ( int j=0 ; j<numBlocks ; j++ ) {
                float q1;
                iss >> q1;
                blockCharge[j] = q1;
            }
        }// s1==charge

        else if ( s1 == "bondType" || s1 == "bondtype" ) {
            for ( int j=0 ; j<numBlocks ; j++ ) {
                int t1; 
                iss >> t1;
                blockBondType[j] = t1;
            }
            // make sure to decide on how to handle junction cases, document it
        }
        else if ( s1 == "angleType" || s1 == "angletype" ) {
            for ( int j=0 ; j<numBlocks ; j++ ) {
                int t1;
                iss >> t1;
                blockAngleType[j] = t1;
                std::cout << "  block ang type: " << j << " " << blockAngleType[j] << std::endl;
            }
        }
        else if ( s1 == "xrange" ) {
            iss >> Rmin[0];
            iss >> Rmax[0];
        }
        else if ( s1 == "yrange" ) {
            iss >> Rmin[1];
            iss >> Rmax[1];
        }
        else if ( s1 == "zrange" ) {
            if ( Dim != 3 ) { die("z-range defined for a non-3D simulation!"); }
            iss >> Rmin[2];
            iss >> Rmax[2];
        }



        else if ( s1 == "grafted" ) {
            
            SC_Nblocks.resize(numBlocks);
            SC_blockBondTypes.resize(numBlocks);
            SC_blockAngleType.resize(numBlocks);
            SC_speciesBlocks.resize(numBlocks);
            SC_intSpeciesBlocks.resize(numBlocks);

            for ( int m=0 ; m<numBlocks ; m++ ) {
                iss >> SC_style[m] ;

                if ( SC_style[m] == "linear" ) {
                    iss >> SC_numBlocks[m] ;

                    SC_Nblocks[m].resize(SC_numBlocks[m],0);
                    SC_blockBondTypes[m].resize(SC_numBlocks[m], 1);
                    SC_blockAngleType[m].resize(SC_numBlocks[m], 0);
                    SC_speciesBlocks[m].resize(SC_numBlocks[m]);
                    SC_intSpeciesBlocks[m].resize(SC_numBlocks[m]);

                    for ( int n=0 ; n<SC_numBlocks[m] ; n++ ) {
                        iss >> SC_Nblocks[m][n];
                        iss >> SC_speciesBlocks[m][n];
                    }
                }

                else { 
                    die("PS_Box_moleculeMaker:makeGrafted: INVALID GRAFT TYPE, only linear grafts supported so far!");
                }
            }
        }// s1==grafted

    } // j=0:numBlcoks


    // Backbone length is current value of Ntot
    int Nbb = Ntot;
    if ( SC_Nblocks.size() == numBlocks ) {
        
        for ( int m=0 ; m<numBlocks ; m++ ) {
            for ( int n=0 ; n<SC_numBlocks[m] ; n++ ) {
                Ntot += Nblocks[m] * SC_Nblocks[m][n];
            }
        }
        
    }


    // Compute number of molecules of this type to add
    // if volume fraction was read
    if ( phi > 0.0 ) { nmolecs = int( rho0 * V * phi / float(Ntot) ); }

    std::cout << "Generating " << nmolecs << " molecules each with " << Ntot << " sites" << std::endl;

    // particle index to be incremented as particles added
    int ind = nstot;

    

    // Update number of sites in the box
    nstot += nmolecs * Ntot;
    allocHostParticleArrays(nstot);
    std::cout << "nstot changed values to: " << nstot << ", starting index: " << ind << std::endl;



    // Find starting molecule ID
    int molecInd = -1;
    for ( int i=0 ; i<nstot ; i++ ) {
        if (mID[i] > molecInd) molecInd = mID[i];
    }
    if ( molecInd < 0 ) molecInd = 0;


    std::cout << "   MAX ANGLE INDEX: " << 3 * nstot * MAXANGLES << std::endl;
    std::cout << "   MAX BOND INDEX: " << nstot * MAXBONDS << std::endl;



    // Storage for indices of the backbone sites
    std::vector<int> bb_inds(Nbb,0);


    ///////////////////////////////////////////////////////////
    // Main loop over molecules, blocks, sites on each block //
    ///////////////////////////////////////////////////////////
    for ( int i=0 ; i<nmolecs ; i++ ) {

        // backbone monomer index for molecule i
        int bbIndex = 0;
        
        double ru[3], rg[3];

        random_unit_vec(ru, Dim);   // Used for main backbone
        random_unit_vec(rg, Dim);   // used for graft after removing component along ru
        
        // Dot ru with rg
        double rug_dot = 0.0;
        for ( int j=0 ; j<Dim ; j++ ) { rug_dot += ru[j]*rg[j]; }
        
        // Remove magnitude of ru * rug_dot
        double rg2 = 0.0;
        for ( int j=0 ; j<Dim ; j++ ) { 
            rg[j] = rg[j] - rug_dot * ru[j]; 
            rg2 += rg[j] * rg[j];
        }

        // re-normalize rg
        double mag_rg = sqrt(rg2);
        for ( int j=0 ; j<Dim ; j++ ) { rg[j] *= 1.0 / mag_rg ; }


        for ( int j=0 ; j<numBlocks; j++ ) {
            int speciesVal = findSpeciesInteger(speciesBlocks[j]);
            
            for ( int s=0 ; s<Nblocks[j]; s++ ) {

                // Track species info
                intSpecies[ind] = speciesVal;
                
                
                // Is this a chain end? 
                // If so, place randomly in the box
                if ( j==0 && s==0 ) {
                    for ( int a=0 ; a<Dim ; a++ ) {
                        x[ind*Dim+a] = ran2() * (Rmax[a] - Rmin[a]) + Rmin[a];
                    }
                }

                // Not a chain end: place monomer a unit vector away
                // from previous monomer
                else {
                    
                    for ( int a=0 ; a<Dim ; a++ ) {
                        int prevXInd = (ind-1)*Dim+a;
                        int Xind = ind*Dim+a;

                        x[Xind] = x[prevXInd] + ru[a];

                        if ( x[Xind] > L[a] ) x[Xind] -= L[a];
                        else if ( x[Xind] < 0.0 ) x[Xind] += L[a];
                    }
                }

                // Initialize velocities, forces to 0.0
                for ( int a=0 ; a<Dim ; a++ ) {
                    v[ind*Dim+a] = f[ind*Dim+a] = 0.0;
                }


                if ( doMolecCharges ) {
                    charges[ind] = blockCharge[j];
                }

                nBonds[ind] = 0;
                nAngles[ind] = 0;
                
                // Initialize bonds
                // If not the first monomer on a chain, make a bond to previous monomer
                if ( j != 0 || s != 0 ) {
                    int nb = nBonds[ind];
                    
                    bondedTo[ind*MAXBONDS+nb] = ind-1;
                    bondType[ind*MAXBONDS+nb] = blockBondType[j] - 1;
                    nBonds[ind]++;
                }

                // if not the last monomer on a chain, make a bond to next monomer
                if ( j != (numBlocks-1) || s != (Nblocks[j]-1 ) ) {
                    int nb = nBonds[ind];
                    
                    bondedTo[ind*MAXBONDS+nb] = ind+1;
                    bondType[ind*MAXBONDS+nb] = blockBondType[j] - 1;
                    nBonds[ind]++;
                }


                // Initialize angles
                // logic is different from bonds. If *central* particle is
                // in an angle, then i1 and i3 angle lists are both updated.
                // This ensures angles accounted for when one site is on a
                // different block that may have different angleTypes (or none)
                if ( ( blockAngleType[j] != 0 ) &&                  // this block has angles
                     ( j > 0 || s > 0 ) &&                          // not the first monomer
                     ( j < numBlocks-1 || s < Nblocks[j]-1 ) ) {    // not the final monomer

                    // std::cout << "j: " << j << " s: " << s << " ind: " << ind << std::endl;
                    int i1 = ind-1;
                    int i2 = ind;
                    int i3 = ind+1;

                    // std::cout << "  ivals: " << i1 << " " << i2 << " " << i3 << std::endl;

                    int n1 = nAngles[i1];
                    int n2 = nAngles[i2];
                    int n3 = nAngles[i3];

                    // std::cout << "  n vals: " << n1 << " " << n2 << " " << n3 << std::endl;

                    int index1 = i1*MAXANGLES*3+3*n1;
                    int index2 = i2*MAXANGLES*3+3*n2;
                    int index3 = i3*MAXANGLES*3+3*n3;

                    // std::cout << "  index: " << index1 << " " << index2 << " " << index3 << std::endl;

                    
                    angleGroup[index1+0] = i1;
                    angleGroup[index1+1] = i2;
                    angleGroup[index1+2] = i3;

                    angleGroup[index2+0] = i1;
                    angleGroup[index2+1] = i2;
                    angleGroup[index2+2] = i3;

                    angleGroup[index3+0] = i1;
                    angleGroup[index3+1] = i2;
                    angleGroup[index3+2] = i3;

                    angleType[i1*MAXANGLES+n1] = blockAngleType[j] - 1;
                    angleType[i2*MAXANGLES+n2] = blockAngleType[j] - 1;
                    angleType[i3*MAXANGLES+n3] = blockAngleType[j] - 1;
                    
                    nAngles[i1]++;
                    nAngles[i2]++;
                    nAngles[i3]++;

                    
                }// blockAngleType[j]



                mID[ind] = molecInd;

                // Store the index of the s backbone
                bb_inds[bbIndex] = ind;

                // Increment the particle index
                ind++;
                bbIndex++;

            }// s=0:N[j]

        }// j=0:numBlocks; 



        bbIndex = 0;

        // Now, loop over backbone and attach grafts
        for ( int j=0 ; j<numBlocks ; j++ ) {
            
            for ( int s=0 ; s<Nblocks[j]; s++ ) {
                // 'partner' index for bonding and positioning
                int pind = bb_inds[bbIndex];

                for ( int k=0; k<SC_numBlocks[j]; k++ ) {
                    int speciesVal = findSpeciesInteger(SC_speciesBlocks[j][k]);
                    for ( int t=0 ; t<SC_Nblocks[j][k]; t++ ) {

                        intSpecies[ind] = speciesVal;
                        
                        // if not the first grafted bead, partner is ind-1
                        if ( t > 0 ) { 
                            pind = ind - 1; 
                        }

                        
                        for ( int a=0 ; a<Dim ; a++ ) {
                            int prevXInd = pind*Dim+a;
                            int Xind = ind*Dim+a;

                            x[Xind] = x[prevXInd] + rg[a];

                            if ( x[Xind] > L[a] ) x[Xind] -= L[a];
                            else if ( x[Xind] < 0.0 ) x[Xind] += L[a];

                            v[Xind] = f[Xind] = 0.0;
                        }

                        nBonds[ind] = 0;
                        nAngles[ind] = 0;

                        std::cout << "  t, pind, ind: " << t << " " << pind << " " << ind << " specInd: " << speciesVal << std::endl;
                        
                        // ind always bonded to pind, no chain end effect
                        int nb = nBonds[ind];
                        bondedTo[ind*MAXBONDS+nb] = pind;
                        bondType[ind*MAXBONDS+nb] = SC_blockBondTypes[j][k] - 1;
                        nBonds[ind]++;

                        // if 'pind' is a backbone, add bond to pind
                        if ( pind == bb_inds[bbIndex] ) {
                            nb = nBonds[pind];
                            bondedTo[pind*MAXBONDS+nb] = ind;
                            bondType[pind*MAXBONDS+nb] = SC_blockBondTypes[j][k] - 1;
                            nBonds[pind]++;
                        }
                        
                        // if ( ind not end of graft )
                        if ( k != (SC_numBlocks[j]-1) || t != (SC_Nblocks[j][k]-1 ) ) {
                            nb = nBonds[ind];
                            bondedTo[ind*MAXBONDS+nb] = ind+1;
                            bondType[ind*MAXBONDS+nb] = SC_blockBondTypes[j][k] - 1;
                            nBonds[ind]++;
                        }
                        
                        mID[ind] = molecInd;

                        ind++;                        
                    }// t=0:SC_Nblocks[j][k] loop over N for each side chain block)
                }// k=0:SC_numBlocks (loop over blocks of side chain)

                
                bbIndex++;
            }
        }

        // Increment molecule index
        molecInd++;
    }// i=0:nmolecs

    std::cout << "nstot is " << nstot << " after molecule creation" << std::endl;
}




// Generate a new linear polymer with side-chain LCs
void PS_Box::makeSCLC(std::istringstream& iss ) {
    if ( rho0 < 0.0 ) die("rho0 must be defined before molecules created!");

    if ( MAXBONDS < 3 ) { die("MAXBONDS must be >= 3 for SCLCs!"); }
    if ( MAXANGLES < 9 ) { die("MAXANGLES must be >= 9 for SCLCs!"); }

    int numBlocks, Ntot = 0;

    // Both set to negative values to determine which keyword given
    double phi = -1.0;  
    int nmolecs = -1; 

    // Flag for dealing with charges on this molec type
    int doMolecCharges = 0;

    std::string s1;

    // read either phi or nmolecs keyword
    iss >> s1;

    if ( s1 == "phi" ) {
        iss >> phi;
    }
    else if ( s1 == "nmolecs" ) {
        iss >> nmolecs;
    }
    else { 
        std::cout << "invalid keyword: " << s1 << std::endl;
        die("invalid keyword after molecule linear. must be 'phi' or 'nmolecs'"); 
    }

    // Read in the number of blocks
    iss >> numBlocks; 
    
    std::vector<int> Nblocks(numBlocks);
    std::vector<int> blockBondType(numBlocks,1);
    std::vector<int> blockAngleType(numBlocks,0);
    std::vector<int> drudeFlag(numBlocks,0);
    std::vector<int> chargeFlag(numBlocks,0);
    std::vector<std::string> speciesBlocks(numBlocks);
    std::vector<int> intSpeciesBlocks(numBlocks);
    std::vector<float> blockCharge(numBlocks, 0.0);

    
    // Storage for side chain graft polymers
    std::vector<int> SC_numBlocks(numBlocks);
    std::vector<std::string> SC_style(numBlocks);       // "linear" or (eventually) "sclc"
    // std::vector<std::string> SC_attachStyle(numBlocks);

    std::vector<int> SC_Nspace(numBlocks);
    std::vector<int> SC_spaceBondType(numBlocks);
    std::vector<int> SC_spaceAngleType(numBlocks);
    std::vector<std::string> SC_spaceSpecies(numBlocks);
    std::vector<int> SC_intSpaceSpecies(numBlocks);
    
    std::vector<int> SC_NLC(numBlocks);
    std::vector<int> SC_LCBondType(numBlocks);
    std::vector<int> SC_LCAngleType(numBlocks);
    std::vector<std::string> SC_LCSpecies(numBlocks);
    
    std::vector<int> SC_hingeAngleType(numBlocks);


    std::string lcFileName;
    std::string lcStyle;
    std::vector<int> lcCenters;
    std::vector<int> lcPartner;
    int lc_counter;
    int lc_file_flag = 0;

    
    std::vector<float> Rmin(Dim,0.0);
    std::vector<float> Rmax(Dim);
    for ( int j=0 ; j<Dim ; j++ ) Rmax[j] = L[j];

    // Store the basic block info
    for (int j=0 ; j<numBlocks; j++ ) {
        iss >> Nblocks[j];
        iss >> speciesBlocks[j];

        Ntot += Nblocks[j];
    }

    // Check for optional arguments
    while ( iss.tellg() != -1 ) {

        iss >> s1;
        if ( s1 == "drude" ) {
            die("drude oscillators not implemented!");
        }

        else if ( s1 == "charge" ) {
            if ( this->doCharges == 0 ) {
                this->enableCharges();
            }

            doMolecCharges = 1;
            
            for ( int j=0 ; j<numBlocks ; j++ ) {
                float q1;
                iss >> q1;
                blockCharge[j] = q1;
            }
        }// s1==charge

        else if ( s1 == "bondType" || s1 == "bondtype" ) {
            for ( int j=0 ; j<numBlocks ; j++ ) {
                int t1; 
                iss >> t1;
                blockBondType[j] = t1;
            }
            // make sure to decide on how to handle junction cases, document it
        }
        else if ( s1 == "angleType" || s1 == "angletype" ) {
            for ( int j=0 ; j<numBlocks ; j++ ) {
                int t1;
                iss >> t1;
                blockAngleType[j] = t1;
                std::cout << "  block ang type: " << j << " " << blockAngleType[j] << std::endl;
            }
        }
        else if ( s1 == "xrange" ) {
            iss >> Rmin[0];
            iss >> Rmax[0];
        }
        else if ( s1 == "yrange" ) {
            iss >> Rmin[1];
            iss >> Rmax[1];
        }
        else if ( s1 == "zrange" ) {
            if ( Dim != 3 ) { die("z-range defined for a non-3D simulation!"); }
            iss >> Rmin[2];
            iss >> Rmax[2];
        }



        else if ( s1 == "grafted" ) {

            for ( int m=0 ; m<numBlocks ; m++ ) {
                iss >> SC_style[m] ;

                if ( SC_style[m] == "end-on" || SC_style[m] == "side-on" ) {
                    std::string s2;
                    iss >> s2;
                    if ( s2 != "spacer" ) { die("wrong argument order"); }
                    iss >> SC_Nspace[m];
                    iss >> SC_spaceSpecies[m];
                    iss >> SC_spaceBondType[m];
                    iss >> SC_spaceAngleType[m];

                    iss >> s2;
                    if ( s2 != "mesogen" ) { die("wrong argument order"); }
                    iss >> SC_NLC[m];
                    if ( SC_NLC[m] != 3 ) { 
                        die("PS_Box::makeSCLC hard-coded to require side-chain mesogens to have length 3"); 
                    }
                    iss >> SC_LCSpecies[m];
                    iss >> SC_LCBondType[m];
                    iss >> SC_LCAngleType[m];

                    iss >> s2;
                    if ( s2 != "hingeAngleType" ) { die("wrong argument order"); }

                    iss >> SC_hingeAngleType[m];
                
                }// end-on or side-on grafting

                else if ( SC_style[m] == "none" ) {
                    SC_NLC[m] = SC_Nspace[m] = 0;
                }// no grafting on this block

                else { 
                    die("PS_Box_moleculeMaker:makeGrafted: INVALID SCLC TYPE, only end-on or side-on grafts supported!");
                }

            }// m=0:numBlocks
        }// s1==grafted

        else if ( s1 == "make-lc-input" ) {
            lc_file_flag = 1;
            iss >> lcFileName;
            iss >> lcStyle;

            if ( lcStyle != "all" && lcStyle != "middle" ) {
                die("invalid lc style!");
            }
        }

    } // while (!end of iss)


    // Backbone length is current value of Ntot
    int Nbb = Ntot;
      
    for ( int m=0 ; m<numBlocks ; m++ ) {
        Ntot += Nblocks[m] * (SC_Nspace[m] + SC_NLC[m]);
    }
    
    

    // Compute number of molecules of this type to add
    // if volume fraction was read
    if ( phi > 0.0 ) { nmolecs = int( rho0 * V * phi / float(Ntot) ); }

    std::cout << "Generating " << nmolecs << " molecules each with " << Ntot << " sites" << std::endl;

    // particle index to be incremented as particles added
    int ind = nstot;

    int prev_nstot = nstot;
    

    // Update number of sites in the box
    nstot += nmolecs * Ntot;
    allocHostParticleArrays(nstot);
    std::cout << "nstot changed values to: " << nstot << ", starting index: " << ind << std::endl;



    // Initialize lc-input variables
    lcCenters.resize(nstot);
    lcPartner.resize(nstot);
    lc_counter = 0;


    // Find starting molecule ID
    int molecInd = -1;
    for ( int i=0 ; i<nstot ; i++ ) {
        if (mID[i] > molecInd) molecInd = mID[i];
    }
    if ( molecInd < 0 ) molecInd = 0;


    std::cout << "   MAX ANGLE INDEX: " << 3 * nstot * MAXANGLES << std::endl;
    std::cout << "   MAX BOND INDEX: " << nstot * MAXBONDS << std::endl;


    // Zero angle counters
    for ( int i=prev_nstot; i<nstot; i++ ) {
        nAngles[i] = 0;
    }


    // Storage for indices of the backbone sites
    std::vector<int> bb_inds(Nbb,0);


    ///////////////////////////////////////////////////////////
    // Main loop over molecules, blocks, sites on each block //
    ///////////////////////////////////////////////////////////
    for ( int i=0 ; i<nmolecs ; i++ ) {

        // backbone monomer index for molecule i
        int bbIndex = 0;
        
        double ru[3], rg[3];

        random_unit_vec(ru, Dim);   // Used for main backbone
        random_unit_vec(rg, Dim);   // used for graft after removing component along ru
        
        // Dot ru with rg
        double rug_dot = 0.0;
        for ( int j=0 ; j<Dim ; j++ ) { rug_dot += ru[j]*rg[j]; }
        
        // Remove magnitude of ru * rug_dot
        double rg2 = 0.0;
        for ( int j=0 ; j<Dim ; j++ ) { 
            rg[j] = rg[j] - rug_dot * ru[j]; 
            rg2 += rg[j] * rg[j];
        }

        // re-normalize rg
        double mag_rg = sqrt(rg2);
        for ( int j=0 ; j<Dim ; j++ ) { rg[j] *= 1.0 / mag_rg ; }


        for ( int j=0 ; j<numBlocks; j++ ) {
            int speciesVal = findSpeciesInteger(speciesBlocks[j]);
            
            for ( int s=0 ; s<Nblocks[j]; s++ ) {

                // Track species info
                intSpecies[ind] = speciesVal;
                
                
                // Is this a chain end? 
                // If so, place randomly in the box
                if ( j==0 && s==0 ) {
                    for ( int a=0 ; a<Dim ; a++ ) {
                        x[ind*Dim+a] = ran2() * (Rmax[a] - Rmin[a]) + Rmin[a];
                    }
                }

                // Not a chain end: place monomer a unit vector away
                // from previous monomer
                else {
                    
                    for ( int a=0 ; a<Dim ; a++ ) {
                        int prevXInd = (ind-1)*Dim+a;
                        int Xind = ind*Dim+a;

                        x[Xind] = x[prevXInd] + ru[a];

                        if ( x[Xind] > L[a] ) x[Xind] -= L[a];
                        else if ( x[Xind] < 0.0 ) x[Xind] += L[a];
                    }
                }

                // Initialize velocities, forces to 0.0
                for ( int a=0 ; a<Dim ; a++ ) {
                    v[ind*Dim+a] = f[ind*Dim+a] = 0.0;
                }


                if ( doMolecCharges ) {
                    charges[ind] = blockCharge[j];
                }

                nBonds[ind] = 0;
                
                // Initialize bonds
                // If not the first monomer on a chain, make a bond to previous monomer
                if ( j != 0 || s != 0 ) {
                    int nb = nBonds[ind];
                    
                    bondedTo[ind*MAXBONDS+nb] = ind-1;
                    bondType[ind*MAXBONDS+nb] = blockBondType[j] - 1;
                    nBonds[ind]++;
                }

                // if not the last monomer on a chain, make a bond to next monomer
                if ( j != (numBlocks-1) || s != (Nblocks[j]-1 ) ) {
                    int nb = nBonds[ind];
                    
                    bondedTo[ind*MAXBONDS+nb] = ind+1;
                    bondType[ind*MAXBONDS+nb] = blockBondType[j] - 1;
                    nBonds[ind]++;
                }

        // std::cout << "bonds FINISHED for " << ind << "!!" << std::endl;

                // Initialize angles
                // logic is different from bonds. If *central* particle is
                // in an angle, then i1 and i3 angle lists are both updated.
                // This ensures angles accounted for when one site is on a
                // different block that may have different angleTypes (or none)
                if ( ( blockAngleType[j] != 0 ) &&                  // this block has angles
                     ( j > 0 || s > 0 ) &&                          // not the first monomer
                     ( j < numBlocks-1 || s < Nblocks[j]-1 ) ) {    // not the final monomer

                    // std::cout << "j: " << j << " s: " << s << " ind: " << ind << std::endl;
                    int i1 = ind-1;
                    int i2 = ind;
                    int i3 = ind+1;

                    // std::cout << "  ivals: " << i1 << " " << i2 << " " << i3 << std::endl;

                    int n1 = nAngles[i1];
                    int n2 = nAngles[i2];
                    int n3 = nAngles[i3];

                    // std::cout << "  n vals: " << n1 << " " << n2 << " " << n3 << std::endl;

                    int index1 = i1*MAXANGLES*3+3*n1;
                    int index2 = i2*MAXANGLES*3+3*n2;
                    int index3 = i3*MAXANGLES*3+3*n3;

                    // std::cout << "  index: " << index1 << " " << index2 << " " << index3 << std::endl;

                    
                    angleGroup[index1+0] = i1;
                    angleGroup[index1+1] = i2;
                    angleGroup[index1+2] = i3;

                    angleGroup[index2+0] = i1;
                    angleGroup[index2+1] = i2;
                    angleGroup[index2+2] = i3;

                    angleGroup[index3+0] = i1;
                    angleGroup[index3+1] = i2;
                    angleGroup[index3+2] = i3;

                    angleType[i1*MAXANGLES+n1] = blockAngleType[j] - 1;
                    angleType[i2*MAXANGLES+n2] = blockAngleType[j] - 1;
                    angleType[i3*MAXANGLES+n3] = blockAngleType[j] - 1;
                    
                    nAngles[i1]++;
                    nAngles[i2]++;
                    nAngles[i3]++;

                    if ( nAngles[i1] >= MAXANGLES || nAngles[i2] >= MAXANGLES || nAngles[i3] >= MAXANGLES ) { die("too many angles on a particle, increase MAXANGLES nd try again"); }

                    
                }// blockAngleType[j]

        // std::cout << "angles FINISHED for " << ind << "!!" << std::endl;

                mID[ind] = molecInd;

                // Store the index of the s backbone
                bb_inds[bbIndex] = ind;

                // Increment the particle index
                ind++;
                bbIndex++;

            }// s=0:N[j]

        }// j=0:numBlocks; 

        ///////////////////////
        // END OF BACKBONE   //
        ///////////////////////

        // std::cout << "BACKBONE FINISHED!!" << std::endl;







        bbIndex = 0;

        // Now, loop over backbone and attach grafts
        for ( int j=0 ; j<numBlocks ; j++ ) {
            
            if ( SC_style[j] == "none" ) {
                bbIndex += Nblocks[j];
                continue;
            }

            for ( int s=0 ; s<Nblocks[j]; s++ ) {
                
                // 'partner' index for bonding and positioning
                int pind = bb_inds[bbIndex];

                // Place the spacer particles
                for ( int t=0 ; t<SC_Nspace[j]; t++ ) {
                    int speciesVal = findSpeciesInteger(SC_spaceSpecies[j]);
                
                    intSpecies[ind] = speciesVal;
                    
                    // if not the first grafted bead, partner is ind-1
                    if ( t > 0 ) { 
                        pind = ind - 1; 
                    }

                    
                    for ( int a=0 ; a<Dim ; a++ ) {
                        int prevXInd = pind*Dim+a;
                        int Xind = ind*Dim+a;

                        x[Xind] = x[prevXInd] + rg[a];

                        if ( x[Xind] > L[a] ) x[Xind] -= L[a];
                        else if ( x[Xind] < 0.0 ) x[Xind] += L[a];

                        v[Xind] = f[Xind] = 0.0;
                    }

                    nBonds[ind] = 0;
                    // nAngles[ind] = 0;

                    // std::cout << "  t, pind, ind: " << t << " " << pind << " " << ind << " specInd: " << speciesVal << std::endl;
                    
                    // ind always bonded to pind, no chain end effect
                    int nb = nBonds[ind];
                    bondedTo[ind*MAXBONDS+nb] = pind;
                    bondType[ind*MAXBONDS+nb] = SC_spaceBondType[j] - 1;
                    nBonds[ind]++;

                    // if 'pind' is a backbone, add bond to pind
                    if ( pind == bb_inds[bbIndex] ) {
                        nb = nBonds[pind];
                        bondedTo[pind*MAXBONDS+nb] = ind;
                        bondType[pind*MAXBONDS+nb] = SC_spaceBondType[j] - 1;
                        nBonds[pind]++;
                    }

                    // Bonds 'ind' to ind+1 for each spacer bead
                    // First bead of the mesogen will be 'ind+1'
                    nb = nBonds[ind];
                    bondedTo[ind*MAXBONDS+nb] = ind+1;
                    bondType[ind*MAXBONDS+nb] = SC_spaceBondType[j] - 1;
                    nBonds[ind]++;


                    // Set up angles along the spacer section
                    int i1 = pind;
                    int i2 = ind;
                    int i3 = ind+1;

                    // std::cout << "  t=" << t << "  ivals: " << i1 << " " << i2 << " " << i3 << std::endl;

                    int n1 = nAngles[i1];
                    int n2 = nAngles[i2];
                    int n3 = nAngles[i3];

                    // std::cout << "  n vals: " << n1 << " " << n2 << " " << n3 << std::endl;

                    int index1 = i1*MAXANGLES*3+3*n1;
                    int index2 = i2*MAXANGLES*3+3*n2;
                    int index3 = i3*MAXANGLES*3+3*n3;

                    // std::cout << "  index: " << index1 << " " << index2 << " " << index3 << std::endl;
                    
                    angleGroup[index1+0] = i1;
                    angleGroup[index1+1] = i2;
                    angleGroup[index1+2] = i3;

                    angleGroup[index2+0] = i1;
                    angleGroup[index2+1] = i2;
                    angleGroup[index2+2] = i3;

                    angleGroup[index3+0] = i1;
                    angleGroup[index3+1] = i2;
                    angleGroup[index3+2] = i3;

                    angleType[i1*MAXANGLES+n1] = SC_spaceAngleType[j] - 1;
                    angleType[i2*MAXANGLES+n2] = SC_spaceAngleType[j] - 1;
                    angleType[i3*MAXANGLES+n3] = SC_spaceAngleType[j] - 1;
                    
                    nAngles[i1]++;
                    nAngles[i2]++;
                    nAngles[i3]++;

                    if ( nAngles[i1] >= MAXANGLES || nAngles[i2] >= MAXANGLES || nAngles[i3] >= MAXANGLES ) { die("too many angles on a particle, increase MAXANGLES nd try again"); }

                    
                    // First spacer monomer adds hinge angles
                    if ( t == 0 ) {

                        // If not the starting chain end, angle between
                        // bbIndex-1, bbIndex, and ind
                        if ( j != 0 || s != 0 ) {
                            // Set up angles along the spacer section
                            int i1 = bb_inds[bbIndex]-1;
                            int i2 = bb_inds[bbIndex];
                            int i3 = ind;

                            // std::cout << "  ivals: " << i1 << " " << i2 << " " << i3 << std::endl;

                            int n1 = nAngles[i1];
                            int n2 = nAngles[i2];
                            int n3 = nAngles[i3];

                            // std::cout << "  n vals: " << n1 << " " << n2 << " " << n3 << std::endl;

                            int index1 = i1*MAXANGLES*3+3*n1;
                            int index2 = i2*MAXANGLES*3+3*n2;
                            int index3 = i3*MAXANGLES*3+3*n3;

                            // std::cout << "  index: " << index1 << " " << index2 << " " << index3 << std::endl;
                            
                            angleGroup[index1+0] = i1;
                            angleGroup[index1+1] = i2;
                            angleGroup[index1+2] = i3;

                            angleGroup[index2+0] = i1;
                            angleGroup[index2+1] = i2;
                            angleGroup[index2+2] = i3;

                            angleGroup[index3+0] = i1;
                            angleGroup[index3+1] = i2;
                            angleGroup[index3+2] = i3;

                            angleType[i1*MAXANGLES+n1] = SC_hingeAngleType[j] - 1;
                            angleType[i2*MAXANGLES+n2] = SC_hingeAngleType[j] - 1;
                            angleType[i3*MAXANGLES+n3] = SC_hingeAngleType[j] - 1;
                            
                            nAngles[i1]++;
                            nAngles[i2]++;
                            nAngles[i3]++;
                            if ( nAngles[i1] >= MAXANGLES || nAngles[i2] >= MAXANGLES || nAngles[i3] >= MAXANGLES ) { die("too many angles on a particle, increase MAXANGLES nd try again"); }
                        }

                        // If not the last monomer on the last backbone block
                        if ( j < numBlocks-1 || s < Nblocks[j]-1 ) {
                            // Set up angles along the spacer section
                            int i1 = bb_inds[bbIndex]+1;
                            int i2 = bb_inds[bbIndex];
                            int i3 = ind;

                            // std::cout << "  ivals: " << i1 << " " << i2 << " " << i3 << std::endl;

                            int n1 = nAngles[i1];
                            int n2 = nAngles[i2];
                            int n3 = nAngles[i3];

                            // std::cout << "  n vals: " << n1 << " " << n2 << " " << n3 << std::endl;

                            int index1 = i1*MAXANGLES*3+3*n1;
                            int index2 = i2*MAXANGLES*3+3*n2;
                            int index3 = i3*MAXANGLES*3+3*n3;

                            // std::cout << "  index: " << index1 << " " << index2 << " " << index3 << std::endl;
                            
                            angleGroup[index1+0] = i1;
                            angleGroup[index1+1] = i2;
                            angleGroup[index1+2] = i3;

                            angleGroup[index2+0] = i1;
                            angleGroup[index2+1] = i2;
                            angleGroup[index2+2] = i3;

                            angleGroup[index3+0] = i1;
                            angleGroup[index3+1] = i2;
                            angleGroup[index3+2] = i3;

                            angleType[i1*MAXANGLES+n1] = SC_hingeAngleType[j] - 1;
                            angleType[i2*MAXANGLES+n2] = SC_hingeAngleType[j] - 1;
                            angleType[i3*MAXANGLES+n3] = SC_hingeAngleType[j] - 1;
                            
                            nAngles[i1]++;
                            nAngles[i2]++;
                            nAngles[i3]++;

                            if ( nAngles[i1] >= MAXANGLES || nAngles[i2] >= MAXANGLES || nAngles[i3] >= MAXANGLES ) { die("too many angles on a particle, increase MAXANGLES nd try again"); }


                        }

                    }




                    
                    mID[ind] = molecInd;

                    ind++;                        
                }// t=0:SC_Nspace[j] loop over spacer beads





                // Place mesogen beads. Entering this section, 'ind' is the center of the mesogen
                int speciesVal = findSpeciesInteger(SC_LCSpecies[j]);

                intSpecies[ind] = intSpecies[ind+1] = intSpecies[ind+2] = speciesVal;


                nBonds[ind] = nBonds[ind+1] = nBonds[ind+2] = 0;
                // nAngles[ind] = nAngles[ind+1] = nAngles[ind+2] = 0;

                pind = ind - 1;

                // Positions of mesogen beads
                for ( int a=0 ; a<Dim ; a++ ) {
                    int prevXInd = pind*Dim+a;
                    int Xind = ind*Dim+a;

                    // bead at end of mesogen
                    int X2ind = (ind+1)*Dim + a;    // Forward mesogen bead
                    int X3ind = (ind+2)*Dim + a;    // backward mesogen bead

                    x[Xind] = x[prevXInd] + rg[a];

                    if ( x[Xind] > L[a] ) x[Xind] -= L[a];
                    else if ( x[Xind] < 0.0 ) x[Xind] += L[a];

                    v[Xind] = f[Xind] = 0.0;
                    

                    if ( SC_style[j] == "side-on" ) {
                        x[X2ind] = x[Xind] + ru[a];
                        x[X3ind] = x[Xind] - ru[a];
                    }
                    else if ( SC_style[j] == "end-on" ) {
                        x[X2ind] = x[Xind] + rg[a];
                        x[X3ind] = x[X2ind] + rg[a];
                    }
                    
                    if ( x[X2ind] > L[a] ) x[X2ind] -= L[a];
                    else if ( x[X2ind] < 0.0 ) x[X2ind] += L[a];
                    v[X2ind] = f[X2ind] = 0.0;

                    
                    if ( x[X3ind] > L[a] ) x[X3ind] -= L[a];
                    else if ( x[X3ind] < 0.0 ) x[X3ind] += L[a];
                    v[X3ind] = f[X3ind] = 0.0;

                }

                nBonds[ind]  = nBonds[ind+1]  = nBonds[ind+2] = 0;
                // nAngles[ind] = nAngles[ind+1] = nAngles[ind+2] = 0;


                // Bond of central mesogen to spacer//backbone
                int nb = nBonds[ind];
                bondedTo[ind*MAXBONDS+nb] = pind;
                bondType[ind*MAXBONDS+nb] = SC_LCBondType[j] - 1;
                nBonds[ind]++;


                // if 'pind' is a backbone, add bond to pind
                if ( pind == bb_inds[bbIndex] ) {
                    nb = nBonds[pind];
                    bondedTo[pind*MAXBONDS+nb] = ind;
                    bondType[pind*MAXBONDS+nb] = SC_LCBondType[j] - 1;
                    nBonds[pind]++;
                }

                // central mesogen to ind+1 end
                nb = nBonds[ind];
                bondedTo[ind*MAXBONDS+nb] = ind+1;
                bondType[ind*MAXBONDS+nb] = SC_LCBondType[j] - 1;
                nBonds[ind]++;

                if ( SC_style[j] == "side-on" ) {
                    // central mesogen to ind+2 end
                    nb = nBonds[ind];
                    bondedTo[ind*MAXBONDS+nb] = ind+2;
                    bondType[ind*MAXBONDS+nb] = SC_LCBondType[j] - 1;
                    nBonds[ind]++;

                    // ind+1 end bonded to center
                    nb = nBonds[ind+1];
                    bondedTo[(ind+1)*MAXBONDS+nb] = ind;
                    bondType[(ind+1)*MAXBONDS+nb] = SC_LCBondType[j] - 1;
                    nBonds[ind+1]++;
                
                    // ind+2 end bonded to center
                    nb = nBonds[ind+2];
                    bondedTo[(ind+2)*MAXBONDS+nb] = ind;
                    bondType[(ind+2)*MAXBONDS+nb] = SC_LCBondType[j] - 1;
                    nBonds[ind+2]++;


                    // side-on 'hinges' 
                    int i1 = ind-1;
                    int i2 = ind;
                    int i3 = ind+1;

                    int n1 = nAngles[i1];
                    int n2 = nAngles[i2];
                    int n3 = nAngles[i3];

                    int index1 = i1*MAXANGLES*3+3*n1;
                    int index2 = i2*MAXANGLES*3+3*n2;
                    int index3 = i3*MAXANGLES*3+3*n3;

                    angleGroup[index1+0] = i1;
                    angleGroup[index1+1] = i2;
                    angleGroup[index1+2] = i3;

                    angleGroup[index2+0] = i1;
                    angleGroup[index2+1] = i2;
                    angleGroup[index2+2] = i3;

                    angleGroup[index3+0] = i1;
                    angleGroup[index3+1] = i2;
                    angleGroup[index3+2] = i3;

                    angleType[i1*MAXANGLES+n1] = SC_hingeAngleType[j] - 1;
                    angleType[i2*MAXANGLES+n2] = SC_hingeAngleType[j] - 1;
                    angleType[i3*MAXANGLES+n3] = SC_hingeAngleType[j] - 1;
                    
                    nAngles[i1]++;
                    nAngles[i2]++;
                    nAngles[i3]++;


                    i1 = ind-1;
                    i2 = ind;
                    i3 = ind+2;

                    n1 = nAngles[i1];
                    n2 = nAngles[i2];
                    n3 = nAngles[i3];

                    index1 = i1*MAXANGLES*3+3*n1;
                    index2 = i2*MAXANGLES*3+3*n2;
                    index3 = i3*MAXANGLES*3+3*n3;

                    angleGroup[index1+0] = i1;
                    angleGroup[index1+1] = i2;
                    angleGroup[index1+2] = i3;

                    angleGroup[index2+0] = i1;
                    angleGroup[index2+1] = i2;
                    angleGroup[index2+2] = i3;

                    angleGroup[index3+0] = i1;
                    angleGroup[index3+1] = i2;
                    angleGroup[index3+2] = i3;

                    angleType[i1*MAXANGLES+n1] = SC_hingeAngleType[j] - 1;
                    angleType[i2*MAXANGLES+n2] = SC_hingeAngleType[j] - 1;
                    angleType[i3*MAXANGLES+n3] = SC_hingeAngleType[j] - 1;
                    
                    nAngles[i1]++;
                    nAngles[i2]++;
                    nAngles[i3]++;
                    if ( nAngles[i1] >= MAXANGLES || nAngles[i2] >= MAXANGLES || nAngles[i3] >= MAXANGLES ) { die("too many angles on a particle, increase MAXANGLES nd try again"); }



                }// side-on bonding

                else if ( SC_style[j] == "end-on" ) {
                    // ind+1 bonded to ind
                    nb = nBonds[ind+1];
                    bondedTo[(ind+1)*MAXBONDS+nb] = ind;
                    bondType[(ind+1)*MAXBONDS+nb] = SC_LCBondType[j] - 1;
                    nBonds[ind+1]++;

                    // ind+1 bonded to ind+2
                    nb = nBonds[ind+1];
                    bondedTo[(ind+1)*MAXBONDS+nb] = ind+2;
                    bondType[(ind+1)*MAXBONDS+nb] = SC_LCBondType[j] - 1;
                    nBonds[ind+1]++;

                    // ind+2 bonded to ind+1
                    nb = nBonds[ind+2];
                    bondedTo[(ind+2)*MAXBONDS+nb] = ind+1;
                    bondType[(ind+2)*MAXBONDS+nb] = SC_LCBondType[j] - 1;
                    nBonds[ind+2]++;


                    // end-on 'hinge' is just the spacer potential
                    int i1 = ind-1;
                    int i2 = ind;
                    int i3 = ind+1;

                    int n1 = nAngles[i1];
                    int n2 = nAngles[i2];
                    int n3 = nAngles[i3];

                    int index1 = i1*MAXANGLES*3+3*n1;
                    int index2 = i2*MAXANGLES*3+3*n2;
                    int index3 = i3*MAXANGLES*3+3*n3;

                    angleGroup[index1+0] = i1;
                    angleGroup[index1+1] = i2;
                    angleGroup[index1+2] = i3;

                    angleGroup[index2+0] = i1;
                    angleGroup[index2+1] = i2;
                    angleGroup[index2+2] = i3;

                    angleGroup[index3+0] = i1;
                    angleGroup[index3+1] = i2;
                    angleGroup[index3+2] = i3;

                    angleType[i1*MAXANGLES+n1] = SC_spaceAngleType[j] - 1;
                    angleType[i2*MAXANGLES+n2] = SC_spaceAngleType[j] - 1;
                    angleType[i3*MAXANGLES+n3] = SC_spaceAngleType[j] - 1;
                    
                    nAngles[i1]++;
                    nAngles[i2]++;
                    nAngles[i3]++;
                    if ( nAngles[i1] >= MAXANGLES || nAngles[i2] >= MAXANGLES || nAngles[i3] >= MAXANGLES ) { die("too many angles on a particle, increase MAXANGLES nd try again"); }
                }

                // Set order for side-on mesogen
                int i1 = ind+1;
                int i2 = ind;
                int i3 = ind+2;

                // Reset variables if end-on
                if ( SC_style[j] == "end-on" ) {
                    i1 = ind;
                    i2 = ind+1;
                    i3 = ind+2;
                }

                // std::cout << "  ivals: " << i1 << " " << i2 << " " << i3 << std::endl;

                int n1 = nAngles[i1];
                int n2 = nAngles[i2];
                int n3 = nAngles[i3];

                // std::cout << "  n vals: " << n1 << " " << n2 << " " << n3 << std::endl;

                int index1 = i1*MAXANGLES*3+3*n1;
                int index2 = i2*MAXANGLES*3+3*n2;
                int index3 = i3*MAXANGLES*3+3*n3;

                // std::cout << "  index: " << index1 << " " << index2 << " " << index3 << std::endl;
                
                angleGroup[index1+0] = i1;
                angleGroup[index1+1] = i2;
                angleGroup[index1+2] = i3;

                angleGroup[index2+0] = i1;
                angleGroup[index2+1] = i2;
                angleGroup[index2+2] = i3;

                angleGroup[index3+0] = i1;
                angleGroup[index3+1] = i2;
                angleGroup[index3+2] = i3;

                angleType[i1*MAXANGLES+n1] = SC_LCAngleType[j] - 1;
                angleType[i2*MAXANGLES+n2] = SC_LCAngleType[j] - 1;
                angleType[i3*MAXANGLES+n3] = SC_LCAngleType[j] - 1;
                
                nAngles[i1]++;
                nAngles[i2]++;
                nAngles[i3]++;
                
                if ( nAngles[i1] >= MAXANGLES || nAngles[i2] >= MAXANGLES || nAngles[i3] >= MAXANGLES ) { die("too many angles on a particle, increase MAXANGLES nd try again"); }


                if ( lc_file_flag ) {
                    if ( SC_style[j] == "side-on" ) {
                        lcCenters[lc_counter] = ind;
                        lcPartner[lc_counter] = ind+1;
                        lc_counter++;

                        if ( lcStyle == "all" ) {
                            lcCenters[lc_counter] = ind+1;
                            lcPartner[lc_counter] = ind;
                            lc_counter++;

                            lcCenters[lc_counter] = ind+2;
                            lcPartner[lc_counter] = ind;
                            lc_counter++;
                        }
                    }

                    else if ( SC_style[j] == "end-on" ) {
                        lcCenters[lc_counter] = ind+1;
                        lcPartner[lc_counter] = ind+2;
                        lc_counter++;

                        if ( lcStyle == "all" ) {
                            lcCenters[lc_counter] = ind;
                            lcPartner[lc_counter] = ind+1;
                            lc_counter++;

                            lcCenters[lc_counter] = ind+2;
                            lcPartner[lc_counter] = ind+1;
                            lc_counter++;
                        }
                    }
                }//lc_file_flag

                
                mID[ind] = mID[ind+1] = mID[ind+2] = molecInd;
                
                ind += 3;

                // Move on to next backbone bead
                bbIndex++;

            }//s=0:Nblock[j]

        }//j=0:numBlocks

        // Increment molecule index
        molecInd++;
    }// i=0:nmolecs

    if ( lc_file_flag ) make_lc_file(lcFileName, lc_counter, lcCenters, lcPartner);

    std::cout << "nstot is " << nstot << " after molecule creation" << std::endl;
}

void make_lc_file(
    std::string name,           // file name
    int nlc,                    // number of lc_pairs
    std::vector<int> centers,   // lc centers list
    std::vector<int> partner   // lc partner list
) {
    std::ofstream out(name);
    out << nlc << "\n";
    for ( int i=0 ; i<nlc; i++ ) {
        out << i+1 << " " << centers[i]+1 << " " << partner[i]+1 << "\n";
    }
    out.close();
}