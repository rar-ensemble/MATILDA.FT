# MATILDA.FT


README

This is documentation for MATILDA.FT, Theorethically Informed Langevin Dynamics and Field-Theorethic simulations software. More detailed documentation can be found in the doc sub-directory.

Copyright (c) 2023 University of Pennsylvania

MATILDA.FT is open-source code, and distributed 
Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

----------------------------------------------------------------------

Quickstart guide:
edit src/makefile so that the variable CUDA_LOC points to the installation of CUDA version 11.2 or higher. matilda.ft should then compile with a simple 'make' command from within the src folder, and the executable will be matilda.ft in this folder. 

Further details can be found in doc/MATILDA_FT_documentation.pdf, and example input files can be found in the subdirectories of examples/.

----------------------------------------------------------------------

The following files and directories are part of MATILDA.FT:

README                     
LICENSE                    
doc                        
examples                   
include
src                        
utils   

Directory details
doc: contains the documentation of the code
examples: example systems that can be directly run with the compiled code
src: all source and header files
utils: some post-processing utilities.
include: has the eigen-3.4.0 library dependency
