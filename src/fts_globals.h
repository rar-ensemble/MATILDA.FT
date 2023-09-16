// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).



#ifndef _FTS_GLOBALS
#define _FTS_GLOBALS

#include "fts_type.h"
#include "fts_molecule.h"
#include "fts_molecule_linear.h"

#ifndef MAIN
extern
#endif
std::vector<FTS_Type> Types;

#ifndef MAIN
extern
#endif
std::vector<FTS_Molec*> Molecs;

#ifndef MAIN
extern
#endif
std::string fts_style;  // Either scft or cl

#endif