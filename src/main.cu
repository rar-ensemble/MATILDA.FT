// Copyright (c) 2024 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#define MAIN
#include "timing.h"
#include "random.h"
#include <vector>
#include <fstream>
#include "Box.h"

void die(const char*);
#ifndef MAIN
extern
#endif
std::vector<Box*> box;

Box* BoxFactory(std::istringstream&);

// To be replaced by 'box' run options
void run_fts_sim(void);


int main(int argc, char** argv)
{

	// Store start time
	main_t_in = int(time(0));
	init_t_in = main_t_in;


	// Convert arguments to strings
	std::vector<std::string> string_vec;
	std::cout << std::flush;
	for (int i = 0; i < argc; i++)
	{
		std::string arg = argv[i];
		string_vec.push_back(arg);
	}

	// Default input file name "input"
	std::string input_file = "input";

	// Check for non-default input file name
	if ( string_vec.size() >= 3 && string_vec[1] == "-in" ) {
		input_file = string_vec[2];
	}
	
	
	// open input file
	std::ifstream in2(input_file);


	// storage for file info
	std::string word, line, rname;


	// Loop over all lines in the input file
	while (!in2.eof()) {
		getline(in2, line);

		// Blank or commented line
		if (line.length() == 0 || line.at(0) == '#')
			continue;

		std::istringstream iss(line);
		// Loop over words in line
		while (iss >> word) {
			if( word == "box" ) {
				box.push_back(BoxFactory(iss));
				box.back()->readInput(in2);
			}
		}

		run_fts_sim();

	} // while (!in2.eof())


	return 0;
}
