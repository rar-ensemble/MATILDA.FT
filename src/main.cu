// Copyright (c) 2024 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#define MAIN
#include "timing.h"
#include "random.h"
#include <vector>
#include <fstream>
#include "Box.h"
#include "quotes.h"
#include<random>

void die(const char*);
#ifndef MAIN
extern
#endif
std::vector<Box*> box;

Box* BoxFactory(std::istringstream&);


int main(int argc, char** argv)
{
	int nBoxes = 0;

	int devCount, curDevice;
	cudaGetDevice(&curDevice);
	cudaGetDeviceCount(&devCount);


	// Store start time
	main_t_in = int(time(0));
	init_t_in = main_t_in;

	// Default input file name "input"
	std::string input_file = "input";




	/////////////////////
	// PARSE ARGUMENTS //
	/////////////////////

	// Convert arguments to strings
	std::vector<std::string> string_vec;
	std::cout << std::flush;
	for (int i = 0; i < argc; i++)
	{
		std::string arg = argv[i];
		string_vec.push_back(arg);
	}

	int argIndex = 1;
	while ( argIndex < string_vec.size() ) {
		// non-default input file flag
		if ( string_vec[argIndex] == "-in" ) {
			input_file = string_vec[argIndex+1];
			argIndex += 2;
			std::cout << "Operating from input file " << input_file << std::endl;
		}

		// manual GPU selection
		else if ( string_vec[argIndex] == "-device" ) {
			int deviceIndex = stoi( string_vec[argIndex+1] );
			argIndex += 2;
			if ( deviceIndex >= devCount ) {
				std::string LW = "Device " + std::to_string(deviceIndex) + " not found; only detecting " + \
					std::to_string(devCount) + " devices";
				die(LW);
			}
			cudaSetDevice(deviceIndex);
			cudaGetDevice(&curDevice);
			if ( curDevice != deviceIndex) { die("Failed to set CUDA device"); }
			else { std::cout << "Device manually set to " << curDevice << std::endl; }

		}

		
		else {
			std::string LW = string_vec[argIndex] + " is not a valid argument";
			die(LW);
		}
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
		iss >> word;

		if ( word == "box" ) {
			box.push_back(BoxFactory(iss));
			box.back()->readInput(in2);
			nBoxes++;
		}


		else if ( word == "run" ) {
			iss >> word;
			if ( word == "nvt" ) {
				int runSteps;
				iss >> runSteps;
			
				box.at(0)->NVT(runSteps);
			}

			else if ( word == "vt" ) {
				if ( box.at(0)->returnBoxStyle() != "fts" ) {
					die("invalid box style trying to run const 'vt' simulation");
				}
				int runSteps;
				iss >> runSteps;

				box.at(0)->NVT(runSteps);
			}


			else {  
				std::string lastWords = word + " is an invalid global command!";
				die(lastWords);
			}
		}// first word == "run"

		
		else if ( word == "modify" ) {
			box.at(0)->modifyBox(iss);
		}

	} // while (!in2.eof())

	std::cout << "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n" << giveQuote() << std::endl << std::endl;

	return 0;
}

std::string giveQuote() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distrib_int(0, quoteDB.size());

	int id = distrib_int(gen);

	return quoteDB[id];
}