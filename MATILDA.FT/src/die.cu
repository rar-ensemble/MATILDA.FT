// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
void die(const char* msg) {
	std::cout << msg << std::endl;
	exit(1);
}

void check_cudaError(const char* tag) {
	cudaReturn = cudaGetLastError();
	if (cudaReturn != cudaSuccess) {
		char cherror[90];
		sprintf(cherror, "Cuda failed with error \"%s\" while %s ran\n", cudaGetErrorString(cudaReturn), tag);
		die(cherror);
	}
}