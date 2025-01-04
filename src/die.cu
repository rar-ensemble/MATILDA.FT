// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "include_libs.h"
std::string giveQuote(void);

void die(const char* msg) {
	std::cout << msg << std::endl;
	std::cout << "\n\nSorry, your job finished prematurely. Maybe these wise words will cheer you up:\n" << \
			giveQuote() << std::endl;
	exit(1);
}

void check_cudaError(const char* tag) {
	cudaError_t cudaReturn = cudaGetLastError();
	if (cudaReturn != cudaSuccess) {
		char cherror[90];
		sprintf(cherror, "Cuda failed with error \"%s\" while %s ran\n", cudaGetErrorString(cudaReturn), tag);
		die(cherror);
	}
}