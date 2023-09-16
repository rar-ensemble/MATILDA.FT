// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"

float integ_trapPBC(float* dat) {
	float sum = 0.f;

	for (int i = 0; i < M; i++)
		sum += dat[i];

	for (int i = 0; i < Dim; i++)
		sum *= dx[i];

	return sum;
}