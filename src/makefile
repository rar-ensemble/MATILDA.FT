# Copyright (c) 2023 University of Pennsylvania
# Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
#
####################################################
# Modify the line below to point to the CUDA library
CUDA_LOC = /usr/local/cuda-12.1


CC = ${CUDA_LOC}/bin/nvcc
# GIT_VERSION := $(shell git describe --abbrev=9 --dirty --always --tags)
# GIT_DIRECTORY := $(shell git rev-parse --show-toplevel)

EIGEN_LOC = ../include/eigen-3.4.0

CFLAGS = -std=c++14 -rdc=true -O3 -Wno-deprecated-gpu-targets -lineinfo
# --default-stream per-thread
CPPFLAGS = -I${EIGEN_LOC} --expt-relaxed-constexpr
# CPPFLAGS = --expt-relaxed-constexpr
#CFLAGS = -std=c++11 -rdc=true -arch=sm_37 -dlto -O3 -I${CUDA_LOC}/include -lineinfo -g 

LIBS = -lm -lcudart -lcufft -L${CUDA_LOC}/lib

       
SRCS = array_utils.cu bonds.cu calc_properties.cu config_utils.cu cuda_random_posits.cu \
	device_array_utils.cu device_bonds.cu device_comm_utils.cu \
	device_grid_utils.cu device_utils.cu  field_component.cu forces.cu \
	initialize.cu integ_utils.cu io_utils.cu main.cu potential.cu \
	potential_fieldphases.cu potential_gaussian.cu pbc_utils.cu read_input.cu \
	potential_erf.cu reduce_utils.cu update_potentials.cu device_GJF_integrator.cu \
	potential_charges.cu potential_gaussian_erf.cu \
	Compute.cu angles.cu device_angles.cu random.cu grid_utils.cu \
	Extraforce.cu die.cu \
	Extraforce_dpd.cu Extraforce_langevin.cu Extraforce_dynamic.cu Extraforce_midpush.cu Extraforce_wall.cu Extraforce_lewis.cu Extraforce_lewis_serial.cu\
	group.cu\
	group_id.cu group_region.cu group_type.cu\
	integrator.cu \
	integrator_VV.cu integrator_EM.cu integrator_GJF.cu\
	Measure.cu Measure_surface_tension.cu\
	nlist.cu \
	nlist_half_distance.cu nlist_bonding.cu \
	Compute_widom.cu Compute_avg_sk.cu Compute_sk.cu Compute_chempot.cu\
	tensor_potential.cu tensor_potential_MaierSaupe.cu device_ms_utils.cu \
	Box.cu FTS_Box.cu fts_species.cu fts_potential.cu \
	fts_potential_helfand.cu fts_potential_flory.cu fts_molecule.cu fts_molecule_linear.cu \
	fft_wrapper.cu run_fts_sim.cu fts_updateSchemes.cu \

SRCS_C = gsd.c
SRCS_CPP = tensor_potential_MaierSaupe_eigenvalue.cpp

OBJFOLDER := objects
OBJS_MAKE  	  = $(addprefix $(OBJFOLDER)/, ${OBJS})
OBJS_MAKE_C   = $(addprefix $(OBJFOLDER)/, ${OBJS_C})
OBJS_MAKE_CPP = $(addprefix $(OBJFOLDER)/, ${OBJS_CPP})

OBJS     = $(SRCS:.cu=.cu.o)
OBJS_C   = $(SRCS_C:.c=.c.o)
OBJS_CPP = $(SRCS_CPP:.cpp=.cpp.o)

$(OBJFOLDER)/%.cu.o: %.cu git-version.h
	@mkdir -p $(dir $@)
	${CC} -g ${CFLAGS} -c $< -o $@

$(OBJFOLDER)/%.cpp.o: %.cpp git-version.h
	@mkdir -p $(dir $@)
	${CC} -g ${CFLAGS} ${CPPFLAGS} -c $< -o $@

$(OBJFOLDER)/%.c.o: %.c git-version.h
	@mkdir -p $(dir $@)
	${CC} -g ${CFLAGS} -c $< -o $@
	 
../matilda.ft: $(OBJS_MAKE) $(OBJS_MAKE_CPP) $(OBJS_MAKE_C)
	$(CC) $(CFLAGS) -o $@ $(OBJS_MAKE) $(OBJS_MAKE_CPP) $(OBJS_MAKE_C) $(LIBS) -g

git-version.h: 
	sed "s/@GIT_VERSION@/$(GIT_VERSION)/"  git-version.h.in > git-version.h

clean:
	rm -fr *.o objects
	rm -f git-version.h
	rm -f *~

