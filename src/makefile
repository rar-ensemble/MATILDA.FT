# Copyright (c) 2023 University of Pennsylvania
# Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
#
####################################################
# Modify the line below to point to the CUDA library


GIT_VERSION := $(shell git describe --abbrev=9 --dirty --always --tags)
GIT_DIRECTORY := $(shell git rev-parse --show-toplevel)

VERSION = 12.1
HPC_HOME = /opt/nvidia/hpc_sdk/Linux_x86_64/23.5
CUDA_HOME = /usr/local/cuda-${VERSION}
NVCC = ${CUDA_HOME}/bin/nvcc
EIGEN_LOC = ../include/eigen-3.4.0

CFLAGS = -rdc=true -O3 -std=c++14 -Wno-deprecated-gpu-targets -lineinfo #--generate-code arch=compute_80,code=sm_80 

LIBS =  -L${CUDA_HOME}/lib64 -lcuda -lcufft -L/lib/x86_64-linux-gnu -lm -L${HPC_HOME}/comm_libs/${VERSION}/openmpi4/openmpi-4.1.5/lib -lmpi -L${HPC_HOME}/comm_libs/${VERSION}/nvshmem_cufftmp_compat/lib -lcufftMp -L${HPC_HOME}/comm_libs/${VERSION}/nvshmem_cufftmp_compat/lib  -lnvshmem_device -lnvshmem_host -L${HPC_HOME}/comm_libs/${VERSION}/nccl/lib -L${CUDA_HOME}/extras/CUPTI/lib64 -L${HPC_HOME}/math_libs/${VERSION}/lib64 

INCS = -I${CUDA_HOME}/include -I${HPC_HOME}/comm_libs/${VERSION}/openmpi4/openmpi-4.1.5/include -I${HPC_HOME}/math_libs/${VERSION}/include/cufftmp -I${HPC_HOME}/comm_libs/${VERSION}/nvshmem_cufftmp_compat/include

LD_LIB=${CUDA_HOME}/lib64:${HPC_HOME}/comm_libs/${VERSION}/openmpi4/openmpi-4.1.5/lib:${HPC_HOME}/comm_libs/${VERSION}/nvshmem_cufftmp_compat/lib:${CUDA_HOME}/extras/CUPTI/lib64:${HPC_HOME}/comm_libs/${VERSION}/nccl/lib:${HPC_HOME}/math_libs/${VERSION}/lib64

CPPFLAGS = -I${EIGEN_LOC} --expt-relaxed-constexpr

# CUDA_LOC = /usr/local/cuda-12.1
# CC = ${CUDA_LOC}/bin/nvcc

# GIT_VERSION := $(shell git describe --abbrev=9 --dirty --always --tags)
# GIT_DIRECTORY := $(shell git rev-parse --show-toplevel)

# EIGEN_LOC = ../include/eigen-3.4.0

# CFLAGS = -std=c++14 -rdc=true -O3 -I${CUDA_LOC}/include -Wno-deprecated-gpu-targets -lineinfo
# # --default-stream per-thread
# CPPFLAGS = -I${EIGEN_LOC} --expt-relaxed-constexpr

#CFLAGS = -std=c++11 -rdc=true -arch=sm_37 -dlto -O3 -I${CUDA_LOC}/include -lineinfo -g 

# LIBS = -lm -lcudart -lcufft -L${CUDA_LOC}/lib -L${MPI_HOME}/lib -lmpi

SRCS = array_utils.cu bonds.cu calc_properties.cu config_utils.cu cuda_random_posits.cu \
	device_array_utils.cu device_bonds.cu device_comm_utils.cu \
	device_grid_utils.cu device_utils.cu  field_component.cu forces.cu \
	initialize.cu integ_utils.cu io_utils.cu main.cu potential.cu \
	potential_fieldphases.cu potential_gaussian.cu pbc_utils.cu read_input.cu \
	potential_erf.cu reduce_utils.cu update_potentials.cu device_GJF_integrator.cu \
	potential_charges.cu potential_gaussian_erf.cu \
	Compute.cu angles.cu device_angles.cu random.cu grid_utils.cu \
	Extraforce.cu die.cu \
	Extraforce_dpd.cu Extraforce_langevin.cu Extraforce_lewis.cu Extraforce_midpush.cu Extraforce_wall.cu \
	group.cu \
	group_id.cu group_region.cu group_type.cu\
	integrator.cu \
	integrator_VV.cu integrator_EM.cu integrator_GJF.cu\
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
	${NVCC} -g ${INCS} ${CFLAGS} -c $< -o $@


# $(OBJFOLDER)/%.cpp.o: %.cpp git-version.h
# 	@mkdir -p $(dir $@)
# 	${CC} -g ${CFLAGS} ${CPPFLAGS} -c $< -o $@

# $(OBJFOLDER)/%.c.o: %.c git-version.h
# 	@mkdir -p $(dir $@)
# 	${CC} -g ${CFLAGS} -c $< -o $@
	 
# ../matilda-mpi: $(OBJS_MAKE) $(OBJS_MAKE_CPP) $(OBJS_MAKE_C)
# 	$(CC) $(CFLAGS) -o $@ $(OBJS_MAKE) $(OBJS_MAKE_CPP) $(OBJS_MAKE_C) $(LIBS) -g

$(OBJFOLDER)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	${NVCC} -g ${INCS} ${CFLAGS} -c $< -o $@

$(OBJFOLDER)/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	${NVCC} -g ${INCS} ${CFLAGS} ${CPPFLAGS} -c $< -o $@

$(OBJFOLDER)/%.c.o: %.c
	@mkdir -p $(dir $@)
	${NVCC} -g ${INCS} ${CFLAGS} -c $< -o $@
	 
../matilda-mpi: $(OBJS_MAKE) $(OBJS_MAKE_CPP) $(OBJS_MAKE_C)
	LD_LIBRARY_PATH=$(LD_LIB)  $(NVCC) $(CFLAGS) -o $@ $(OBJS_MAKE) $(OBJS_MAKE_CPP) $(OBJS_MAKE_C) $(LIBS) -g


git-version.h: 
	sed "s/@GIT_VERSION@/$(GIT_VERSION)/"  git-version.h.in > git-version.h

clean:
	rm -fr *.o objects
	rm -f matilda-mpi ../matilda-mpi
	rm -f git-version.h
	rm -f *~

