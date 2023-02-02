CC 	   = mpic++
#FFTW_LOC = /opt/seas/pkg/gcc/fftw3/mpi/double/3.3.6
FFTW_LOC = ${HOME}/Install/fftw3
#EIGEN_LOC = ${HOME}/Install/eigen
CFLAGS     = -std=c++11 -I${FFTW_LOC}/include -I${EIGEN_LOC} -O3 -Wno-unused-result -Wno-write-strings
LIBS      = -lm -lfftw3_mpi -lfftw3 -O3 -L${FFTW_LOC}/lib
#CFLAGS     = -g -I${FFTW_LOC}/include -I${EIGEN_LOC} -Wno-unused-result -Wno-write-strings
#LIBS      = -g -lm -lfftw3_mpi -lfftw3 -L${FFTW_LOC}/lib


#############################################################################
# nothing should be changed below here

SRCS = calc-sk_per_frame.cpp fftw_mpi_wrappers.cpp array_utils.cpp

OBJS = ${SRCS:.cpp=.o}

.cpp.o:
	${CC} ${CFLAGS} ${DFLAGS} -c  $<

calc-sk-per-frame:  ${OBJS}
	$(CC) ${CFLAGS} ${DFLAGS} -o $@ ${OBJS} $(LIBS)

clean:
	rm -f *.o
	rm -f calc-sk
	rm -f *~

