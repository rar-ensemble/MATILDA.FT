#!/bin/sh
# bash commands to extract the final frame from traj.lammpstrj into
# resume.lammpstrj for the read_resume command. 
# This routine WILL OVERWRITE any existing output file
# if arguments given, then the read filename is the first one
# and the output filename is the second argument

if [ "$#" -ne 2 ]; then
    ns=`head -n 4 traj.lammpstrj | tail -n 1`
    nlines=$(($ns+9))
    tail -n ${nlines} traj.lammpstrj >resume.lammpstrj
  else
    ns=`head -n 4 $1 | tail -n 1`
    nlines=$(($ns+9))
    tail -n ${nlines} $1 > $2
fi
