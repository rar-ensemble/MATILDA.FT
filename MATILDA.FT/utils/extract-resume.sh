#!/bin/sh
# bash commands to extract the final frame from traj.lammpstrj into
# resume.lammpstrj for the read_resume command. 
# This routine WILL OVERWRITE any existing resume.lammpstrj

ns=`head -n 4 traj.lammpstrj | tail -n 1`
nlines=$(($ns+9))

tail -n ${nlines} traj.lammpstrj >resume.lammpstrj
