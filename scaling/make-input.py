import numpy as np
from random import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-L',type=float)
parser.add_argument('-N',type=int)

args = parser.parse_args()

print(args.N,args.L)
N = args.N
L = args.L

str = f"""Madatory string --> First rule of programing: if it works then don't touch it!

{N} atoms
0 bonds
0 angles

1 atom types
0 bond types
0 angle types

0.000 {L} xlo xhi
0.000 {L} ylo yhi
0.000 {L} zlo zhi

Masses

1 1.000000 

Atoms

"""
with open("input.data",'w') as f:
    f.writelines(str)
    for i in range(N):
        f.writelines(f"{i+1} {i+1} 1  {random()*L}  {random()*L}  {random()*L}\n")


str=f"""Dim 3

max_steps 500000
log_freq 10000000
binary_freq 500000
traj_freq 10000000
pmeorder 1

delt 0.005000

read_data input.data
integrator all GJF


Nx {int(L)}
Ny {int(L)}
Nz {int(L)}


potential gaussian 1 1 0.5  1.0"""

with open("input",'w') as f:
    f.writelines(str)
