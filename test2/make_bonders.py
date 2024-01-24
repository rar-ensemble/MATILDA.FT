import numpy as np
from random import random
N = 1000
L = 20
# print(N/L**3,10664/(25*25*180))
with open("ad_input", "w") as f:
    for n in range(400):
        f.writelines(f"1 ")
    for n in range(600):
        f.writelines(f"0 ")



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
        f.writelines(f"{i+1} {i+1} 1  {0.0} {random()*L}  {random()*L}  {random()*L}\n")
