import numpy as np
from random import random
N = 20
L = 20
# print(N/L**3,10664/(25*25*180))
with open("ad_file1", "w") as f:
    for n in range(5):
        f.writelines(f"1 ")
    for n in range(5):
        f.writelines(f"0 ")


with open("ad_file2", "w") as f:
    for n in range(5):
        f.writelines(f"1 ")
    for n in range(5):
        f.writelines(f"0 ")


with open("id_file1", "w") as f:
    for n in range(N//2):
        f.writelines(f"{n+1} ")


with open("id_file2", "w") as f:
    for n in range(N//2, N):
        f.writelines(f"{n +1} ")

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
