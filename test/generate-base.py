#!/usr/bin/env python3

import random
import numpy as np
import copy
import argparse


parser = argparse.ArgumentParser()

###### DEFAULTS #########

parser.add_argument('--lx', default = 15.0, type = float)
parser.add_argument('--ly', default = 15.0, type = float)
parser.add_argument('--lz', default = 70.0, type = float)

parser.add_argument('--max_steps', default = 10000001, type = int)
parser.add_argument('--log_freq', default = 5000, type = int)
parser.add_argument('--binary_freq', default = 20000, type = int)
parser.add_argument('--traj_freq', default = 500000, type = int)

###### EXTRAS ########

parser.add_argument('--salt', default= 0.0, type = float)
parser.add_argument('--no_bonds', action = 'store_true')
parser.add_argument('--eq', action = 'store_true')
###### DYNAMIC BONDS ########

parser.add_argument('--e_bond', default = 0.0, type = float)
parser.add_argument('--bond_log_freq', default = 1000, type = int)
parser.add_argument('--bond_freq', default = 100, type = int)

###############################

args = parser.parse_args()

# calculate num,ber of atoms

number_fraction = 2.00
N = 75

active = [0,N-1,24,49]

rin = 1.0
rskin = 1.0
k_spring = 1.5

zmin = 1/4
zmax = 3/4
r0 = 0.0
qind = 0.0
ad_hoc_density = 50

vol = args.lx * args.ly * args.lz
n_mon = number_fraction * vol
n_pol = np.ceil(n_mon/N)
n_pol_positive = int(n_pol//2)
n_pol_negative = n_pol_positive

if args.salt == 0:
    n_types = 1
    
else:
    n_types = 2
n_salt_2 = int(n_pol_positive* args.salt)


# donors are positively charged
# acceptors are negatively charged

donor_list = []
acceptor_list =  []

molecule_types = 1
atom_count = 1
mol_count = 1
bond_count = 1

properties = []
bonds = []
angles = []

if args.salt == 0:
    # id mol type charge x y z
    for m_num in range(n_pol_positive):
        for chain_pos in range(N):
            props = [atom_count,mol_count,1,1.0]
            
            if chain_pos == 0:
                props.append(np.random.uniform(0,args.lx))
                props.append(np.random.uniform(0,args.ly))
                props.append(np.random.uniform(zmin * args.lz, zmax * args.lz))
                
            else:
                theta = random.uniform(-np.pi, np.pi)
                phi = random.uniform(- 2 * np.pi, 2 * np.pi)
                x = 1.0 * np.cos(phi)*np.sin(theta) + properties[-1][4]
                y = 1.0 * np.sin(phi)*np.sin(theta) + properties[-1][5]
                z = 1.0 * np.cos(theta) + properties[-1][6]
                
                props.append(x)
                props.append(y)
                props.append(z)
                
            properties.append(copy.deepcopy(props))

            if chain_pos != (N-1):
                bonds.append([bond_count,1,atom_count,atom_count+1])
                bond_count += 1

            if chain_pos in active:
                donor_list.append(atom_count)

            # advance the atom count
            atom_count += 1
        mol_count += 1

    # id mol type charge x y z
    for m_num in range(n_pol_negative):
        for chain_pos in range(N):
            props = [atom_count,mol_count,1,-1.0]
            
            if chain_pos == 0:
                props.append(np.random.uniform(0,args.lx))
                props.append(np.random.uniform(0,args.ly))
                props.append(np.random.uniform(zmin * args.lz, zmax * args.lz))

            else:
                theta = random.uniform(-np.pi, np.pi)
                phi = random.uniform(- 2 * np.pi, 2 * np.pi)
                x = 1.0 * np.cos(phi)*np.sin(theta) + properties[-1][4]
                y = 1.0 * np.sin(phi)*np.sin(theta) + properties[-1][5]
                z = 1.0 * np.cos(theta) + properties[-1][6]
                
                props.append(x)
                props.append(y)
                props.append(z)
                
                # add atom properties to the list
            properties.append(copy.deepcopy(props))

            if chain_pos != (N-1):
                bonds.append([bond_count,1,atom_count,atom_count+1])
                bond_count += 1

            if chain_pos in active:
                acceptor_list.append(atom_count)

                # advance the atom count
            atom_count += 1
        mol_count += 1 

else:

    for m_num in range(n_pol_positive):
        for chain_pos in range(N):
            props = [atom_count,mol_count,1,+1.0,0,0,0]
            properties.append(props)

            if chain_pos != (N-1):
                bonds.append([bond_count,1,atom_count,atom_count+1])
                bond_count += 1

            if chain_pos in active:
                donor_list.append(atom_count)

            # advance the atom count
            atom_count += 1
        mol_count += 1

    # id mol type charge x y z
    for m_num in range(n_pol_negative):
        for chain_pos in range(N):
            props = [atom_count,mol_count,1,-1.0,0,0,0]
            properties.append(props)

            if chain_pos != (N-1):
                bonds.append([bond_count,1,atom_count,atom_count+1])
                bond_count += 1

            if chain_pos in active:
                acceptor_list.append(atom_count)

                # advance the atom count
            atom_count += 1
        mol_count += 1 
        

# id mol type charge x y z
for m_num in range(n_salt_2):
    props = [atom_count,mol_count,2,1.0]

    x = np.random.uniform(0,args.lx)
    y = np.random.uniform(0,args.ly)
    z = np.random.uniform(0,args.lz)
    
    props.append(x)
    props.append(y)
    props.append(z)
    
    # add atom properties to the list
    properties.append(copy.deepcopy(props))

    # advance the atom count
    atom_count += 1
    mol_count += 1 

for m_num in range(n_salt_2):
    props = [atom_count,mol_count,2,-1.0]
    x = np.random.uniform(0,args.lx)
    y = np.random.uniform(0,args.ly)
    z = np.random.uniform(0,args.lz)
    
    props.append(x)
    props.append(y)
    props.append(z)
    
    # add atom properties to the list
    properties.append(copy.deepcopy(props))

    # advance the atom count
    atom_count += 1
    mol_count += 1 

with open(file="input.data",mode='w') as fout:
    
    fout.writelines("Madatory string --> First rule of programing: if it works then don't touch it!\n\n")
    fout.writelines(f'{atom_count - 1} atoms\n')
    fout.writelines(f'{bond_count - 1} bonds\n')
    fout.writelines(f'0 angles\n')
    fout.writelines('\n')

    fout.writelines(f'{n_types} atom types\n')
    fout.writelines(f'{1} bond types\n')
    fout.writelines(f'{0} angle types\n')
    fout.writelines('\n')

    fout.writelines(f'0.000 {args.lx} xlo xhi\n')
    fout.writelines(f'0.000 {args.ly} ylo yhi\n')
    fout.writelines(f'0.000 {args.lz} zlo zhi\n')
    fout.writelines('\n')

    fout.writelines('Masses\n')
    fout.writelines('\n')

    for i in range(n_types):
        fout.writelines(f'{i + 1} {1.000:3f} \n')
    fout.writelines('\n')

    fout.writelines('Atoms\n')
    fout.writelines('\n')

    for atom in properties:
        line = atom
        if len(line) == 7:
            fout.writelines(f"{line[0]} {line[1]} {line[2]}  {line[3]}  {line[4]}  {line[5]}  {line[6]}\n")

    fout.writelines('\n')
    fout.writelines('Bonds\n')
    fout.writelines('\n')
    for line in bonds:
        fout.writelines(f"{line[0]} {line[1]}  {line[2]} {line[3]}\n")

# if not args.no_bonds:
#     input_file = f"""Dim 3

# max_steps {args.max_steps}
# log_freq {args.log_freq}
# binary_freq {args.binary_freq}
# traj_freq {args.traj_freq}
# pmeorder 1

# charges 0.43484 1.0

# delt {0.005:7f}

# read_data input.data
# integrator all GJF


# group bonding_group id id_file
# nlist bonding_group bonding bonding_list {rin} {rskin} {ad_hoc_density} {args.bond_freq} ad_file
# extraforce bonding_group lewis bonding_list {k_spring} {args.e_bond} {r0} {qind} {args.bond_freq} {args.bond_log_freq} bonds

# Nx {int(np.ceil(args.lx * 1.5))}
# Ny {int(np.ceil(args.ly * 1.5))}
# Nz {int(np.ceil(args.lz * 1.5))}

# bond 1 harmonic {k_spring} 0.0

# potential gaussian 1 1 0.003688  1.0
# """

# else:
#     input_file = f"""Dim 3

# max_steps {args.max_steps}
# log_freq {args.log_freq}
# binary_freq {args.binary_freq}
# traj_freq {args.traj_freq}
# pmeorder 1

# charges 0.43484 1.0

# delt {0.005:7f}

# read_data input.data
# integrator all GJF


# Nx {int(np.ceil(args.lx * 1.5))}
# Ny {int(np.ceil(args.ly * 1.5))}
# Nz {int(np.ceil(args.lz * 1.5))}

# bond 1 harmonic {k_spring} 0.0

# potential gaussian 1 1 0.003688  1.0
# """


# # if args.salt:
# #     input_file += f"potential gaussian 1 2 {2 * 0.003688}  1.0\npotential gaussian 2 2 0.003688  1.0"

# if args.eq:
#     input_file = f"""Dim 3

# max_steps 10000001
# log_freq 1000000
# binary_freq 1000000
# traj_freq 1000000
# pmeorder 1

# charges 0.43484 1.0

# delt {0.005:7f}

# read_data input.data
# integrator all GJF


# Nx {int(np.ceil(args.lx * 1.5))}
# Ny {int(np.ceil(args.ly * 1.5))}
# Nz {int(np.ceil(args.lz * 1.5))}

# bond 1 harmonic {k_spring} 0.0

# potential gaussian 1 1 0.003688  1.0
# """


# with open('input', 'w') as fout:       
#     fout.writelines(input_file)


print("Writing ad_file and id_file")
ad_list = []
with open("id_file", 'w') as f:
    for i in donor_list:
        f.writelines(f"{i} ")
        ad_list.append(1)
    for i in acceptor_list:
        f.writelines(f"{i} ")
        ad_list.append(0)

with open("ad_file", 'w') as f:
    for i in ad_list:
        f.writelines(f"{i} ")

