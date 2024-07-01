#!/bin/bash

if ! grep -qF 'read_resume resume_bonds.lammpstrj' in1; then
    sed -i '/read_data input.data/a read_resume resume_bonds.lammpstrj' in1
fi

sed -i "s/max_steps .*/max_steps 800000001/g" in1

awk '{

if ($3=="dynamic")
{
    print $0 " resume "$13"_resume"
} 
else
{
    print $0
}
}' in1 > in2