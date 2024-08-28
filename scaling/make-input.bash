#!/bin/bash
for L in 64_128_128;do
mkdir $L
cd $L
for N in 100 1000 5000 10000 25000 50000;do
mkdir $N
cd $N
python /home/marshmallow/MATILDA.FT/scaling/make-input.py -N $N -L $L
cd ../
done
cd ../
done