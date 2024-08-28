#!/bin/bash
for L in 8 16 32 64 64_128_128 64_128 128;do
cd $L
for N in 100 1000 5000 10000 25000 50000;do
cd $N
~/MATILDA.FT/matilda.ft > out
cd ../
done
cd ../
done
