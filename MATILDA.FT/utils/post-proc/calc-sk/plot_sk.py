#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

t = "0" #type
max_frame = 200

os.system("/home/jello/matilda/MATILDA.FT/utils/post-proc/calc-sk/calc-sk_per_frame grid_densities.bin")
os.system("mkdir sk_plots")
files = []
for x in os.listdir():
    if x.endswith(".sk") and x[2]==t:
        files.append(x)
for file in files:
    if int(file[4:-3]) < max_frame:
        with open(file,'r') as f:
            lines = f.readlines()
        with open(file,'w') as f:
            for line in lines:
                if len(line)!= 0:
                    f.writelines(line)
        data = np.loadtxt(file)
        data = data[:,2:4]
        df = pd.DataFrame(data)
        res = df.groupby(1)[0].mean()
        val = res.to_numpy()[1:]
        k = res.index.to_numpy()[1:]
        plt.plot(k,val,'-', lw = 0.75, c = 'tab:blue')
        plt.xlabel("|k|")
        plt.ylabel(r"$S(k)$")
        plt.text( 0.5 * k.max() , 0.8 * val.max(), f"time step: {int(file[4:-3]) * 2000}")
        plt.savefig("sk/" + file +".png",dpi = 500)
        plt.close()









