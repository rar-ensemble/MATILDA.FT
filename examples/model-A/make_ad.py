import numpy as np
N = 300
with open('ad_input','w') as f:
    for i in np.arange(0,N,1,dtype = int):
        f.writelines(f"{np.random.choice((0,1))} ")
        
with open('id_file','w') as f:
    for i in np.arange(0,N,1,dtype = int):
        f.writelines(f"{i+1} ")