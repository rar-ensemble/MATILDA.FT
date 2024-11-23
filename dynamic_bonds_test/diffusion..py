import numpy as np
import matplotlib.pyplot as plt

D = 1/10
koff = 1/100
# for e in energies:
#     kon.append(np.exp(e)*koff)
# kon= np.array(kon)

rm = 0.002
rg = 0.01
Keq = 300
kon = koff * Keq
ton = 1/kon
print(kon)

dx = np.sqrt(D * ton * 6)
print(f"delta_r/monomer_size during bond time = {dx/rg:3.1f}, time on (1/k_on) = {ton:1.2f} seconds")

