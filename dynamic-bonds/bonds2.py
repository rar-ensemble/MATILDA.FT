import numpy as np
import matplotlib.pyplot as plt
MEANS = []
M = []

def logW(nB, nA, nD):
    if ( nB == nA or nB == nD or nB == 0 ):
        return 0
    else:
        Aentropy = nA*np.log(nA) - nB*np.log(nB) - (nA-nB)*np.log(nA-nB) 
        Dentropy = nD*np.log(nD) - nB*np.log(nB) - (nD-nB)*np.log(nD-nB)
        lnW = Aentropy + Dentropy
        return lnW

e = [0,1,2,3]
f = [0.5110399999999999,0.6107799999999999,0.7295900000000001,0.8116999999999998]
e2 = [0,1,2,3]
f2 = [0.6002624999999999,0.73834375,0.8487375,0.9157437500000001]
energies = np.arange(0,4.0,0.1)
nD = 25
nA = 25

k_spring = 1.5
N = nD+nA
L = 5
V =L**3
nmin = np.min([nA, nD])
for E in energies:
    E = E * -1.0
    # print( E)
    Z = 0.0
    nvals = np.linspace(0,nmin, nmin+1)
    N = np.shape(nvals)[0]
    pE = np.zeros(N, 'd')
    for nB in range(nmin+1):
        Etot = (nB * E)
        # print(-Etot + logW(nB, nA, nD))
        bFactor = np.exp(-Etot + logW(nB, nA, nD))
        pE[nB] = bFactor
        Z = Z + bFactor

    frac = 0
    for i in range(len(nvals)):
        frac+= (nvals[i]/nmin* (pE[i]/Z))
    # frac = nvals[np.argmax(pE/Z)]/nmin
    MEANS.append(frac)
    M.append((nmin * frac)/V / ((nA - nmin * frac)/V * (nD - nmin * frac)/V))
#     print(f"Energy {E}, Max: {nvals[np.argmax(pE/Z)]/nmin} Mean: {frac}")



nD = 120
nA = 80


N = nD+nA
V =(5**3)


nmin = np.min([nA, nD])
nmax = np.max([nA,nD])

M2 = []
MEANS2 = []

for E in energies:
    E = E * -1.0
    # print( E)
    Z = 0.0
    nvals = np.linspace(0,nmin, nmin+1)
    N = np.shape(nvals)[0]
    pE = np.zeros(N, 'd')
    for nB in range(nmin+1):
        Etot = (nB * E)
        # print(-Etot + logW(nB, nA, nD))
        bFactor = np.exp(-Etot + logW(nB, nA, nD))
        pE[nB] = bFactor
        Z = Z + bFactor

    frac = 0
    for i in range(len(nvals)):
        frac+= (nvals[i]/nmin* (pE[i]/Z))

    # frac = nvals[np.argmax(pE/Z)]/nmin
    MEANS2.append(frac)
    M2.append((nmin * frac)/V / ((nA - nmin * frac)/V * (nD - nmin * frac)/V))


plt.plot(energies,MEANS, label = "Analytical solution 1:1", color = 'k', lw = 1.2)
plt.plot(e,f, label = "Simulation (Lewis Bonds) 1:1", color = 'tab:red', marker = 'o', ls = '')
plt.plot(energies,MEANS2, label = "Analytical solution 3:2 / 2:3", color = 'k', ls = '--', lw = 1.2)
plt.plot(e2,f2, label = "Simulation (Lewis Bonds) 3:2", color = 'tab:blue', marker = 'o', ls = '')
plt.title("Penn State")
plt.legend(fontsize=8,ncol = 2)
plt.ylabel("Bonded Fraction")
plt.xlabel(r"$\epsilon_a$")
plt.savefig("Bonds.png",dpi=500)