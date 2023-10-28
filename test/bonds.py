import numpy as np
import matplotlib.pyplot as plt
print(len(np.loadtxt('ad_file')))
data = np.loadtxt('bonds')
print(np.average(data[-5:,1]))
# print(np.sum(data[:,1]))
# print(np.sum(data[:,1]))
# print(data[-1,:])
# print((data[-1,2]-data[-1,3]))
# print((data[-1,2]-data[-1,3])/min(data[-1,2:]))


plt.plot(data[:,0],data[:,1])
plt.show()
e2 = np.array([-3,-2,-1,0,1,2,3,4,5,6,7])
f2 = [0.18366666666666667,0.26804761904761903,0.3816190476190476,0.5011538461538462,0.6208571428571427,0.7278461538461538,0.8112307692307693,0.8763076923076922,0.9184378109452737,0.9491044776119402,0.9686245847176077]



e3 = [-3,-2,-1,0,1,2,3,4]
f3 = [0.22607843137254902,0.3304411764705883,0.44691176470588234,0.6008333333333333,0.7415686274509804,0.8532843137254903,0.9216176470588234,0.9673529411764706]


f4 = [0.22446078431372551,0.33034313725490194,0.45284313725490194,0.5964215686274511,0.7367156862745098,0.8496078431372549,0.9244607843137254,0.9687745098039215]

# Energy 0.0, Max: 0.5 Mean: 0.5000000000000003
# Energy -0.25, Max: 0.53 Mean: 0.531053061384217
# Energy -0.5, Max: 0.56 Mean: 0.5618650838565366
# Energy -1.0, Max: 0.62 Mean: 0.6218459308043619
# Energy -2.0, Max: 0.73 Mean: 0.7299007855937585
# Energy -3.0, Max: 0.82 Mean: 0.8159820342980876
# Energy -4.0, Max: 0.88 Mean: 0.8788851851608989

#[0.2263217974345926, 0.27453295826696533, 0.32966791645066773, 0.3911903402753372, 0.4579463019320729, 0.5281500780383072, 0.5994817592127134, 0.6692979859934219, 0.7349234183758193, 0.7939686087873712, 0.8446095573827136, 0.8857256099619182, 0.9169062561669054, 0.9387152973841723, 0.9528824885197201, 0.9616525069577165, 0.9669614604448533, 0.9701538290461421, 0.9720735320081683, 0.9732302263362852, 0.9739286078263739]

nD = 100
nA = 100

N = nD+nA
V =(5**3)


nmin = np.min([nA, nD])
def logW(nB, nA, nD):
    if ( nB == nA or nB == nD or nB == 0 ):
        return 0
    else:
        Aentropy = nA*np.log(nA) - nB*np.log(nB) - (nA-nB)*np.log(nA-nB)
        Dentropy = nD*np.log(nD) - nB*np.log(nB) - (nD-nB)*np.log(nD-nB)
        lnW = Aentropy + Dentropy
        return lnW

M = []
MEANS = []
energies = np.arange(-3,7.5,0.5)
for E in energies:
    E = E * -1.0
    # print( E)
    Z = 0.0
    nvals = np.linspace(0,nmin, nmin+1)
    N = np.shape(nvals)[0]
    pE = np.zeros(N, 'd')
    for nB in range(nmin+1):
        Etot = nB * E
        # print(-Etot + logW(nB, nA, nD))
        bFactor = np.exp(-Etot + logW(nB, nA, nD))
        pE[nB] = bFactor
        Z = Z + bFactor

    frac = 0
    for i in range(len(nvals)):
        frac+= (nvals[i]/nmin* (pE[i]/Z))
    MEANS.append(frac)
    M.append((frac * N/2 / V) / ((1-frac)* N/2/V)**2)
    # (frac * N/2 / V) / ((1-frac)* N/2/V)**2
    # print(f"Energy {E}, Max: {nvals[np.argmax(pE/Z)]/nmin} Mean: {mean}")

nD = 60
nA = 40


N = nD+nA
V =(5**3)


nmin = np.min([nA, nD])

M2 = []
MEANS2 = []
energies = np.arange(-3,7.5,0.5)
for E in energies:
    E = E * -1.0
    # print( E)
    Z = 0.0
    nvals = np.linspace(0,nmin, nmin+1)
    N = np.shape(nvals)[0]
    pE = np.zeros(N, 'd')
    for nB in range(nmin+1):
        Etot = nB * E
        # print(-Etot + logW(nB, nA, nD))
        bFactor = np.exp(-Etot + logW(nB, nA, nD))
        pE[nB] = bFactor
        Z = Z + bFactor

    frac = 0
    for i in range(len(nvals)):
        frac+= (nvals[i]/nmin* (pE[i]/Z))
    MEANS2.append(frac)
    M2.append((frac * N/2 / V) / ((1-frac)* N/2/V)**2)
    # (frac * N/2 / V) / ((1-frac)* N/2/V)**2
    # print(f"Energy {E}, Max: {nvals[np.argmax(pE/Z)]/nmin} Mean: {mean}")


# plt.plot(energies,MEANS, label = "Rob", color = 'tab:red')
# plt.plot(energies,MEANS2, label = "Rob 60d:40a / 40d:60a", color = 'tab:red', ls = '--')
# plt.plot(e2,f2, label = "Zuzanna", color = 'tab:blue', marker = 'o')
# plt.plot(e3,f3, label = "Zuzanna 60d:40s", color = 'tab:green', marker = 'o')
# plt.plot(e3,f4, label = "Zuzanna 40d:60s", color = 'tab:purple', marker = 'o')
# # plt.ylim(0.49,1)
# plt.legend()
# plt.ylabel("Bonded Fraction")
# plt.xlabel("-Energy")
# # plt.yscale('log')
# plt.show()


rates = [10,100,1000]
ebond2=[0.10501782178217822,0.10047128712871288,0.10511683168316831]
ebond5 = [0.35901782178217834,0.35679405940594067,0.35898613861386136]
ebond10=[0.8878475247524754,0.8852019801980197,0.882481188118812]


# plt.plot(rates,ebond2,label = "e_bond = 2", color = 'tab:green', marker = 'o')
# plt.plot(rates,ebond5,label = "e_bond = 5", color = 'tab:blue', marker = 'o')
# plt.plot(rates,ebond10,label = "e_bond = 10", color = 'tab:red', marker = 'o')
# plt.xscale('log')
# plt.ylabel("Bond fraction (k_spring = 1.5) + gaussian")
# plt.legend()
# plt.xlabel("Rate")
# plt.show()
# M2 = []
# for frac in f:
#     M2.append((frac * N/2 / V) / ((1-frac)* N/2/V)**2)
    

# M2b = []
# for frac in f2:
#     M2b.append((frac * N/2 / V) / ((1-frac)* N/2/V)**2)


# M2c = []
# for frac in f3:
#     M2c.append(frac*scale/(scale*(1-frac))**2)


# plt.title("Plot should be a straight line (black line as a guide)")
# plt.plot(energies,M, c = 'tab:blue', label = 'Rob', marker = 'o')
# plt.plot(e,M2, c = 'tab:green', label = "Zuzanna", marker = 'o')
# # plt.plot(e2,M2b, c = 'tab:purple', label = "Zuzanna (no competition)", marker = 'o')
# # plt.plot(e3,M2c, c = 'tab:orange', label = "Zuzanna (1/2)", marker = 'o')
# plt.plot(energies,np.exp(energies), color = 'black', ls = '--', label = 'theory')
# plt.ylabel("K_eq")
# plt.xlabel("-Energy")
# plt.yscale('log')
# plt.legend()
# plt.show()

