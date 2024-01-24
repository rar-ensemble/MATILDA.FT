import numpy as np
import matplotlib.pyplot as plt

e = [2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5]
m = [31.49000247825787,46.125417716124964,69.06203180248357,108.03777982631063,172.30276367607104,267.8810721123556,417.97138528172826,686.9507780382144,1062.054163566676,1918.108429910862,3180.9856303916667,4740.467419264671]

energies = np.arange(2,8,0.1)
# e = [2,3,4,5]
# m = [31.08667995092147,74,200,545.2405336231305]
plt.plot(energies,np.exp(energies)*(3.0), color = 'black', ls = '--', label = r'theory: $A \cdot \mathrm{exp}(\epsilon_a)$')
# plt.plot(energies,M, c = 'tab:blue', label = 'Rob')
plt.plot(e,m, c = 'tab:green', label = "simulation",marker = 'o', ls='')
# # plt.plot(e2,M2b, c = 'tab:purple', label = "Zuzanna (no competition)", marker = 'o')
# plt.plot(e3,M2c, c = 'tab:orange', label = "Zuzanna (1/2)", marker = 'o')

plt.ylabel(r"$K_{eq}$", fontsize=10)
plt.xlabel(r"$\epsilon_a$", fontsize = 10)
plt.yscale('log')
plt.legend()
plt.savefig("Figure2A.png",dpi = 500)
plt.close()
data = np.loadtxt('bonds')


# print(np.sum(data[:,1]))
# print(np.sum(data[:,1]))
# print(data[-1,:])
# print((data[-1,2]-data[-1,3]))
# print((data[-1,2]-data[-1,3])/min(data[-1,2:]))

N = 100
L = 20
V = L**3
nc = 400
f = np.average(data[-nc:,1])
print(f)

# f = 0.6044

Kc = (N * f)/V / ((1-f)*N/V)**2
print(Kc)
plt.axvline((data[-nc,0]))
plt.plot(data[:,0],data[:,1])
plt.show()
# 60 -- >
# 0.14383999999999997
# 85.37952519875402

# 100 --> 

# 50 -->

# e2 = np.array([-3,-2,-1,0,1,2,3,4,5,6,7])
# f2 = [0.18366666666666667,0.26804761904761903,0.3816190476190476,0.5011538461538462,0.6208571428571427,0.7278461538461538,0.8112307692307693,0.8763076923076922,0.9184378109452737,0.9491044776119402,0.9686245847176077]
# KC = [107,110.48607599865296,111.03417022035703,112.22411024821666,112.11389150058616]
# V = np.array([30**3,40**3,50**3,80**3,100**3])
# C = N/V
# plt.plot(C,KC,'o', color = 'tab:blue')
# plt.ylabel(r"$K_{eq}$")
# plt.xlabel(r"[$D_0$]")
# plt.ylim(100,120)
# # plt.xscale('log')
# plt.show()




f4 = [0.22446078431372551,0.33034313725490194,0.45284313725490194,0.5964215686274511,0.7367156862745098,0.8496078431372549,0.9244607843137254,0.9687745098039215]

# Energy 0.0, Max: 0.5 Mean: 0.5000000000000003
# Energy -0.25, Max: 0.53 Mean: 0.531053061384217
# Energy -0.5, Max: 0.56 Mean: 0.5618650838565366
# Energy -1.0, Max: 0.62 Mean: 0.6218459308043619
# Energy -2.0, Max: 0.73 Mean: 0.7299007855937585
# Energy -3.0, Max: 0.82 Mean: 0.8159820342980876
# Energy -4.0, Max: 0.88 Mean: 0.8788851851608989

#[0.2263217974345926, 0.27453295826696533, 0.32966791645066773, 0.3911903402753372, 0.4579463019320729, 0.5281500780383072, 0.5994817592127134, 0.6692979859934219, 0.7349234183758193, 0.7939686087873712, 0.8446095573827136, 0.8857256099619182, 0.9169062561669054, 0.9387152973841723, 0.9528824885197201, 0.9616525069577165, 0.9669614604448533, 0.9701538290461421, 0.9720735320081683, 0.9732302263362852, 0.9739286078263739]


# r = np.arange(0,10,0.01)
# y = np.exp(-k_spring * r**2) * cA * 4 * np.pi * r**2

# Pfind = np.trapz(y,r)
# print(Pfind)


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
dU = 0
# r = np.arange(0,10,0.01)
# y = np.exp(-k_spring * r**2) * k_spring * r**2 * 4 * np.pi * r**2
# dU = np.trapz(y, r)


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


# plt.plot(energies,MEANS,'o')
# plt.show()

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
    # (frac * N/2 / V) / ((1-frac)* N/2/V)**2
    # print(f"Energy {E}, Max: {nvals[np.argmax(pE/Z)]/nmin} Mean: {frac}")

# e3 = [-3,-2,-1,0,1,2,3,4]
# f3 = [0.22607843137254902,0.3304411764705883,0.44691176470588234,0.6008333333333333,0.7415686274509804,0.8532843137254903,0.9216176470588234,0.9673529411764706]

e3 = [0,0.5,1,1.5,2,2.5,3,3.5]
f3 = [0.5956,0.6650375000000001,0.73645,0.7949949999999999,0.8463749999999999,0.8949125,0.9233500000000001,0.94808]
f2 = [0.49528000000000005,0.56247,0.61869,0.68104,0.72781,0.77522,0.8121600000000001,0.8512339999999999]
plt.plot(energies,MEANS, label = "Analytical solution 1:1", color = 'k', lw = 1)
plt.plot(energies,MEANS2, label = "Analytical solution 3:2 / 2:3", color = 'k', ls = '--', lw = 1)
plt.plot(e3,f2, label = "Simulation 1:1", color = 'tab:red', marker = 'o', ls = '')
plt.plot(e3,f3, label = "Simulation 3:2 / 2:3", color = 'tab:green', marker = 'o', ls = '')
# # # plt.plot(e3,f4, label = "Zuzanna 40d:60s", color = 'tab:purple', marker = 'o')
# # # plt.ylim(0.49,1)
plt.legend(fontsize=8,ncol = 2)
plt.ylabel("Bonded Fraction")
plt.xlabel(r"$\epsilon_a$")
# # plt.yscale('log')
plt.savefig("Figure1A.png",dpi = 500)
plt.close()

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



e = [3,4,5,7]

f_15 = [0.3477979591836735,0.7829199999999998,0.9087259999999998,0.9646080000000002]
f_175 = [0.318498,0.7685959999999998,0.9104879999999999,0.9630658227848101]
f_18 = [0.30773483146067415,0.7610234042553191,0.9089944444444443,0.9646854166666666]
f_2 = [0.2261136842105263,0.5666114285714287,0.7813080000000001,0.908056]


# f_3 = [0.1400648401826484,0.4162229508196721,0.6631516666666666,0.841364,0.9559102040816327]


f_35 = [0.6100105263157894,0.9485435897435898]
f_35 = [0.612352,0.9512959999999999]
f_4 = [0.5630248908296944,0.9386939244663384]
f_5 = [0.47801906643615705,0.9162932544378699,]
f_10 = [0.2434701317715959,0.7931608944357774,]
f_inf = [0.07402228756248162,0.4647400534045394]


f = [0.05598000000000001,0.10624000000000003,0.19095999999999996,0.4734570361145704]


# plt.plot(e,f_35, label =3.5)
# plt.plot(e,f_4, label =4)
# plt.plot(e,f_5, label = 5)
# plt.plot(e,f_10, label = 10)
# plt.plot(e,f_inf, label = r"$\inf$")
# # plt.plot(e,np.exp(e),color = 'black',label = 'theory')
# plt.ylabel("Bond fraction")
# plt.xlabel("Energy")
# plt.legend(title = "r_cutoff")
# plt.show()

F = []
F3 = []
F15 = []
F175 = []
F18 = []
F10=[]
Finf=[]


V = 10**3
N = 500
for frac in f_2:
    F.append((frac * N/2 / V) / ((1-frac)* N/2/V)**2)
    

# for frac in f_3:
#     F3.append((frac * N/2 / V) / ((1-frac)* N/2/V)**2)
    
for frac in f_15:
    F15.append((frac * N/2 / V) / ((1-frac)* N/2/V)**2)
    
for frac in f_175:
    F175.append((frac * N/2 / V) / ((1-frac)* N/2/V)**2)
    
for frac in f_18:
    F18.append((frac * N/2 / V) / ((1-frac)* N/2/V)**2)
    
for frac in f_10:
    F10.append((frac * N/2 / V) / ((1-frac)* N/2/V)**2)
    
for frac in f_inf:
    Finf.append((frac * N/2 / V) / ((1-frac)* N/2/V)**2)
    
# M2b = []
# for frac in f2:
#     M2b.append((frac * N/2 / V) / ((1-frac)* N/2/V)**2)


# M2c = []
# for frac in f3:
#     M2c.append(frac*scale/(scale*(1-frac))**2)


# for i,scale in enumerate([0.27]):#,300,2e3,1e4):
#     if i == 0:
#         plt.plot(e,np.exp(e)/scale,color = 'black', ls = '--', label = 'theory')
#     else:
#         plt.plot(e,np.exp(e)/scale,color = 'black', ls = '--')


# plt.plot(energies,M, label = r"Rob, $k_{spring}= 0$", color = 'tab:red')
# # plt.plot(energies,M2, color = "tab:red")
# plt.plot(e,F)
# # plt.plot(e,F175, label =r"r_n=" + "1.75")
# # plt.plot(e,F18, label =r"r_n=" + "1.80")
# # plt.plot(e,F2, label =r"r_n=" + "2.0")
# # plt.plot(e,F3, label =r"r_n=" + "3.0")
# # plt.plot(e,F35, label =3.5)
# # plt.plot(e,F4, label =4)
# # plt.plot(e,F5, label = 5)
# # plt.plot(e,F10, label = 10)
# # plt.plot(e,Finf, label = r"$\inf$")
# plt.plot(e,np.exp(e)/1,color = 'black', ls = '--',label = 'theory K_eq ~ exp(e)')

# plt.yscale('log')
# plt.ylabel("K_eq")
# plt.xlabel("Energy")
# # plt.legend(title="r_cutoff")
# plt.show()

# x = np.arange(0,2.5,0.01)
# y = np.exp(-1.5 * x**2)
# plt.plot(x,y)
# plt.xlabel("r")
# plt.ylabel(r"$e^{-(k_{spring} \times r^2)}$")
# plt.axhline(0,ls = '--',c="black")
# plt.show()

# print(x[np.where(y == 0)])
# print(y)


# rate = [50,100,500,1000]
# f = [0.764457142857143,0.7637526881720431,0.7638180000000001,0.7642700000000001]

# plt.plot(rate,f,marker = '.')
# plt.title(r"$k_{spring},r_{cutoff}=1.8$ check frequency independence")
# plt.xlabel("Bond rate")
# plt.ylabel("Bond fraction")
# plt.ylim(0.74,0.78)
# plt.xscale('log')
# plt.show()

# e_8 = [0.6408607594936709,0.6429365079365079,0.6401081081081079,0.6437219251336899]
# r_cutoff = [2,3,4,20]

# plt.plot(r_cutoff,e_8, label = "E_bond = constant, k_spring = 1.5",marker = 'o')
# plt.ylabel("Bond fraction")
# plt.xlabel("r_cutoff")
# plt.ylim(0.2,0.7)
# plt.legend()
# plt.title("Effect of increasing n-list cutoff (no effect!)")
# plt.show()