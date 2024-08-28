import matplotlib.pyplot as plt
import re
import numpy as np

yticks = [30,40,50,60,70,80,90,100,200,300,400,500, 600,700,800,900]
joint = []
sjoint =[]
for y in yticks:
    plt.axhline(y,ls='--',color="tab:gray",alpha=0.5,lw=1)
colors = ['tab:red','tab:orange','tab:green','tab:blue','tab:purple','tab:gray']
# sizes = np.array([20,30,50,80,100])
# sizes = np.array([8,16, 32, 64,128])
sizes = np.array([8**3,16**3, 32**3, 64**3,64*64*128,64*128*128,128**3])
names = ["8","16","32","64","64_128","64_128_128","128"]
particles = [100,1000,5000,25000,50000]
# (well, really M*log(M) with M = Nx^Dim)
for _ in particles:
    joint.append([])
for n in range(len(sizes)):
    tmp = []
    for m in range(len(particles)):
        with open(f"{names[n]}/{particles[m]}/out2") as infile:
            lines=infile.readlines()
            line = lines[37].split()
            time = re.split(r'([a-z]+)',line[-1])
            time = int(time[0])*60+int(time[2])
            joint[m].append(time)
            tmp.append(time)
    sjoint.append(np.array(tmp))

ref = []

for m in range(len(joint)):
    joint[m] = np.array(joint[m])
    plt.plot(sizes,joint[m],color=colors[m], linewidth=1.5, label=particles[m], marker='o')
plt.xlabel(r'$M$', fontsize=10)
plt.ylabel('Total run time [seconds]',fontsize=10)

plt.gca().set_aspect(1.5)
plt.xscale('log')
plt.yscale('log')
yticks = [50,100,500,1000]
plt.yticks(yticks,labels=[r"$50$",r"$10^2$",r"$500$",r"$10^3$"])
plt.legend(fontsize=10, title="Number of particles", loc='upper left')
plt.title(r'Simulation time for 3D boxes with $M$ grid points')
plt.savefig('scaling.png',dpi=500,bbox_inches='tight')
plt.close()


for m in range(len(joint)):
    joint[m] = np.array(joint[m])
    plt.plot(sizes,joint[m],color=colors[m], linewidth=1.5, label=particles[m], marker='o')
plt.xlabel(r'$M$', fontsize=10)
plt.ylabel('Total run time [seconds]',fontsize=10)
# plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.yticks(yticks)


plt.legend(fontsize=10, title="Number of particles", loc='upper left')
plt.title(r'Simulation time for 3D boxes with $M$ grid points')
plt.savefig('scaling_norm.png',dpi=500,bbox_inches='tight')
plt.close()


# for y in yticks:
#     plt.axhline(y,ls='--',color="tab:gray",alpha=0.5)

# for m in range(len(joint)):
#     plt.plot(np.array(sizes)**3,joint[m],color=colors[m], linewidth=1.5, label=particles[m], marker='o')
    
# plt.xlabel(r'$N_x^3$', fontsize=10)
# plt.ylabel('Total run time [seconds]',fontsize=10)
# plt.legend(fontsize=10, title="N (No. of partciles)", loc='upper left')
# # plt.xticks(sizes**3,labels=[r"$20^3$",r"$30^3$",r"$50^3$",r"$80^3$",r"$100^3$"])
# # plt.yticks(ticks=yticks)
# plt.title(r'Simulation time for boxes with $(N_x)^3$ grid points')
# plt.savefig('scaling_3.png',dpi=500,bbox_inches='tight')
# plt.close()

nscale = []
lscale = []
Ns = []
ls = []
for i in range(1,len(joint)):
    nscale.append(np.average(joint[i]/joint[0]))
    Ns.append(particles[i]/particles[0])
    
for i in range(1,len(sjoint)):
    lscale.append(np.average(sjoint[i]/sjoint[0]))
    ls.append(sizes[i]/sizes[0])

print(f"Particle ratio {Ns}\nTime ratio {nscale}")
print("----")

print(f"Nx^3 ratio {ls}\nTime ratio {lscale}\n")
print("----")
m=250
plt.plot(ls,lscale,marker='o',ls='')
plt.plot(ls,np.array(ls)/m + (lscale[0]) - np.array(ls[0])/m)
plt.savefig("scale_ratio.png",dpi=500)
