import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

# make figure and assign axis objects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
fig.subplots_adjust(wspace=0)

f1 = 20.4
f2 = 5.8+6.2
f3 = 1.7+1.8+3.0+4.2
frac = [f1+f2+f3,23+12.3,14.2,7.4]
labels = ["Force\nRoutines","FFT", "Particle\nto\nMesh","Integrator"]
explode=(0.1,0,0,0)

# pie chart parameters

# rotate so that first wedge is split by the x-axis
angle = 250
wedges, texts ,autotexts = ax1.pie(x=frac,autopct='%1.1f%%',explode=explode,startangle=angle,wedgeprops=dict(width=0.75,edgecolor='white',linewidth=2),pctdistance=0.6)

angles = [0,0,30,55]

for i,t in enumerate(autotexts):
    t.set_rotation(angles[i])


plt.setp(autotexts, size=14, weight="bold",color='black')
# plt.setp(texts, size=14,color='black')

bbox_props = dict(boxstyle="square,pad=0.6", fc="w", ec="k", lw=1)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center",ha='center',size=12)

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = f"angle,angleA=0,angleB={ang}"
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax1.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

# bar chart parameters
forces = np.array([f3,f2,f1])
forces = forces/sum(forces)
force_labels = ['Fourier\nSpace Calc','Zero all /\naccumulate\nforces','Apply forces',]
bottom = 1
width = .2

# Adding from the top matches the legend.
for j, (height, label) in enumerate(reversed([*zip(forces, force_labels)])):
    bottom -= height
    bc = ax2.bar(0, height, width, bottom=bottom, color='tab:blue', label=label,
                 alpha=0.8 - 0.25 * j)
    ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center',size=14)

ax2.set_title('Force Routines',fontsize=14,weight='bold')
ax2.legend(loc = (-0.05,0.4),fontsize=11)
ax2.axis('off')
ax2.set_xlim(- 2.5 * width, 2.5 * width)

# use ConnectionPatch to draw lines between the two plots
theta1, theta2 = wedges[0].theta1, wedges[0].theta2
center, r = wedges[0].center, wedges[0].r
bar_height = sum(forces)

# draw top connecting line
x = r * np.cos(np.pi / 180 * theta2) + center[0]
y = r * np.sin(np.pi / 180 * theta2) + center[1]
con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)
con.set_color([0, 0, 0])
con.set_linewidth(2)
con.set_linestyle('--')
ax2.add_artist(con)

# draw bottom connecting line
x = r * np.cos(np.pi / 180 * theta1) + center[0]
y = r * np.sin(np.pi / 180 * theta1) + center[1]
con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)
con.set_color([0, 0, 0])
ax2.add_artist(con)
con.set_linewidth(2)
con.set_linestyle('--')
plt.savefig("Pie.png",dpi=500,bbox_inches='tight')