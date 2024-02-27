import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.optimize import curve_fit
colors = ['tab:red','tab:orange','tab:blue','tab:green','tab:purple']
res = [125,175,225,275,375]
dx = [1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,15,20,30]
U125 = np.array([0.197512,0.309930,0.462924,0.656495,0.829799,0.852519,0.918256,1.027011,1.092760,1.109118,1.206114, 1.207808,1.266184, 1.270145,1.341904,1.431795,1.455297])
# U175 = np.array([0.544389,1.049814,1.732251,2.046483,2.414098])#3
U175 = np.array([0.544389,1.049814,1.732251,2.046483,2.414098,2.823387,2.895868,3.116183,3.285426,3.498922,3.560762,3.682424,3.873437,3.873637,4.027478,4.237269,4.236001])
U225 = np.array([1.256971,2.614735,3.658464,4.794061,5.498481,6.148189,6.598410,6.945704,7.285631,7.761410,8.118972,8.405162,8.534664,8.635187,8.951104,9.219532,9.388823])
U275 = np.array([2.468169,4.815770,7.105054,8.903038,10.616741,11.469414,12.543417,13.020687,13.614301,14.346480,14.901311,15.361765,15.774632,16.168262,16.768963,17.132996,17.667479])
U375 = np.array([6.756089,12.850498,18.565096,23.454306,27.408941,30.470028,32.341141,33.926556,35.340645,37.545612,38.773048,40.258568,40.708126,41.603729,43.708252,44.506378,45.407047])


#sigma = sqrt(2/alpha)
alpha = 1/100
y = np.arange(1,30.0,0.1)
a0 = -2.1
a1 = 1.5
a2 = alpha

# plt.plot(dx,U175 * 120**3/175**3, marker = 'o', ls = '',label = "175", color = colors[1])
# plt.plot(dx,U225 * 120**3/225**3, marker = 'o', ls = '',label = "225", color = colors[2])
# plt.plot(dx,U275 * 120**3/275**3, marker = 'o', ls = '',label = "275", color = colors[3])

def fun(dx,a1,a2):
    return -2* special.erfc(a2 * dx)/dx + a1

PARAMS = []
for i,data in enumerate([U125,U175,U225,U275,U375]):
    plt.plot(dx,data * 120**3/res[i]**3, marker = 'o', ls = '',label = res[i], color=colors[i])
    param, cov = curve_fit(fun, dx[3:], data[3:] * 120**3/res[i]**3, p0 = (1.6,0.01), bounds=[(0.0,0),(2,2)])
    print(param)
    PARAMS.append(param[0])
    # param = (1.5,0.01)
    plt.plot(y,fun(y,*param), label = "fit", color = colors[i])
    # plt.plot(y,-1/y, label = "y = 1/r")

plt.xlabel("r")
plt.ylabel("Electrostatic energy")
plt.legend()
plt.show()

PARAMS = np.array(PARAMS)
plt.plot(res,PARAMS,marker = 'o', lw = 1)
plt.xlabel('resolution')
plt.ylabel("self-interaction term")
plt.show()

# print(U275 * 120**3/275**3)
# with open("grace-in.data","w") as f:
#     for i in range(len(U275)):
#         f.writelines(f"{dx[i]} {U275[i] * 120**3/275**3}\n")