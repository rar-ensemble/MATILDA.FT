import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
'exec(%matplotlib inline)'
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.optimize import curve_fit
mpl.rcParams['figure.dpi'] = 150

def average_sq(bmax,binsize,sq):
    nbins=int(bmax / binsize)
    p=np.zeros((nbins,3))
    for j in range(nbins):
        p[j,0]=j*binsize

    for i in range(len(sq)):
        b=int(sq[i, 4]/binsize)
        p[b, 1]+=sq[i, 5]
        p[b, 2]+=1
        
    print(p[:,1])
    print(p[:,2])
    p[:, 1]=p[:, 1]/p[:, 2]
    print(p[:,1])
    
    where_are_NaNs = np.isnan(p)
    p[where_are_NaNs] = 0
    p=remove_zero(p)
    return p

def remove_zero(x):
    p=[]
    for i in x:
        
        if i[1]!=0:
            p.append(i)
    p=np.array(p)
    return p

def func(K_space,prefactor,chi):
    #chi=0.5
    N=17.7
    f=numberA/totalUnits
    R=N/6
    gr1=2*(K_space**2*f*R**2+np.exp(-f*K_space**2*R**2)-1)/(K_space*R)**4
    gr2=2*(K_space**2*(1-f)*R**2+np.exp(-(1-f)*K_space**2*R**2)-1)/(K_space*R)**4

    gr=2*(K_space**2*R**2+np.exp(-K_space**2*R**2)-1)/(K_space*R)**4

    sq_fit=N/(gr/(gr1*gr2-0.25*(gr-gr1-gr2)**2)-2*chi*N)
#    print("prefactor:", prefactor)
    return prefactor*sq_fit  

#________________________________________________________________________
# Input Section: This contains all the paramaters to adjust.
numberA = 5                         # The number of charged units
totalUnits = 20                     # The total number of units
xFitCutoffMin = 1.00                # Minimum cutoff k for peak fitting
xFitCutoffMax = 2.50                # Maximum cutoff k for peak fitting
xFitCutoffDelta = 0.01              # Change in k for peak fitting
# note: How the previous three bits of code work is that it selects different numbers of points to do the
# peak fitting to, then choses the peak with the lowest mean squared error.

msePeakCutoff = 2.50                # Cutoff k for MSE peak fit (so it only does MSE on the peak)
#________________________________________________________________________

# Retreive file information
# Convert the number of A to string for file recognition
numberAStr=str(numberA)

# Ask for input chi. Note that it is one oom larger for file name purposes, i.e. 01 input gives chi = 0.1
chiInput=input('Enter input chi value (up OOM, use index value):')
fileHeader='./avg_sk_A_20_'+numberAStr+'_2_Chi'+chiInput+'.dat'
print('Using data from file: '+fileHeader)

# Load the data
sq1=np.loadtxt(fileHeader)

# Calculate the avrage values for each x, and drop the zeros
q1=average_sq(40,0.01,sq1 )
#print(np.shape(q1))

#Convert the chi input down an OOM to actual value for calculations
chiInput=(float(chiInput)/10)
chiInputStr=str(chiInput)

# Data manimpulation to fit curve to part of data
new_x = sq1[:,4]
new_y = sq1[:,5]

#### Loop to test different points

#Loop condition- loops through cutting of fit inputs from min to max every delta
loopValues = np.arange(xFitCutoffMin, xFitCutoffMax, xFitCutoffDelta)
#print(loopValues)
mseXValues = q1[:,0] #np.arange(0,0.01*len(q1),0.01)


# adjustment for better fit (only does MSE calculations on peak)
mseXValuesIndex = np.where(mseXValues < msePeakCutoff)
mseXValues = mseXValues[mseXValuesIndex]
mseXValuesIndex = np.shape(mseXValuesIndex)[1]

#empty matrix to store MSE data
MSE_data=[]

#Loop
for j in loopValues:
    fit_points = np.where (new_x < j)
    poptLoop, pcovLoop = curve_fit(func, new_x[fit_points], new_y[fit_points], p0=(0.0001, chiInput), bounds=([ 0, 0], [ 1, 1]))
    fitYValues = func(mseXValues, *poptLoop)
    MSE_data.append(mean_squared_error(q1[0:mseXValuesIndex,1], fitYValues))

# Calculate minimum MSE and take its index
min_index=MSE_data.index(min(MSE_data))
print('Cutoff Value: ' , loopValues[min_index])
print('MSE Value: ', MSE_data[min_index])

#once the minumum is determined, do the fit with that value
fit_points = np.where(new_x < loopValues[min_index])
popt, pcov = curve_fit(func, new_x[fit_points], new_y[fit_points], p0=(0.0001, chiInput), bounds=([ 0, 0], [ 1, 1]))
print("prefactor:", popt[0], "chi:", popt[1])
plt.plot(q1[:,0], q1[:,1], 'o', color='r', markersize=5,mfc='w', label ='$\chi_{input}=$'+chiInputStr)

# Plot the average data
x=np.linspace(0,15,1000)
tag = '$\chi_{fit}$=%f' % (popt[1])
plt.plot(x,func(x,*popt),'k--',label=tag)

#Misc plot stuff
plt.legend(loc = 'best',fontsize=12,frameon=False)
plt.ylabel('S(k)', fontsize=15)
plt.xlabel(r'$k (b^{-1})$', fontsize=15)
plt.show()
