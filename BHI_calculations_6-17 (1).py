#!/usr/bin/env python
# coding: utf-8

# In[133]:


from pylab import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


# In[2]:


lats      = np.load('CESM_LME_lats.npy')
lons      = np.load('CESM_LME_lons.npy')
# time = np.arange(1850, 2005, 1)

PSLmean   = np.load('PSL_LME_ENSMEAN_850-1850.npy')
PSL       = np.load('PSL_LME_1850-2005.npy')
Z3850     = np.load('Z3_850_LME_1850-2005.npy')
Z3850Mean = np.load('Z3_850_LME_ENSMEAN_850-1850.npy')


# In[204]:


# Save only U.S. 
latbounds = [ 25, 50] # 25 - 55
lonbounds = [250 , 340] # 110 - 10 degrees east ? 

# latitude lower and upper index
latli = np.argmin( np.abs( lats - latbounds[0] ) ) # the index of the lats array that is 25
latui = np.argmin( np.abs( lats - latbounds[1] ) ) # the index of the lats array that is 55

# longitude lower and upper indexf
lonli = np.argmin( np.abs( lons - lonbounds[0] ) ) # the index of the lons array that is 250
lonui = np.argmin( np.abs( lons - lonbounds[1] ) ) # the index of the lons array that is 300


SubsetPSLmean = PSLmean[ : ,latli:latui , lonli:lonui ]
SubsetPSL = PSL[ : ,latli:latui , lonli:lonui ]
SubsetZ3850Mean = Z3850Mean[ : ,latli:latui , lonli:lonui ]
SubsetZ3850 = Z3850[ : ,latli:latui , lonli:lonui ]

latrange = latui - latli
lonrange = lonui - lonli

yrAvg = 10

print(latrange)
print(lonrange)


# In[205]:


#====================================================================#
# WRITE 5 FUNCTIONS, ONE FOR EACH METHOD ABOVE
#====================================================================#

# 1. BHI  by Stahle and Cleaveland (1992) is defined as the normalized SLP 
# difference between Bermuda (40°N, 60°W) and New Orleans (30°N, 90°W).

BHIStahle = SubsetPSL[ : , 7, 19] - SubsetPSL[ : , 1, 3]
BHIStahle850 = SubsetPSLmean[ : , 7, 19] - SubsetPSLmean[ : , 1, 3]

print(BHIStahle850)

s1 = size(BHIStahle, axis = 0)
m  = s1 - mod(s1, 12 * yrAvg)
BHIStahle = BHIStahle[0:m]
newTerms = m/(12 * yrAvg)
y = BHIStahle.reshape(12 * yrAvg, int(newTerms))

tenYrAvgBHIStahle = transpose(sum(y, axis = 0) / (12 * yrAvg))
tenYrAvgBHIStahle = tenYrAvgBHIStahle.reshape(1,int(150/yrAvg))


s1 = size(BHIStahle850, axis = 0)
m  = s1 - mod(s1, (12 * yrAvg))
BHIStahle850 = BHIStahle850[0:m]
newTerms = m/(12 * yrAvg)
y  = BHIStahle850.reshape((12 * yrAvg), int(newTerms))

tenYrAvgBHIStahle850 = transpose(sum(y, axis = 0) / (12 * yrAvg))
tenYrAvgBHIStahle850 = tenYrAvgBHIStahle850.reshape(1, int(1000/yrAvg))

tenYrAvgBHIStahle = np.concatenate((tenYrAvgBHIStahle850[0], tenYrAvgBHIStahle[0]))

print(tenYrAvgBHIStahle)


# In[183]:


#=============================================================================#
# 2. BHI by Ortegren et al. (2011) is defined as the normalized SLP difference 
# between New Orleans and locations near Bermuda, such as (32.5°N, 65°W) 

BHIOrtegren = SubsetPSL[ : , 7, 19] - SubsetPSL[ : , 3, 17]
BHIOrtegren850 = SubsetPSLmean[ : , 7, 19] - SubsetPSLmean[ : , 3, 17]

s1 = size(BHIOrtegren, axis = 0)
m  = s1 - mod(s1, (12 * yrAvg))
BHIOrtegren = BHIOrtegren[0:m]
newTerms = m/(12 * yrAvg)
y = BHIOrtegren.reshape((12 * yrAvg), int(newTerms))

tenYrAvgBHIOrtegren = transpose(sum(y, axis = 0) / (12 * yrAvg))
tenYrAvgBHIOrtegren = tenYrAvgBHIOrtegren.reshape(1,int(150/yrAvg))

s1 = size(BHIOrtegren850, axis = 0)
m  = s1 - mod(s1, (12 * yrAvg))
BHIOrtegren850 = BHIOrtegren850[0:m]
newTerms = m/(12 * yrAvg)
y  = BHIOrtegren850.reshape((12 * yrAvg), int(newTerms))


tenYrAvgBHIOrtegren850 = transpose(sum(y, axis = 0) / (12 * yrAvg))
tenYrAvgBHIOrtegren850 = tenYrAvgBHIOrtegren850.reshape(1, int(1000 / yrAvg))

# tenYrAvgBHIOrtegren = np.concatenate((tenYrAvgBHIOrtegren850[0], tenYrAvgBHIOrtegren[0]))


# In[184]:


#============================================================================#
# 3. BHI (or NASH) by Li et al. (2011) is defined as the maximum geopotential 
# height at 850 hPa within the mean area (20°–45°N, 80°–10°W), seasonal mean 
# or monthly mean. 


BHILi = np.amax(SubsetZ3850[ :, 0:9, 11:35], axis = (1,2))
BHILimean = np.amax(SubsetZ3850Mean[ :, 0:9, 11:35], axis = (1,2))

s1 = size(BHILi, axis = 0)
m  = s1 - mod(s1, (12 * yrAvg))
BHILi = BHILi[0:m]
newTerms = m/(12 * yrAvg)
y = BHILi.reshape((12 * yrAvg), int(newTerms))

tenYrAvgBHILi = transpose(sum(y, axis = 0) / (12 * yrAvg))
tenYrAvgBHILi = tenYrAvgBHILi.reshape(1, int(150 / yrAvg))


s1 = size(BHILimean, axis = 0)
m  = s1 - mod(s1, (12 * yrAvg))
BHILimean = BHILimean[0:m]
newTerms = m/(12 * yrAvg)
y  = BHILimean.reshape((12 * yrAvg), int(newTerms))


tenYrAvgBHILimean = transpose(sum(y, axis = 0) / (12 * yrAvg))
tenYrAvgBHILimean = tenYrAvgBHILimean.reshape(1, int(1000 / yrAvg))

# tenYrAvgBHILi = np.concatenate((tenYrAvgBHILimean[0], tenYrAvgBHILi[0]))


# In[185]:


#====================================================
# 4. BHI by Zhu and Liang (2012) is defined as the difference of regional-mean 
# SLP between the Gulf of Mexico (25.3°–29.3°N, 95°–90°W) and the southern Great 
# Plains (35°-39°N, 105.5°-100°W)

GulfMex  = np.average(SubsetPSL[ :, 0:1, 5:7], axis = (1,2))
GreatPla = np.average(SubsetPSL[ :, 4:6, 1:3], axis = (1,2))
BHIZhu   = GulfMex - GreatPla

GulfMexMean  = np.average(SubsetPSLmean[ :, 0:1, 5:7], axis = (1,2))
GreatPlaMean = np.average(SubsetPSLmean[ :, 4:6, 1:3], axis = (1,2))
BHIZhuMean   = GulfMexMean - GreatPlaMean


s1 = size(BHIZhu, axis = 0)
m  = s1 - mod(s1, (12 * yrAvg))
BHIZhu = BHIZhu[0:m]
newTerms = m/(12 * yrAvg)
y = BHIZhu.reshape((12 * yrAvg), int(newTerms))

tenYrAvgBHIZhu = transpose(sum(y, axis = 0) / (12 * yrAvg))
tenYrAvgBHIZhu = tenYrAvgBHIZhu.reshape(1,int(150/yrAvg))


s1 = size(BHIZhuMean, axis = 0)
m  = s1 - mod(s1, (12 * yrAvg))
BHIZhuMean = BHIZhuMean[0:m]
newTerms = m/(12 * yrAvg)
y  = BHIZhuMean.reshape((12 * yrAvg), int(newTerms))


tenYrAvgBHIZhuMean = transpose(sum(y, axis = 0) / (12 * yrAvg))
tenYrAvgBHIZhuMean = tenYrAvgBHIZhuMean.reshape(1, int(1000/yrAvg))

# tenYrAvgBHIZhu = np.concatenate((tenYrAvgBHIZhuMean[0], tenYrAvgBHIZhu[0]))


# In[180]:


plt.style.use('ggplot')

print(tenYrAvgBHIStahle)
print(tenYrAvgBHIOrtegren)
print(tenYrAvgBHILi)
print(tenYrAvgBHILi)


y = np.arange(0,int(1150/yrAvg))

figure(1)
plt.plot( y, tenYrAvgBHIStahle)
plt.ylabel('BHI index')
plt.title('BHI Stahle')
plt.xlabel('Years since 850 in 50 year increments')

#plt.subplot(412)
figure(2)
plt.plot( y, tenYrAvgBHIOrtegren)
plt.ylabel('BHI index')
plt.title('BHI Ortegren')
plt.xlabel('Years since 850 in 50 year increments')

figure(3)
plt.plot( y, tenYrAvgBHILi)
plt.ylabel('BHI index')
plt.title('BHI Li')
plt.xlabel('Years since 850 in 50 year increments')

figure(4)
plt.plot( y, tenYrAvgBHIZhu)
plt.ylabel('BHI index')
plt.title('BHI Zhu')
plt.xlabel('Years since 850 in 50 year increments')


# In[89]:


BHIStahleSTD = np.std(tenYrAvgBHIStahle,axis = 0)
print(BHIStahleSTD)


# In[166]:


t2, p2 = stats.ttest_ind(tenYrAvgBHIStahle[0], tenYrAvgBHIStahle850[0] ,equal_var=False)
print("t = " + str(t2))
print("p = " + str(p2))


# In[191]:


plt.style.use('ggplot')
#======================================================================
fig=plt.figure()
#======================================================================
figure(1)

distribution1 = tenYrAvgBHIStahle[0]
distribution2 = tenYrAvgBHIStahle850[0]
distribution3 = tenYrAvgBHIOrtegren[0]
distribution4 = tenYrAvgBHIOrtegren850[0]
distribution5 = tenYrAvgBHILi[0]
distribution6 = tenYrAvgBHILimean[0]
distribution7 = tenYrAvgBHIZhu[0]
distribution8 = tenYrAvgBHIZhuMean[0]


print(distribution1)

sns.kdeplot(distribution1, shade=True, linewidth=3) #label = "Stahle"

sns.kdeplot(distribution2, shade=True, linewidth=3) #label = "Stahle850"
plt.xlabel('BHI index')
plt.ylabel('Probability')
plt.title('BHI Stahle')

# plt.axvline(0.31, 0,20,color='Navy',label='LIA NINO3 Full Ensemble Mean')
# plt.axvline(0.39, 0,20,color='DodgerBlue',label='LIA NINO3 Mean of MC Realizations')

# plt.axvline(0.32, 0,20,color='Coral',label='MCA NINO3 Full Ensemble Mean')
# plt.axvline(0.36, 0,20,color='Orange',label='MCA NINO3 Mean of MC Realizations')

    #plt.title('A. NINO3',fontsize=20)
    #plt.xlabel('NINO3 1-$\sigma$ spread',fontsize=20)
    #plt.ylabel('Density',fontsize=20)
    #plt.xticks(fontsize=20)
    #plt.yticks(fontsize=20)
#plt.legend(fontsize=14)

# plt.xlim([-0.9,0.9])
    #plt.ylim([-0.1,22])
#plt.legend()
#======================================================================

figure(2)

sns.kdeplot(distribution3,shade=True, linewidth=3)#, label = "MCA")

sns.kdeplot(distribution4, shade=True, linewidth=3)#, label = "LIA")
plt.xlabel('BHI index')
plt.ylabel('Probability')
plt.title('BHI Ortegren')


# sns.distplot(sigma_full_ens, hist = True, kde = True,
#                  kde_kws = {'shade': True, 'linewidth': 3}, 
#                   label = "Last2k")

# In [20]: MCA_amplitude
# Out[20]: 0.35166755

# In [21]: LIA_amplitude
# Out[21]: 0.32226029

# plt.axvline(0.32, 0,20,color='Navy')#,label='LIA NINO3.4 Full Ensemble Mean')
# plt.axvline(0.38, 0,20,color='DodgerBlue',linewidth=3.0)#,label='LIA NINO3.4 Mean of MC Realizations')

# plt.axvline(0.35, 0,20)#,color='Coral',label='MCA NINO3.4 Full Ensemble Mean')
# plt.axvline(0.38, 0,20)#,color='Orange',label='MCA NINO3.4 Mean of MC Realizations')

    #plt.title('B. NINO3.4',fontsize=20)
    #plt.xlabel('NINO3.4 1-$\sigma$ spread',fontsize=20)
    #plt.ylabel('Density',fontsize=20)
# plt.xlim([-0.9,0.9])
    # plt.ylim([-0.1,22])
#plt.legend()
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    #plt.legend(fontsize=14)

fig.legend(loc='upper center',
          fancybox=True, shadow=True, ncol=3)


figure(3)

sns.kdeplot(distribution5,shade=True, linewidth=3)#, label = "MCA")

sns.kdeplot(distribution6, shade=True, linewidth=3)#, label = "LIA")
plt.xlabel('BHI index')
plt.ylabel('Probability')
plt.title('BHI Li')


figure(4)

sns.kdeplot(distribution7,shade=True, linewidth=3)#, label = "MCA")

sns.kdeplot(distribution8, shade=True, linewidth=3)#, label = "LIA")

plt.xlabel('BHI index')
plt.ylabel('Probability')
plt.title('BHI Zhu')

#======================================================================
plt.show()


# In[ ]:





# In[ ]:




