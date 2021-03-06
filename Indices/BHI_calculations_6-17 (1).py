#!/usr/bin/env python
# coding: utf-8

# In[79]:


from pylab import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


# In[80]:


lats      = np.load('CESM_LME_lats.npy')
lons      = np.load('CESM_LME_lons.npy')
# time = np.arange(1850, 2005, 1)

PSLmean   = np.load('PSL_LME_ENSMEAN_850-1850.npy')
PSL       = np.load('PSL_LME_1850-2005.npy')
Z3850     = np.load('Z3_850_LME_1850-2005.npy')
Z3850Mean = np.load('Z3_850_LME_ENSMEAN_850-1850.npy')


# In[81]:


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


# In[127]:


#====================================================================#
# WRITE 5 FUNCTIONS, ONE FOR EACH METHOD ABOVE
#====================================================================#

# 1. BHI  by Stahle and Cleaveland (1992) is defined as the normalized SLP 
# difference between Bermuda (40°N, 60°W) and New Orleans (30°N, 90°W).

BHIStahle = SubsetPSL[ : , 7, 19] - SubsetPSL[ : , 1, 3]
BHIStahle850 = SubsetPSLmean[ : , 7, 19] - SubsetPSLmean[ : , 1, 3]

print(size(BHIStahle))

# running normalization
bhistahle=pd.Series(BHIStahle)
# normalization
BHIStahle_std=(bhistahle-bhistahle.rolling(window=30).mean())/bhistahle.rolling(window=30).std()

bhistahle850=pd.Series(BHIStahle850)
BHIStahle850_std=(bhistahle850-bhistahle850.rolling(window=30).mean())/bhistahle850.rolling(window=30).std()

BHIStahle_std = pd.Series.to_numpy(BHIStahle_std)
BHIStahle850_std = pd.Series.to_numpy(BHIStahle850_std)

BHIStahle_std = np.nan_to_num(BHIStahle_std)
BHIStahle850_std = np.nan_to_num(BHIStahle850_std)


print(size(BHIStahle_std))
print(size(BHIStahle850_std))


s1 = size(BHIStahle_std, axis = 0)
m  = s1 - mod(s1, 12 * yrAvg)
BHIStahle_std = BHIStahle_std[0:m]
newTerms = m/(12 * yrAvg)
y = BHIStahle_std.reshape(12 * yrAvg, int(newTerms))

tenYrAvgBHIStahle = transpose(sum(y, axis = 0) / (12 * yrAvg))
tenYrAvgBHIStahle = tenYrAvgBHIStahle.reshape(1,int(150/yrAvg))


s1 = size(BHIStahle850_std, axis = 0)
m  = s1 - mod(s1, (12 * yrAvg))
BHIStahle850_std = BHIStahle850_std[0:m]
newTerms = m/(12 * yrAvg)
y  = BHIStahle850_std.reshape((12 * yrAvg), int(newTerms))

tenYrAvgBHIStahle850 = transpose(sum(y, axis = 0) / (12 * yrAvg))
tenYrAvgBHIStahle850 = tenYrAvgBHIStahle850.reshape(1, int(1000/yrAvg))

tenYrAvgBHIStahleCombined = np.concatenate((tenYrAvgBHIStahle850[0], tenYrAvgBHIStahle[0]))

print(tenYrAvgBHIStahleCombined)


# In[129]:


#=============================================================================#
# 2. BHI by Ortegren et al. (2011) is defined as the normalized SLP difference 
# between New Orleans and locations near Bermuda, such as (32.5°N, 65°W) 

BHIOrtegren = SubsetPSL[ : , 7, 19] - SubsetPSL[ : , 3, 17]
BHIOrtegren850 = SubsetPSLmean[ : , 7, 19] - SubsetPSLmean[ : , 3, 17]

BHIOrtegrenSeasons = np.reshape(BHIOrtegren, [156,12])
BHIOrtegrenJJA = BHIOrtegrenSeasons[:, 5:8]
BHIOrtegrenJJAavg = np.average(BHIOrtegrenSeasons, axis = 1)

BHIOrtegren850Seasons = np.reshape(BHIOrtegren850, [1000,12])
BHIOrtegren850JJA = BHIOrtegren850Seasons[:, 5:8]
BHIOrtegren850JJAavg = np.average(BHIOrtegren850Seasons, axis = 1)


# running normalization
bhiortegren=pd.Series(BHIOrtegrenJJAavg)
# normalization
BHIOrtegren_std=(bhiortegren-bhiortegren.rolling(window=30).mean())/bhiortegren.rolling(window=30).std()

BHIOrtegrenJJAavg_STD = pd.Series.to_numpy(BHIOrtegren_std)

BHIOrtegrenJJAavg_STD = np.nan_to_num(BHIOrtegrenJJAavg_STD)

s1 = size(BHIOrtegrenJJAavg_STD, axis = 0)
m  = s1 - mod(s1, (yrAvg))
BHIOrtegrenJJAavg_STD = BHIOrtegrenJJAavg_STD[0:m]
newTerms = m/(yrAvg)
y = BHIOrtegrenJJAavg_STD.reshape((yrAvg), int(newTerms))

tenYrAvgBHIOrtegren = transpose(sum(y, axis = 0) / (yrAvg))
tenYrAvgBHIOrtegren = tenYrAvgBHIOrtegren.reshape(1,int(150/yrAvg))


# running normalization
bhiortegren=pd.Series(BHIOrtegren850JJAavg)
# normalization
BHIOrtegren_std=(bhiortegren-bhiortegren.rolling(window=30).mean())/bhiortegren.rolling(window=30).std()

BHIOrtegren850JJAavg_STD = pd.Series.to_numpy(BHIOrtegren_std)

BHIOrtegren850JJAavg_STD = np.nan_to_num(BHIOrtegren850JJAavg_STD)


s1 = size(BHIOrtegren850JJAavg_STD, axis = 0)
m  = s1 - mod(s1, (yrAvg))
BHIOrtegren850JJAavg_STD = BHIOrtegren850JJAavg_STD[0:m]
newTerms = m/(yrAvg)
y  = BHIOrtegren850JJAavg_STD.reshape((yrAvg), int(newTerms))


tenYrAvgBHIOrtegren850 = transpose(sum(y, axis = 0) / (yrAvg))
tenYrAvgBHIOrtegren850 = tenYrAvgBHIOrtegren850.reshape(1, int(1000 / yrAvg))

tenYrAvgBHIOrtegrenCombined = np.concatenate((tenYrAvgBHIOrtegren850[0], tenYrAvgBHIOrtegren[0]))





# In[98]:


#============================================================================#
# 3. BHI (or NASH) by Li et al. (2011) is defined as the maximum geopotential 
# height at 850 hPa within the mean area (20°–45°N, 80°–10°W), seasonal mean 
# or monthly mean. 


BHILi_subset = SubsetZ3850[ :, 0:9, 11:35]
x = 0
y = 0
maxLat = np.zeros(1873)

years = size(BHILi_subset, axis = 0)
lat = size(BHILi_subset, axis = 1)

while x < years:
    BHILi_lat = BHILi_subset[x, :, :]
    maxMonth = 0
    y = 0
    x = x + 1
    while y < lat:
        BHILi_lon = BHILi_lat[y, :]
        for z in BHILi_lon:
            if z > maxMonth:
                maxMonth = z
                maxLat[x] = y
        y = y + 1        
        
                
s1 = size(maxLat, axis = 0)
m  = s1 - mod(s1, (12 * yrAvg))
maxLat = maxLat[0:m]
newTerms = m/(12 * yrAvg)
y = maxLat.reshape((12 * yrAvg), int(newTerms))

tenYrAvgBHILi = transpose(sum(y, axis = 0) / (12 * yrAvg))

tenYrAvgBHILi = tenYrAvgBHILi.reshape(1, int(150 / yrAvg))


BHILi_subset = SubsetZ3850Mean[ :, 0:9, 11:35]
x = 0
y = 0
maxLat = np.zeros(12001)

years = size(BHILi_subset, axis = 0)
lat = size(BHILi_subset, axis = 1)

while x < years:
    BHILi_lat = BHILi_subset[x, :, :]
    maxMonth = 0
    y = 0
    x = x + 1
    while y < lat:
        BHILi_lon = BHILi_lat[y, :]
        for z in BHILi_lon:
            if z > maxMonth:
                maxMonth = z
                maxLat[x] = y
        y = y + 1        
        
s1 = size(maxLat, axis = 0)
m  = s1 - mod(s1, (12 * yrAvg))
maxLat = maxLat[0:m]
newTerms = m/(12 * yrAvg)
y  = maxLat.reshape((12 * yrAvg), int(newTerms))        

tenYrAvgBHILimean = transpose(sum(y, axis = 0) / (12 * yrAvg))   
                              
tenYrAvgBHILimean = tenYrAvgBHILimean.reshape(1, int(1000 / yrAvg))
                                        
tenYrAvgBHILi = np.concatenate((tenYrAvgBHILimean[0], tenYrAvgBHILi[0]))

# running normalization
bhili=pd.Series(tenYrAvgBHILi)
# normalization
BHILi_std=(bhili-bhili.rolling(window=30).mean())/bhili.rolling(window=30).std()

BHILi_std = pd.Series.to_numpy(BHILi_std)

print(BHILi_std)


# In[47]:


#====================================================
# 4. BHI by Zhu and Liang (2012) The BHI is defined as 
# the difference of regional-mean SLP between the Gulf 
# of Mexico  (25.3°–29.3°N, 95°–90°W) and the southern
# Great Plains (35°–39°N, 105.5°–100°W)

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

tenYrAvgBHIZhu = np.concatenate((tenYrAvgBHIZhuMean[0], tenYrAvgBHIZhu[0]))

# running normalization
bhizhu=pd.Series(tenYrAvgBHIZhu)
# normalization
BHIZhu_std=(bhizhu-bhizhu.rolling(window=30).mean())/bhizhu.rolling(window=30).std()

BHIZhu_std = pd.Series.to_numpy(BHIZhu_std)


# In[130]:


plt.style.use('ggplot')

print(tenYrAvgBHIStahle)
print(tenYrAvgBHIOrtegren)
print(tenYrAvgBHILi)
print(tenYrAvgBHILi)


years = np.arange(855, 2005, yrAvg)
print(size(years))

print(size(tenYrAvgBHIStahleCombined, axis = 0))
figure(1)
plt.plot( years, tenYrAvgBHIStahleCombined)
plt.ylabel('BHI index')
plt.title('BHI Stahle')
plt.xlabel('Year')

#plt.subplot(412)
figure(2)
plt.plot( years, tenYrAvgBHIOrtegrenCombined)
plt.ylabel('BHI index')
plt.title('BHI Ortegren')
plt.xlabel('Year')

figure(3)
plt.plot( years, BHILi_std)
plt.ylabel('BHI index')
plt.title('BHI Li')
plt.xlabel('Year')

figure(4)
plt.plot( years, BHIZhu_std)
plt.ylabel('BHI index')
plt.title('BHI Zhu')
plt.xlabel('Year')


# In[106]:


BHIStahleSTD = np.std(tenYrAvgBHIStahle,axis = 0)
print(BHIStahleSTD)


# In[107]:


t2, p2 = stats.ttest_ind(tenYrAvgBHIStahle[0], tenYrAvgBHIStahle850[0] ,equal_var=False)
print("t = " + str(t2))
print("p = " + str(p2))


# In[109]:

plt.style.use('ggplot')
#======================================================================
fig=plt.figure()
#======================================================================
figure(1)

distribution1 = tenYrAvgBHIStahle[0]
distribution2 = tenYrAvgBHIStahle850[0]
distribution3 = tenYrAvgBHIOrtegren[0]
distribution4 = tenYrAvgBHIOrtegren850[0]


#distribution5 = tenYrAvgBHILi
#distribution6 = tenYrAvgBHILimean
#distribution7 = tenYrAvgBHIZhu[0]
#distribution8 = tenYrAvgBHIZhuMean[0]


print(distribution1)

sns.kdeplot(distribution1, shade=True, linewidth=3, label = "Stahle 1850-2005")

sns.kdeplot(distribution2, shade=True, linewidth=3, label = "Stahle 850-1850")
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

sns.kdeplot(distribution3,shade=True, linewidth=3, label = "Ortegren 1850-2005")

sns.kdeplot(distribution4, shade=True, linewidth=3, label = "Ortegren 850-1850")
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

#fig.legend(loc='upper center',fancybox=True, shadow=True, ncol=3)
plt.show()
