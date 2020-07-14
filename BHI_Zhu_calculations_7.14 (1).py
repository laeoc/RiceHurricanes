#!/usr/bin/env python
# coding: utf-8

# In[8]:


from pylab import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


# In[9]:


lats      = np.load('CESM_LME_lats.npy')
lons      = np.load('CESM_LME_lons.npy')
# time = np.arange(1850, 2005, 1)

PSLmean   = np.load('PSL_LME_ENSMEAN_850-1850.npy')
PSL       = np.load('PSL_LME_1850-2005.npy')
Z3850     = np.load('Z3_850_LME_1850-2005.npy')
Z3850Mean = np.load('Z3_850_LME_ENSMEAN_850-1850.npy')


# In[10]:


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


yrAvg = 10  # for averaging 10 = average of ten years


# In[11]:


#====================================================
# 4. BHI by Zhu and Liang (2012) The BHI is defined as 
# the difference of regional-mean SLP between the Gulf 
# of Mexico  (25.3°–29.3°N, 95°–90°W) and the southern
# Great Plains (35°–39°N, 105.5°–100°W)

GulfMex  = np.average(SubsetPSL[ :, 0:1, 5:7], axis = (1,2))
GreatPla = np.average(SubsetPSL[ :, 4:6, 1:3], axis = (1,2))
BHIZhu   = GulfMex - GreatPla
print(size(BHIZhu))

GulfMex850  = np.average(SubsetPSLmean[ :, 0:1, 5:7], axis = (1,2))
GreatPla850 = np.average(SubsetPSLmean[ :, 4:6, 1:3], axis = (1,2))
BHIZhu850   = GulfMex850 - GreatPla850

print(size(BHIZhu850))

## 850 to 2005
BHIZhuTot = pd.Series(BHIZhu)
BHIZhu850Tot = pd.Series(BHIZhu850)

# combine both series so it becomes 850 to 2005
BHIZhuTot = BHIZhu850Tot.append(BHIZhuTot)


# normalization
BHIZhuTot=(BHIZhuTot-BHIZhuTot.rolling(window=30).mean())/BHIZhuTot.rolling(window=30).std()

BHIZhuTot = pd.Series.to_numpy(BHIZhuTot)
BHIZhuTot = np.nan_to_num(BHIZhuTot)

# average every yrAvg years
yrAvg = 10
s1 = size(BHIZhuTot, axis = 0)
m  = s1 - mod(s1, (12 * yrAvg))
BHIZhuTot = BHIZhuTot[0:m]
newTerms = m/(12 * yrAvg)
y = BHIZhuTot.reshape((12 * yrAvg), int(newTerms))

tenYrAvgBHIZhu = transpose(sum(y, axis = 0) / (12 * yrAvg))
tenYrAvgBHIZhu = tenYrAvgBHIZhu.reshape(1,int(1150/yrAvg))


# In[12]:


## plotting
plt.style.use('ggplot')

years = np.arange(855, 2005, yrAvg)

plt.plot( years, tenYrAvgBHIZhuCombined)
plt.ylabel('BHI index')
plt.title('BHI Zhu')
plt.xlabel('Year')


# In[ ]:




