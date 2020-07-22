#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[22]:


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


# In[23]:


#============================================================================#
# 3. BHI (or NASH) by Li et al. (2011) is defined as the maximum geopotential 
# height at 850 hPa within the mean area (20°–45°N, 80°–10°W), seasonal mean 
# or monthly mean. 


## 850 to 1850
BHILi850_subset = SubsetZ3850Mean[ :, 0:9, 11:35]
x = 0
y = 0

months = size(BHILi850_subset, axis = 0)

# empty array to be filled with the longitude values of maximum geopotential 
# height at 850 hPa
maxLat850 = np.zeros(months)

# number of lat steps in area
lat = size(BHILi850_subset, axis = 1)

# while loop that extracts the longitude value of the maximum geopotential 
# height at 850 hPa
while x < months:  # go through months
    BHILi_sheet = BHILi850_subset[x, :, :]
    maxMonth = 0
    y = 0
    while y < lat:
        BHILi_lon = BHILi_sheet[y, :]
        for z in BHILi_lon:
            if z > maxMonth:
                maxMonth = z
                maxLat850[x] = y
        y = y + 1 
    x = x + 1

# saving the raw data before smoothing for appending in tot
maxLat850Tot = maxLat850

# running normalization
bhili=pd.Series(maxLat850)
# normalization
maxLat850 =(bhili-bhili.rolling(window=30).mean())/bhili.rolling(window=30).std()
maxLat850 = pd.Series.to_numpy(maxLat850)
maxLat850 = np.nan_to_num(maxLat850)

# averaging over the yrAvg 
s1 = size(maxLat850, axis = 0)
m  = s1 - mod(s1, (12 * yrAvg))
maxLat850 = maxLat850[0:m]
newTerms = m/(12 * yrAvg)
y  = maxLat850.reshape((12 * yrAvg), int(newTerms))        

tenYrAvgBHILimean = transpose(sum(y, axis = 0) / (12 * yrAvg))   
                              
tenYrAvgBHILimean = tenYrAvgBHILimean.reshape(1, int(1000 / yrAvg))
                                        

## 1850 to 2005
BHILi_subset = SubsetZ3850[ :, 0:9, 11:35]
x = 0
y = 0

months = size(BHILi_subset, axis = 0)

# empty array to be filled with the longitude values of maximum geopotential 
# height at 850 hPa
maxLat = np.zeros(months)

# while loop that extracts the longitude value of the maximum geopotential 
# height at 850 hPa

while x < months:
    BHILi_sheet = BHILi_subset[x, :, :]
    maxMonth = 0
    y = 0
    while y < lat:
        BHILi_lon = BHILi_sheet[y, :]
        for z in BHILi_lon:
            if z > maxMonth:
                maxMonth = z
                maxLat[x] = y
        y = y + 1        
    x = x + 1

# saving the raw data before smoothing for appending in tot
maxLatTot = maxLat

# averaging over the yrAvg 
s1 = size(maxLat, axis = 0)
m  = s1 - mod(s1, (12 * yrAvg))
maxLat = maxLat[0:m]
newTerms = m/(12 * yrAvg)
y = maxLat.reshape((12 * yrAvg), int(newTerms))

tenYrAvgBHILi = transpose(sum(y, axis = 0) / (12 * yrAvg))

tenYrAvgBHILi = tenYrAvgBHILi.reshape(1, int(150 / yrAvg))
             
## 850 to 2005
maxLatTot = pd.Series(maxLatTot)
maxLat850Tot = pd.Series(maxLat850Tot)

maxLatTot = maxLat850Tot.append(maxLatTot)

# running normalization
# normalization
maxLatTot =(maxLatTot-maxLatTot.rolling(window=30).mean())/maxLatTot.rolling(window=30).std()
maxLatTot = pd.Series.to_numpy(maxLatTot)

# fill in nan with 0 s so averaging works
maxLatTot = np.nan_to_num(maxLatTot)

# averaging over the yrAvg 
s1 = size(maxLatTot, axis = 0)
m  = s1 - mod(s1, (12 * yrAvg))
maxLatTot = maxLatTot[0:m]
newTerms = m/(12 * yrAvg)
y  = maxLatTot.reshape((12 * yrAvg), int(newTerms))        

tenYrAvgBHILiTot = transpose(sum(y, axis = 0) / (12 * yrAvg))   
                              
tenYrAvgBHILiTot = tenYrAvgBHILiTot.reshape(1, int(1150 / yrAvg))
             

print(tenYrAvgBHILiTot)


# In[19]:


## plotting 
plt.style.use('ggplot')

years = np.arange(855, 2005, yrAvg)

plt.plot( years, tenYrAvgBHILiTot[0])
plt.ylabel('BHI index')
plt.title('BHI Li')
plt.xlabel('Year')


# In[ ]:




