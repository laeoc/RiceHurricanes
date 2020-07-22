#!/usr/bin/env python
# coding: utf-8

# In[14]:


from pylab import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys


# In[5]:


#=============================================================================
# LOAD DATA
#=============================================================================

# Sea level pressure 
PSL_850_1850 = np.load('PSL_LME_ENSMEAN_850-1850.npy')
PSL_1850_2005 = np.load('PSL_LME_1850-2005.npy')

# Concatenate the two into one large array
PSL = np.concatenate((PSL_850_1850, PSL_1850_2005))

# Geopotential height
GH_850_1850 = np.load('Z3_850_LME_ENSMEAN_850-1850.npy')
GH_1850_2005 = np.load('Z3_850_LME_1850-2005.npy')

# Concatenate the two into one large array
GH = np.concatenate((GH_850_1850, GH_1850_2005))

# Latitudes and longitudes
lats      = np.load('CESM_LME_lats.npy')
lons      = np.load('CESM_LME_lons.npy')


# In[6]:


#=============================================================================
# CONFINE THE LAT/LON TO ONLY THE U.S.
#=============================================================================

# Lat/lon bounds
latbounds = [25, 50] # 25 - 55
lonbounds = [250 , 340] # 110 - 10 degrees east ? 

# latitude lower and upper index
latli = np.argmin(np.abs(lats - latbounds[0])) # the index of the lats array that is 25
latui = np.argmin(np.abs(lats - latbounds[1])) # the index of the lats array that is 55

# longitude lower and upper indexf
lonli = np.argmin(np.abs(lons - lonbounds[0])) # the index of the lons array that is 250
lonui = np.argmin(np.abs(lons - lonbounds[1])) # the index of the lons array that is 300

lats = lats[latli:latui]
lons = lons[lonli:lonui]
PSL = PSL[:, latli:latui, lonli:lonui]
GH = GH[:, latli:latui, lonli:lonui]


# In[ ]:





# In[7]:


#=============================================================================
# 1. BHI  by Stahle and Cleaveland (1992)
#=============================================================================

diff_stahle = PSL[: , 7, 19] - PSL[: , 1, 3]
pd_stahle = pd.Series(diff_stahle)
np_stahle = (pd_stahle - pd_stahle.mean())/pd_stahle.std()
bhi_stahle = pd.Series.to_numpy(np_stahle)


# In[8]:


#=============================================================================
# 2. BHI by Ortegren et al. (2011)
#=============================================================================

diff_ort = PSL[: , 7, 19] - PSL[: , 3, 17]
ort_seasons = np.reshape(diff_ort, [1156, 12])
ort_jja = np.mean(ort_seasons[:, 5:8], axis=1)

pd_ort = pd.Series(ort_jja)
np_ort = (pd_ort - pd_ort.mean())/pd_ort.std()
bhi_ort = pd.Series.to_numpy(np_ort)


# In[9]:


#=============================================================================
# 3. BHI by Li et al. (2011) 
#=============================================================================


# In[10]:


#=============================================================================
# 4. BHI by Zhu and Liang (2012)
#=============================================================================


# In[20]:


plt.style.use('ggplot')

months = np.arange(850, 2006, 1/12)
years = np.arange(850, 2006, 1)

figure(1)
plt.plot( months, bhi_stahle)
plt.ylabel('BHI index')
plt.title('BHI Stahle 100 Year Average')
plt.xlabel('Year')

figure(2)
plt.plot( years, bhi_ort)
plt.ylabel('BHI index')
plt.title('BHI Ortegren 100 Year Average')
plt.xlabel('Year')


# In[ ]:




