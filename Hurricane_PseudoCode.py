#PSEUDOCODE - BHI

#===========================================================================
# INITIALIZATION:

from pylab import *
import numpy as np
import matplotlib.pyplot as plt


# 1. BHI  by Stahle and Cleaveland (1992) is defined as the normalized SLP 
# difference between Bermuda (40°N, 60°W) and New Orleans (30°N, 90°W).  
# https://journals.ametsoc.org/doi/abs/10.1175/1520-0477%281992%29073%3C1947%3ARAAOSR%3E2.0.CO%3B2

# 2. BHI by Ortegren et al. (2011) is defined as the normalized SLP difference 
# between New Orleans and locations near Bermuda, such as (32.5°N, 65°W) by 
# Katz et al. (2003) and (35°N, 65°W). https://journals.ametsoc.org/doi/10.1175/2010JAMC2566.1

# 3. BHI (or NASH) by Li et al. (2011) is defined as the maximum geopotential 
# height at 850 hPa within the mean area (20°–45°N, 80°–10°W), seasonal mean 
# or monthly mean.  https://journals.ametsoc.org/doi/pdf/10.1175/2010JCLI3829.1

# 4. BHI by Zhu and Liang (2012) is defined as the difference of regional-mean 
# SLP between the Gulf of Mexico (25.3°–29.3°N, 95°–90°W) and the southern Great 
# Plains (35°-39°N, 105.5°-100°W), https://journals.ametsoc.org/doi/full/10.1175/JCLI-D-12-00168.1

# 5. WBHI by Diem (2012) is defined similar to BHI (Stahle and Cleaveland, 1992)
#  but using monthly 850‐hPa heights.  https://rmets.onlinelibrary.wiley.com/doi/10.1002/joc.3421

#====================================================
# LOAD DATA
#====================================================
cd /XXX # change into your directory, where the data are.

PSL=np.load('PSL_LME_ENSMEAN_850-1850.npy')
GH =np.load('Z3_850_LME_ENSMEAN_850-1850.npy')

lats=np.load('CESM_LME_lats.npy')
lons=np.load('CESM_LME_lons.npy')

#time=np.arange(0,2001,1) (change to time in months)


#====================================================
# TO GET THE LAT/LON ranges for each of the methods above:
#====================================================
# Save only U.S. 
latbounds = [ 25 , 55]
lonbounds = [250 , 300] # degrees east ? 

# latitude lower and upper index
latli = np.argmin( np.abs( lats - latbounds[0] ) )
latui = np.argmin( np.abs( lats - latbounds[1] ) ) 

# longitude lower and upper index
lonli = np.argmin( np.abs( lons - lonbounds[0] ) )
lonui = np.argmin( np.abs( lons - lonbounds[1] ) )  

# Air (time, latitude, longitude) 

SUBSET = PSL[ : ,latli:latui , lonli:lonui ]

#====================================================
# WRITE 5 FUNCTIONS, ONE FOR EACH METHOD ABOVE
#====================================================

# 1. BHI  by Stahle and Cleaveland (1992) is defined as the normalized SLP 
# difference between Bermuda (40°N, 60°W) and New Orleans (30°N, 90°W).  

#====================================================

# 2. BHI by Ortegren et al. (2011) is defined as the normalized SLP difference 
# between New Orleans and locations near Bermuda, such as (32.5°N, 65°W) 


#====================================================


# 3. BHI (or NASH) by Li et al. (2011) is defined as the maximum geopotential 
# height at 850 hPa within the mean area (20°–45°N, 80°–10°W), seasonal mean 
# or monthly mean.  


#====================================================

# 4. BHI by Zhu and Liang (2012) is defined as the difference of regional-mean 
# SLP between the Gulf of Mexico (25.3°–29.3°N, 95°–90°W) and the southern Great 
# Plains (35°-39°N, 105.5°-100°W)

#======================================================================
# EXAMPLE OF HOW TO PLOT MAPS
#======================================================================
from mpl_toolkits.basemap import Basemap, shiftgrid
plt.style.use('ggplot')

#======================================================================
levels=np.arange(-1.2,1.22,0.1)
vmin=-0.7
vmax=0.7

# REPLACE WITH LATS/LONS from ABOVE

plats=np.array(pdsi_mn.getLatitude())
plons=np.array(pdsi_mn.getLongitude())
#======================================================================

ax1=plt.subplot(3, 2, 1)
# create Basemap instance for Robinson projection.
#m = Basemap(projection='robin',lon_0=0,resolution='l') #globe
m = Basemap(projection='cyl', llcrnrlon=190, llcrnrlat=10, urcrnrlon=340, urcrnrlat=75)

# draw costlines and coutries
m.drawcoastlines(linewidth=1.5)
m.drawcountries(linewidth=1.5)

# compute the lons and lats to fit the projection
x, y = m(*np.meshgrid(plons, plats))

#ax1 = m.contourf(x,y,MCA_nino_pdsi,31,cmap=plt.cm.BrBG,levels=levels,vmin=vmin,vmax=vmax)
ax1 = m.contourf(x,y,PY_SP_NINO,31,cmap=plt.cm.BrBG,levels=levels,vmin=vmin,vmax=vmax)

cbar = m.colorbar(ax1,location='bottom',pad="20%")
cbar.set_label('SPEI')
#cbar.set_label(r'MCA-LIA $\delta D_{PR}$')
cbar.ax.tick_params(labelsize=12) 
plt.title('A. PHYDA [LIA-MCA] NINO SPEI',fontsize=16)
#======================================================================
#======================================================================

ax1=plt.subplot(3, 2, 2)
# create Basemap instance for Robinson projection.
#m = Basemap(projection='robin',lon_0=0,resolution='l') #globe
m = Basemap(projection='cyl', llcrnrlon=190, llcrnrlat=10, urcrnrlon=340, urcrnrlat=75)

# draw costlines and coutries
m.drawcoastlines(linewidth=1.5)
m.drawcountries(linewidth=1.5)

# compute the lons and lats to fit the projection
x, y = m(*np.meshgrid(plons, plats))

ax1 = m.contourf(x,y,PY_SP_NINA,31,cmap=plt.cm.BrBG,levels=levels,vmin=vmin,vmax=vmax)
cbar = m.colorbar(ax1,location='bottom',pad="20%")
cbar.set_label('SPEI')
#cbar.set_label(r'MCA-LIA $\delta D_{PR}$')
cbar.ax.tick_params(labelsize=12) 
plt.title('B. PHYDA [LIA-MCA] NINA SPEI',fontsize=16)
#======================================================================
#======================================================================

ax1=plt.subplot(3, 2, 3)
# create Basemap instance for Robinson projection.
#m = Basemap(projection='robin',lon_0=0,resolution='l') #globe
m = Basemap(projection='cyl', llcrnrlon=190, llcrnrlat=10, urcrnrlon=340, urcrnrlat=75)

# draw costlines and coutries
m.drawcoastlines(linewidth=1.5)
m.drawcountries(linewidth=1.5)

# compute the lons and lats to fit the projection
x, y = m(*np.meshgrid(plons, plats))

ax1 = m.contourf(x,y,PY_PDSI_NINO,31,cmap=plt.cm.BrBG,levels=levels,vmin=vmin,vmax=vmax)
cbar = m.colorbar(ax1,location='bottom',pad="20%")
cbar.set_label('PDSI')
#cbar.set_label(r'MCA-LIA $\delta D_{PR}$')
cbar.ax.tick_params(labelsize=12) 
plt.title('C. PHYDA [LIA-MCA] NINO PDSI',fontsize=16)
#======================================================================
