from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
#np.set_printoptions(threshold=sys.maxsize)

#=============================================================================
# LOAD DATA
#=============================================================================

# Sea level pressure 
PSL_850_1850 = np.load('../Data/PSL_LME_ENSMEAN_850-1850.npy')
PSL_1850_2005 = np.load('../Data/PSL_LME_1850-2005.npy')

# Concatenate the two into one large array
PSL = np.concatenate((PSL_850_1850, PSL_1850_2005))

# Geopotential height
GH_850_1850 = np.load('../Data/Z3_850_LME_ENSMEAN_850-1850.npy')
GH_1850_2005 = np.load('../Data/Z3_850_LME_1850-2005.npy')

# Concatenate the two into one large array
GH = np.concatenate((GH_850_1850, GH_1850_2005))

# Latitudes and longitudes
lats = np.load('../Data/CESM_LME_lats.npy')
lons = np.load('../Data/CESM_LME_lons.npy')

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

try_new = PSL_850_1850[:, latli:latui, lonli:lonui]

#=============================================================================
# 1. BHI  by Stahle and Cleaveland (1992)
#=============================================================================

diff_stahle = PSL[: , 7, 19] - PSL[: , 1, 3]
pd_stahle = pd.Series(diff_stahle)
np_stahle = (pd_stahle - pd_stahle.mean())/pd_stahle.std()
bhi_stahle = pd.Series.to_numpy(np_stahle)

stah_sigma = np.std(bhi_stahle)
stah_pos = np.where(bhi_stahle > stah_sigma)
stah_neg = np.where(bhi_stahle < -stah_sigma)

#=============================================================================
# 2. BHI by Ortegren et al. (2011)
#=============================================================================

diff_ort = PSL[: , 7, 19] - PSL[: , 3, 17]
ort_seasons = np.reshape(diff_ort, [1156, 12])
ort_jja = np.mean(ort_seasons[:, 5:8], axis=1)

pd_ort = pd.Series(ort_jja)
np_ort = (pd_ort - pd_ort.mean())/pd_ort.std()
bhi_ort = pd.Series.to_numpy(np_ort)

ort_sigma = np.std(bhi_ort)
ort_pos = np.where(bhi_ort > bhi_ort)
ort_neg = np.where(bhi_ort < -bhi_ort)

#=============================================================================
# 3. BHI by Li et al. (2011) (CHANGE) !!!!!!  (COMPLETELY WRONG)
#=============================================================================

diff_li = PSL[: , 9, 21] - PSL[: , 3, 5]
pd_li = pd.Series(diff_li)
np_li = (pd_li - pd_li.mean())/pd_li.std()
bhi_li = pd.Series.to_numpy(np_li)

li_sigma = np.std(bhi_li)
li_pos = np.where(bhi_li > li_sigma)
li_neg = np.where(bhi_li < -li_sigma)

#=============================================================================
# 4. BHI by Zhu and Liang (2012)
#=============================================================================

gulf_mex  = np.mean(PSL[ :, 0:1, 5:7], axis = (1,2))
great_pla = np.mean(PSL[ :, 4:6, 1:3], axis = (1,2))
diff_zhu   = gulf_mex - great_pla
pd_zhu = pd.Series(diff_zhu)
np_zhu = (pd_zhu - pd_zhu.mean())/pd_zhu.std()
bhi_zhu = pd.Series.to_numpy(np_zhu)

zhu_sigma = np.std(bhi_zhu)
zhu_pos = np.where(bhi_zhu > zhu_sigma)
zhu_neg = np.where(bhi_zhu < -zhu_sigma)

#=============================================================================
# EXTRACT SLP FOR N. ATLANTIC/GULF OF MEXICO AREA
# N. ATLANTIC: 35°N, 320°E
# GULF OF MEXICO: 25°N, 265°E
#=============================================================================

# lat_bounds = [25, 35]
# lon_bounds = [265, 320]

# longitude lower and upper index
# latli = np.argmin(np.abs(lats - lat_bounds[0]))
# latui = np.argmin(np.abs(lats - lat_bounds[1])) 
# longitude lower and upper index
# lonli = np.argmin(np.abs(lons - lon_bounds[0]))
# lonui = np.argmin(np.abs(lons - lon_bounds[1]))  

# slp_region = PSL[:, latli:latui, lonli:lonui]
slp_pos_stah = PSL[stah_pos[0], :, :]
slp_pos_ort = PSL[ort_pos[0], :, :]
slp_pos_li = PSL[li_pos[0], :, :]
slp_pos_zhu = PSL[zhu_pos[0], :, :]
slp_neg_stah = PSL[stah_neg[0], :, :]
slp_neg_ort = PSL[ort_neg[0], :, :]
slp_neg_li = PSL[li_neg[0], :, :]
slp_neg_zhu = PSL[zhu_neg[0], :, :]

#=============================================================================
# FUNCTION TO PLOT THE COMPOSITE MAP FOR EACH SLP
#=============================================================================

# If positive = 0, bhi
def slp_composite(bhi, title):
    lon, lat = np.meshgrid(lons, lats) 
    m = Basemap(projection='mill',llcrnrlat=25.5,urcrnrlat=48, llcrnrlon=260, \
                urcrnrlon=320, resolution='c')
    x, y = m(lon, lat)
    plt.figure(figsize=(5,3))
    plt.title(title, pad=25)
    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,10.), labels=[0,1,0,0])
    m.drawmeridians(np.arange(-180.,81.,15.), labels=[0,0,1,0])
    m.drawmapboundary(fill_color='white')
    m.contourf(x,y,(bhi.mean(axis=0)-PSL.mean(axis=0)), cmap='PuOr', extend='neither')
    cb = plt.colorbar(orientation='horizontal', pad=0.05)
    cb.set_label('Sea Level Pressure (hPa)')
    plt.show()

slp_composite(slp_pos_stah, 'SLP During Positive BHI (Stahle)')
slp_composite(slp_pos_ort, 'SLP During Positive BHI (Ortegren)')
slp_composite(slp_pos_li, 'SLP During Positive BHI (Li)')
slp_composite(slp_pos_zhu, 'SLP During Positive BHI (Zhu)')
slp_composite(slp_neg_stah, 'SLP During Negative BHI (Stahle)')
slp_composite(slp_neg_ort, 'SLP During Negative BHI (Ortegren)')
slp_composite(slp_neg_li, 'SLP During Negative BHI (Li)')
slp_composite(slp_neg_zhu, 'SLP During Negative BHI (Zhu)')