import pyWECcast as wc 

buoyFile = 'noaa_Buoy_info.csv'
#buoys = wc.buoys.download(buoyFile)

modelRegions = ['wc_10m']
#models = wc.models.download(modelRegions)
#wc.models.processH5('t00z',search_host=False)

db = f'./model_downloads.h5'
ww = wc.WW3(db)
ww.get_location(1,2)
