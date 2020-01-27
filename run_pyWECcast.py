import pyWECcast as wc 

buoyFile = 'noaa_Buoy_info.csv'
#buoys = wc.buoys.download(buoyFile)

modelRegions = ['wc_10m']
models = wc.models.download(modelRegions)
