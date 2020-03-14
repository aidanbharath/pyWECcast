import pyWECcast as wc 

buoyFile = 'noaa_Buoy_info.csv'
#buoys = wc.buoys.download(buoyFile)

modelRegions = ['wc_10m']
#models = wc.models.download(modelRegions)
#wc.models.processH5('t00z',search_host=False)

db = f'./model_downloads.h5'
ww = wc.WW3(db,235,46)
ww.get_location()
ww.get_times('2020-03-11','2020-03-12')
ww.reduce_db('test.h5')
