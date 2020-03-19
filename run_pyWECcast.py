import pyWECcast as wc 

buoyFile = 'noaa_Buoy_info.csv'
#buoys = wc.buoys.download(buoyFile)

modelRegions = ['wc_10m']
#models = wc.models.download(modelRegions)
#wc.models.processH5('t00z',search_host=False)

db = f'./model_downloads.h5'
#ww = wc.WW3(db,235,46)
#ww.get_location()
#ww.get_times('2020-03-11','2020-03-12')
#ww.reduce_db('random_forecast_slice.h5')


cutDB = f'random_forecast_slice.h5'
wecSim = f'WECSim_dataset_RM3_scale_0-1873.hdf5'
mf = wc.power.read_forecast(cutDB)
#wc.power.read_buoys('buoy_downloads.h5')

wc.power.calculate_powerseries(cutDB,wecSim)
