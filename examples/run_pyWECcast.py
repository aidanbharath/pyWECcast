import pyWECcast as wc 

#buoyFile = 'noaa_Buoy_info.csv'
#buoys = wc.buoys.download(buoyFile)

#modelRegions = ['wc_4m','ak_4m']
#models = wc.models.download(modelRegions,processH5s=True)
#wc.models.processH5('t00z',search_host=False)

locationFile = 'fairweather_grounds.h5'
resultDB = 'fairweather_power_forecast.h5'

db = f'./ak-wc-4m.h5'
ww = wc.WW3(db,360-137.9,58.3,model_id='ak_4m')
ww.get_location()
ww.get_times('2020-03-20 00:00:00','2020-03-20 04:00:00')
ww.reduce_db(locationFile)

wecSim = f'../data/WECSim_dataset_RM3_scale_0-1873.hdf5'
#mf = wc.power.read_forecast(locationFile)
#wc.power.read_buoys('buoy_downloads.h5')

wc.power.forecast_powerseries(locationFile,wecSim,resultDB=resultDB)
