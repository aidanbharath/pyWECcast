
import os
import re
import glob
import ftplib
import urllib
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import shutil
import urllib.request as request
from contextlib import closing
from tqdm import tqdm

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


host = 'ftp.ncep.noaa.gov'
dataDir =  'pub/data/nccf/com/wave/prod'

def get_buoy_data(buoys,buoySave):
    try:
        with closing(request.urlopen(buoys)) as buoy:
            savedFile = f'{buoySave}'
            with open(savedFile, 'wb') as F:
                shutil.copyfileobj(buoy, F)
            buoy = pd.read_csv(savedFile,delim_whitespace=True)
            data = buoy.iloc[1::,5::].values
            cols = buoy.columns[5::]
            b = buoy.iloc[1::,0:5] 
            Y,M,D,h,m = b['#YY'],b['MM'],b['DD'],b['hh'],b['mm']
            times = [pd.to_datetime(f"{Y[i]}-{M[i]}-{D[i]}T{h[i]}:{m[i]}:00") 
                        for i in Y.index]
            buoy = pd.DataFrame(data,index=pd.to_datetime(times),columns=cols)
            buoy = buoy.iloc[::-1]  
        buoy.to_pickle(savedFile)
    except urllib.error.HTTPError as e:
        print(f'{buoys} -- was not found')


def retrieve_model_files(host,dataDir):

    try:
        ftp = ftplib.FTP(host)
        ftp.login()
        ftp.cwd(dataDir)
        print(f'connection successful to - {dataDir}')
    except ftplib.all_errors as e:
        print(e)

    data = []
    ftp.retrlines('LIST',data.append)
    ftpFileList = [line.split(None,8)[-1] for line in data]
    ftp.close()
    return ftpFileList

def strip_Files(dataList,exc,exclude=True):
    returnData = []
    for data in dataList:
        clearList = []
        for d in data:
            if exclude:
                if exc not in d:
                    clearList.append(d)
            else:
                if exc in d:
                    clearList.append(d)
        returnData.append(clearList)
    return returnData

def get_files(host,dataDir,ftpDays,ftpFiles,dlFiles=f'noaa_downloads',
                perminentSave=True):

    for i,days in enumerate(ftpDays):
        sf = f'{dlFiles}/{days}'
        if not os.path.exists(sf):
            os.makedirs(sf)
        print(f'Downloading from {days}:')
        for ftpFile in tqdm(ftpFiles[i]):
            if not os.path.isfile(f'{sf}/{ftpFile}'):
                Url = f'https://www.{host}/{re.sub("pub/","",dataDir)}/{days}/{ftpFile}'
                with closing(request.urlopen(Url)) as url:
                    savedFile = f'{sf}/{ftpFile}'
                    with open(savedFile, 'wb') as File:
                        shutil.copyfileobj(url, File)


def combine_datasets(days,concats=[''],regions=[''],dlDir=f'noaa_downloads',
                    sf=f'combine_noaa_nc',eagleDir=f'/projects/wec/pyWecPredict_data',
                    parallel=False):

    dlDir=f'{eagleDir}/{dlDir}'
    sf=f'{eagleDir}/{sf}'
    
    for day in days:
        for concat in concats:
            for region in regions: 
                for g in glob.glob(f'{dlDir}/{day}/*'):
                    if  'idx' in g:
                        os.remove(g)
                try:
                    if not os.path.exists(sf):
                        os.makedirs(sf)
                    fname = f'Combined_{day}_{region}_{concat}.nc'
                    if os.path.isfile(f'{sf}/{fname}'):
                        #print(f'{fname} -- Already Exists')
                        pass
                    else:
                        print(f'Processing {day}/{concat}/{region}')
                        ds = xr.open_mfdataset(f'{dlDir}/{day}/*{region}.{concat}*',
                                        combine='nested',
                                        concat_dim=['sim_start'],
                                        engine='cfgrib',
                                        parallel=parallel)
                        print(ds.swh.shape)
                        ds.to_netcdf(f'{sf}/{fname}')
                        print(f'Processed and saved {fname}')
                except OSError as e:
                    print(e)


def extract_buoy_from_model(buoy, nc=f'combine_noaa_nc',
                                buoyNames=f'combine_noaa_buoys',
                                ncbuoySave=f'sliced_noaa_nc',
                                eagleDir='/projects/wec/pyWecPredict_data',
                                negLon=True):


    buoyName = glob.glob(f'{eagleDir}/{buoyNames}/*')
    ncFiles = glob.glob(f'{eagleDir}/{nc}/*')
    ncBuoys = glob.glob(f'{eagleDir}/{ncbuoySave}/*')
    try:
        bn = [bname for bname in buoyName if buoy.name in bname]
        bn = sorted(bn)[-1].split(f'/')[-1][:-4]
        if f'{eagleDir}/{ncbuoySave}/{bn}.nc' not in ncBuoys:
            dates = np.unique([nf[-23::][:8] for nf in ncFiles])
            ncSel = []
            for date in dates:
                for File in ncFiles:
                    if date in File:
                        nc = xr.open_dataset(File)
                        tolerance = np.abs(nc.latitude.values[1]-nc.latitude.values[0])
                        if negLon:
                            newLon = -buoy['longitude']
                        else:
                            newLon = buoy['longitude']
                        buoyLat,buoyLon = (np.abs(nc.latitude.values-buoy['latitude']),
                                            np.abs((nc.longitude.values-360)-newLon))
                        if buoyLat.min() <= tolerance and buoyLon.min() <= tolerance:
                            ncData = nc.isel(latitude=buoyLat.argmin(),
                                            longitude=buoyLon.argmin())
                            if not np.isnan(ncData.swell).all():
                                ncSel.append(ncData)

            if ncSel:
                buoyData = xr.concat(ncSel,dim='time')
                buoyData = buoyData.rename({'time':'sim_start'})
                buoyData.to_netcdf(f"{eagleDir}/{ncbuoySave}/{bn}.nc")
                print(f'{bn} -- has been processed and saved')
                return buoyData
            else:
                return []
        else:
            return xr.open_dataset(f"{eagleDir}/{ncbuoySave}/{bn}.nc")
            print(f'{bn} -- has been loaded from file')
    except IndexError as e:
        print(f'{buoy.name} -- is not in files')


def update_buoys(buoys,oSys='linux',buoyFile = f'combine_noaa_buoys',
                    eagleDir=f'/projects/wec/pyWecPredict_data'):

    buoyFile = f'{eagleDir}/{buoyFile}'

    updatedBuoys = {}
    for buoy in buoys:
        bFiles = glob.glob(f'{buoyFile}/*')
        buoySave = f'{buoyFile}/{buoy}_{re.sub(" ","T",str(pd.Timestamp.now(tz="gmt"))[:-22])}.csv'
        if not buoySave in bFiles:
            bn = int(buoys.loc['buoy_number',buoy])
            b = f'https://www.ndbc.noaa.gov/data/realtime2/{bn}.spec'
            get_buoy_data(b,buoySave)
            print(f'retrieved {b}')
        try:        
            updatedBuoys[buoy] = pd.read_pickle(buoySave)
        except FileNotFoundError as e:
            pass

    return updatedBuoys

def load_latest_ncBuoy(buoy,buoyFile=f'sliced_noaa_nc',
            eagleDir=f'/projects/wec/pyWecPredict_data',
            date=''):
    
    if date:
        date = pd.to_datetime(date)

    buoyDir = f'{eagleDir}/{buoyFile}'
    fNames = glob.glob(f'{buoyDir}/*')    
    availFiles = [fname for fname in fNames if buoy in fname]
    availDates = np.array([pd.to_datetime(fname[-13:-3]) 
                    for fname in fNames if buoy in fname])
    
    if date:
        if date in availDates:
            loadFile = availFiles[np.argwhere(date==np.array(availDates)).flatten()]
            loadedFile = xr.open_dateset(loadFile)
        else:
            print(f'NC buoy files for {date} - {buoy} - NOT FOUND')
            loadedFile = None

    else:
        loadedFile = xr.open_dataset(availFiles[availDates.argmax()])

    return loadedFile
                
                 














