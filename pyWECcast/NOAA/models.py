import os
import ftplib
import urllib
import shutil
import requests
import pandas as pd
import urllib.request as request
import requests

from h5py import File
from collections import defaultdict
from abc import ABCMeta, abstractmethod
from numpy import int32, array, float32, nan, where, string_, fromstring, newaxis
from contextlib import closing
from xarray import open_mfdataset, open_dataset
from glob import glob
from re import sub
from tqdm import tqdm


__default_host__ = 'ftp.ncep.noaa.gov'
__default_host_dir__ =  'pub/data/nccf/com/wave/prod'

class __MODEL__(object):

    """
    Model Base Class
    """

    __metaclass__ = ABCMeta

    def __init__(self,*args,**kwargs):
        
        self.args = args
        self.kwargs = kwargs
        self.last_update = None
        self.chunks = {'time':100,'surface':1,'latitude':10,
                    'longitude':10,'level':1,'step':1,'valid_time':10}
        self.dims = [dim for dim in self.chunks.keys()]
        if not self.args:
            print('WARNING: -- No model regions passed, Everything Will Be Downloaded! --')
            self.regions = list([''])
        else:
            self.regions = list(self.args[0])

        """
        host is looking to get to the host and directory
        within host where things are listed by day.

        It will then sort the days.
        """
        if 'host' in kwargs.keys():
            self.host = kwargs['model_ftp']
            self.__dhost = False
        else:
            self.host = __default_host__
            self.__dhost = True
            
        if 'host_dir' in kwargs.keys():
            self.host_dir = kwargs['host_dir']
            self.__dhost_dir = False
        else:
            self.host_dir = __default_host_dir__
            self.__dhost_dir = True
        
        if 'model_hdf5' in kwargs.keys():
            self.h5fname = kwargs['model_hdf5']
        else:
            self.h5fname = './model_downloads.h5'
        if 'ftp_prefix' in kwargs.keys():
            """
            Currently limited to looking at one prefix only
            """
            self.ftp_prefix = kwargs['ftp_prefix']
        else:
            self.ftp_prefix = 'multi_1'
        if 'tempGRIBFile' in kwargs.keys():
            self.tempGRIB = kwargs['tempGRIBFile']
        else:
            self.tempGRIB = './tempGRIB'
        try: 
            assert self.__test_ftp_connection__()
            self.days = []
            self.ftp_url = []
            for files in self.__retrieve_dirs__(self.host_dir):
                if self.ftp_prefix in files:
                    self.days.append(files)
                    if self.__dhost and self.__dhost_dir:
                        self.ftp_url.append(
                            f"{self.host_dir}/{files}"
                            )
                    else:
                        self.ftp_url.append(f'{self.host_dir}/{files}')
            self.model_files = defaultdict(list)
            print(f'-- Discovering Available Data --')
            for i in tqdm(self.ftp_url):
                day = [sub(f'{self.ftp_prefix}.','',d) for d in self.days if d in i]
                for urls in self.__retrieve_dirs__(f'{i}/'):
                    if any(reg in urls for reg in self.regions) and 'idx' not in urls:
                        if self.__dhost and self.__dhost_dir:
                            self.model_files[f'{day[0]}'].append(
                                f'{self.host}/{sub("pub/","",i)}/{urls}'
                                )
                        else:
                            self.model_files[f'{day[0]}'].append(
                                f'{self.host}/{i}/{urls}'
                                )
        except AssertionError as e:
            print(f'FTP host: {self.host}/{self.host_dir}, not available')

    def __attrs__(self):
        return [d for d in self.__dict__ if '__' not in d]

    def __retrieve_dirs__(self,ftpDir):
        
        data = []
        ftp = ftplib.FTP(self.host)
        ftp.login()
        ftp.cwd(ftpDir)
        ftp.retrlines('LIST',data.append)
        ftpFileList = [line.split(None,8)[-1] for line in data]
        ftp.close()

        return ftpFileList


    def __test_ftp_connection__(self):
        """
        Method to test ftp connection to host
        """
        try:
            ftp = ftplib.FTP(self.host)
            ftp.login()
            ftp.cwd(self.host_dir)
            success = True
            ftp.close()
        except ftplib.all_errors as e:
            success = False
            raise e
        finally:
            return success

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def processH5(self):
        pass


class NOAA_Forecast(__MODEL__):

    def download(self,processH5=True, groups=('t00z','t06z','t12z','t18z')):
        """
        Check if the ftp host is available
        """
        
        for reg in self.regions:
            print('')
            print(f'-- Begining download and processing on: {reg}')
            if reg is '':
                print('No regions set - cannot be configured')

            else: 
                for key in self.model_files:
                    print(f'-- Collection data from: {key} ')
                    for file in tqdm(self.model_files[key]):
                         
                        Url = f'https://www.{file}'
                        if not os.path.isfile(f'{self.tempGRIB}/{Url.split("/")[-1]}'):
                            with closing(request.urlopen(Url)) as url:
                                with open(f'{self.tempGRIB}/{Url.split("/")[-1]}', 'wb') as tempFile:
                                    shutil.copyfileobj(url, tempFile)
                        
                    if processH5:
                        print('')
                        print(f'-- Processing into HDF5 Format --')
                        for tz in groups:   
                            self.processH5(tz)
                            break
                    break
    
    def processH5(self,group,find_groupName=True,compression=None):

        def stpTime(time):
            times = [pd.to_datetime(time).strftime(
                    '%Y-%m-%d %H:%M:%S')]
            return [string_(t) for t in times]
        
        if not os.path.isfile(self.h5fname): # Create File if doesn't exist
            h5f  = File(self.h5fname,'w')
            h5f.close()

        for g in glob(f'{self.tempGRIB}/*'):
            if  'idx' in g:
                os.remove(g)
        try:
            groupNames = glob(f'{self.tempGRIB}/*{group}*')
            print(f'Loading Group: {group}')
            ds = open_mfdataset(groupNames,
                combine='nested',
                concat_dim=['valid_time'],
                engine='cfgrib',
                )
            #ds.to_netcdf('test.nc')
            #ds = open_dataset('test.nc')
            with File(self.h5fname,'a') as hdf:
                if find_groupName:
                    model = groupNames[0].split('.')[2]
                else:
                    model = group
                if not model in list(hdf.keys()):
                    grp = hdf.create_group(f'{model}')
                    oddBalls = ['time','step','valid_time']
                    for var in self.dims:
                        if var not in oddBalls:
                            grp.create_dataset(f'{var}',data=ds[var].values,
                                    dtype=ds[var].dtype,
                                    compression=compression
                                )
                        '''        
                        else:
                            if var == 'time':
                                times = stpTime(ds[var].values)
                                grp.create_dataset(f'{var}',data=times,
                                        dtype='S19',chunks=(100,),maxshape=(None,)
                                    )
                        '''
                else:
                    grp = hdf[f'{model}']
                keyTimes = ['valid_time','time']
                variables = keyTimes+[var for var in ds.variables
                                if var not in self.dims]
                print(f'Processing GRIB file Variables for reference time: {ds["time"].values}')
                for var in tqdm(variables):
                    if var not in grp.keys() and var not in keyTimes:
                        chunks = [100]
                        for i,dim in enumerate(ds[var].dims): 
                            chunks.append(int(ds[var].shape[i]/self.chunks[dim]))
                        maxshape=tuple(None for i in chunks)
                        grp.create_dataset(f'{var}',data=ds[var].values[newaxis,:],
                                    dtype=ds[var].dtype,
                                    chunks=tuple(chunk for chunk in chunks),
                                    maxshape=maxshape,
                                    compression=compression
                                )
                        for attr in ds[var].attrs.keys():
                            grp[var].attrs[attr] = ds[var].attrs[attr]
                        grp[var].attrs['dims'] = ['time']+list(ds[var].dims)
                        for i,dim in enumerate(grp[var].attrs['dims']):
                            grp[var].dims[i].label = dim
                    
                    elif var == 'valid_time' and var not in grp.keys():
                        times = stpTime(ds[var].values)
                        grp.create_dataset(f'{var}',data=times,
                                dtype='S19',chunks=(100,
                                int(ds[var].shape[-1]/self.chunks[var])),
                                maxshape=(None,ds[var].shape[-1]),
                                compression=compression
                            )
                        for attr in ds[var].attrs.keys():
                            grp[var].attrs[attr] = ds[var].attrs[attr]
                        grp[var].attrs['dims'] = ['time']+list(ds[var].dims)
                        for i,dim in enumerate(grp[var].attrs['dims']):
                            grp[var].dims[i].label = dim
                
                    else:
                        if var not in keyTimes:
                            grp[f'{var}'].resize((grp[f'{var}'].shape[0] 
                                                    +ds['time'].size), axis = 0)
                            grp[f'{var}'][-ds['time'].size:,:] = ds[var].values
                        elif var == 'time':
                            if var not in grp.keys():
                                times = stpTime(ds[var].values)
                                grp.create_dataset(f'{var}',data=times,
                                        dtype='S19',chunks=(100,),maxshape=(None,)
                                    )
                            else:
                                grp[f'{var}'].resize((grp[f'{var}'].shape[0] 
                                                    +ds['time'].size), axis = 0)
                                times = stpTime(ds[var].values)
                                grp[f'{var}'][-ds['time'].size:] = times
                        elif var == 'valid_time':
                            grp[f'{var}'].resize((grp[f'{var}'].shape[0] 
                                                    +ds['time'].size), axis = 0)
                            times = stpTime(ds[var].values)
                            grp[f'{var}'][-ds['time'].size:,:] = times

                    for attr in ds.attrs.keys():
                        if attr not in grp.attrs.keys():
                            grp.attrs[attr] = ds.attrs[attr]
                
        except OSError as e:
            print(f'No files in {group} Available to open')

"""
Direct Functions
"""
def download(*args,**kwargs):

    """
    Direct Download function which will construct and return a model Class obj
    """
    model = NOAA_Forecast(*args,**kwargs)
    model.download() 
    
    return model

def processH5(group,*args,**kwargs):

    """
    Direct Download function which will construct and return a Model Class obj
    """

    model = NOAA_Forecast(*args,**kwargs)
    model.processH5(group)

    return model





