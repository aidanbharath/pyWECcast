import os
import sys
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
from numpy import int32, nansum, array, float32, nan, argwhere, where, string_, fromstring, newaxis
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

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.last_update = None
        self.chunks = {'time':100, 'surface':1, 'latitude':10, 'longitude':10,
                       'level':1, 'step':1, 'valid_time':30}
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

        if 'processH5s' in list(kwargs.keys()):
            self.processH5s = kwargs['processH5s']
        else:
            self.processH5s = True

        if 'processVars' in list(kwargs.keys()):
            self.processVars = kwargs['processVars']
        else:
            self.processVars = ['swh', 'perpw']
        if 'processFiles' in list(kwargs.keys()):
            self.processFiles = kwargs['processFiles']
        else:
            self.processFiles = None            
        if 'host' in list(kwargs.keys()):
            self.host = kwargs['model_ftp']
            self.__dhost = False
        else:
            self.host = __default_host__
            self.__dhost = True

        if 'host_dir' in list(kwargs.keys()):
            self.host_dir = kwargs['host_dir']
            self.__dhost_dir = False
        else:
            self.host_dir = __default_host_dir__
            self.__dhost_dir = True

        if 'model_hdf5' in list(kwargs.keys()):
            self.h5fname = kwargs['model_hdf5']
        else:
            self.h5fname = './model_downloads.h5'

        if 'ftp_prefix' in list(kwargs.keys()):
            """
            Currently limited to looking at one prefix only
            """
            self.ftp_prefix = kwargs['ftp_prefix']
        else:
            self.ftp_prefix = 'multi_1'

        if 'tempGRIBFile' in list(kwargs.keys()):
            # create and automatic file build if doesn't exist
            self.tempGRIB = kwargs['tempGRIBFile']
        else:
            self.tempGRIB = './tempGRIB'

        if 'search_host' in list(kwargs.keys()):
            self.search_host = kwargs['search_host']
        else:
            self.search_host = True

        if self.search_host:
            try:
                assert self.__test_ftp_connection__()
                self.days = []
                self.ftp_url = []
                for files in self.__retrieve_dirs__(self.host_dir):
                    if self.ftp_prefix in files:
                        self.days.append(files)
                        if self.__dhost and self.__dhost_dir:
                            self.ftp_url.append(f"{self.host_dir}/{files}")
                        else:
                            self.ftp_url.append(f'{self.host_dir}/{files}')
                self.model_files = defaultdict(list)

                print(f'-- Discovering Available Data --')
                for i in tqdm(self.ftp_url):
                    day = [sub(f'{self.ftp_prefix}.', '', d)
                           for d in self.days if d in i]
                    for urls in self.__retrieve_dirs__(f'{i}/'):
                        if any(reg in urls for reg in self.regions) and 'idx' not in urls:
                            if self.__dhost and self.__dhost_dir:
                                self.model_files[f'{day[0]}'].append(
                                    f'{self.host}/{sub("pub/", "", i)}/{urls}')
                            else:
                                self.model_files[f'{day[0]}'].append(
                                    f'{self.host}/{i}/{urls}')

            except AssertionError as e:
                print(f'FTP host: {self.host}/{self.host_dir}, not available')

    def __attrs__(self):
        return [d for d in self.__dict__ if '__' not in d]

    def __retrieve_dirs__(self,ftpDir):
        data = []
        ftp = ftplib.FTP(self.host)
        ftp.login()
        ftp.cwd(ftpDir)
        ftp.retrlines('LIST', data.append)
        ftpFileList = [line.split(None, 8)[-1] for line in data]
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
    def download(self, groups=('t00z', 't06z', 't12z', 't18z'), remove=False):
        """
        Check if the ftp host is available
        """
        for reg in self.regions:
            print('')
            print(f'-- Begining download and processing on: {reg}')
            if reg == '':
                print('No regions set - cannot be configured')
            else:
                for key in self.model_files:
                    self.key = key
                    print(f'')
                    print(f'-- Collecting data from: {self.key} ')
                    for file in tqdm(self.model_files[self.key]):
                        Url = f'https://www.{file}'
                        if not os.path.isdir(f'{self.tempGRIB}/{self.key}'):
                            os.mkdir(f'{self.tempGRIB}/{self.key}')
                        if not os.path.isfile(f'{self.tempGRIB}/{key}/{Url.split("/")[-1]}'):
                            with closing(request.urlopen(Url)) as url:
                                with open(f'{self.tempGRIB}/{self.key}/{Url.split("/")[-1]}', 'wb') as tempFile:
                                    shutil.copyfileobj(url, tempFile)
                    if self.processH5s:
                        print(f'-- Processing into HDF5 Format --')
                        for tz in groups:
                            self.processH5(tz)
                    if remove:
                        shutil.rmtree(f'{self.tempGRIB}/{key}')

    def processH5(self, group, find_groupName=False, compression=None):
        def stpTime(time):
            times = [pd.to_datetime(time).strftime('%Y-%m-%d %H:%M:%S')]
            return [string_(t) for t in times]

        if not os.path.isdir(self.tempGRIB):
            os.mkdir(self.tempGRIB)
        if not os.path.isfile(self.h5fname):
            h5f  = File(self.h5fname, 'w')
            h5f.close()
        
        if hasattr(self,'key'):
            for g in glob(f'{self.tempGRIB}/{self.key}/*'):
                if 'idx' in g:
                    os.remove(g)
        elif self.processFiles:
            for pf in self.processFiles: 
                for fIdx in glob(f'{self.tempGRIB}/{pf}/*'):
                    if 'idx' in fIdx:
                        os.remove(fIdx)
        else:
            print('No Files Available To Process')
            sys.exit()


        try:
            if hasattr(self,'key'):
                groupNames = glob(f'{self.tempGRIB}/{self.key}/*{group}*')
                fhr = array(sorted(list(set([n.split('.')[4] for n in groupNames]))))
            elif self.processFiles:
                groupNames = [glob(f'{self.tempGRIB}/{pf}/*{group}*')
                                 for pf in self.processFiles]
                groupNames = [j for i in groupNames for j in i]
                fhr = array(sorted(list(set([n.split('.')[3] for n in groupNames]))))

            print(f'Loading Group: {group}')
            print(groupNames[0].split('.'))
            
            for name in tqdm(groupNames):
                ds = open_dataset(name, engine='cfgrib')
                with File(self.h5fname, 'a') as hdf:
                    if find_groupName:
                        model = name.split('.')[2]
                        sim_start = stpTime(ds['time'].values)
                    else:
                        model = group
                        sim_start = stpTime(ds['time'].values)

                    hdfKey = f'{model}/{sim_start[0].decode("ascii")}'
                    try:
                        grp = hdf[hdfKey]
                        if 'status' not in list(hdf[hdfKey].attrs.keys()):
                            hdf[hdfKey].attrs['status'] = ['working']

                        if 'complete' not in hdf[hdfKey].attrs['status']:
                            variables = [var for var in ds.variables if var not in self.dims]
                            for var in variables:
                                if var in self.processVars:
                                    grp[var][argwhere(name.split('.')[4] == fhr), :] = ds[var].values
                        else:
                            pass

                    except KeyError as notMade:
                        if not model in list(hdf.keys()):
                            GRP = hdf.create_group(model)
                        else:
                            GRP = hdf[model]

                        if not sim_start[0].decode('ascii') in list(GRP.keys()):
                            grp = GRP.create_group(sim_start[0].decode("ascii"))
                            oddBalls = ['time', 'step', 'valid_time']
                            for var in self.dims:
                                if var not in oddBalls:
                                    grp.create_dataset(f'{var}',
                                                       data=ds[var].values,
                                                       dtype=ds[var].dtype,
                                                       compression=compression)
                                elif var == 'valid_time' and var not in grp.keys():
                                    grp.create_dataset(f'{var}',
                                                       data=array([string_(f) for f in fhr]),
                                                       dtype='S10',
                                                       compression=compression)
                                    for attr in ds[var].attrs.keys():
                                        grp[var].attrs[attr] = ds[var].attrs[attr]
                                    grp[var].attrs['dims'] = ['valid_time']
                                    for i, dim in enumerate(grp[var].attrs['dims']):
                                        grp[var].dims[i].label = dim

                            variables = [var for var in ds.variables
                                         if var not in self.dims]

                            for var in variables:
                                if var in self.processVars:
                                    if var not in grp.keys():
                                        chunks = [self.chunks['valid_time']]
                                        data = ds[var].values[newaxis, :]
                                        for i, dim in enumerate(ds[var].dims):
                                            chunks.append(int(ds[var].shape[i]/self.chunks[dim]))
                                        maxshape = tuple(None for i in chunks)
                                        grp.create_dataset(f'{var}',
                                                           fhr.shape + ds[var].shape,
                                                           dtype=ds[var].dtype,
                                                           chunks=tuple(chunk for chunk in chunks),
                                                           maxshape=maxshape,
                                                           compression=compression)
                                        grp[var][0, :] = ds[var].values
                                        for attr in ds[var].attrs.keys():
                                            grp[var].attrs[attr] = ds[var].attrs[attr]
                                        grp[var].attrs['dims'] = ['valid_time'] + list(ds[var].dims)
                                        for i, dim in enumerate(grp[var].attrs['dims']):
                                            grp[var].dims[i].label = dim

                    for attr in ds.attrs.keys():
                        if attr not in grp.attrs.keys():
                            grp.attrs[attr] = ds.attrs[attr]

            with File(self.h5fname, 'a') as hdf:
                hdf[hdfKey].attrs['status'][0] = 'complete'

        except OSError as e:
            print(f'No files in {group} Available to open', e)


"""
Direct Functions
"""
def download(*args, **kwargs):
    """
    Direct Download function which will construct and return a model Class obj
    """
    model = NOAA_Forecast(*args, **kwargs)
    model.download()
    return model


def processH5(group, *args, **kwargs):
    """
    Direct Download function which will construct and return a Model Class obj
    """
    model = NOAA_Forecast(*args, **kwargs)
    model.processH5(group)
    return model