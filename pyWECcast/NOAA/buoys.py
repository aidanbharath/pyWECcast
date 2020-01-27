import os
import ftplib
import urllib
import shutil
import requests
import pandas as pd
import urllib.request as request
import requests

from h5py import File
from abc import ABCMeta, abstractmethod
from numpy import int32, array, float32, nan, where, string_, fromstring
from contextlib import closing
from tqdm import tqdm


class __BUOY__(object):

    """
    Buoy Base Class

    only arg is mean to be the filename, everythign else is
    a kwarg
    """

    __metaclass__ = ABCMeta

    def __init__(self,*args,**kwargs):
        
        self.args = args
        self.kwargs = kwargs
        self.last_update = None
        try:
            self.buoyFile = args[0]
        except (FileNotFoundError,IndexError) as e:
            if e is FileNotFoundError:
                raise e
            else:
                self.buoyFile = None
                
        if self.buoyFile:
            self.__buoys = pd.read_csv(self.buoyFile,index_col=0)
            self.buoy_number = list(int32(self.__buoys.loc['buoy_number',:].values))
            self.names = list(self.__buoys.columns)
            self.coords = self.__buoys.loc[('latitude','longitude'),:].values
        else:
            self.__buoys = None 
            if 'buoy_number' in kwargs.keys(): 
                self.buoy_number = kwargs['buoy_number']
            else: self.buoy_number = None
            if 'names' in kwargs.keys():
                self.names = kwargs['names']
            else: self.names = None
            if 'latitude' in kwargs.keys() and 'longitude' in kwargs.keys():
                self.coords = array([kwargs['latitude'],kwargs['longitude']])
            else: self.coords = None

        if 'buoy_ftp' in kwargs.keys():
            self.buoy_ftp = kwargs['buoy_ftp']
        else:
            self.buoy_ftp = f'https://www.ndbc.noaa.gov/data/realtime2'

        if 'buoy_hdf5' in kwargs.keys():
            self.h5fname = kwargs['buoy_hdf5']
        else:
            self.h5fname = './buoy_downloads.h5'

    def __attrs__(self):
        return [d for d in self.__dict__ if '__' not in d]

    def select(self,buoy_number='',names='',coords=''):
        pass

    @abstractmethod
    def download(self):
        pass


class Buoy(__BUOY__):

    """
    Public Buoy class for buoys from NOAA site
    """

    def download(self):

        temp = f'.tempBuoy.pkl'
        ftp = (f'{self.buoy_ftp}/{bn}.spec' for bn in self.buoy_number)
        
        if not os.path.isfile(self.h5fname): # Create File if doesn't exist
            h5f  = File(self.h5fname,'w')
            h5f.close()

        with File(self.h5fname,'a') as hdf: # run with hdf5 file open
            for i,f in tqdm(enumerate(ftp)):
                try:
                    """
                    Cycle through all buoys associated with the Class.
                    Downloading and processing happens individually
                    """
                    with closing(request.urlopen(f)) as buoy:
                        if os.path.isfile(temp): os.remove(temp)
                        with open(temp, 'wb') as F:
                            shutil.copyfileobj(buoy, F)
                    buoy = pd.read_csv(temp,delim_whitespace=True)
                    data = buoy.iloc[1::,5::].values
                    cols = buoy.columns[5::]
                    b = buoy.iloc[1::,0:5] 
                    Y,M,D,h,m = b['#YY'],b['MM'],b['DD'],b['hh'],b['mm']
                    times = [pd.to_datetime(f"{Y[i]}-{M[i]}-{D[i]}T{h[i]}:{m[i]}:00") 
                                for i in Y.index]
                    buoy = pd.DataFrame(data,index=pd.to_datetime(times),columns=cols)
                    buoy = buoy.iloc[::-1]  
                    
                    bn = self.buoy_number[i]
                    if f'{bn}' not in list(hdf.keys()):
                        """ 
                        Create a new group is the buoy number selected
                        does not exist in the hdf5 dataset
                        """
                        hdf.create_group(f'{bn}')
                        times = buoy.index.strftime(
                                    '%Y-%m-%d %H:%M:%S').values
                        times = [string_(t) for t in times]
                        hdf[f'{bn}'].create_dataset(f'time_index',data=times,
                                    dtype='S19',chunks=(10000,),maxshape=(None,)
                                )

                        for col in buoy:
                            """
                            On the assumption of first run for the bn, fields 
                            are created based on what exists in the downloaded
                            dataset
                            """
                            df = buoy[col].values
                            try:
                                df[where(df=='MM')] = nan #'MM' assumed = nan
                                df = float32(df)
                            except ValueError as e:
                                df = array([str(d) for d in df],dtype='S')
                            
                            # Chunksizes are assumed 10k for no good reason
                            hdf[f'{bn}'].create_dataset(f'{col}',data=df,
                                        chunks=(10000,),maxshape=(None,)
                                    )
                            key = f'{bn}/{col}'
                            hdf[key].dims[0].attach_scale(hdf[f'{bn}/time_index'])

                        """
                        all attributes here are associated to the group
                        These can be expanded considerably as default values
                        """                        
                        hdf[f'{bn}'].attrs['coords'] = self.coords[:,i]
                        hdf[f'{bn}'].attrs['name'] = self.names[i]
                        hdf[f'{bn}'].attrs['buoy_number'] = self.buoy_number[i]
                    
                    else:
                        """
                        Here will append to an existing dataset
                        """
                        tAvail = pd.to_datetime([k.decode('ascii') for k in 
                                    hdf[f'{bn}/time_index']]
                                    )
                        buoy = buoy[~buoy.index.isin(tAvail)]
                        if not buoy.empty:
                            for col in buoy:
                                df = buoy[col].values
                                try:
                                    df[where(df=='MM')] = nan
                                    df = float32(df)
                                except ValueError as e:
                                    df = array([str(d) for d in df],dtype='S')
                                
                                hdf[f'{bn}/{col}'].resize((hdf[f'{bn}/{col}'].shape[0] 
                                                            +df.shape[0]), axis = 0)
                                """
                                will catch any odd value and replace with 
                                binary 'nan'
                                """
                                try:
                                    hdf[f'{bn}/{col}'][-df.shape[0]:] = df
                                except OSError as e:
                                    hdf[f'{bn}/{col}'][-df.shape[0]:] = b'nan'

                except urllib.error.HTTPError as e:
                    print(f'{f} -- was not found')
        
                finally:
                    if os.path.isfile(temp): os.remove(temp)

                    self.last_update = pd.Timestamp.now(tz='GMT')


"""
Direct Functions
"""
def download(*args,**kwargs):

    """
    Direct Download function which will construct and return a Buoy Class obj
    """
    buoy = Buoy(*args,*kwargs)
    buoy.download() 
    
    return buoy