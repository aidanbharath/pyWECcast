
from numpy import array, sqrt, meshgrid, dstack, argmin, unravel_index
from pandas import to_datetime
from abc import ABCMeta, abstractmethod
from h5py import File
from tqdm import tqdm


class __DB_HANDLER__(object):

    __metaclass__ = ABCMeta

    def __init__(self,db,*args,**kwargs):
        
        self.args = args
        self.kwargs = kwargs
        self.db = db
        self.loc_args, self.time_args = {}, {}
        self.begin, self. end = None, None
        self.variables = None

        if len(self.args) >= 2:
            self.lon, self.lat = array(self.args[0]), array(self.args[1])
        if len(self.args) >= 3:
            self.begin = to_datetime(self.args[2]) 
        if len(self.args) >= 4:
            self.end = to_datetime(self.args[3])

        if 'variables' in list(self.kwargs.keys()):
            self.variables = list(self.kwargs['variables'])

        with File(self.db,'r+') as hdf:
            
            if 'model_id' in list(self.kwargs.keys()):
                self.model_id = list(self.kwargs['model_id'])
            else:
                self.model_id = list(hdf.keys())
            
            self.times = {model:to_datetime(list(hdf[model].keys()))
                                for model in self.model_id}
    
    def __repr__(self):
        for model in self.time:
            return f'{model}: has data from -> {self.time[model][0]} - {self.time[model][-1]}'
    
    def __attrs__(self):
        return [d for d in self.__dict__ if '__' not in d]
    
    @abstractmethod
    def get_location(self,*args):
        pass

    @abstractmethod
    def get_times(self,*args):
        pass


class WW3(__DB_HANDLER__):

    def get_location(self,*args):

        # This needs to be made to work with lists of lon lats

        if args:
            self.lon, self.lat = array(args[0]), array(args[1])
        attrs = self.__attrs__()
        assert 'lon' in attrs or 'lat' in attrs, 'No Location Provided'
        with File(self.db,'r') as hdf:
            for model in self.model_id:
                key = list(hdf[model].keys())[0]
                lons, lats = meshgrid(hdf[f'{model}/{key}']['longitude'][:],
                                hdf[f'{model}/{key}']['latitude'][:])

                distance = dstack([sqrt((lons-[self.lon][i])**2+(lats-[self.lat][i])**2)
                                    for i,j in enumerate([self.lon])])
                self.loc_args[model] = [unravel_index(argmin(distance[:,:,i],axis=None),
                                    distance[:,:,i].shape)
                                    for i in range(distance.shape[-1])]
                                    

    def get_times(self,*args):

        assert args or self.begin, 'No Times have been set'

        if len(args) >= 1:
            self.begin = to_datetime(args[0]) 
        if len(args) >= 2:
            self.end = to_datetime(args[1])

        for model in self.model_id:
            if self.begin:
                begin = self.times[model] >= self.begin
            else:
                begin = True
            if self.end:
                end = self.times[model] <= self.end
            else:
                end = True

            self.time_args[model] = self.times[model][begin*end]            


    def reduce_db(self,saveName):

        dims = ['longitude','latitude','valid_time','level','surface']
        levels = ['swdir','swell','swper']

        with File(saveName,'a') as hdf:
            for model in self.model_id:
                for time in self.time_args[model].strftime('%Y-%m-%d %H:%M:%S'):
                    if self.variables:
                        self.trans = self.variables+dims
                    else:
                        with File(self.db,'r') as db:
                            self.trans = list(db[f'{model}/{time}'].keys())
                    
                    lon = self.loc_args[model][0][0]
                    lat = self.loc_args[model][0][1]
                    with File(self.db,'r') as db:
                        prefix = f'{model}/{time}'
                        hdf.create_dataset(f'{prefix}/valid_time',
                                            data=db[f'{prefix}/valid_time'],
                                            dtype=db[f'{prefix}/valid_time'].dtype)
                        hdf.create_dataset(f'{prefix}/level',
                                            data=db[f'{prefix}/level'],
                                            dtype=db[f'{prefix}/level'].dtype)
                        hdf.create_dataset(f'{prefix}/longitude',
                                            data=db[f'{prefix}/longitude'][lon],
                                            dtype=db[f'{prefix}/longitude'].dtype)
                        hdf.create_dataset(f'{prefix}/latitude',
                                            data=db[f'{prefix}/latitude'][lat],
                                            dtype=db[f'{prefix}/latitude'].dtype)
                        for var in self.trans:
                            if var not in dims and var not in levels:
                                hdf.create_dataset(f'{prefix}/{var}',
                                        data=db[f'{prefix}/{var}'][:,lon,lat],
                                                dtype=db[f'{prefix}/{var}'].dtype)
                            elif var in levels:
                                hdf.create_dataset(f'{prefix}/{var}',
                                        data=db[f'{prefix}/{var}'][:,:,lon,lat],
                                                dtype=db[f'{prefix}/{var}'].dtype)


def get_data(db,saveName,*args,**kwargs):

    ww = WW3(db,*args,**kwargs)
    ww.get_location()
    ww.get_times()
    ww.reduce_db(saveName)











