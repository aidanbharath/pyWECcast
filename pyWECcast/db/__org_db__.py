
from numpy import array, sqrt, meshgrid, dstack, argmin, unravel_index, float32
from pandas import to_datetime, read_csv, DataFrame
from h5py import File
from tqdm import tqdm


class WW3(object):

    def __init__(self,db,*args,**kwargs):
        
        self.args = args
        self.kwargs = kwargs
        self.db = db
        self.loc_args, self.time_args = {}, {}
        self.begin, self.end = None, None
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
                self.model_id = [self.kwargs['model_id']]
            else:
                self.model_id = list(hdf.keys())

            self.times = {model:to_datetime(list(hdf[model].keys()))
                                for model in self.model_id}
    
    def __repr__(self):
        for model in self.time:
            return f'{model}: has data from -> {self.time[model][0]} - {self.time[model][-1]}'
    
    def __attrs__(self):
        return [d for d in self.__dict__ if '__' not in d]
    
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

        if len(args) >= 1:
            self.begin = to_datetime(args[0]) 
        if len(args) >= 2:
            self.end = to_datetime(args[1])

        if not self.begin:
            for model in self.model_id:
                self.time_args[model] = self.times[model]
        else:
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
                    
                    lat = self.loc_args[model][0][0]
                    lon = self.loc_args[model][0][1]
                    with File(self.db,'r') as db:
                        prefix = f'{model}/{time}'
                        vt = db[f'{prefix}/valid_time']
                        hdf.create_dataset(f'{prefix}/valid_time',
                                            data=vt,
                                            dtype=vt.dtype)
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
                                        data=db[f'{prefix}/{var}'][:,lat,lon],
                                                dtype=db[f'{prefix}/{var}'].dtype)
                            elif var in levels:
                                hdf.create_dataset(f'{prefix}/{var}',
                                        data=db[f'{prefix}/{var}'][:,:,lat,lon],
                                                dtype=db[f'{prefix}/{var}'].dtype)


class Buoy(object):
    
    def __init__(self,db,*args,**kwargs):
        
        self.args = args
        self.kwargs = kwargs
        self.db = db
        self.names, self.latitude, self.longitude = {}, {}, {}
        self.variables = None
        
        if 'buoyFile' not in list(self.kwargs.keys()):
            self.buoyFile = f'./noaa_Buoy_info.csv'
        else:
            self.buoyFile = self.kwargs['buoyFile']

        if 'buoys' in list(self.kwargs.keys()):
            self.buoys = list(self.kwargs['buoys'])
        else:
            with File(self.db,'r') as hdf:
                self.buoys = list(hdf.keys()) 
            
        if 'variables' in list(self.kwargs.keys()):
            self.variables = list(self.kwargs['variables'])
        
        with File(self.db,'r') as hdf:
            self.times = {b:to_datetime([a.decode() for a in hdf[b]['time_index']])
                                for b in self.buoys}
            
        try:
            name = read_csv(self.buoyFile,index_col=f'Unnamed: 0').astype(object)
            for b in self.buoys:
                col = name.loc[:,name.loc['buoy_number']==int(b)]
                self.names[b] = col.columns[0]
                self.latitude[b] = col.loc['latitude'][0]
                self.longitude[b] = col.loc['longitude'][0]
        except:
            self.names = f'-- Name Association File Not Found --'
    
    def __repr__(self):
        if type(self.names) is not str(): 
            return f'Buoys: {self.names}'
            
    def __attrs__(self):
        return [d for d in self.__dict__ if '__' not in d]
    
    def add_DWP(self):
        
        def calc_DWP(buoy):
            he = float32(buoy.loc[:,'WVHT'].values)
            swell = float32(buoy.loc[:,'SwH'].values)
            windw = float32(buoy.loc[:,'WWH'].values)
            DWP = []
            for i,j in enumerate(swell>=windw):
                if j:
                    DWP.append(float32(buoy.iloc[i].loc['SwP']))
                elif not j:
                    DWP.append(float32(buoy.iloc[i].loc['WWP']))

            return array(DWP)
                                
        with File(self.db,'a') as hdf:
            for b in self.buoys:
                #try:
                df = DataFrame(array([hdf[b]['WVHT'][:],hdf[b]['SwH'][:],
                                         hdf[b]['WWH'][:],
                                         hdf[b]['SwP'][:],hdf[b]['WWP'][:]]).T,
                                         columns=['WVHT','SwH','WWH','SwP','WWP'],
                                         index=self.times[b]
                                     )
                DWP = calc_DWP(df)
                try:
                    hdf.create_dataset(f'{b}/DWP',data=DWP,dtype=DWP.dtype)
                except RuntimeError as e:
                    hdf[f'{b}/DWP'][:] = DWP
    
    def load(self,*buoy):
        if not buoy:
            for b in self.buoys:
                with File(self.db,'r') as hdf:
                    yield hdf[b]
        else:
            for b in buoy:
                with File(self.db,'r') as hdf:
                    yield hdf[b]
                                
        
        
def get_data(db,saveName,*args,**kwargs):

    ww = WW3(db,*args,**kwargs)
    ww.get_location()
    ww.get_times()
    ww.reduce_db(saveName)











