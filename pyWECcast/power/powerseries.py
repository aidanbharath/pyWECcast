import matplotlib.pyplot as plt

from numpy import array, string_, zeros_like, pi, int64, unique, abs, arctan2, mean, datetime64, sum, cos, argwhere, sin, zeros
from random import choice
from numba import jit, typeof, float64, int32
from abc import ABCMeta, abstractmethod
from pandas import to_datetime, to_timedelta, date_range, DatetimeIndex, DataFrame
from h5py import File
from tqdm import tqdm
from scipy.fft import fft, fftfreq
from dask.array import from_array
from dask.dataframe import from_pandas


class __POWERSERIES__(object):
    __metaclass__ = ABCMeta

    def __init__(self,db, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.db = db
        self.seaState = {}
        self.forecast_time = {}
        self.resultDB = './tempResultDB.h5'

        if len(self.args):
            self.ws = args[0]
        else:
            self.ws = None
        if 'resultDB' in list(self.kwargs.keys()): 
            self.resultDB = self.kwargs['resultDB']
        if 'Hs' not in list(self.kwargs.keys()):
            self.h0 = ['swh']
        else:
            self.h0 = list(self.kwargs['Hs'])
        if 'Tp' not in list(self.kwargs.keys()):
            self.te = ['perpw']
        else:
            self.te = list(self.kwargs['Tp'])
        if 'deg' not in list(self.kwargs.keys()):
            self.deg = '0'
        else:
            self.deg = self.kwargs['deg']
        if 'tempFFTFiles' not in list(self.kwargs.keys()):
            self.tempFFT = f'./tempFFTFile.h5'
        else:
            self.tempFFT = self.kwargs['tempFFTFiles']
        if 'time_var' not in list(self.kwargs.keys()):
            self.time_var = 'Time'
        else:
            self.time_var = self.kwargs['time_var']
        if 'variable' not in list(self.kwargs.keys()):
            self.var = 'Power'
        else:
            self.var = self.kwargs['variable']
        if 'freq_limit' not in list(self.kwargs.keys()):
            self.freq_limit = 1
        else:
            self.freq_limit = self.kwargs['freq_limit']
        if 'freq' not in list(self.kwargs.keys()):
            self.freq = self.kwargs['freq']
        else:
            self.freq = '10S'

    @abstractmethod
    def read_db(self):
        pass


class buoy(object):
    def __init__(self,db,*args,**kwargs):
        self.db = db
        self.args = args
        self.kwargs = kwargs
        self.seaState, self.recon_times = {}, {}

        if 'buoyFile' not in list(self.kwargs.keys()):
            self.buoyFile = f'./noaa_Buoy_info.csv'
        else:
            self.buoyFile = self.kwargs['buoyFile']

        if 'buoys' in list(self.kwargs.keys()):
            self.buoys = list(self.kwargs['buoys'])
        else:
            with File(self.db,'r') as hdf:
                self.buoys = list(hdf.keys())

        if 'WEC_variable' in list(self.kwargs.keys()):
            self.WEC_var = self.kwargs['WEC_variable']
        else:
            self.WEC_var = 'Power'
        
        if 'freq' not in list(self.kwargs.keys()):
            self.freq = self.kwargs['freq']
        else:
            self.freq = '10S'

        try:
            name = read_csv(self.buoyFile,index_col=f'Unnamed: 0').astype(object)
            for b in self.buoys:
                col = name.loc[:, name.loc['buoy_number']==int(b)]
                self.names[b] = col.columns[0]
                self.latitude[b] = col.loc['latitude'][0]
                self.longitude[b] = col.loc['longitude'][0]
        except:
            self.names = f'-- Name Association File Not Found --'

        if len(self.args):
            self.ws = args[0]
        else:
            self.ws = None
        if 'resultDB' in list(self.kwargs.keys()): 
            self.resultDB = self.kwargs['resultDB']
        else:
            self.resultDB = 'tempBuoyResults.h5'
        if 'Hs' not in list(self.kwargs.keys()):
            self.h0 = 'WVHT'
        else:
            self.h0 = self.kwargs['Hs']
        if 'Tp' not in list(self.kwargs.keys()):
            self.te = 'DWP'
        else:
            self.te = self.kwargs['Tp']
        if 'deg' not in list(self.kwargs.keys()):
            self.deg = '0'
        else:
            self.deg = self.kwargs['deg']
        if 'tempFFTFiles' not in list(self.kwargs.keys()):
            self.tempFFT = f'./tempFFTFile_buoy.h5'
        else:
            self.tempFFT = self.kwargs['tempFFTFiles']

    def get_times(self):
        with File(self.db, 'r') as hdf:
            self.times = {b:to_datetime([a.decode() for a in hdf[b]['time_index']])
                                for b in self.buoys}

    def load(self, *buoy):
        if not buoy:
            for b in self.buoys:
                with File(self.db, 'r') as hdf:
                    yield hdf[b]
        else:
            for b in buoy:
                with File(self.db, 'r') as hdf:
                    yield hdf[b]

    def calculate_seaStates(self, *args, **kwargs):
        def find_ss(x, ss, seeds):
            seed = choice(seeds)
            Hs = abs(x['Hs'] - ss['Hs'].values).argmin()
            Tp = abs(x['Tp'] - ss['Tp'].values).argmin()
            return f'{seed}/Hs_{ss["Hs"].values[Hs]}/Tp_{ss["Tp"].values[Tp]}'

        with File(self.ws, 'r') as ws:
            sskeys = {}
            seeds = list(ws.keys())
            for seed in ws.keys():
                for hs in ws[f'{seed}'].keys():
                    for tp in ws[f'{seed}/{hs}'].keys():
                        sskeys[f'{seed}/{hs}/{tp}'] = [float(hs[3::]), float(tp[3::])]
                                
        self.keys = DataFrame(sskeys,index = ['Hs', 'Tp']).T

        self.get_times()
        with File(self.db, 'r') as hdf:
            for b in tqdm(self.buoys):
                data = array([self.times[b], hdf.get(f'{b}/{self.te}'),
                                 hdf.get(f'{b}/{self.h0}')]).T
                df = DataFrame(data, columns = ['time', 'Tp', 'Hs'])
                df['fftKey'] = df.apply(find_ss, axis=1, args=(self.keys,seeds,))
                self.seaState[b] = df

    def calculate_ffts(self, *args, **kwargs):
        with File(self.ws, 'r') as ws:
            for b in tqdm(self.buoys):
                for x in self.seaState[b]['fftKey'].unique():
                    time, data = ws[f'{x}/Time'][:], ws[f'{x}/{self.WEC_var}'][:]
                    coefs, freq = fft(data), fftfreq(time.shape[0], d=time[1]-time[0])
                    try:
                        with File(self.tempFFT,'a') as tmpSave:
                            tmpSave.create_dataset(f'{x}/frequency', data=freq, dtype=freq.dtype)
                            tmpSave.create_dataset(f'{x}/coefficients', data=coefs, dtype=coefs.dtype)
                    except (RuntimeError,OSError) as e:
                        pass

    def set_times(self, *args, **kwargs):
        for b in self.buoys:
            fcTimes = date_range(to_datetime(self.seaState[b]['time'].values[0]),
                                 to_datetime(self.seaState[b]['time'].values[-1]),
                                 freq=self.freq)
            fcInts = fcTimes.astype(int64)/10**9
            dt = self.seaState[b].set_index('time')
            dff = dt.reindex(fcTimes, method='pad')
            dff['fcInts'] = fcInts
            self.recon_times[b] = dff

    def reconstruct_powerseries(self, *args, **kwargs):
        
        @jit(nopython = True)
        def reconstruct(coef,freq,t):
            return sum(coef.real*cos(2*pi*freq*t)+coef.imag*sin(2*pi*freq*t))

        for b in self.buoys:
            result = zeros(self.recon_times[b].shape[0])
            j = 0
            with File(self.tempFFT, 'r') as fft:
                for i, item in tqdm(self.recon_times[b].iterrows()):
                    coef, freq, t = (fft[item['fftKey']]['coefficients'][:],
                                 fft[item['fftKey']]['frequency'][:],
                                 item['fcInts'])
                    print(coef.shape,freq.shape,t.shape)
                    result[j] = (1/coef.shape[0])*reconstruct(coef,freq,t)
                    j += 1
            try:
                saveTimes = string_([t.encode('utf-8')
                            for t in self.recon_times[b].index.strftime('%Y-%m-%d %H:%M:%S')])
                with File(self.resultDB, 'a') as resultDB:
                    resultDB.create_dataset(f'{b}/powerseries', data=result, dtype=result.dtype)
                    resultDB.create_dataset(f'{b}/time_index', data=saveTimes, dtype=saveTimes.dtype)
            except (RuntimeError, OSError) as e:
                pass


class forecast(__POWERSERIES__):
    def read_db(self):
        models, dates, var = [], [], []
        with File(self.db, 'r') as hdf:
            for model in hdf.keys():
                models.append(model)
                for t0 in hdf[model]:
                    dates.append(t0)
                    for Vars in hdf[model][t0]:
                        var.append(Vars)
        self.models = unique(models)
        self.start_time = unique(dates)
        self.vars = unique(var)

    def calculate_seaStates(self, *args, **kwargs):
        with File(self.db, 'r') as hdf:
            print(hdf.keys())
            with File(self.ws, 'r') as ws:
                for model in hdf.keys():
                    for t0 in hdf[model]:
                        print(t0)
                        '''
                        h0, te = (hdf[model][t0][self.h0],
                                  hdf[model][t0][self.te])
                        '''
                        print(hdf[model][t0].keys())
                        h0 = [hdf[model][t0][h0] for h0 in self.h0]
                        te = [hdf[model][t0][te] for te in self.te]
                        deg = [hdf[model][t0][d] for d in self.deg]
                        
                        print(h0[0],te,deg)
                        
                        if 'Tp' in ws.attrs.keys() and 'Hs' in ws.attrs.keys():
                            pass
                        else:
                            Hs = list(ws.keys())
                            Tp = list(ws[Hs[0]].keys())
                            HsVal = array([float(h[3:]) for h in Hs])
                            TpVal = array([float(t[3:]) for t in Tp])
                        self.seaState[f'{model}/{t0}'] = [(Hs[abs(HsVal - j[0]).argmin()],
                                                           Tp[abs(TpVal - j[1]).argmin()])
                                                           for i, j in enumerate(zip(h0, te))]
                        

    def calculate_ffts(self, *args, **kwargs):
        def powers2(shape):
            powers = 0
            while 2**powers <= shape:
                powers += 1
            return shape - 2**(powers-1)

        def freqLimit(A, p, freq, limit):
            idx = argwhere(freq <= limit).max()
            return A[:idx], p[:idx], freq[:idx]

        with File(self.db, 'r') as hdf:
            with File(self.ws, 'r') as ws:
                print('')
                print(' ---------- Calculating FFTs --------')
                seaStates = set([s for sea in self.seaState.values() for s in sea])
                for sea in tqdm(seaStates):
                    state = f'{sea[0]}/{sea[1]}'
                    try:
                        with File(self.tempFFT, 'r') as fftFile:
                            temp = fftFile[state]
                    except:
                        try:
                            with File(self.tempFFT, 'a') as fftFile:
                                grp = fftFile.create_group(state)
                                time = ws[state][self.time_var][:]
                                data = ws[state][self.var][:]
                                A,freq = fft(data), fftfreq(time.shape[0], d=time[1]-time[0])
                                grp.create_dataset('frequency', data=freq, dtype=freq.dtype)
                                grp.create_dataset('Amplitude', data=A, dtype=A.dtype)

                        except OSError as e:
                            pass

    def set_times(self, *args, **kwargs):
        with File(self.db, 'r') as hdf:
            for model in hdf.keys():
                for t0 in hdf[model]:
                    key = f'{model}/{t0}'
                    times = DatetimeIndex([to_datetime(t0) +
                                           to_timedelta(int(advTime.decode('UTF-8')[1:]),
                                           unit='h')
                                           for advTime in hdf[key]['valid_time']])
                    Hs = [self.seaState[key][i][0]
                            for i in range(hdf[key]['valid_time'].shape[0])]
                    Tp = [self.seaState[key][i][1]
                            for i in range(hdf[key]['valid_time'].shape[0])]
                    self.forecast_time[key] = DataFrame(array([Hs, Tp]).T,
                                                        index=times,
                                                        columns=['Hs', 'Tp'])

    def reconstruct_powerseries(self, *args, **kwargs):
        @jit(nopython = True)
        def recon(A, f, t):
            return sum(A.real*cos(2*pi*f*t) + A.imag*sin(2*pi*f*t))

        with File(self.tempFFT, 'r') as fftFile:
            for key, values in self.forecast_time.items():
                fcTimes = date_range(values.index[0], values.index[-1], freq=self.freq)
                fcInts = fcTimes.astype(int64)/10**9
                fcInts = fcInts - fcInts[0]
                with File(self.resultDB, 'a') as resultDB:
                    result = zeros_like(fcInts)
                    grp = resultDB.create_group(key)
                    print('')
                    print(f'Calculating {key}')
                    for i,time in tqdm(enumerate(fcTimes)):
                        fftTime = values[values.index <= time]
                        fftkey = f'{fftTime["Hs"].values[-1]}/{fftTime["Tp"].values[-1]}'
                        data = fftFile[fftkey]
                        A,f = (data['Amplitude'][:],data['frequency'][:])
                        result[i] = (1/A.shape[0])*recon(A,f,fcInts[i])
                    saveTimes = string_([t.encode('utf-8') 
                                        for t in fcTimes.strftime('%Y-%m-%d %H:%M:%S')])
                    grp.create_dataset('time_index', data=saveTimes, dtype=saveTimes.dtype)
                    grp.create_dataset('powerseries', data=result, dtype=result.dtype)


def read_buoys(db,*args,**kwargs):
    b = buoy(db)
    b.read_db()


def read_forecast(db, *args, **kwargs):
    f = forecast(db)
    f.read_db()
    return f


def forecast_powerseries(db, wecSim, *args, **kwargs):
    f = forecast(db, wecSim, *args, **kwargs)
    f.calculate_seaStates()
    f.calculate_ffts()
    f.set_times()
    f.reconstruct_powerseries()
    
def multi_sea_powerseries(db, wecSim, *args, **kwargs):
    f = forecast(db, wecSim, *args, **kwargs)
    f.calculate_seaStates()

def buoy_powerseries(db, wecSim, *args, **kwargs):
    b = buoy(db, wecSim, *args, **kwargs)
    b.calculate_seaStates()
    b.calculate_ffts()
    b.set_times()
    b.reconstruct_powerseries()
