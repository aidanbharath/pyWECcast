from numpy import array, string_, zeros_like, pi, int64, unique, abs, arctan2, mean, datetime64, sum, cos
from numba import jit, typeof, float64, int32
from abc import ABCMeta, abstractmethod
from pandas import to_datetime, to_timedelta, date_range, DatetimeIndex, DataFrame
from h5py import File
from tqdm import tqdm
from scipy.fftpack import fft, fftfreq
from dask.array import from_array


class __POWERSERIES__(object):

    __metaclass__ = ABCMeta

    def __init__(self,db,*args,**kwargs):

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
        if 'resultDB' in self.kwargs.keys(): self.resultDB = self.kwargs['resultDB']
        if 'Hs' not in self.kwargs.keys():
            self.h0 = 'swh'
        else:
            self.h0 = self.kwargs['Hs']
        if 'Tp' not in self.kwargs.keys():
            self.te = 'perpw'
        else:
            self.te = self.kwargs['Tp']
        if 'tempFFTFiles' not in self.kwargs.keys():
            self.tempFFT = f'./tempFFTFile.h5'
        else:
            self.tempFFT = self.kwargs['tempFFTFiles']
        if 'time_var' not in self.kwargs.keys():
            self.time_var = 'Time'
        else:
            self.time_var = self.kwargs['time_var']
        if 'variable' not in self.kwargs.keys():
            self.var = 'Power'
        else:
            self.var = self.kwargs['variable']
        

    @abstractmethod
    def read_db(self):
        pass


class buoy(__POWERSERIES__):

    def read_db(self):
        with File(self.db,'r') as hdf:
            for model in hdf.keys():
                for t0 in hdf[model]:
                    print(hdf[model][t0])

class forecast(__POWERSERIES__):
    
    def read_db(self):
        models,dates,var = [],[],[]
        with File(self.db,'r') as hdf:
            for model in hdf.keys():
                models.append(model)
                for t0 in hdf[model]:
                    dates.append(t0)
                    for Vars in hdf[model][t0]:
                        var.append(Vars)

        self.models = unique(models)
        self.start_time = unique(dates)
        self.vars = unique(var)

    def calculate_seaStates(self,*args,**kwargs):

        with File(self.db,'r') as hdf:
            with File(self.ws,'r') as ws:
                for model in hdf.keys():
                    for t0 in hdf[model]:
                        h0, te = (hdf[model][t0][self.h0],
                                    hdf[model][t0][self.te])
                        if 'Tp' in ws.attrs.keys() and 'Hs' in ws.attrs.keys(): 
                            pass
                        else:
                            Hs = list(ws.keys())
                            Tp = list(ws[Hs[0]].keys())
                            HsVal = array([float(h[3:]) for h in Hs])
                            TpVal = array([float(t[3:]) for t in Tp])
                        self.seaState[f'{model}/{t0}'] = [(Hs[abs(HsVal-j[0]).argmin()],
                                                    Tp[abs(TpVal-j[1]).argmin()])
                                                    for i,j in enumerate(zip(h0,te))]
                
    def calculate_ffts(self,*args,**kwargs):

        def powers2(shape):
            powers = 0 
            while 2**powers <= shape:
                powers += 1
            return shape-2**(powers-1)
        
        with File(self.db,'r') as hdf:
            with File(self.ws,'r') as ws:
                print('')
                print(' ---------- Calculating FFTs --------')
                seaStates = set([s for sea in self.seaState.values() for s in sea])
                for sea in tqdm(seaStates):
                    state = f'{sea[0]}/{sea[1]}'
                    try:
                        with File(self.tempFFT,'r') as fftFile:
                            temp = fftFile[state]
                    except:
                        try:
                            with File(self.tempFFT,'a') as fftFile:
                                grp = fftFile.create_group(state)
                                time = ws[state][self.time_var]
                                length = powers2(time.shape[0])
                                data = ws[state][self.var][length:]
                                dmean = mean(data) #mean must be added to result
                                d = data-dmean
                                freq, FFT = (fftfreq(length,time[1]-time[0])[1:length//2],
                                        fft(data)[1:length//2])
                                A, phase = (2/length)*abs(FFT),arctan2(FFT.imag,FFT.real)
                                
                                grp.create_dataset('frequency',data=freq,dtype=freq.dtype) 
                                grp.create_dataset('Amplitude',data=A,dtype=A.dtype)
                                grp.create_dataset('phase',data=phase,dtype=phase.dtype)
                        except OSError as e:
                            pass


    def set_times(self,*args,**kwargs):
        with File(self.db,'r') as hdf:
            for model in hdf.keys():
                for t0 in hdf[model]:
                    key = f'{model}/{t0}'
                    times = DatetimeIndex([to_datetime(t0)+to_timedelta(
                                            int(advTime.decode('UTF-8')[1:]),
                                            unit='h')
                                            for advTime in hdf[key]['valid_time']])
                    Hs = [self.seaState[key][i][0] 
                            for i in range(hdf[key]['valid_time'].shape[0])] 
                    Tp = [self.seaState[key][i][1] 
                            for i in range(hdf[key]['valid_time'].shape[0])] 
                    
                    self.forecast_time[key] = DataFrame(array([Hs,Tp]).T,
                                                        index=times,
                                                        columns=['Hs','Tp']
                                                        )

    
    def reconstruct_powerseries(self,*args,**kwargs):
       
        with File(self.tempFFT,'r') as fftFile:
            for key, values in self.forecast_time.items():
                fcTimes = date_range(values.index[0],values.index[-1],freq='10S')
                fcInts = fcTimes.astype(int64)/10**9
                #try:
                with File(self.resultDB,'a') as resultDB:
                    result = zeros_like(fcInts)
                    grp = resultDB.create_group(key)
                    print('')
                    print(f'Calculating {key}')
                    for i,time in tqdm(enumerate(fcTimes)):
                        fftTime = values[values.index <= time]
                        fftkey = f'{fftTime["Hs"].values[-1]}/{fftTime["Tp"].values[-1]}'
                        data = fftFile[fftkey]
                        A,p,f = (data['Amplitude'][:],
                                    data['phase'][:], 
                                    data['frequency'][:])
                        result[i] = sum(A*cos(2*pi*f*fcInts[i]+p)) #missing the mean
                    saveTimes = string_([t.encode('utf-8') 
                                        for t in fcTimes.strftime('%Y-%m-%d %H:%M:%S')])
                    grp.create_dataset('time',data=saveTimes,dtype=saveTimes.dtype)
                    grp.create_dataset('powerseries',data=result,dtype=result.dtype)
                #except:
                #    pass


@jit('void(float64[::1],float64[::1],float64[::1],float64,int32)', nopython=True)
def __build_sum__(A,p,f,t,i):
    pass

def read_buoys(db,*args,**kwargs):
    b = buoy(db)
    b.read_db()

def read_forecast(db,*args,**kwargs):
    f = forecast(db)
    f.read_db()
    return f

def calculate_powerseries(db,wecSim,*args,**kwargs):
    f = forecast(db,wecSim)
    f.calculate_seaStates()
    f.calculate_ffts()
    f.set_times()
    f.reconstruct_powerseries()
