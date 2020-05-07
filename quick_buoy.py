import pyWECcast as wc
import h5py as h5
import numpy as np
import pandas as pd

from scipy.fft import fft, fftfreq
from tqdm import tqdm

if __name__ == "__main__":
    
    baseDir = f'/mnt/c/Users/abharath/Documents/Projects/Grid_Value'
    tempFFTsave = f'{baseDir}/git/fftTemp.h5'
    
    strp = lambda t: t.decode('utf-8')
    
    fwg = h5.File(f'{baseDir}/git/buoy_downloads.h5','r')
    
    index = fwg['46083']['time_index'][:]
    index = pd.DatetimeIndex([strp(i) for i in index])

    data = np.array([index,fwg['46083']['APD'][:],fwg['46083']['WVHT'][:]]).T
    df = pd.DataFrame(data,columns = ['time','Tp','Hs'])
    
    with h5.File(f'{baseDir}/data/WECSim_dataset_RM3_scale_0-1873.hdf5','r') as ws:
        keys = pd.DataFrame({f'{hs}/{tp}':[float(hs[3::]),float(tp[3::])] 
                            for hs in ws.keys() for tp in ws[hs].keys()},
                            index = ['Hs','Tp']).T

    def find_ss(x,ss):
        Hs = abs(x['Hs']-ss['Hs'].values).argmin()
        Tp = abs(x['Tp']-ss['Tp'].values).argmin()
        return f'Hs_{ss["Hs"].values[Hs]}/Tp_{ss["Tp"].values[Tp]}'

    df['fftKey'] = df.apply(find_ss,axis=1,args=(keys,))
    
    def calc_fft(x):
        ws = h5.File(f'{baseDir}/data/WECSim_dataset_RM3_scale_0-1873.hdf5','r')
        time, data = ws[x]['Time'][:], ws[x]['Power'][:]
        coefs,freq = fft(data), fftfreq(time.shape[0],d=time[1]-time[0])
        try:
            tmpSave = h5.File(tempFFTsave,'a')
            tmpSave.create_dataset(f'{x}/frequency',data=freq,dtype=freq.dtype)
            tmpSave.create_dataset(f'{x}/coefficients',data=coefs,dtype=coefs.dtype)
        except (RuntimeError,OSError) as e:
            pass

    
    for fftK in df['fftKey'].unique():
        calc_fft(fftK)
        
    fcTimes = pd.date_range(pd.to_datetime(df['time'].values[0]),
                         pd.to_datetime(df['time'].values[-1]),
                         freq='10S')
    fcInts = fcTimes.astype(np.int64)/10**9
    
    tDD = pd.DataFrame(np.array([fcInts,fcTimes]).T,columns = ['fcInts','fcTimes'])
    dt = df.set_index('time')
    dff = dt.reindex(fcTimes,method='pad')
    dff['fcInts'] = fcInts
    
    def reconstruct(fft,item):
        coef,freq,t = (fft[item['fftKey']]['coeficients'][:], 
                        fft[item['fftKey']]['frequency'][:], 
                           item['fcInts'])
        shape = coef.shape[0]
        Sum = np.sum(coef.real*np.cos(2*np.pi*freq*t)+coef.imag*np.sin(2*np.pi*freq*t))
        return (2/shape)*Sum

    #result = pd.DataFrame(np.zeros(dff.shape[0]),index=fcTimes,columns=['powerseries'])
    print(np.zeros(dff.shape[0]))
    result = np.zeros(dff.shape[0])
    with h5.File(tempFFTsave,'r') as fft:
        j = 0
        for i,item in tqdm(dff.iterrows()):
            #result[i] = reconstruct(fft,item)
            #print(item)
            coef,freq,t = (fft[item['fftKey']]['coeficients'][:], 
                        fft[item['fftKey']]['frequency'][:], 
                           item['fcInts'])
            shape = coef.shape[0]
            result[j] = (2/shape)*np.sum(coef.real*np.cos(2*np.pi*freq*t)+
                                         coef.imag*np.sin(2*np.pi*freq*t))
            #print(result[i])
            j += 1
            
    result = pd.DataFrame(result,index=fcTimes,columns=['powerseries'])
    result.to_csv('./fairweather_grounds_buoy.csv')