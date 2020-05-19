import os
import h5py as h5
import xarray as xr
import pandas as pd

from random import choice
from numpy import nan, array, abs, float32, arctan2, argwhere, int64, sum, pi, cos
from scipy.fftpack import fft, fftfreq
from tqdm import tqdm
from shutil import rmtree


def extract_variables(File, Hs, Te, dwp=True):
    if type(File) is xr.Dataset:
        thp = {}
        start_time = File.sim_start.values
        print(f'Beginning file slicing on {type(File)}')

        for st in tqdm(start_time):
            t = File.sel(sim_start = st)
            hs, te = (t[Hs].dropna(dim = 'valid_time'),
                      t[Te].dropna(dim = 'valid_time'))
            startGroup = []
            times = hs.valid_time.values

            for i, time in enumerate(times):
                try:
                    startGroup.append((time, times[i+1],
                                       str(hs.sel(valid_time = time).values),
                                       str(te.sel(valid_time = time).values)))
                except IndexError as e:
                    pass
            thp[st] = startGroup

    elif type(File) is pd.core.frame.DataFrame:
        times = File.index
        thp = []
        print(f'Beginning file slicing on {type(File)}')

        if dwp:
            DWP = calc_DWP(File)
            for i, time in tqdm(enumerate(times)):
                try:
                    thp.append((time, times[i+1], File.loc[time, Hs], f'{round(DWP[time], 2)}'))
                except IndexError as e:
                    pass
        else:
            for i, time in tqdm(enumerate(times)):
                try:
                    thp.append((time, times[i+1], File.loc[time, Hs], File.loc[time, Te]))
                except IndexError as e:
                    pass
    return thp


def timeseries_params(ts, model_db, freq=10, f='s', closest=True):
    strip = lambda x: [float(i.split('-')[-1]) for i in x]
    model = {}
    for db in model_db:
        ws = h5.File(db, 'r+')
        if closest:
            if type(ts) is list:
                selection = []
                print('Beginning timeseries parameter collection')
                for t in tqdm(ts):
                    seed = choice(ws.attrs['Seeds'])
                    Hs = list(ws[seed].keys())
                    Hs = Hs[abs(array(strip(Hs)) - float32(t[2])).argmin()]
                    Te = list(ws[seed][Hs].keys())
                    Te = Te[abs(array(strip(Te)) - float32(t[3])).argmin()]
                    selection.append(((seed, Hs, Te), pd.date_range(t[0], t[1], freq=f'{freq}{f}')))
                output = selection

            elif type(ts) is dict:
                output = {}
                print('Beginning timeseries parameter collection')
                for tkey, tdata in tqdm(ts.items()):
                    selection = []
                    for t in tdata:
                        seed = choice(ws.attrs['Seeds'])
                        Hs = list(ws[seed].keys())
                        print(Hs, t)
                        Hs = Hs[abs(array(strip(Hs)) - float32(t[2])).argmin()]
                        Te = list(ws[seed][Hs].keys())
                        Te = Te[abs(array(strip(Te)) - float32(t[3])).argmin()]
                        selection.append(((seed, Hs, Te), pd.date_range(t[0], t[1], freq=f'{freq}{f}')))
                    output[tkey] = selection
        ws.close()
        model[db] = output
    return model


def cuts(shape):
    powers = 0
    while 2**powers <= shape:
        powers += 1
    return shape-2**(powers - 1)


def FFT(time, data):
    samples = time.shape[0]
    y = fft(data)
    spacing = (time[1] - time[0])
    freq = fftfreq(samples, spacing)
    y = y[1:samples//2]
    return (2/samples)*abs(y), arctan2(y.imag, y.real), freq[1:samples//2]


def freqLimit(A, p, freq, limit):
    idx = argwhere(freq<=limit).max()
    return A[:idx], p[:idx], freq[:idx]


def calculate_ffts(selections, variable='EPower', freqCut=1):
    FFTs = {}
    for db,selection in selections.items():
        if type(selection) is list:
            values = set([select[0] for select in selection])
            fft = {}
            ws = h5.File(db,'r+')
            print(f"Calculating NOAA buoy FFTs on - {db.split('/')[-1]}")

            for v in tqdm(values):
                d = ws[f'{v[0]}/{v[1]}/{v[2]}/{variable}']
                time = ws[f'{v[0]}/{v[1]}/{v[2]}/time']
                cut = cuts(time.shape[0])
                dmean = d[cut:].mean()
                d = d[cut:]-dmean
                A, p, freq = FFT(time[cut:], d)
                A, p, freq = freqLimit(A, p, freq, freqCut)
                fft[v] = [A, p, freq, dmean]
            ws.close()
            ffts = [[select[0], fft[select[0]], select[1]] for select in selection]

        elif type(selection) is dict:
            ffts, values = {}, []
            print(f"Calculating Model FFTs on - {db.split('/')[-1]}")
            for skey, sdata in selection.items():
                values.append(list(set([select[0] for select in sdata])))
            ws = h5.File(db, 'r+')

            for v in tqdm(set([v for value in values for v in value])):
                d = ws[f'{v[0]}/{v[1]}/{v[2]}/{variable}']
                time = ws[f'{v[0]}/{v[1]}/{v[2]}/time']
                cut = cuts(time.shape[0])
                dmean = d[cut:].mean()
                d = d[cut:]-dmean
                A, p, freq = FFT(time[cut:], d)
                A, p, freq = freqLimit(A, p, freq, freqCut)
                ffts[v] = [A, p, freq, dmean]
            ws.close()
        FFTs[db] = ffts
    return FFTs


def linear_transition_timeseries(datasets, transLength=1800):
    for db, dataset in datasets.items():
        fullTS = [flat for ds in dataset for flat in ds[-1]]
        fullnTS = [flat for ds in dataset for flat in array(ds[-1]).astype(int64)/10**9]
        df = pd.Series(index=fullnTS)
        print(f'Reconstructing NOAA Buoy Timeseries - {db.split("/")[-1]}')

        for ds in tqdm(dataset):
            times = array(ds[-1]).astype(int64)/10**9
            for i, time in enumerate(times):
                df[time] = sum(ds[1][0]*cos(2*pi*ds[1][2]*time+ds[1][1]))+ds[1][-1]

        return pd.Series(df.values, index=fullTS)


def linear_transition_timeseries_model(ffts, times, buoy, transitionLength=1800):
    for keys, starts in times.items():
        if not os.path.isdir('./working/'):
            os.mkdir('./working/')
        print(f'Reconstructing Model Timeseries - {keys.split("/")[-1]}')

        for skey, start in tqdm(starts.items()):
            fullTS, fullnTS = [], []
            for i, ds in enumerate(start):
                fullTS.append([flat for flat in ds[-1]])
                fullnTS.append([(ds[0],d) for d in array(ds[-1]).astype(int64)/10**9])

            fullTS = [ts for tsgrp in fullTS for ts in tsgrp[:-1]]
            fullnTS = [ts for tsgrp in fullnTS for ts in tsgrp[:-1]]
            df = pd.Series(index=fullTS,name=f"{skey}")

            for i, j in enumerate(fullnTS):
                fft = ffts[keys][j[0]]
                df[fullTS[i]] = sum(fft[0]*cos(2*pi*fft[2]*j[-1]+fft[1]))+fft[-1]

            df.to_pickle(f'./working/epower-{skey}.pkl')
            #da = xr.DataArray.from_series(df)
            #da.to_netcdf(f'./working/epower-{skey}.nc',engine='netcdf4')
        '''
        df = xr.open_mfdataset(f'./working/*',combine='nested',concat_dim=None)
        df.load()
        saved, ext = False, 0
        while saved is False:
            try:
                df.to_netcdf(f'./Model_Forecast-{buoy}-{keys.split("/")[-1].split(".")[0]}{ext}.nc',
                        engine='netcdf4')
                saved = True
            except PermissionError as e:
                ext += 1
        df.close()
        #rmtree('./working', ignore_errors=True)
        '''


def calc_DWP(buoy):
    he = float32(buoy.loc[:, 'WVHT'].values)
    swell = float32(buoy.loc[:, 'SwH'].values)
    windw = float32(buoy.loc[:, 'WWH'].values)
    DWP = []

    for i, j in enumerate(swell >= windw):
        if j:
            DWP.append(float32(buoy.iloc[i].loc['SwP']))
        elif not j:
            DWP.append(float32(buoy.iloc[i].loc['WWP']))

    return pd.Series(DWP, index=buoy.index, name='DWP')