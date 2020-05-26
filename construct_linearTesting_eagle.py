import re
import sys
import random as rand
import numpy as np
import construct_timeseries as ct
import h5py as h5
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

from scipy.fftpack import fft
from itertools import product
from glob import glob

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def clean_wording(w):
    return re.sub('./WECSim_dataset_', '', re.sub('.hdf5', '', w))


if __name__ == "__main__":

    func = np.mean
    plotData = True
    plotWSstats = True
    plotRolling = True
    plotMeans = True
    constantSeed = False

    freq, variable,f  = 10, 'EPower', 's'
    transitionLength = 2000 # Seconds
    varianceWindow = 1 # freq*60*varianceWindow
    method = 'linear'

    data = pd.read_excel('./aug_08_14-forecasts.xlsx')
    times = ct.Times(pd.to_datetime(data['date']), freq, f=f)
    if constantSeed:
        seeds = [1 for i in range(len(times))]
    else:
        seeds = [rand.choice([1, 2, 3]) for i in range(len(times))]
    choice = ct.choices(seeds, data['Hm'], data['Tm'])
    selection = ct.create_selection(choice, times, freq)

    h5File = glob(f'./*.hdf5')

    ts,label = [], []
    for h5F in h5File:
        label.append(f'{h5F}')
        with h5.File(h5F, 'r+') as ds:
            print(f'Begin Processing on - {ds}')
            for c in selection:
                try:
                    a = ds[f'{c[0][0]}/{c[0][1]}/{c[0][2]}/{variable}']
                except KeyError:
                    sys.exit(f'{c[0][0]}/{c[0][1]}/{c[0][2]} - not in Dataset')

            timeseries = pd.date_range(times[0][0], times[-1][-1], freq=f'{freq}{f}')
            selections = {'selections':selection,'timeseries':timeseries}

            fft = ct.calculate_fft(ds, selections, variable)
            ts.append(ct.linear_transition_timeseries(fft,
                                        transitionLength=transitionLength,
                                        type=method))
            print(f'Complete Processing')

    color = ['r', 'b']
    if plotData:
        skip = 2
        plt.figure()
        for i,t in enumerate(ts):
            plt.plot(t[0].iloc[::skip].index, t[0].iloc[::skip],
                     label=f'{clean_wording(label[i])}', alpha=0.4,
                     color=color[i])

    if plotWSstats:
    # raw data variance
        sets = []
        for h5F in h5File:
            label.append(f'{h5F}')
            variance,time = [],[]
            with h5.File(h5F, 'r+') as ds:
                for s in selections['selections']:
                    dvar = func(ds[s[0][0]][s[0][1]][s[0][2]][variable][5000::])
                    variance.append([dvar for i in range(s[1].shape[0])])
                    time.append(s[1])
            variance = np.array(variance).flatten()
            time = np.array(time).flatten()
            sets.append(pd.Series(np.stack(variance, axis=0),
                                index=np.stack(time, axis=0)))
        for s in sets:
            plt.plot(s.index, s, label='Raw Data Variance')

    if plotRolling:
        for i, t in enumerate(ts):
            rv = t[0].rolling(int(freq*60*varianceWindow), center=True).apply(func, raw=True)
            plt.plot(rv.index, rv, label=f'{clean_wording(label[i])} STD',
                     color=color[i])

    if plotMeans:
        means = []
        for i, s in enumerate(sets):
            subMeans = []
            df = pd.DataFrame(s)
            for j in df.groupby(0):
                start = j[1].index[0] + pd.Timedelta(f'{transitionLength/2}s')
                stop = j[1].index[-1] - pd.Timedelta(f'{transitionLength/2}s')
                mean = ts[i][0].loc[start:stop].apply(func, raw=True).values[0]
                subMeans.append(pd.Series([mean for k in range(j[1].index.shape[0])],
                                index=j[1].index))
            means.append(pd.concat(subMeans, axis=0).sort_index())
        for mean in means:
            plt.plot(mean.index, mean.values, label='predicted group mean')

    plt.title(f'Transistion Length {transitionLength} s, STD Window {varianceWindow} hrs')
    plt.ylabel(f'{variable} STD')
    plt.xlabel('Time')
    plt.legend(loc='upper right')
    plt.show()