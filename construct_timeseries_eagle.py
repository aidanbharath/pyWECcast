import h5py as h5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.fftpack import fft, fftfreq, ifft
from glob import glob
from random import choice, randrange
from itertools import permutations, product
from scipy.interpolate import interp1d


def cuts(shape):
    powers = 0
    while 2**powers <= shape:
        powers += 1
    return shape - 2**(powers - 1)


def FFT(time, data):
    samples = time.shape[0]
    y = fft(data)
    spacing = (time[1] - time[0])
    freq = fftfreq(samples, spacing)
    y = y[1:samples//2]
    return (2/samples)*np.abs(y), np.arctan2(y.imag, y.real), freq[1:samples//2]


def freqLimit(A, p, freq, limit):
    idx = np.argwhere(freq <= limit).max()
    return A[:idx], p[:idx], freq[:idx]


def activator(on, times, transType, transitionLength, type):
    tl = transitionLength
    start = min(on)+tl/2
    stop = max(on)-tl/2
    values = []
    for t in times:
        if t >= start and t < stop:
                values.append(1)
        elif t >= start-tl and t < start:
            if transType[0] == 'linear':
                x0, x1 = start-tl, start
                y0, y1 = 0, 1
                values.append(y0 + (t - x0)*((y1 - y0)/(x1 - x0)))
            elif transType[0] == 'sqrt':
                values.append(np.sqrt((t - (start - tl))/(tl)))
        elif t >= stop and t < stop+tl:
            if transType[1] == 'linear':
                x0, x1 = stop, stop+tl
                y0, y1 = 1, 0
                values.append(y0 + (t - x0)*((y1 - y0)/(x1 - x0)))
            elif transType[1] == 'sqrt':
                values.append(np.sqrt(1 - (t - (stop))/(tl)))
        else:
            values.append(0)
    return np.array(values)


def linear_transition_timeseries(ffts, transitionLength=1800, type='linear'):
    unix = ffts['timeseries'].astype(np.int64)/10**9
    summs,actives = [], []
    for fft in ffts['fft']:
        fftActive = fft[-1].astype(np.int64)//10**9
        active = activator(fftActive, unix, fft[5], transitionLength, type)
        series = np.zeros_like(unix)
        for t in range(unix.shape[0]):
            if active[t] != 0.0:
                ftp = (2*np.pi*fft[3]*unix[t]+fft[2])
                series[t] = (fft[4]+np.sum(fft[1]*np.cos(ftp)))*active[t]
            else:
                series[t] = 0
        actives.append(active)
        summs.append(series)
    series = np.array(summs).sum(axis=0)
    return pd.DataFrame(series, index=ffts['timeseries']), actives


def spline_coef_transition(ffts, transitionLength=1800, init_phase=0,
                           processPhase=True, ampMax=False):

    def Phase(T, tmi, tma, ampMax):
        if ampMax:
            f = T[3][T[1].argmax()]
        else:
            f = 1/np.float64(T[0][-1][2::])
        periods = (f*(tma - tmi))/(2*np.pi)
        return (periods - int(periods))*2*np.pi, f

    coefs = ffts['fft'][0][3]
    tl = transitionLength
    unix = ffts['timeseries'].astype(np.int64)//10**9
    unix = unix - unix.min()
    timeseries = np.zeros_like(unix)

    if processPhase:
        phase = init_phase
        for i, fft in enumerate(ffts['fft']):
            t = fft[-1].astype(np.int64)//10**9
            tmi, tma = t.min(), t.max()
            P, f  = Phase(fft, tmi, tma, ampMax=ampMax)
            fIdx = np.abs(coefs - f).argmin()
            pPhase = fft[2][fIdx]
            phaseDiff = phase - pPhase
            ffts['fft'][i][2] = fft[2] + (phaseDiff*(fft[2]/f))
            phase = P

    for i in range(len(coefs)):
        A, p, time = [], [], []

        for fft in ffts['fft']:
            t = fft[-1].astype(np.int64)//10**9
            tmi, tma = t.min(), t.max()
            tmin, tmax = tmi+tl/2, tma-tl/2
            t = t[(t > tmin) & (t < tmax)]

            for j in t:
                A.append(fft[1][i])
                p.append(fft[2][i])
                time.append(j)

        f = fft[3][i]
        kind = 'cubic'
        fill = 'extrapolate'
        time  = np.array(time)
        time = time-time.min()
        Af = interp1d(time, A, kind, fill_value=fill)
        pf = interp1d(time, p, kind, fill_value=fill)
        timeseries = timeseries+Af(unix)*np.cos(np.pi*2*f*unix+pf(unix))

    timeseries = pd.DataFrame(timeseries, index=ffts['timeseries'])

    for fft in ffts['fft']:
        t = fft[-1]
        timeseries[t[0]:t[-1]] = timeseries[t[0]:t[-1]] + fft[4]

    return timeseries


def random_selection(ds, timeRange, nStates, freq='10S'):
    choices = list(product(ds.attrs['Seeds'], ds.attrs['H'], ds.attrs['T']))
    start,stop = timeRange
    timeseries = pd.date_range(start, stop, freq=freq)
    selections = []

    for state in range(nStates):
        slice = randrange(int(timeseries.shape[0]/(nStates - state)))
        if state != nStates-1:
            selections.append([choice(choices), timeseries[0:slice]])
            timeseries = timeseries[slice::]
        else:
            selections.append([choice(choices), timeseries])

    return {'selections':selections, 'timeseries':pd.date_range(start, stop, freq=freq)}


def specific_selection(ds, timeRange, nStates, freq='10S'):
    pass


def calculate_fft(ds, selections, var, limit=1):

    def calc(i, select, transistion):
        d = ds[f'{select[0][0]}/{select[0][1]}/{select[0][2]}/{var}']
        time = ds[f'{select[0][0]}/{select[0][1]}/{select[0][2]}/time']
        cut = cuts(time.shape[0])
        dmean = d[cut:].mean()
        d = d[cut:] - dmean
        A, p, freq = FFT(time[cut:], d)
        A, p, freq = freqLimit(A, p, freq, limit)
        return [select[0], A, p, freq, dmean, transistion, select[1]]

    fft = []
    selects = selections['selections']

    for i,select in enumerate(selects):
        transistion = ['sqrt', 'sqrt']
        try:
            fc = ((selects[i][0][1]==selects[i+1][0][1]) &
                  (selects[i][0][2]==selects[i+1][0][2]))
            bc = ((selects[i][0][1]==selects[i-1][0][1]) &
                  (selects[i][0][2]==selects[i-1][0][2]))
            if fc: transistion[1] = 'linear'
            if bc: transistion[0] = 'linear'
            fft.append(calc(i, select, transistion))
        except IndexError:
            if i == 0:
                fc = ((selects[i][0][1] == selects[i+1][0][1]) &
                      (selects[i][0][2] == selects[i+1][0][2]))
                if fc: transistion[1] = 'linear'
                transistion[0] = 'linear'
            else:
                bc = ((selects[i][0][1] == selects[i-1][0][1]) &
                      (selects[i][0][2]==selects[i-1][0][2]))
                transistion[1] = 'linear'
                if bc: transistion[0] = 'linear'
            fft.append(calc(i, select, transistion))
    return {'fft':fft, 'timeseries':selections['timeseries']}


def Times(date, freq, gap='3h', f='s'):
    times = []
    for i in range(len(date)):
        try:
            times.append([date[i], date[i+1] - pd.Timedelta(f'{freq}{f}')])
        except KeyError:
            endTime = date[i] - pd.Timedelta(f'{freq}s') + pd.Timedelta(gap)
            times.append([date[i], endTime])
    return times


def choices(seeds, H, T):
    seeds = [f'Seed-{s}' for s in seeds]
    H = [f'H-{h}' for h in H]
    T = [f'T-{t}' for t in T]
    return zip(seeds, H, T)


def create_selection(choice, times, freq):
    selection = []
    for i, pick in enumerate(choice):
        selection.append([pick, pd.date_range(times[i][0], times[i][1],
                          freq=f'{freq}s')])
    return selection


def calc_DWP(buoy):
    he = np.float64(buoy.loc[:, 'WVHT'].values)
    swell = np.float64(buoy.loc[:, 'SwH'].values)
    windw = np.float64(buoy.loc[:, 'WWH'].values)
    DWP = []

    for i, j in enumerate(swell>=windw):
        if j:
            DWP.append(np.float64(buoy.iloc[i].loc['SwP']))
        elif not j:
            DWP.append(np.float64(buoy.iloc[i].loc['WWP']))

    return pd.Series(DWP, index=buoy.index, name='DWP')


if __name__ == "__main__":
    pass