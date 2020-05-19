import os
import subprocess
import threading
import numpy as np
import h5py as h5

from glob import glob
from re import sub


def group_Hs_Te(ncbuoy, noaabuoy, ds_name, h_rounds=1, t_rounds=0):
    try:
        noaa_hc = noaabuoy['WVHT']
        noaa_te = noaabuoy['DPD']
    except KeyError as e:
        print(f'NOAA Variable DPD no available - Openning APD')
        noaa_hc = noaabuoy['WVHT']
        noaa_te = noaabuoy['APD']

    nc_hc, nc_te = (np.round(ncbuoy.swh.values, decimals=h_rounds),
                    np.round(ncbuoy.perpw.values, decimals=t_rounds))
    modelGroup = []

    for i,j in np.ndenumerate(nc_hc):
        x,y = nc_hc[i[0],i[1]], nc_te[i[0],i[1]]
        if ~np.isnan(x) and ~np.isnan(y):
            modelGroup.append((x, y))

    modelGroup = set(modelGroup)
    buoyGroup = []

    for i in range(noaa_hc.shape[0]):
        buoyGroup.append((np.round(noaa_hc.iloc[i], decimals=h_rounds),
                            np.round(noaa_te.iloc[i], decimals=t_rounds)))

    buoyGroup = set(buoyGroup)
    required = set(list(buoyGroup) + list(modelGroup))
    dbName = f'./WECSim_db/WECSim_dataset_{ds_name}.hdf5'
    sfloat = lambda x: float(x.split('-')[-1])
    wecSims = []

    with h5.File(dbName, 'r') as ds:
        seeds = list(ds.keys())
        for s in seeds:
            H = list(ds[s].keys())
            for h in H:
                T = list(ds[s][h].keys())
                for t in T:
                    wecSims.append((sfloat(h), sfloat(t)))

    wecSims = list(set(wecSims))

    return [i for i in required if i not in wecSims]


def update_database(ds_name, oSys='linux', compression=None):
    if oSys == 'windows':
        spl = '\\'
        wecSim = f'C:\\Users\\abharath\\Documents\\wecSIM\\grid-Project-Sims\\{ds_name}\\'
    elif oSys == 'linux':
        spl = '/'
        wecSim = f'/mnt/c/Users/abharath/Documents/wecSIM/grid-Project-Sims/{ds_name}/'

    fname = f'*.mat'
    dbName = f'.{spl}WECSim_db{spl}WECSim_dataset_{ds_name}.hdf5'
    matFiles = wecSim
    fname = f'*.mat'
    ds_name = matFiles.split(f'{spl}')[-2]

    print(f'Opening file -- {ds_name}')
    with h5.File(dbName, 'a') as ds:
        for name in glob(f'{matFiles}{spl}output*{spl}{fname}'): # sometimes this needs to be specific
            with h5.File(name, 'r') as mat:
                for key in mat['Power'].keys():
                    case = [sub('.mat', '', n)
                            for n in name.split(f'{spl}')[-1].split('_')[1:]]
                    if type(mat['Power'][key]) == h5._hl.dataset.Dataset:
                        grp = f'{case[0]}/{case[1]}/{case[2]}/{key}'
                        if grp not in ds:
                            ds.create_dataset(grp, data=mat['Power'][key][0,:],
                                              compression=compression)
                            ds[grp].dims[0].label = 'time'
                    else:
                        for subkey in mat['Power'][key].keys():
                            grp = f'{case[0]}/{case[1]}/{case[2]}/{key}/{subkey}'
                            if grp not in ds:
                                ds.create_dataset(grp,
                                                  data = mat['Power'][key][subkey][0,0],
                                                  compression = compression)
        seeds = list(ds.keys())
        for s in seeds:
            H = list(ds[s].keys())
            for h in H:
                T = list(ds[s][h].keys())
                for t in T:
                    var = list(ds[s][h][t].keys())

        ds.attrs.create('Seeds', seeds)
        ds.attrs.create('H', H)
        ds.attrs.create('T', T)
        ds.attrs.create('Vars', var)
        ds.attrs.create('index', ['/seed/H/T/var'])


def run_WECSim(ds_name, requiredRuns, oSys='linux', compression=None):
    wecSim_done = threading.Event()

    def calc():
        p = subprocess.Popen('matlab -nojvm -nosplash -wait -r "disp(2+2);exit"')
        wecSim_done.set()

    if oSys == 'windows':
        wecSim = f'C:\\Users\\abharath\\Documents\\wecSIM\\grid-Project-Sims\\{ds_name}-Run\\'
    elif oSys == 'linux':
        wecSim = f'/mnt/c/Users/abharath/Documents/wecSIM/grid-Project-Sims/{ds_name}-Run/'

    cwd = os.getcwd()
    i = 0

    for he,tc in requiredRuns:
        He = f'waves.H = {he};                     % Significant Wave Height [m]\n'
        Tc = f'waves.T = {tc};                     % Peak Period [s]\n'
        os.chdir(wecSim)
        print(os.getcwd())

        with open(f'wecSimInputFile.m', 'r') as File:
            wecSimInput = File.readlines()

        wecSimInput[24] = He
        wecSimInput[25] = Tc

        with open(f'wecSimInputFile.m', 'w') as File:
            File.writelines(wecSimInput)
        '''
        thread = threading.Thread(target=calc)
        thread.start()
        wecSim_done.wait()
        '''

        system = 'RM#_hydraulicPTO'
        p = os.system(f'matlab -nojvm -nosplash -wait -r "wecSim;save_system({system});exit"')
        i += 1
        print(p)