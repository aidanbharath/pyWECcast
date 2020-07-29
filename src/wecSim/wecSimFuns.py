import os
import sys
import subprocess
import threading
import numpy as np
import h5py as h5

from os.path import join, exists, normpath
from os import makedirs, getcwd
from glob import glob
from re import sub


def group_Hs_Te(ncbuoy, noaabuoy, ds_name, h_rounds=1, t_rounds=0):
    try:
        noaa_hc = noaabuoy['WVHT']
        noaa_te = noaabuoy['DPD']
    except KeyError as e:
        print(f'NOAA Variable DPD not available - Openning APD')
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


def wecsim_mats_to_hdf(wecSimDatDir, modelName, outputDir=None, compression=None,
                       top_label='mcr', seed_label='Seed', H_label='H',
                       T_label='T', power_label='Power', time_label='Time',
                       power_multiplier=-1, varName=f'MechPower'):
    """Create a single .hdf5 file from multiple WEC-Sim .mat output files

    Parameters
    ----------
    wecSimDatDir: str
        path to the raw WEC-Sim data (.mat files containing time-series - from
        WEC-Sim mcr)
    modelName: str
        name of the WEC-Sim model
    outputDir: str
        path to directory to save .hdf5 file (will create directory in cwd if
        none provided)

    Returns
    -------
    wecSimHdf: hdf5
        hdf5 file containing selected variables from WEC-Sim .mat output

    Notes
    -----
    - The .mat files from WEC-Sim must be saved as v7.3 format!
    - power_multiplier: WEC-Sim currently outputs RM3 power as -ve
                        power_multiplier performs power*-1 to get +ve power
                        this may change in future WEC-Sim versions
    - remaining 'label' arguments are defined in the WEC-Sim user's
          userDefinedFunctionsMCR.m file
    """

    # retrieve list of WEC-Sim .mat files from specified directory
    wecSimDatDir = normpath(wecSimDatDir)
    fileExtension = f'*.mat'
    wecSimMatFiles = glob(join(wecSimDatDir, fileExtension))
    numWecSimMatFiles = len(wecSimMatFiles)
    print(f'\npyWECcast: Number of WEC-Sim .mat files found in {wecSimDatDir}: {numWecSimMatFiles}\n')

    # define path/folder to save .hdf5 file to
    if outputDir == None:
        outputDir = join(getcwd(), 'WECSim_db')
    if not exists(outputDir):
        makedirs(outputDir)

    # check if .hdf5 already exists, and if user wants to overwrite
    dbName = join(outputDir, f'WECSim_dataset_{modelName}.hdf5')
    if exists(dbName):
        overwrite = input(f'\npyWECcast: {dbName} already exists, enter Y to overwrite : ')
        if (overwrite == 'Y' or 'y'):
            os.remove(dbName)
            print(f'\npyWECcast: old {dbName} file deleted.')
        else:
            sys.exit()

    # create hdf5 file from the separate .mat files
    for matFile in wecSimMatFiles:
        with h5.File(matFile, 'r') as mat: #shifted to a context managed approach
            # variables may need to be changed according to wec-sim .mat output
            # format - this is defined in userDefinedFunctions.m for wecSimMCR
            # Made setable in kwargs
            seed = mat[top_label][seed_label][0,0]
            Hs = round(mat[top_label][H_label][0,0],2)
            Tp = round(mat[top_label][T_label][0,0],2)
            power = mat[top_label][power_label][0,:]
            time = mat[top_label][time_label][0,:]
        with h5.File(dbName, 'a') as hdf:
            if f'Time' not in hdf.keys():
                hdf.create_dataset(f'Time', data=time, compression=compression)
            hdf.create_dataset(f'Hs_{Hs}/Tp_{Tp}/Seed_{seed}/{varName}',
                               data=power*power_multiplier,
                               compression=compression)
            # define attributes
            hsList = list(hdf.keys())[:-1] # remove 'Time' key
            for hs in hsList:
                tpList = list(hdf[hs].keys())
                for tp in tpList:
                    seedList = list(hdf[hs][tp].keys())
                    for seed in seedList:
                        varList = list(hdf[hs][tp][seed].keys())
            attrsDict = {'Hs' : hsList,
                         'Tp' : tpList,
                         'Seeds' : seedList,
                         'Vars' : varList}
            for a in attrsDict:
                hdf.attrs.create(a, attrsDict[a])
            hdf.attrs.create('index', ['/Hs/Tp/Seed/Var'])
    print(f'\npyWECcast: new {dbName} file created.')


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