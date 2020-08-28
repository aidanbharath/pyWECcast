import os
import warnings

from h5py import File
from numpy import array, abs, zeros, zeros_like, unique, pi, cos, sin, sum, newaxis, vstack, hstack, where, equal, all, nan, where, arange, isnan, empty
from numpy import int64 as i64
from numpy import float64 as f64
from numpy.ma import masked_equal
from scipy.spatial import cKDTree
from scipy.fft import fft, fftfreq
from numba import njit, typeof, prange, float64, complex128, int64, b1, optional, intp
from numba.types.misc import Omitted
from pandas import date_range
from random import randint, seed, choice
from tqdm import tqdm


@njit(['float64[:,:](complex128[:,:],float64[:],float64[:],float64[:,:],int64,int64,b1)'],
         nogil=True,parallel=True)
def __reconstruction__(spectrum,frequency,times,result,lntime,nWECs,inPhase):
    """
        Parameters
        ----------
        spectrum : complex128 ndarray[:,:] (frequency_components, direction)
            Scaled FFT components in the desired directions
        frequency : float64 ndarray[:] (frequencies)
            frequency values associated with the spectrum
        times : float64 ndarray[:] (times)
            times values associated with the reconstruction
        result : float64 ndarray[:] (reconstrustion,direction)
            result array filled during function execution
        lntime : int64 int
            length of the time output array
        ndeg : int64 int
            length of the direction output array
        inPhase : boolean
            choose if all reconstructions should be in phase or not

        Returns:
            result : float64 ndarray[:,:] (times,direction)
                reconstruction array
    """

    for j in range(nWECs):
        if not inPhase:
            rInt = randint(0,1e8)
        else:
            rInt = 0
        for i in prange(lntime):
            Cos = cos(2*pi*frequency*(times[i]+rInt))
            Sin = sin(2*pi*frequency*(times[i]+rInt))
            s = spectrum[:,j]
            result[i,j] = sum(s.real*Cos+s.imag*Sin)
    return result


@njit(['float64[:,:](complex128[:,:],float64[:],float64[:],float64[:,:],int64,int64,int64[:])'],
         nogil=True,parallel=True)
def __reconstruction_seed__(spectrum,frequency,times,result,lntime,nWECs,Seed):
    """
        Parameters
        ----------
        spectrum : complex128 ndarray[:,:] (frequency_components, direction)
            Scaled FFT components in the desired directions
        frequency : float64 ndarray[:] (frequencies)
            frequency values associated with the spectrum
        times : float64 ndarray[:] (times)
            times values associated with the reconstruction
        result : float64 ndarray[:] (reconstrustion,direction)
            result array filled during function execution
        lntime : int64 int
            length of the time output array
        ndeg : int64 int
            length of the direction output array
        Seed : int64 ndarray[:]
            choose if all reconstructions should be in phase or not

        Returns:
            result : float64 ndarray[:,:] (times,direction)
                reconstruction array
    """

    for j in range(nWECs):
        seed(Seed[j])
        rInt = randint(0,1e8)
        for i in prange(lntime):
            Cos = cos(2*pi*frequency*(times[i]+rInt))
            Sin = sin(2*pi*frequency*(times[i]+rInt))
            s = spectrum[:,j]
            result[i,j] = sum(s.real*Cos+s.imag*Sin)
    return result


def link_sea_states(WECSim,Hs,Tp,Dir=None,kdTree=False,Seed=None):
    """
    Parameters:

    WECSim : str
        filename associated with the WECSim powerseries matrix
    Hs : float64 ndarray[:]
        array of waveheights from the sea-state data
    Tp : float64 ndarray[:]
        array of wave periods from teh sea-state data
    Dir : optional float64 ndarray[:]
        array of wave direction from the sea-state data
    kdTree : optional boolean default:False
        option to use kdtree method for finding nearest neighbours

    Returns:
        ndarray : (WECSim_Hs,WECSim_Tp)
            Hs and Tp values corresponding to available simulation in powerseries matrix
    """

    def __find_shortest__(values,wsHs,wsTp,wsseeds,Seed):
        
        flag, count = False, 0
        wsA = array([array(wsHs), array(wsTp), array(wsseeds)]).T
        Hs, Tp, seedlist = [], [], []
        for i in range(values.shape[0]):
            if isnan(values[i,:]).any() or sum(values[i,:]) == 0.:
                Hs.append(nan)
                Tp.append(nan)
                seedlist.append(nan)
                flag = True
                count += 1
            else:
                HsDiff = abs(values[i,0]-wsA[:,0])
                HsMins = where(HsDiff==HsDiff.min())[0].flatten()
                wsA_Hs = wsA[HsMins,:]
                TeDiff = abs(values[i,1]-wsA_Hs[:,1])
                TeMins = where(TeDiff==TeDiff.min())[0].flatten()
                wsA_Te = wsA_Hs[TeMins,:]

                if wsA_Te.shape[0] > 1:
                    seed(Seed)
                    wsA_Te = wsA_Te[choice(arange(0,wsA_Te[:,0].shape[0],1)),:]

                wsA_seed = wsA_Te.flatten()
                Hs.append(wsA_seed[0])
                Tp.append(wsA_seed[1])
                seedlist.append(wsA_seed[2])
            
        if flag:
            warn = f'Dataset passed contains {count} instances of null values'
            warnings.warn(warn,UserWarning)
            
        return array([Hs,Tp]), array(seedlist)


    if not Dir:
        wsHs, wsTp, wsseedlist = [], [], []
        with File(WECSim,'r') as pMatrix:
            for hs in pMatrix.keys():
                if hs != 'Time':
                    for tp in pMatrix[hs].keys():
                        for s in pMatrix[hs][tp].keys():
                            wsHs.append(float(hs[3::]))
                            wsTp.append(float(tp[3::]))
                            wsseedlist.append(float(s[5::]))


        values = array([Hs,Tp]).T
        if kdTree:
            tree = cKDTree(seas)
            _, idx = tree.query(values)
            return array([seas[int(i-1),:] for i in idx]), seedlist, values

        else:
            result, seedlist = __find_shortest__(values,wsHs,wsTp,wsseedlist,Seed)
            return result.T, seedlist, values


    elif Dir:
        pass


def calculate_fft_matrix(WECSim, Hs, Tp, seedlist, Dir=None, fftFname=f'./tempFFT.h5', inMemory=False,
                        WS_time=f'Time',WS_variable=f'Power',cutoff=None,dt=None):
    """
    Parameters:

    WECSim : str
        filename associated with the WECSim powerseries matrix
    Hs : float64 ndarray[:]
        array of waveheights available in the WECSim database
    Tp : float64 ndarray[:]
        array of wave periods available in the WECSim database
    seedlist : str list
        list of string seed names available in the WECSim db
    Dir : optional float64 ndarray[:]
        array of wave direction available in teh WECSim database
    fftFname : optional str
        filename to save the fft coeffients and frequencies on disk
    inMemory : optional boolean default:False
        option the hold the fft coefficients and frequencies in active memory and return,
        fft file will not be saved if True
    WS_Time : optional str
        WECSim time variable name in database
    WS_variable : optional str
        WECSim variable name to use in FFT
    cutoff : optional int
        Number of values to remove for the start of the timeseries
    dt : optional float64
        optional parameter to specify the dt value directly which removes
        the need for the time array

    Returns:
        dict : (coefficients complex128 ndarray[:], frequencies float64 ndarray[:])
            if inMemory:True ; returns FFT results
    """

    if inMemory: results = {}

    if not Dir:
        uni = unique([i for i in zip(Hs,Tp,seedlist)],axis=0)
        for i in range(uni.shape[0]):
            if isnan(uni[i,:]).any():
                pass
            else:
                label = f'Hs_{uni[i,0]}/Tp_{uni[i,1]}/Seed_{uni[i,2]}'
                with File(WECSim,'r') as ws:
                    time = ws[WS_time][cutoff:]
                    variable = ws[label][WS_variable][cutoff:]
                if not inMemory:
                    if os.path.isfile(fftFname):
                        os.remove(fftFname)
                    with File(fftFname,'a') as FFT:
                        if dt:  
                            coefs,freq = fft(variable), fftfreq(variable.shape[0],d=dt)
                        else:
                            coefs,freq = fft(variable), fftfreq(time.shape[0],d=time[1]-time[0])

                        FFT.create_dataset(f'{label}/frequency', data=freq, dtype=freq.dtype)
                        FFT.create_dataset(f'{label}/coefficients', data=coefs, dtype=coefs.dtype)
                else:
                    if dt:
                        results[f'{label}/frequency'] = fftfreq(variable.shape[0],d=dt)
                        results[f'{label}/coefficients'] = fft(variable)
                    else:
                        results[f'{label}/frequency'] = fftfreq(time.shape[0],d=time[1]-time[0])
                        results[f'{label}/coefficients'] = fft(variable)

    else:
        pass

    if inMemory:
        return results


def construct_powerseries(timestamps,freq,Hs,Tp,seedlist,Dir=None,fft_matrix=f'./tempFFT.h5',
                         recFile=f'./tempRecon.h5',inMemory=False,inPhase=False,WSseed=None):
    """
    Parameters:

    timestamps : datetimeIndex
        timestamps associated with the reconstruction times
    freq : str
        tiem frequency of the returned dataset
    Hs : float64 ndarray[:]
        array of wave heights available in the WECSim database
    Tp : float64 ndarray[:]
        array of wave periods available in the WECSim database
    seedlist : str list
        list of string seed names available in the WECSim db
    Dir : optional float64 ndarray[:]
        array of wave direction available in the WECSim database
    nWECs : int
        number of wecs to build into the reconstruction
    fft_matrix : str / dict
        filename or in memory data containing fft coeffients and frequencies on disk
    recFile : optional str
        filename to save the reconstructed data on disk
    inMemory : optional boolean default:False
        option the hold the fft coefficients and frequencies in active memory and return,
        fft file will not be saved if True
    inPhase : optional boolean default:False
        parameter to generate the reconstruction with fixed or random phase values
    WSseed : optional int
        value to specify what seed to use when choosing WECSim seeds

    Returns:
        dict : (coefficients complex128 ndarray[:], frequencies float64 ndarray[:])
            if inMemory:True ; returns FFT results
    """


    for i in tqdm(range(timestamps.shape[0]-1)):
        times = date_range(timestamps[i], timestamps[i+1], freq=freq)[:-1]
        intTimes = array(times.astype(i64)/10**9)
        lnTime = intTimes.shape[0]
        if isnan(array([Hs[i],Tp[i],seedlist[i]])).any():
            if type(fft_matrix) is not type(''):
                clist = [i for i in list(fft_matrix.keys()) if 'coefficients' in i]
                coefs = fft_matrix[clist[0]].shape
                construct = empty((times.shape[0],coefs[-1]))
                construct[:] = nan
            
            else:
                with File(fft_matrix,'r') as ws:
                    clist = [i for i in list(ws.keys()) if 'coefficients' in i]
                    coefs = ws[list(clist[0])].shape
                    construct = empty((times.shape[0],coefs[-1]))
                    construct[:] = nan
                
                
        else:
            ffts = f'Hs_{Hs[i]}/Tp_{Tp[i]}/Seed_{seedlist[i]}/coefficients'
            freqs = f'Hs_{Hs[i]}/Tp_{Tp[i]}/Seed_{seedlist[i]}/frequency'
            if Dir is None:
                if type(fft_matrix) is not type(''):
                    coefs, f = fft_matrix[ffts], fft_matrix[freqs]
                    if len(coefs.shape) == 1:
                        coefs = coefs[:,newaxis]
                    cShape = coefs.shape[-1]
                    construct = zeros([times.shape[0],cShape],dtype=f64)
                    N = 1/coefs.shape[0]
                    #if Seeds is None:
                    construct = N*__reconstruction__(coefs,f,intTimes,construct,lnTime,cShape,inPhase)
                    #else:
                    #    construct = N*__reconstruction_seed__(coefs,f,intTimes,construct,lnTime,cShape,Seeds)

                else:
                    with File(fft_matrix,'r') as ws:
                        coefs, f = ws[ffts][:], ws[freqs][:]
                    if len(coefs.shape) == 1:
                        coefs = coefs[:,newaxis]
                    cShape = coefs.shape[-1]
                    construct = zeros([times.shape[0],cShape],dtype=f64)
                    N = 1/coefs.shape[0]
                    #if Seeds is None:
                    construct = N*__reconstruction__(coefs,f,intTimes,construct,lnTime,cShape,inPhase)
                    #else:
                    #    construct = N*__reconstruction_seed__(coefs,f,intTimes,construct,lnTime,cShape,Seeds)

            else:
                if type(fft_matrix) is not type(''):
                    coefs, f = fft_matrix[ffts], fft_matrix[freqs]
                    if len(coefs.shape) == 1:
                        coefs = coefs[:,newaxis]
                    ceofs = Dir[i,:]*coefs
                    cShape = coefs.shape[-1]
                    construct = zeros([times.shape[0],cShape],dtype=f64)
                    N = 1/coefs.shape[0]
                    #if Seeds is None:
                    construct = N*__reconstruction__(coefs,f,intTimes,construct,lnTime,cShape,inPhase)
                    #else:
                    #    construct = N*__reconstruction_seed__(coefs,f,intTimes,construct,lnTime,cShape,Seeds)

                else:
                    with File(fft_matrix,'r') as ws:
                        coefs, f = ws[ffts][:], ws[freqs][:]
                    if len(coefs.shape) == 1:
                        coefs = coefs[:,newaxis]
                    ceofs = Dir[i,:]*coefs
                    cShape = coefs.shape[-1]
                    construct = zeros([times.shape[0],cShape],dtype=f64)
                    N = 1/coefs.shape[0]
                    #if Seeds is None:
                    construct = N*__reconstruction__(coefs,f,intTimes,construct,lnTime,cShape,inPhase)
                    #else:
                    #    construct = N*__reconstruction_seed__(coefs,f,intTimes,construct,lnTime,cShape,Seeds)

        if i == 0:
            if not inMemory:
                with File(recFile,'a') as recon:
                    t = array([time.encode() for time in times.strftime('%Y-%m-%d %H:%M:%S')])
                    recon.create_dataset(f'time', data=t, dtype=t.dtype,chunks=True,maxshape=(None,))
                    recon.create_dataset(f'reconstruction', data=construct, dtype=construct.dtype,
                                        chunks=True,maxshape=(None,construct.shape[-1]))
            else:
                results = construct
                time = times

        else:
            if not inMemory:
                with File(recFile, 'a') as recon:
                    t = array([time.encode() for time in times.strftime('%Y-%m-%d %H:%M:%S')])
                    recon['time'].resize((recon['time'].shape[0] + t.shape[0]), axis = 0)
                    recon['time'][-t.shape[0]:] = t
                    recon['reconstruction'].resize((recon['reconstruction'].shape[0] + construct.shape[0]), axis = 0)
                    recon['reconstruction'][-construct.shape[0]:] = construct
            else:
                results = vstack([results,construct])
                time = hstack([time, times])


    if inMemory: return results, time
