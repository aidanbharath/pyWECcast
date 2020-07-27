import os
import warnings

from h5py import File
from numpy import array, abs, zeros, zeros_like, unique, pi, cos, sin, sum, newaxis, vstack, hstack, where, equal, all
from numpy import int64 as i64 
from numpy import float64 as f64
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
    
    def __find_shortest__(values,seas,result,idx,Seed):
        for i in prange(idx):
            seed(Seed)
            result[i,0] = seas[abs(values[i,0]-seas[:,0]).argmin(),0]
            result[i,1] = seas[abs(values[i,1]-seas[choice(where(seas[:,0]==result[i,0])[0]),1]).argmin(),1]
        return result
            
    
    if not Dir:
        seas, seedlist = [], []
        with File(WECSim,'r') as pMatrix:
            for i in range(3): ## testing only
                for seedName in pMatrix.keys():
                    seas.append(array([[float(hs[3::]),float(tp[3::])] 
                                for hs in pMatrix[seedName].keys() 
                                  for tp in pMatrix[seedName][hs].keys()]
                                ))
                    seedlist.append(seedName)
        
        # Equal sea-state check between seeds
        check = array(seas).sum(axis=1)
        Check = lambda x: [i for i in range(len(x)-1) if x[i]!=x[i+1]]
    
        if Check(check[:,0]) and not Check(check[:,1]):
            warn = f' -- Modeled sea-state missing --'
            warnings.warn(warn,UserWarning)
            
        values = array([Hs,Tp]).T
                
        if kdTree:
            tree = cKDTree(seas)
            _, idx = tree.query(values)
            return array([seas[int(i-1),:] for i in idx]), seedlist, values

        else:
            result = zeros_like(values,dtype=f64)
            result = __find_shortest__(values,seas[0],result,result.shape[0],Seed)
            return result, seedlist, values
            
            
    elif Dir:
        pass

    
def calculate_fft_matrix(WECSim, Hs, Tp, seedlist, Dir=None, fftFname=f'./tempFFT.h5', inMemory=False,
                        WS_time=f'Time',WS_variable=f'Power',cutoff=None):
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
        
    Returns:
        dict : (coefficients complex128 ndarray[:], frequencies float64 ndarray[:])
            if inMemory:True ; returns FFT results
    """
    
    if inMemory: results = {}
    
    if not Dir:
        uni = unique([i for i in zip(Hs,Tp)],axis=0)
        for seed in seedlist:
            for i in range(uni.shape[0]):
                label = f'{seed}/Hs_{uni[i,0]}/Tp_{uni[i,1]}'
                with File(WECSim,'r') as ws:
                    time = ws[label][WS_time][cutoff:]
                    variable = ws[label][WS_variable][cutoff:]
                if not inMemory:
                    if os.path.isfile(fftFname):
                        os.remove(fftFname)
                    with File(fftFname,'a') as FFT:
                        coefs,freq = fft(variable), fftfreq(time.shape[0],d=time[1]-time[0])
                        FFT.create_dataset(f'{label}/frequency', data=freq, dtype=freq.dtype)
                        FFT.create_dataset(f'{label}/coefficients', data=coefs, dtype=coefs.dtype)
                else:
                    results[f'{label}/frequency'] = fftfreq(time.shape[0],d=time[1]-time[0])
                    results[f'{label}/coefficients'] = fft(variable) 
                    
    else:
        pass
    
    if inMemory:
        return results
    
    
def construct_powerseries(timestamps,freq,Hs,Tp,seedlist,Dir=None,fft_matrix=f'./tempFFT.h5',
                         recFile=f'./tempRecon.h5',inMemory=False,inPhase=False,Seeds=None,WSseed=None):
       
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
    Seeds : optional int64 ndarray[:]
        seeds to assign to the random phase value, used for repeated random seed generation
        Length of Seeds must be equal to the nWECs
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
        seed(WSseed)
        Choice = choice(seedlist)
        ffts = f'{Choice}/Hs_{Hs[i]}/Tp_{Tp[i]}/coefficients'
        freqs = f'{Choice}/Hs_{Hs[i]}/Tp_{Tp[i]}/frequency'
        
        if Dir is None:
            if type(fft_matrix) is not type(''):
                coefs, f = fft_matrix[ffts], fft_matrix[freqs]
                if len(coefs.shape) == 1:
                    coefs = coefs[:,newaxis]
                cShape = coefs.shape[-1]
                construct = zeros([times.shape[0],cShape],dtype=f64)
                N = 1/coefs.shape[0]
                if Seeds is None:
                    construct = N*__reconstruction__(coefs,f,intTimes,construct,lnTime,cShape,inPhase)
                else:
                    construct = N*__reconstruction_seed__(coefs,f,intTimes,construct,lnTime,cShape,Seeds)
        
            else:
                with File(fft_matrix,'r') as ws:
                    coefs, f = ws[ffts][:], ws[freqs][:]
                if len(coefs.shape) == 1:
                    coefs = coefs[:,newaxis]
                cShape = coefs.shape[-1]
                construct = zeros([times.shape[0],cShape],dtype=f64)
                N = 1/coefs.shape[0]
                if Seeds is None:
                    construct = N*__reconstruction__(coefs,f,intTimes,construct,lnTime,cShape,inPhase)
                else:
                    construct = N*__reconstruction_seed__(coefs,f,intTimes,construct,lnTime,cShape,Seeds)
                    
        else:
            if type(fft_matrix) is not type(''):
                coefs, f = fft_matrix[ffts], fft_matrix[freqs]
                if len(coefs.shape) == 1:
                    coefs = coefs[:,newaxis]
                ceofs = Dir[i,:]*coefs
                cShape = coefs.shape[-1]
                construct = zeros([times.shape[0],cShape],dtype=f64)
                N = 1/coefs.shape[0]
                if Seeds is None:
                    construct = N*__reconstruction__(coefs,f,intTimes,construct,lnTime,cShape,inPhase)
                else:
                    construct = N*__reconstruction_seed__(coefs,f,intTimes,construct,lnTime,cShape,Seeds)
                    
            else:
                with File(fft_matrix,'r') as ws:
                    coefs, f = ws[ffts][:], ws[freqs][:]
                if len(coefs.shape) == 1:
                    coefs = coefs[:,newaxis]
                ceofs = Dir[i,:]*coefs
                cShape = coefs.shape[-1]
                construct = zeros([times.shape[0],cShape],dtype=f64)
                N = 1/coefs.shape[0]
                if Seeds is None:
                    construct = N*__reconstruction__(coefs,f,intTimes,construct,lnTime,cShape,inPhase)
                else:
                    construct = N*__reconstruction_seed__(coefs,f,intTimes,construct,lnTime,cShape,Seeds)
        
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
            