import warnings

from numpy import pi, sqrt, abs, cos, linspace, array, radians, degrees, zeros, argmin, complex128, outer, round
from numpy import hstack, vstack, insert
from scipy.special import gamma
from scipy.integrate import simps
from h5py import File
from collections.abc import Iterable   


def _iter(obj):
    '''
    Convenience method for testing if an object is iterable
    '''
    return isinstance(obj, Iterable)


def __Ds__(O,Omean,Omax,s):
    '''
    Parameters:
    
    O : float64 ndarray[:]
        directional values in radians for the cosine spreading function
    Omean : float64 
        center direction for the cosine spreading function
    Omax : float64
        first sigma value for the cosine spreading function
    s : optional float64
        spreading factor for the cosine spreading function
        
    Returns:
        ndarray[:] (cosine_spreading_values)
            amplitudes for the cosine spreading function
    '''
    
    C = (sqrt(pi)*gamma(s+1))/(2*Omax)*gamma(s+0.5)
    return C*abs(cos(pi*(O-Omean)/(2*Omax)))**(2*s)


def __shift360__(shift,points=36):
    '''
    Parameters:
    
    shift : float64
        parameter in radians to shift a directional window
    window_size : optional float64
        number of points within the directional array
        
    Returns:
        ndarray[:] (direction_rad)
            shifted directional array
    '''
    
    deg = linspace(-pi,pi,points)
    result = []
    for d in deg:
        newD = d+shift
        if newD > pi:
            newD = newD - 2*pi
        elif newD < -pi:
            newD = newD + 2*pi
        result.append(newD)
    return array(result)


def __shift_window__(window,shift,output):
    '''
    Parameters:
    
    window : float64 ndarray[:]
        directional values used to shift
    shift : float64
        parameter in radians to shift a directional window
    output : float64 iterable
        number of points within the directional array
        
    Returns:
        ndarray[:] (direction_rad)
            shifted data array
    '''
    
    inc = window[1] - window[0]
    idx_shift = -int(round(shift/inc))
    l = len(output)
    result = []
    for i in range(l):
        idx = i+idx_shift
        if idx < l:
            result.append(output[idx])
        elif idx >= l:
            result.append(output[int(l-i-idx_shift)])
        elif idx < 0:
            result.append(output[int(l-idx)])
            
    return array(result)


def __cosine_spreading__(Omean,Omax,s=None,points=None,ang='rad',conv_iters=100):
    '''
    Parameters:
    
    
    Returns:
    
    
    '''
    
    if ang is not 'rad':
        Omean, Omax = radians(Omean), radians(Omax)
    O = linspace(-Omax,Omax,int(points/2))
    p = linspace(-pi,-Omax,int(points/4))
    n = linspace(Omax,pi,int(points/4))
    z = zeros([int(points/4)])
    
    S = linspace(0,2,conv_iters)
    if not s:
        Sint = [simps(__Ds__(O,0,Omax,i),O)-1 for i in S]
        s = S[argmin(abs(Sint))]

    ori = linspace(-pi,pi,points)
    temp = hstack([z,__Ds__(O,0,Omax,s),z])
    
    # weird need to make sure array output is correct size
    while ori.shape[0]-temp.shape[0] != 0:
        if (temp.shape[0] % 2) == 0:
            temp = insert(temp,0,0)
        else:
            temp = insert(temp,-1,0)
    
    D = __shift_window__(ori,Omean,temp)
    
    if ang is not 'rad':
        ori = degrees(ori)
    
    return D, ori

    
def wec_window(fft_matrix,window_type,center=0,ang='rad',**kwargs):
    '''
    Parameters:
    
    fft_matrix : str / dict
        filename or in memory data containing fft coeffients and frequencies on disk
    window_type : pyobject window function
        Function to apply as a window to the fft coefficients
    center : int
        center of the window function
    ang : string
        flag used to output values in radians ('rad') or degrees ('deg')
    **kwargs : dict
        additions arguments passed to window_type function
    
    Returns:
        dict : (coefficients complex128 ndarray[:,:], frequencies float64 ndarray[:],
                direction float64 ndarray[:])
                
            if dict passed ; returns FFT results with window applied
            else ; saves FFT results with window applied
    
    '''
    
    if type(fft_matrix) == type({}):
        d = fft_matrix['direction']
        window = window_type(d.shape[-1],**kwargs)
        shift_window = __shift_window__(d,center,window)
        for key in fft_matrix:
            if fft_matrix[key].dtype == complex128:
                fft_matrix[key] = shift_window*fft_matrix[key]
                        
        return fft_matrix
        
    elif type(fft_matrix) == type(''):
        with File(fft_matrix,'a') as hdf:
            d = hdf['direction'][:]
            window = window_type(d.shape[-1],**kwargs)
            shift_window = __shift_window__(d,center,window)
            for key in hdf:
                if hdf[key].dtype == complex128:
                    data = shift_window*fft_matrix[key][:]
                    del hdf[key]                                       
                    hdf.create_dataset(key, data=data, dtype=data.dtype,chunks=True)
                
    else:
        warn = f'-- FFT Matrix datatype not recoognised --'
        warnings.warn(warn,UserWarning)
        
        
def wave_window(fft_matrix, directions, Omax=pi/4,s=None,spreading='cosine',ang='rad'):
    '''
    Parameters:
    
    fft_matrix : str / dict
        filename or in memory data containing fft coeffients and frequencies on disk
    directions : float iterable
        iterable of mean incident wave directions (Omean in sea-state spreading)
    Omax : float / iterable
        Omax parameter used in determining the sea-state spreading function.
        if passed as iterable, len(Omax) = len(directions)
    s : optional float
        spreading parameter
        if not passed, it will be approximated
    spreading : str
        type of spreading function to use
    ang : string
        flag used to output values in radians ('rad') or degrees ('deg')
        
    Returns:
        ndarray[:,:] : (length of direction, length of degrees in FFt matrix)
            Scaling functions for each passed direction based on the spreading functions
    
    '''
    
    if type(fft_matrix) == type({}):
        points = int(fft_matrix['direction'].shape[0])
        if _iter(Omax):
            iterator = zip([directions,Omax])
        else:
            iterator = directions
        
        wave_windows = []
        for i in iterator:
            if _iter(i):
                wave_windows.append(__cosine_spreading__(i[0],i[1],s=s,points=points,ang=ang))
            else:
                wave_windows.append(__cosine_spreading__(i,Omax,s=s,points=points,ang=ang))
        
        return vstack(wave_windows)
               
    elif type(fft_matrix) == type(''):
        with File(fft_matrix,'a') as hdf:
            points = int(fft_matrix['direction'][:].shape[0])
        
        if _iter(Omax):
            iterator = zip([directions,Omax])
        else:
            iterator = directions
        
        wave_windows = []
        for i in iterator:
            if _iter(i):
                wave_windows.append(__cosine_spreading__(i[0],i[1],s=s,points=points,ang=ang))
            else:
                wave_windows.append(__cosine_spreading__(i,Omax,s=s,points=points,ang=ang))
        
        return vstack(wave_windows)
            
    else:
        warn = f'-- FFT Matrix datatype not recoognised --'
        warnings.warn(warn,UserWarning)
        

def wec_direction(fft_matrix,window=None,ang='rad'):
    
    '''
    Parameters:
    
    fft_matrix : str / dict
        filename or in memory data containing fft coeffients and frequencies on disk
    ang : str
        flag used to output values in radians ('rad') or degrees ('deg')
        
    Returns:
        dict : (coefficients complex128 ndarray[:,:], frequencies float64 ndarray[:],
                direction float64 ndarray[:])
                
            if inMemory:True ; returns FFT results with direction matrix added
            else ; direction array added and saved to dataset
    
    '''

    if type(fft_matrix) == type({}):
        for key in fft_matrix:
            if fft_matrix[key].dtype == complex128:
                points = fft_matrix[key].shape[-1]
                
        if ang == 'deg':
            fft_matrix['direction'] = linspace(-180,180,points)
        else:
            fft_matrix['direction'] = linspace(-pi,pi,points)
            
        return fft_matrix
        
    elif type(fft_matrix) == type(''):
        with File(fft_matrix,'a') as hdf:
            for key in fft_matrix:
                if fft_matrix[key].dtype == complex128:
                    points = fft_matrix[key].shape[-1]
                    
            if ang == 'deg':
                Dir = linspace(-180,180,points)
                hdf.create_dataset('direction', data=Dir, dtype=Dir.dtype,chunks=True)
            else:                
                Dir = linspace(-pi,pi,points)
                hdf.create_dataset('direction', data=Dir, dtype=Dir.dtype,chunks=True)
                
    else:
        warn = f'-- FFT Matrix datatype not recoognised --'
        warnings.warn(warn,UserWarning)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    