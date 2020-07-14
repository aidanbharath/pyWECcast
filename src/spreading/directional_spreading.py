import warnings

from numpy import pi, sqrt, abs, cos, linspace, array, radians, degrees, zeros, argmin, complex128, outer, round
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


def __cosine_spreading__(Omean,Omax,s=None,points=None,ang='rad'):
    '''
    Parameters:
    
    
    Returns:
    
    
    '''
    
    if ang is not 'rad':
        Omean, Omax = radians(Omean), radians(Omax)
    
    O = linspace(-Omax,Omax,int(points)/2)
    p = linspace(-pi,-Omax,int(points/4))
    n = linspace(Omax,pi,int(point/4))
    z = zeros([int(points)])
    
    if not s:
        Sint = [simps(Ds(O,0,Omax,i),O)-1 for i in linspace(0,2,1000)]
        s = S[argmin(abs(Sint))]

    ori = linspace(-pi,pi,points)
    D = __shift_window__(ori,Omean,hstack([z,Ds(O,Omean,Omax,s),z]))
    
    if ang is not 'rad':
        ori = degrees(ori)
    
    return D, ori
    
def wec_window(fft_matrix,window_type,center=0,ang='rad',**kwargs):
    '''
    Parameters:
    
    Returns:
    '''
    
    if type(fft_matrix) == type({}):
        d = fft_matrix['direction']
        window = window_type(d.shape[-1],**kwargs)
        shift_window = __shift_window__(d,center,window)
        for key in fft_matrix:
            if fft_matrix[key].dtype == complex128:
                fft_matrix[key] = shift_window*fft_matrix[key]
                    
            
        #return fft_matrix
        
    elif type(fft_matrix) == type(''):
        with File(fft_matrix,'a') as hdf:
            for key in fft_matrix:
                if fft_matrix[key].dtype == complex128:
                    points = fft_matrix[key].shape[-1]
                
    else:
        warn = f'-- FFT Matrix datatype not recoognised --'
        warnings.warn(warn,UserWarning)
        
        

def wec_direction(fft_matrix,window=None,ang='rad'):
    
    '''
    Parameters:
    
    fft_matrix : str / dict
        filename or in memory data containing fft coeffients and frequencies on disk
    ang : string
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
            fft_matrix['direction'] = linspace(-180,180,points)
        else:
            fft_matrix['direction'] = linspace(-pi,pi,points)
                
    else:
        warn = f'-- FFT Matrix datatype not recoognised --'
        warnings.warn(warn,UserWarning)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    