
from numpy import pi, sqrt, abs, cos, linspace, array, radians, degrees, zeros, argmin, complex128, outer
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
    idx_shift = -int(np.round(shift/inc))
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
            
    return np.array(result)


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
    
        

def __unit_spreading__():
    pass

def expand_fft_matrix(fft_matrix,points=36,ang='rad'):
    
    '''
    Parameters:
    
    
    Returns:
    
    
    '''

    if type(fft_matrix) == type({}):
        for key in fft_matrix:
            if fft_matrix[key].dtype == complex128:
                ones = zeros(points)
                ones[:] = 1
                fft_matrix[f'{key}'] = outer(fft_matrix[key],ones)
                print(fft_matrix[f'{key}'].shape)
        
        if ang == 'deg':
            fft_matrix['direction'] = linspace(-180,180,points)
        else:
            fft_matrix['direction'] = linspace(-pi,pi,points)
            
        return fft_matrix
        
    else:
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    