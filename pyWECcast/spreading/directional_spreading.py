
from numpy import pi, sqrt, abs, cos, linspace, array
from scipy.special import gamma



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


def __shift360__(shift,window_size=36):
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
    
    deg = linspace(-pi,pi,window_size)
    result = []
    for d in deg:
        newD = d+shift
        if newD > pi:
            newD = newD - 2*pi
        elif newD < -pi:
            newD = newD + 2*pi
        result.append(newD)
    return array(result)


def __cosine_spreading__(Omean,Omax):
    
    pass


def __unit_spreading__():
    pass

def expand_fft_matrix():
    pass


    
    