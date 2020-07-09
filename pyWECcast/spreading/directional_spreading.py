
from numpy import pi, sqrt, abs, cos, linspace, array
from scipy.special import gamma



def __Ds__(O,Omeam,Omax,s):
    C = (sqrt(pi)*gamma(s+1))/(2*Omax)*gamma(s+0.5)
    return C*abs(cos(pi*(O-Omean)/(2*Omax)))**(2*s)

def __shift360__(shift,window_size=window_size):
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


def expand_fft_matrix():
    pass


def cosine_spreading():
    pass


def unit_spreading():
    pass
    
    