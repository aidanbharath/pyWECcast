from numba import jit, float64, complex128, prange, int64

@jit(['float64[:,:](complex128[:,:],float64[:],float64[:],float64[:,:],int64,int64)'],
         nopython=True, nogil=True,parallel=True)
def reconstruction_spreading(spectrum,frequency,times,result,lntime,ndeg):
    
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
            
        Returns:
            result : float64 ndarray[:,:] (times,direction)
                reconstruction array
    """
    
    for i in prange(tl):
        for j in range(ndeg):
            Cos = np.cos(2*np.pi*F*times[i])
            Sin = np.sin(2*np.pi*F*times[i])
            s = spectral[:,j]
            result[i,j] = np.sum(s.real*Cos+s.imag*Sin)
    return result