import warnings

from numpy import complex128, zeros, outer
from h5py import File




def multiple_wecs(nWECs,fft_matrix=None):
    """
    
    Parameters:
    
    nWECs : int
        Number of WEC you wish too add for future calculations
    fft_matrix : default None, dict/str
        either an in memory dict or filename containing the relavent WEC fft coefficients and freqency
    
    Returns:
        dict : (coefficients complex128 ndarray[:], frequencies float64 ndarray[:])
            if inMemory:True ; returns FFT results
            else ; data replaces original in hdf5 file passed
    
    """
    
    if type(fft_matrix) == type({}):
        for key in fft_matrix:
            if fft_matrix[key].dtype == complex128:
                ones = zeros(nWECs)
                ones[:] = 1
                fft_matrix[f'{key}'] = outer(fft_matrix[key][:],ones)
            
        return fft_matrix
        
    elif type(fft_matrix) == type(''):
        with File(fft_matrix,'a') as hdf:
            for key in fft_matrix:
                if fft_matrix[key].dtype == complex128:
                    ones = zeros(nWECs)
                    ones[:] = 1
                    data = outer(fft_matrix[key][:],ones)
                    del hdf[key]
                    hdf.create_dataset(f'{key}', data=data, dtype=data.dtype,chunks=True)

    else:
        warn = f'-- FFT Matrix datatype not recoognised --'
        warnings.warn(warn,UserWarning)