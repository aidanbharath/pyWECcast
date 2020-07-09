import warnings

from numpy import array, nan
from pandas import to_datetime




def binary_timestamp_convert(timestamps):
    """
    Parameters:
    
    timestamps : binary iterable
        iterable list/ndarry of binary timestamps
        
    Returns:
        timestamps : ndarray[:] (DatetimeIndex)
            pandas datetime index array of timestamps
    """
    
    return to_datetime([time.decode() for time in timestamps])

def datetimeIndex_timestamp_convert(timestamps):
    """
    Parameters:
    
    timestamps : datetimeIndex iterable
        iterable list/ndarry of datetimeIndex timestamps
        
    Returns:
        timestamps : ndarray[:] (binary str)
            array of binary timestamps
    """
    
    return array([time.encode() for time in timestamps.strftime('%Y-%m-%d %H:%M:%S')])


def binary_direction_convert(direction,north=0.,conversion_schema=None):
    """
    Parameters:
    
    direction : binary iterable
        iterable list/ndarray of set NOAA directions
    north : int64 (degrees)
        northern degree direction offset
    convertion_schema : dict
        conversion table to apply to binary direction values.
        default based on NOAA schema
        
    Returns:
        directions : ndarray[:] (float64)
            sea state direction in degrees based on the NOAA reporting schema
    """
    
    flag = False
    if not conversion_schema:
        schema = {
            'N':0+north,'NNE':22.5+north,'NE':45+north,'ENE':67.5+north,'E':90+north,
            'S':180+north,'SSE':157.5+north,'SE':135+north,'ESE':112.5+north,
            'SSW':202.5+north,'SW':225+north,'WSW':247.5+north,'W':270+north,
            'WNW':292.5+north,'NW':315+north,'NNW':337.5+north
        }
    else:
        schema = conversion_schema
        
    results = []    
    for d in direction:
        value = d.decode()
        if value != 'nan':
            results.append(schema[value])
        else:
            flag = True
            results.append(north)
           
    if flag:
        warn = f'Directional ndarray contains NaNs - default value inserted {north}'
        warnings.warn(warn,UserWarning)
    
    return array(results)
    
    