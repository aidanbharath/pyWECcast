from h5py import File


def extract_NOAA_buoy(filename,buoyNum,Hs,Tp,Dir=None):
    """
    Parameters:
    
    filename : str
        Filename of the pyWECcast format NOAA buoy download file
    buoyNum : str
        buoy identifier number
    Hs : str
        Variable name associated with the desired Hs values
    Tp : str
        Variable name associated with the desired Tp values
    Dir : optional str
        Variable name assoociated with the desired Direction values
        
    Returns:
        tuple : (time_index,Hs,Tp, optional Dir)
            - return types may depend on filetype
                - default (binary,float64,float64,binary)
    """
    
    with File(filename, 'r') as buoys:
        buoy = buoys[buoyNum]
        time_index = buoy['time_index'][:]
        hs,tp = buoy[Hs][:], buoy[Tp][:]
        
        if Dir:
            d = buoy[Dir][:]
            return time_index, hs, tp, d
        
        else:
            return time_index, hs, tp