
import os
import re
import glob
import ftplib
import urllib
import numpy as np
import pandas as pd
import xarray as xr

import shutil
import urllib.request as request
from contextlib import closing
from tqdm import tqdm


class forecast(object):

    def __init__(self,*args,**kwargs):
        
        self.args = args
        self.kwargs = kwargs
        self.test = 'yes'

