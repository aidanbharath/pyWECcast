# \\\\\\\\\\\\\\\ pyWecPredict imports ////////////////////////////////
import forecasts_eagle as fc 
import pyWECSim_eagle as ws
import powerseries_eagle as ps

# \\\\\\\\\\\\\\\\\ python imports ///////////////////////////////////
import os
import pickle
from pandas import read_csv
from glob import glob


if __name__ == "__main__":


    eagleDir = f'/mnt/d/eagle'    
    downloadDir = f'noaa_downloads'
    processDir = f'{eagleDir}/predictions'
    
    if not os.path.exists(downloadDir):
        os.makedirs(downloadDir)

    ds_name = f'RM3_PTO_woPS_3hr'
    #ds_name = f'RM3_PTO_wAcc_10PBV3450_Control04_3hr'

    wecSim_db = glob(f'{eagleDir}/WECSim_db/*{ds_name}*')

    regions = ['wc_10m']
    selectedBuoys = [f'clatsop_spit']#f'humboldt_bay',f'fairweather_grounds',f'cape_suckling'] ## select a buoy to analyze
    noaaBuoyInfo = f'./noaa_Buoy_info.csv' ## base buoy info file

    updateDays = False ## update available days 
    extractNCData = False ## If True load buoy data from model data
    combinedNCDatasets = False ## If True the datasets will be combined
    updateNCFiles = False  ## True will update noaa model datafiles locally
    loadLatestNCBuoys = True ## True to load latest or date specific uoy data
    updateBuoyFiles = False ## True will update noaa buoy datafiles locally
    updateWECSim_db = False ## True check to see if new WECSim files are available
    check_WECSim_db = False ## True will compare Hs and Te available to whats needed
    runWECSim = False ## True will run WECSIM to fill db -- Can take a lot of time

    constructTS_noaaBuoys = False  ## Construct WECSIM TS from noaa buoys
    constructTS_ncBuoys = True   ## Construct WECSIM TS from nc buoys
    

    parallel = False ## Used for xarray, True is not working on eagle


    # \\\\\\\\\\\\\\  Update NC files from NOAA ////////////////////////

    if updateDays:
        ftpFiles = fc.retrieve_model_files(fc.host,fc.dataDir)
        ftpDays = [File for File in ftpFiles if 'multi_1' in File]
        
    if updateNCFiles:
        ftpFiles = []
        for ftpFile in (f'{fc.dataDir}/{day}' for day in ftpDays):
            ftpFiles.append(fc.retrieve_model_files(fc.host,ftpFile))
        
        ftpFiles = tempwc = fc.strip_Files(ftpFiles,regions[0],exclude=False)
        #tempak = fc.strip_Files(ftpFiles,regions[1],exclude=False)
        #tempat = fc.strip_Files(ftpFiles,regions[2],exclude=False)
        #ftpFiles = [j+tempak[i]+tempat[i] for i,j in enumerate(tempwc)]
        
        ftpFiles = fc.strip_Files(ftpFiles,'idx')
        fc.get_files(fc.host,fc.dataDir,ftpDays,ftpFiles,
          dlFiles=f'{eagleDir}/{downloadDir}')
        
    if combinedNCDatasets:
        concats = ['t00z','t06z','t12z','t18z']
        fc.combine_datasets(ftpDays,concats=concats,regions=regions,
                            parallel=parallel,eagleDir=eagleDir)


    # \\\\\\\\\\\\\\\\\\ Update Buoy files from NOAA ////////////////////
    buoys = read_csv(noaaBuoyInfo,index_col=0)
    if updateBuoyFiles:     
        noaaBuoys = fc.update_buoys(buoys,eagleDir=eagleDir)

    # \\\\\\\\\\\\\\\\\\ Load NC buoy data /////////////////////////////
    if extractNCData:
        ncBuoys = {}
        for buoy in selectedBuoys:
            ncBuoys[buoy] = fc.extract_buoy_from_model(buoys[buoy],eagleDir=eagleDir)

    if loadLatestNCBuoys:
        ncBuoys = {}
        for buoy in selectedBuoys:
            ncBuoys[buoy] = fc.load_latest_ncBuoy(buoy,eagleDir=eagleDir) 
         

    # \\\\\\\\\\\\\\\\\\\\ Search WECSim database ////////////////////////

    if updateWECSim_db:     
        ws.update_database(ds_name)

    if check_WECSim_db:
        for buoy in selectedBuoys:
            requiredSims = ws.group_Hs_Te(ncBuoys[buoy],noaaBuoys[buoy],ds_name,
                                            h_rounds=0,t_rounds=0)
    
    # \\\\\\\\\\\\\\\\\\\\\ Run WECSIM /////////////////////////////////
    
    if runWECSim:
        ws.run_WECSim(ds_name,requiredSims)

    #  \\\\\\\\\\\\\\\\\\\\ Construct Powerseries  ///////////////////////

    timeGroupsNOAA, timeGroupsNC = {}, {}
    if constructTS_noaaBuoys:
        print(' ')
        print('---- Beginning process on NOAA Buoy Reconstruction ----')
        print(' ')
        timeGroupsNOAA = {} 
        for buoy in selectedBuoys:
            timeGroupsNOAA[buoy] = ps.extract_variables(noaaBuoys[buoy],'WVHT','APD')
        selectNOAA = {}
        for buoy in selectedBuoys:
            selectNOAA[buoy] = ps.timeseries_params(timeGroupsNOAA[buoy],wecSim_db)
        fftNOAA = {}
        for buoy in selectedBuoys:
            fftNOAA[buoy] = ps.calculate_ffts(selectNOAA[buoy])
        timeseriesNOAA = {}
        for buoy in selectedBuoys:
            timeseriesNOAA[buoy] = ps.linear_transition_timeseries(fftNOAA[buoy])
            timeseriesNOAA[buoy].to_pickle(f'{processDir}/{buoy}-{ds_name}.pkl',header=False)

    if constructTS_ncBuoys:
        print(' ')
        print('---- Beginning process on Model Buoy Reconstruction ----')
        print(' ')
        timeGroupsNC = {}
        for buoy in selectedBuoys:
            timeGroupsNC[buoy] = ps.extract_variables(ncBuoys[buoy],'swh','perpw')
        '''
        with open('./model_tg_save.pkl','wb') as f:
            pickle.dump(timeGroupsNC,f)
        with open('./model_tg_save.pkl','rb') as f:
            timeGroupsNC = pickle.load(f)
        '''
        selectNC = {}
        for buoy in selectedBuoys:
            selectNC[buoy] = ps.timeseries_params(timeGroupsNC[buoy],wecSim_db)
        '''
        with open('./model_select_save.pkl','wb') as f:
            pickle.dump(selectNC,f)
        with open('./model_select_save.pkl','rb') as f:
            selectNC = pickle.load(f)
        '''
        fftNC = {}
        for buoy in selectedBuoys:
            fftNC[buoy] = ps.calculate_ffts(selectNC[buoy],freqCut=0.5)
        '''
        with open('./model_time_save.pkl','wb') as f:
            pickle.dump(selectNC,f) 
        with open('./model_test_save.pkl','wb') as f:
            pickle.dump(fftNC,f)
        with open('./model_test_save.pkl','rb') as f:
            fftNC = pickle.load(f)
        with open('./model_time_save.pkl','rb') as f:
            selectNC = pickle.load(f)
        '''
        timeseriesNC = {}
        for buoy in selectedBuoys:
            timeseriesNC[buoy] = ps.linear_transition_timeseries_model(fftNC[buoy],selectNC[buoy],buoy)
