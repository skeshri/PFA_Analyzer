'''
Version for 2022 data
'''
import pandas as pd
import numpy as np
import sys
import os
import json
from PFA_Analyzer_Utils import * 
sys.path.append(os.path.abspath("/eos/project-c/cmsgemonline/public/doc/"))
from chamber_mapping import chamber_mapping as mapping

TRIMDAC2fC = 2.1/63     # VFAT Design --> 63 TRIM DAC = 2.1 fC
THRDAC2fC = 12. / 100   # VFAT Design --> 100 THR DAC = 12 fC

## inverting the map so that the chamber name is the key
mapping = {v: k for k, v in mapping.items()}

## 
def GetCHDict(chamber_ID,run=348832):
        tupl = mapping[chamber_ID]
        print(tupl)
        crate = int(tupl[0])
        amc = int(tupl[1])
        OHLink = int(tupl[2])
        fed = 1466 + crate
        
        file_path = "/afs/cern.ch/user/f/fivone/Test/runParameterGEM/run"+str(run)+"/fed%d" % fed + "-amc%02d" % amc +"_ConfigInfo.json"
        #file_path = "/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/2022/config/global/THR_Latency/run"+str(run)+"/fed%d" % fed + "-amc%02d" % amc +"_ConfigInfo.json"

        try:
                with open(file_path) as json_file:
                        data_dict = json.load(json_file)

        except:
                ##print "Can't open ",file_path
                return {}
        ## check exsistance off all keys
        try:
                data_dict = data_dict['fed'][str(fed)]['slot'][str(amc)]['link'][str(OHLink)]
        except:
                data_dict = {}
        return data_dict

## Returns the THR_DAC set for a given VFAT in a Chamber
## run can be provided
def GetVFAT_THRDAC(chamber_ID,VFATN,run=348832):

        data = GetCHDict(chamber_ID,run)

        try:
                threshold = data['vfat'][str(VFATN)]['THRESHOLD_DAC']
        except:
                threshold = None

        return threshold

def GetVFAT_LATENCY(chamber_ID,VFATN,run=348832):

        data = GetCHDict(chamber_ID,run)

        try:
                latency = data['vfat'][str(VFATN)]['LATENCY']
        except:
                latency = -1

        return latency

## Returns the AVG THR of a chamber       
def GetOverallChamberThreshold(chamberID,run=357329):

        if chamberID not in mapping.keys():
                print(f"Invalid chamberID: {chamberID}")
                print(f"Exiting...")
                sys.exit(0)
        

        data = GetCHDict(chamberID,run)


        temp_array = np.empty(0,dtype=float)
    
        for VFAT_N in range(24):
                # THR data
                try:
                        threshold = data['vfat'][str(VFAT_N)]['THRESHOLD_DAC']
                        if threshold != -1: temp_array = np.append(temp_array,threshold)
                except:
                        continue
                        #print "Can't find THR for\t",chamberID,"\tVFAT:",VFAT_N,"\tSkipping...",run,"\n"
                
        # if overall_vfat_trheshold == 10**6:
        #         continue
        # elif abs(overall_vfat_trheshold) > 255:
        #     if verbose: print "No-sense DAC value for VFAT",VFAT_N," on chamber ",chamberID,"\nPlease check, skipping VFAT value..."
        #     continue
        # else:


        if len(temp_array) != 0:
                return np.mean(temp_array)
        else:
                return None
        

if __name__ == "__main__":
        for k in range(0,24):
                print(f'{k} {GetVFAT_THRDAC("GE11-M-25L1-S",k,361917)}')
        pass

