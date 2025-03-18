from numpy.core.numeric import NaN
import os
import ROOT
import numpy
import subprocess
import re as regularExpression
import pandas as pd
from ROOT_Utils import *
from EtaPartitionBoundaries import *
import sys
import json
from array import array
import yaml
import pathlib
import time
import logging
import math

OUTPUT_PATH = "/afs/cern.ch/work/s/skeshri/GEM_efficiency/PFA_Analyzer_updated/PFA_Analyzer/Output/"

map_cutcode_to_quantity = {-1:"ErrPropR",
                           -2:"ErrPropPhi",
                           -3:"FiducialCuts",
                           -4:"pT",
                           -5:"chi2",
                           -6:"minMExhit",
                           -9:"muon_id"}


#region station IsOdd etaP centerY minY maxY

data_boundaries =[
#long
[1,1,0,1,50.7475,41.045,60.45],
[1,1,0,2,31.2925,21.59,40.995],
[1,1,0,3,13.4865,5.43299,21.54],
[1,1,0,4,-2.6705,-10.724,5.383],
[1,1,0,5,-17.502,-24.23,-10.774],
[1,1,0,6,-31.008,-37.736,-24.28],
[1,1,0,7,-43.4395,-49.093,-37.786],
[1,1,0,8,-54.7965,-60.45,-49.143],
#short
[1,1,1,1,44.7225,36.395,53.05],
[1,1,1,2,28.0175,19.69,36.345],
[1,1,1,3,12.6115,5.583,19.64],
[1,1,1,4,-1.49551,-8.52401,5.53299],
[1,1,1,5,-14.5525,-20.531,-8.574],
[1,1,1,6,-26.5595,-32.538,-20.581],
[1,1,1,7,-37.691,-42.794,-32.588],
[1,1,1,8,-47.947,-53.05,-42.844]
#GE21,Demo
#[1,2,0,1,79.3487,74.3275,84.37],
#[1,2,0,2,69.2863,64.265,74.3075],
#[1,2,0,3,59.2237,54.2025,64.245],
#[1,2,0,4,49.1613,44.14,54.1825],
#[1,2,0,5,34.6937,28.7975,40.59],
#[1,2,0,6,22.8812,16.985,28.7775],
#[1,2,0,7,11.0688,5.17251,16.965],
#[1,2,0,8,-0.743749,-6.64,5.1525],
#[1,2,0,9,-15.0737,-19.9575,-10.19],
#[1,2,0,10,-24.8613,-29.745,-19.9775],
#[1,2,0,11,-34.6488,-39.5325,-29.765],
#[1,2,0,12,-44.4362,-49.32,-39.5525],
#[1,2,0,13,-57.7537,-62.6375,-52.87],
#[1,2,0,14,-67.5413,-72.425,-62.6575],
#[1,2,0,15,-77.3288,-82.2125,-72.445],
#[1,2,0,16,-87.1163,-92,-82.2325],
#GE21,Neg
#[-1,2,0,1,79.855,75.34,84.37],
#[-1,2,0,2,70.805,66.29,75.32],
#[-1,2,0,3,61.755,57.24,66.27],
#[-1,2,0,4,52.705,48.19,57.22],
#[-1,2,0,5,39.25,33.86,44.64],
#[-1,2,0,6,28.45,23.06,33.84],
#[-1,2,0,7,17.65,12.26,23.04],
#[-1,2,0,8,6.85,1.46,12.24],
#[-1,2,0,9,-7.48,-12.87,-2.09],
#[-1,2,0,10,-18.28,-23.67,-12.89],
#[-1,2,0,11,-29.08,-34.47,-23.69],
#[-1,2,0,12,-39.88,-45.27,-34.49],
#[-1,2,0,13,-54.21,-59.6,-48.82],
#[-1,2,0,14,-65.01,-70.4,-59.62],
#[-1,2,0,15,-75.81,-81.2,-70.42],
#[-1,2,0,16,-86.61,-92,-81.22],
]

parameters_default = {
    'phi_cut':{"type":float,"default":0.0025},
    'rdphi_cut':{"type":float,"default":0.5},
    'chi2cut':{"type":float,"default":9999999},
    'minPt':{"type":float,"default":0},
    'outputname':{"type":str,"default":time.strftime("%-y%m%d_%H%M")},
    'fiducialR':{"type":float ,"default":1},
    'fiducialPhi':{"type":float ,"default":0.005},
    'maxErrPropR':{"type":float ,"default":1},
    'maxErrPropPhi':{"type":float,"default":0.005},
    'DLE':{"type":bool,"default":False},
    'FD':{"type":bool,"default":False},
    'verbose':{"type":bool,"default":False},
    'fiducialCut' :{"type":bool,"default":True},
    'doGE21':{"type":bool,"default":False},
    'minME1':{"type":int,"default":0},
    'minME2':{"type":int,"default":0},
    'minME3':{"type":int,"default":0},
    'minME4':{"type":int,"default":0},
    'njobs':{"type":int,"default":1},
    'ID':{"type":int,"default":2}, # 0: loose, 1: medium, 2: tight
}
muon_id = {0: 'loose_id', 1: 'medium_id', 2: 'tight_id'}

#numpy_type = { int:"int32",float:"float32",bool:np.bool_,str:"<S20",tuple:numpy.dtype('int32,int32')}
numpy_type = { int:"int32",float:"float32",bool:np.bool_,str:"<S20",tuple:numpy.dtype('int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32,int32')}
ROOT_type = { int:"I",float:"F",bool:"b",str:"C",tuple:"I"}

## Magic numbers taken from strip geometry
#  They are loc_x/glb_r (basically cos phi) for strip 127 and 255 respectively
LeftMost_iPhi_boundary = -0.029824804
RightMost_iPhi_boundary = 0.029362374

def  get_etaP_boundaries_Y(region, station, chamber, etaP):    
    isOdd = chamber%2
    ymin = -999
    ymax = -999
    for index, row in enumerate(data_boundaries):
        if row[1] == station and row[2] == isOdd and row[3] == etaP:
            ymin = row[5]
            ymax = row[6]
    return ymin,ymax


def load_config(filepath,verbose=True):
    filepath = pathlib.Path(filepath)
    data_ranges = {}
    parameters = {}
    
    with filepath.open() as ymlfile:
        cfg = yaml.full_load(ymlfile)

    cfg_data = cfg["data"]
    cfg_param = cfg["parameters"]

    ## DATA CFG
    for run,value in cfg_data.items():
        data_ranges[run] = value
        data_ranges[run]["lumisection"] = tuple(sorted(data_ranges[run]["lumisection"])) if data_ranges[run]["lumisection"] is not None else (0,0)
        data_ranges[run]["nevts"] = data_ranges[run]["nevts"] if data_ranges[run]["nevts"] is not None else -1
        data_ranges[run]["VFATOFF"] = {} if value["VFATOFF"] is None else importOFFVFAT(value["VFATOFF"])
        data_ranges[run]["chamberOFF"]= {} if value["chamberOFF"] is None else ChamberOFF_byLS(value["chamberOFF"])
        if ("branches" in data_ranges[run].keys()):
            data_ranges[run]["branches"]= [] if value["branches"] is None else BranchesDict(value["branches"])
    ## DATA CFG


    ## PARAM CFG
    # load not None parameters that match default var type
    for param,value in cfg_param.items():
        if param not in parameters_default.keys():
            logging.warning(f"Found {param} in {filepath} is not a known input parameter. Skipped")
            continue
        
        if not isinstance(value, parameters_default[param]['type']) and value is not None: 
            logging.error(f"Found {value} of type {type(value)} for {param}. Not of type {parameters_default[param]['type']}. Skipped")
        
        elif value is None: pass
            
        else: parameters[param] = value
    
    # defaulting the others
    default_param = [x for x in parameters_default if x not in parameters]
    for k in default_param:
        parameters[k] = parameters_default[k]["default"]
    ## PARAM CFG

    
    logging.debug("### CFG Data")
    logging.debug(f"{'Run':<20}{'Param':<20}{'Value':<20}")
    for run in data_ranges:
        logging.debug(f"{run:<20}{'':>20}{'':>20}")
        for k,value in data_ranges[run].items():
            logging.debug(f"{'':<20}{k:<20}{value}")
    logging.debug("### CFG Data")

    logging.debug("### CFG Param")
    logging.debug(f"{'Param':<20}{'Value':>12}")
    for x in [ k for k in parameters if k not in default_param ]:
        logging.debug(f"{x:<20}{parameters[x]:>12}")
    for x in default_param:
        logging.debug(f"{x:<20}{parameters[x]:>12}{' (default)':>10}")
    logging.debug("### CFG Param\n")

    return data_ranges,parameters

def getEtaPID(station,region,chamber,layer,etaP):
    # |------|-----|--|---|-|
    # chamber | etaP | station | layer | region 
    region = 0 if region == 1 else 1

    as_string = f'{chamber:06b}' + f'{etaP:05b}' + f'{station:02b}' + f'{layer:03b}' + f'{region:01b}' 
    return int(as_string,2)

def convert_etaPID(etaPID):
    etaPID_asString = f'{etaPID:017b}'
    chamber = int(etaPID_asString[0:6],2)
    eta_partition = int(etaPID_asString[6:11],2)
    station = int(etaPID_asString[11:13],2)
    layer = int(etaPID_asString[13:16],2)
    region = int(etaPID_asString[16:17],2)
    
    region = 1 if region == 0 else -1
    return station,region,chamber,layer,eta_partition
    

def getInfoFromEtaID(id):
    etaPartition = abs(id)%10
    layer = ((abs(id)-etaPartition)%100)/10
    chamber = (abs(id)-etaPartition-10*layer)/100
    region = id/abs(id)
    return int(region),int(chamber),int(layer),int(etaPartition)


def GE11_ChamberIDs():
    list_of_chambers = []
    for (re,la) in [(1,1),(1,2),(-1,1),(-1,2)]:
        for chamber in range(1,37):
            chID = getChamberName(re,chamber,la)
            list_of_chambers.append(chID)
    return list_of_chambers


## Expects to have a list of file paths each containing a list of "GE11-*-XXLY" to be excluded
def ChamberOFF_byLS(file_path):
    try:
        io = open(file_path,"r")
        dictionary = json.load(io)  
    except IOError:
            print("Couldn't open file : "+file_path+"\nExiting ..")
            sys.exit(0)

    for key,item in list(dictionary.items()):
        if item == []:
            dictionary.pop(key)
    return dictionary

def VFAT2iEta_iPhi(VFATN):
    try:
        vfatPosition = int(VFATN)
    except:
        print("VFAT Number provided is not a number.\nExiting...")
        sys.exit(0)

    if vfatPosition <0 or vfatPosition>23:
        print("Invalid VFAT position.\nExiting...")
        sys.exit(0)

    iEta = (8 - vfatPosition%8)
    iPhi = vfatPosition//8
    return iEta,iPhi

def iEta_iPhi2VFAT(iEta,iPhi):
    try:
        etaP = int(iEta)
        phi = int(iPhi)
    except:
        print("Provided iEta and/or iPhi are not numbers.\nExiting...")
        print(iEta,iPhi)
        sys.exit(0)
    
    if iEta <1 or iEta>8 or iPhi <0 or iPhi>2:
        print("Invalid iEta and/or iPhi provided position.\nExiting...")
        print(iEta,iPhi)
        sys.exit(0)
    
    VFAT = iPhi*8 + (8-iEta)
    return VFAT

def recHit2VFAT(etaP,firstStrip,CLS):
    iphi_first = firstStrip//128
    iphi_last = (firstStrip + CLS - 1 ) // 128
    if iphi_last  == iphi_first:
        iphi =  [iphi_first]
    else: 
        iphi = list(range(iphi_first, iphi_last+1))
    VFATs = []
    for phi in iphi:
        VFATs.append(iEta_iPhi2VFAT(etaP,phi))
    return VFATs




## Associates a propagated hit to a VFAT
## It is based on default GE11 geometry
## Might need refinements after alignment 
def propHit2VFAT(glb_r,loc_x,etaP):
    prophit_cosine = loc_x/glb_r
    iPhi =  0 if  prophit_cosine<=LeftMost_iPhi_boundary else (2 if prophit_cosine>RightMost_iPhi_boundary else 1)
        
    VFAT = iEta_iPhi2VFAT(etaP,iPhi)

    return VFAT

## find VFAT propagated hit belongs to based on the strip
def iEtaStrip_2_VFAT(iEta, strip, station):
#    if station==1:
#        vfat_ge11 = (strip // 128) * 8 + (8 - iEta)   
    vfat_ge11 = (strip // 128) * 8 + (8 - iEta)
    return vfat_ge11




def VFATcompatibleHit(VFATs,glb_r,loc_x,etaP):
    for VFAT in VFATs :
        hit_cosine = loc_x/glb_r
        iPhi_cosine_width = (abs(LeftMost_iPhi_boundary) + abs(RightMost_iPhi_boundary))/2
        iEta,iPhi = VFAT2iEta_iPhi(VFAT)
        if etaP == iEta:
            if iPhi == 0 and abs(abs(hit_cosine)-abs(LeftMost_iPhi_boundary)) <= iPhi_cosine_width*0.05:
                return True
            elif iPhi == 1 and ( abs(abs(hit_cosine)-abs(LeftMost_iPhi_boundary)) <= iPhi_cosine_width*0.05 or abs(abs(hit_cosine)-abs(RightMost_iPhi_boundary)) <= iPhi_cosine_width*0.05 ):
                return True
            elif iPhi == 2 and abs(abs(hit_cosine)-abs(RightMost_iPhi_boundary)) <= iPhi_cosine_width*0.05:
                return True

    return False
## returns a dict with a list of OFF VFAT for each  etapartitionID (key of the dict)
def importOFFVFAT(file_path):
    off_VFAT = {}
    try:
        df = pd.read_csv(file_path, sep='\t')
    except IOError:
        print("Couldn't open file : "+file_path+"\nExiting ..")
        sys.exit(0)

    off_VFAT = {}
    for index, row in df.iterrows():
        region =  -1 if ( row['region']=='N' or row['region']=='M') else (1 if row['region']=='P' else None) ## In the past N was used... currently M indicates the minus endcap
        layer = int(row['layer'])
        chamber = int(row['chamber'])
        VFAT = int(row['VFAT'])
        maskReason = int(row['reason_mask'])

        iEta , iPhi = VFAT2iEta_iPhi(VFAT)
        etaPartitionID = region*(100*chamber+10*layer+iEta)

        ## If the key doesn't exsist create it, then fill the list
        off_VFAT.setdefault(etaPartitionID,[])
        if maskReason == 1:  ## only mask empty VFATs
            off_VFAT[etaPartitionID].append(VFAT)

    return off_VFAT

def generateVFATMaskTH2(station=1):
    if station != 1:
        logging.error(f"function generateVFATMaskTH2() not implemented for station = 2. Exiting...")
        sys.exit()
    else:
        output_dict = {}
        for re,la in [(1,1),(1,2),(-1,1),(-1,2)]:
            endcapTag = EndcapLayer2label(re,la)
            output_dict[endcapTag] = ROOT.TH2F("GE11-"+endcapTag+"  Masked VFAT","GE11-"+endcapTag+"  Masked VFAT",36,0.5,36.5,24,-0.5,23.5)
        for key_1 in output_dict.keys():
            output_dict[key_1].GetYaxis().SetTitle("VFAT Number")
            output_dict[key_1].GetXaxis().SetTitle("Chamber")
            output_dict[key_1].GetXaxis().SetNdivisions(40,0,0,True)
            output_dict[key_1].GetYaxis().SetNdivisions(40,0,0,True)
            output_dict[key_1].GetYaxis().SetLabelSize(0.02)
            output_dict[key_1].GetXaxis().SetLabelSize(0.02)
            output_dict[key_1].GetXaxis().SetTickLength(0.005)
            output_dict[key_1].GetYaxis().SetTickLength(0.005)
            output_dict[key_1].SetStats(False)
            output_dict[key_1].SetMinimum(0)
    
    return output_dict    

def chamberName2ReChLa(chamberName):
    ## Accepts as input either 
        # GE11-M-03L1 or GE11-M-03L1-S
    if len(chamberName)==13:
        chamberName = chamberName[:11]
    re = -1 if "M" in chamberName else 1
    ch = int( chamberName.split("-")[-1][:2] )
    la = int( chamberName.split("-")[-1][-1] )

    return [re,ch,la]


def ReChLa2chamberName(re,ch,la):
    endcap = "M" if re == -1 else "P"
    size = "S" if ch%2 == 1 else "L"
    chID = 'GE11-'+endcap+'-%02d' % ch +"L"+str(la)+"-"+size
    return chID

def getChamberName(re,ch,la,station=1,module=""):
    st = "GE11-" if station == 1 else "GE21-" if station == 2 else "ME0-"
    endcap = "M" if re == -1 else "P"
    if station != 1: size = "-"+module
    else: size = "-L" if ch%2 == 0 else "-S"
    chID = st+endcap+'-%02d' % ch +"L"+str(la)+size
    return chID

def EndcapLayer2label(re,layer):
    label = "ML" if re == -1 else "PL"
    return label+str(int(layer))
    

def ChambersOFFHisto(chamberNumberOFF_byLS,minLS,maxLS):
    GE11_OFF = {-1:{},1:{}}
    GE11_OFF[-1][1] = ROOT.TH1F("GE11-M-L1  Masked Chambers","GE11-M-L1  Masked Chambers",36,0.5,36.5)
    GE11_OFF[-1][2] = ROOT.TH1F("GE11-M-L2  Masked Chambers","GE11-M-L2  Masked Chambers",36,0.5,36.5)
    GE11_OFF[1][1] = ROOT.TH1F("GE11-P-L1  Masked Chambers","GE11-P-L1  Masked Chambers",36,0.5,36.5)
    GE11_OFF[1][2] = ROOT.TH1F("GE11-P-L2  Masked Chambers","GE11-P-L2  Masked Chambers",36,0.5,36.5)

    for region in [-1,1]:
        for layer in [1,2]:
            for chamber in range(1,37):
                ch_ID = getChamberName(region,chamber,layer) 
                n_OFF = 0 if ch_ID not in chamberNumberOFF_byLS else len([k for k in chamberNumberOFF_byLS[ch_ID] if k > minLS and k < maxLS]) if -1 not in chamberNumberOFF_byLS[ch_ID] else (maxLS-minLS)
                percentageOFF = 100*float(n_OFF)/(maxLS-minLS)
                GE11_OFF[region][layer].SetBinContent(chamber,percentageOFF)
    for key_1 in GE11_OFF.keys():
        for key_2 in GE11_OFF[key_1].keys():
            GE11_OFF[key_1][key_2].GetYaxis().SetTickLength(0.005)
            GE11_OFF[key_1][key_2].GetYaxis().SetTitle("Masked Lumisection (%)")
            GE11_OFF[key_1][key_2].SetStats(False)
            GE11_OFF[key_1][key_2].SetMinimum(0)
            GE11_OFF[key_1][key_2].SetMaximum(110)
    return GE11_OFF[-1][1],GE11_OFF[1][1],GE11_OFF[-1][2],GE11_OFF[1][2]



def VFATOFFHisto(VFATOFF_dictionary):
    VFAT_OFFTH2D = {-1:{},1:{}}
    VFAT_OFFTH2D[-1][1] = ROOT.TH2F("GE11-M-L1  Masked VFAT","GE11-M-L1  Masked VFAT",36,0.5,36.5,24,-0.5,23.5)
    VFAT_OFFTH2D[-1][2] = ROOT.TH2F("GE11-M-L2  Masked VFAT","GE11-M-L2  Masked VFAT",36,0.5,36.5,24,-0.5,23.5)
    VFAT_OFFTH2D[1][1] = ROOT.TH2F("GE11-P-L1  Masked VFAT","GE11-P-L1  Masked VFAT",36,0.5,36.5,24,-0.5,23.5)
    VFAT_OFFTH2D[1][2] = ROOT.TH2F("GE11-P-L2  Masked VFAT","GE11-P-L2  Masked VFAT",36,0.5,36.5,24,-0.5,23.5)

    for key_1 in VFAT_OFFTH2D.keys():
        for key_2 in VFAT_OFFTH2D[key_1].keys():
            VFAT_OFFTH2D[key_1][key_2].GetYaxis().SetTitle("VFAT Number")
            VFAT_OFFTH2D[key_1][key_2].GetXaxis().SetTitle("Chamber")
            VFAT_OFFTH2D[key_1][key_2].SetStats(False)
            VFAT_OFFTH2D[key_1][key_2].SetMinimum(0)

    for key, value in VFATOFF_dictionary.items():
        station,region,chamber,layer,etaPartition = convert_etaPID(key)
        for VFATN in value:
            VFAT_OFFTH2D[region][layer].Fill(chamber,VFATN)
    
    return VFAT_OFFTH2D[-1][1],VFAT_OFFTH2D[1][1],VFAT_OFFTH2D[-1][2],VFAT_OFFTH2D[1][2]


def GE11DiscardedSummary(chamberNumberOFF_byLS,minLS,maxLS,VFATOFF_dictionary):
    VFAT_OFFTH2D = {-1:{},1:{}}
    VFAT_OFFTH2D[-1][1] = ROOT.TH2F("GE11-M-L1  Masked","GE11-M-L1  Masked",36,0.5,36.5,24,-0.5,23.5)
    VFAT_OFFTH2D[-1][2] = ROOT.TH2F("GE11-M-L2  Masked","GE11-M-L2  Masked",36,0.5,36.5,24,-0.5,23.5)
    VFAT_OFFTH2D[1][1] = ROOT.TH2F("GE11-P-L1  Masked","GE11-P-L1  Masked",36,0.5,36.5,24,-0.5,23.5)
    VFAT_OFFTH2D[1][2] = ROOT.TH2F("GE11-P-L2  Masked","GE11-P-L2  Masked",36,0.5,36.5,24,-0.5,23.5)


    for key_1 in VFAT_OFFTH2D.keys():
        for key_2 in VFAT_OFFTH2D[key_1].keys():
            VFAT_OFFTH2D[key_1][key_2].GetYaxis().SetTitle("VFAT Number")
            VFAT_OFFTH2D[key_1][key_2].GetYaxis().SetTickLength(0.005)
            VFAT_OFFTH2D[key_1][key_2].GetXaxis().SetTickLength(0.005)
            VFAT_OFFTH2D[key_1][key_2].GetXaxis().SetTitle("Chamber")
            VFAT_OFFTH2D[key_1][key_2].GetZaxis().SetTitle("Masked Lumisection (%)")
            VFAT_OFFTH2D[key_1][key_2].GetZaxis().SetTitleSize(0.03)
            VFAT_OFFTH2D[key_1][key_2].GetZaxis().SetTitleOffset(1.1)
            VFAT_OFFTH2D[key_1][key_2].GetZaxis().SetLabelSize(0.025)
            VFAT_OFFTH2D[key_1][key_2].SetStats(False)
            VFAT_OFFTH2D[key_1][key_2].SetMinimum(0)
            VFAT_OFFTH2D[key_1][key_2].SetMaximum(100)
    
    for region in [-1,1]:
        for layer in [1,2]:
            for chamber in range(1,37):
                current_chamber_ID = getChamberName(region,chamber,layer)
                n_OFF = 0 if current_chamber_ID not in chamberNumberOFF_byLS else len([k for k in chamberNumberOFF_byLS[current_chamber_ID] if k > minLS and k < maxLS]) if -1 not in chamberNumberOFF_byLS[current_chamber_ID] else (maxLS-minLS)
                percentageOFF = 100*float(n_OFF)/(maxLS-minLS)
                for VFATN in range(0,24):
                    eta,phi = VFAT2iEta_iPhi(VFATN)
                    etaPartitionID = region*(100*chamber+10*layer+eta)

                    if etaPartitionID in VFATOFF_dictionary.keys() and VFATN in VFATOFF_dictionary[etaPartitionID]:
                        VFAT_OFFTH2D[region][layer].SetBinContent(chamber,VFATN+1,100.)
                    else:
                        VFAT_OFFTH2D[region][layer].SetBinContent(chamber,VFATN+1,percentageOFF)

    return [VFAT_OFFTH2D[-1][1],VFAT_OFFTH2D[1][1],VFAT_OFFTH2D[-1][2],VFAT_OFFTH2D[1][2]]


def GenerateMetadata(parameters,data_ranges):
    TH1MetaData = {}
    TH1MetaData["Parameters"] = ROOT.TTree("Parameters", "Parameters")
    par_array = {}
    for par,value in parameters.items():
        par_type = parameters_default[par]["type"]
        par_array[par] = np.empty((1), dtype = numpy_type[par_type] )
        TH1MetaData["Parameters"].Branch(par, 
                                        par_array[par], 
                                        f"par_array/{ROOT_type[par_type]}"
                                    )
        par_array[par][0] = value
    TH1MetaData["Parameters"].Fill()

    for run,sub_dict in data_ranges.items():
        TH1MetaData[run] = ROOT.TTree(f"run{run}", f"run{run}")
        data_range_array = {}
        for par,value in sub_dict.items():
            if par in ["chamberOFF","VFATOFF","lumisection"]: continue
            par_type = type(value)
            if par=="path_tag" and value !="":                         #Sumit 
                par_type=type(tuple(value))
                value=tuple(value)
            #print(par_type)
            #print(value)
            if("HI" in sub_dict['tag']):
                data_range_array[par] = np.empty((1), dtype = numpy_type[par_type] )
            else:
                data_range_array[par] = np.empty((1), dtype = numpy_type[par_type] )
            #print(data_range_array[par])
            TH1MetaData[run].Branch(par, 
                                    data_range_array[par], 
                                    f"data_range_array/{ROOT_type[par_type]}"
                                )
            #data_range_array[par][0] = value
        
        lumi_array = np.array(list(sub_dict["lumisection"]), dtype = "float32")
        TH1MetaData[run].Branch("lumisection", lumi_array, "lumi_array[2]/F")
        TH1MetaData[run].Fill()

    return TH1MetaData


def incidenceAngle_vs_Eff(sourceDict,input_region=1,input_layer=1):
    ## Transforming in list
    reg_tag_string = "All" if isinstance(input_region, list) else "P" if input_region == 1 else "M"
    lay_tag_string = "" if isinstance(input_layer, list) else "L1" if input_layer == 1 else "L2"
    input_region = [input_region] if not isinstance(input_region, list) else input_region
    input_layer = [input_layer] if not isinstance(input_layer, list) else input_layer

    title = "GE11"+reg_tag_string+lay_tag_string
    angle_nbins,angle_min,angle_max = 20,0,1.
    
    
    Eff_Plot = ROOT.TGraphAsymmErrors()
    NumTH1F = ROOT.TH1F(title+"_incidenceAngle_Num",title+"_incidenceAngle_Num",angle_nbins,angle_min,angle_max)
    DenTH1F = ROOT.TH1F(title+"_incidenceAngle_Den",title+"_incidenceAngle_Den",angle_nbins,angle_min,angle_max)
        

    for j in range(0,20):
        etaPartitionRecHits = 0
        etaPartitionPropHits = 0
        for etaPartitionID,value in sourceDict.items():
            station,region,chamber,layer,eta = convert_etaPID(etaPartitionID)
            
            if layer not in input_layer or region not in input_region:
                continue

            etaPartitionRecHits  += value[j]['num']
            etaPartitionPropHits += value[j]['den']
        NumTH1F.SetBinContent((j+1),etaPartitionRecHits)
        DenTH1F.SetBinContent((j+1),etaPartitionPropHits)
    
    Eff_Plot.Divide(NumTH1F,DenTH1F,"B")
    Eff_Plot.SetTitle(title+"_incidenceAngle_Eff")
    Eff_Plot.SetName(title+"_incidenceAngle_Eff")
    Eff_Plot.GetXaxis().SetTitle("Cos(#alpha)")
    Eff_Plot.GetYaxis().SetTitle("Efficiency")
    
    return NumTH1F, DenTH1F, Eff_Plot



## Structure:: TH2Fresidual_collector[matchingvar][chambers][residual_of_what][Plot] == Contains the TH2F
## Structure:: TH2Fresidual_collector[matchingvar][chambers][residual_of_what][binx][biny] == list(n entries, sum(abs(residual)))
## Usage:: TH2Fresidual_collector[matchingvar][chambers][residual_of_what][binx][biny] == Fill at each iteration
## Usage:: TH2Fresidual_collector[matchingvar][chambers][residual_of_what][Plot] == Fill at the end of the loop
def generate2DResidualContainer(matching_variables,nbins,minB):
    TH2Fresidual_collector = {}

    for key_1 in matching_variables:
        TH2Fresidual_collector.setdefault(key_1,{'all':{},'short':{},'long':{}})
        for key_2 in ['all','short','long']:
            TH2Fresidual_collector[key_1][key_2] = {}
            for key_3 in ['glb_phi','glb_rdphi']:
                titleTH2 = key_2 + "Chmbrs_MatchedBy_"+key_1+"_2DMap_of_"+key_3+"_res" 
                TH2Fresidual_collector[key_1][key_2][key_3] = {}
                TH2Fresidual_collector[key_1][key_2][key_3]['TH2F'] = ROOT.TH2F(titleTH2,titleTH2,nbins,minB,-minB,nbins,minB,-minB)
                TH2Fresidual_collector[key_1][key_2][key_3]['TH2F'].GetXaxis().SetTitle("Loc_x (cm)")
                TH2Fresidual_collector[key_1][key_2][key_3]['TH2F'].GetYaxis().SetTitle("Loc_y (cm)")
    
                for x_bin in range(1,nbins+1):
                    TH2Fresidual_collector[key_1][key_2][key_3][x_bin] = {}
                    for y_bin in range(1,nbins+1):
                        TH2Fresidual_collector[key_1][key_2][key_3][x_bin].setdefault(y_bin,[0,0])

    return TH2Fresidual_collector
    

def generate2DMap_ExcludedHits(nbins,minB,verbose=False):
    TH2Fresidual_collector = {}

    TH2Fresidual_collector = {'all':{},'short':{},'long':{}}
    for key_2 in TH2Fresidual_collector:
        for key_3,value in map_cutcode_to_quantity.items():
            titleTH2 = key_2 + "chmbr_discardedHits_due2_"+value
            TH2Fresidual_collector[key_2][value] = ROOT.TH2F(titleTH2,titleTH2,nbins,minB,-minB,nbins,minB,-minB)
            TH2Fresidual_collector[key_2][value].GetXaxis().SetTitle("Loc_x (cm)")
            TH2Fresidual_collector[key_2][value].GetYaxis().SetTitle("Loc_y (cm)")
            if verbose: print(f"dict[{key_2}][{value}] = { TH2Fresidual_collector[key_2][value]}")

    return TH2Fresidual_collector


def FillExcludedHits( container, propagatedHits, index, chID, cutPassed):
    key = map_cutcode_to_quantity[cutPassed] ## reason why the prop hit was excluded
    ch = chamberName2ReChLa(chID)[1]
    chamberSize = "short" if ch%2 == 1 else "long"

    loc_y = propagatedHits['loc_y'][index]
    loc_x = propagatedHits['loc_x'][index]

    container["all"][key].Fill(loc_x,loc_y)
    container[chamberSize][key].Fill(loc_x,loc_y)
    
    return container

def generate1DxResidualContainer_CLS(matching_variables,nbins,ResidualCutOff):
    output_dict = {}
    for cls in range(1,20):
        output_dict[cls]={}
        for mv in matching_variables:
            output_dict[cls][mv] = {}
            if mv == "glb_rdphi":
                minB = -ResidualCutOff["glb_rdphi"]
                x_axis_title = "#Deltardphi (cm)"
            else:
                minB = -ResidualCutOff["glb_phi"]
                x_axis_title = "#Deltaphi (rad)"
            for (re,la) in [(1,1),(1,2),(-1,1),(-1,2)]:
                endcap_key = EndcapLayer2label(re,la)
                output_dict[cls][mv][endcap_key] = {}
                for chamber in range(1,37):
                    chID = getChamberName(re,chamber,la)
                    output_dict[cls][mv][endcap_key][chID] = {}
                    for eta in list(range(1,9))+["All"]:
                        titleTH1 = chID+"_eta"+str(eta)+"_"
                        output_dict[cls][mv][endcap_key][chID][eta]={"Residual":ROOT.TH1F(titleTH1+"Residual",titleTH1+"Residual",nbins,minB,-minB)}
                        output_dict[cls][mv][endcap_key][chID][eta]["Residual"].GetXaxis().SetTitle(x_axis_title)
    return output_dict

def generate1DxResidualContainer(matching_variables,nbins,ResidualCutOff):
    output_dict = {}
    for mv in matching_variables:
        output_dict[mv] = {}
        if mv == "glb_rdphi":
            minB = -ResidualCutOff["glb_rdphi"]
            x_axis_title = "#Deltardphi (cm)"
        else:
            minB = -ResidualCutOff["glb_phi"]
            x_axis_title = "#Deltaphi (rad)"
        for (re,la) in [(1,1),(1,2),(-1,1),(-1,2)]:
            endcap_key = EndcapLayer2label(re,la)
            output_dict[mv][endcap_key] = {}

            for chamber in range(1,37):
                chID = getChamberName(re,chamber,la)
                output_dict[mv][endcap_key][chID] = {}
                for eta in list(range(1,9))+["All"]:
                    titleTH1 = chID+"_eta"+str(eta)+"_"
                    output_dict[mv][endcap_key][chID][eta]={"Residual":ROOT.TH1F(titleTH1+"Residual",titleTH1+"Residual",nbins,-10,10)}
                    output_dict[mv][endcap_key][chID][eta]["Residual"].GetXaxis().SetTitle(x_axis_title)
    return output_dict

def generate2DxResidualContainer(matching_variables,nbins,ResidualCutOff):
    output_dict = {}
    for mv in matching_variables:
        output_dict[mv] = {}
        if mv == "glb_rdphi":
            minB = -ResidualCutOff["glb_rdphi"]
            x_axis_title = "pT (GeV)"
            y_axis_title = "#Deltardphi (cm)"
        else:
            minB = -ResidualCutOff["glb_phi"]
            x_axis_title = "pT (GeV)"
            y_axis_title = "#Deltardphi (cm)"
        for (re,la) in [(1,1),(1,2),(-1,1),(-1,2)]:
            endcap_key = EndcapLayer2label(re,la)
            output_dict[mv][endcap_key] = {}

            for chamber in range(1,37):
                chID = getChamberName(re,chamber,la)
                output_dict[mv][endcap_key][chID] = {}
                for eta in list(range(1,9))+["All"]:
                    titleTH1 = chID+"_eta"+str(eta)+"_"
                    output_dict[mv][endcap_key][chID][eta]={"Residual":ROOT.TH2F(titleTH1+"Residual2D",titleTH1+"Residual2D",100,0,100,nbins,-10,10)}
                    output_dict[mv][endcap_key][chID][eta]["Residual"].GetXaxis().SetTitle(x_axis_title)
                    output_dict[mv][endcap_key][chID][eta]["Residual"].GetYaxis().SetTitle(y_axis_title)
    return output_dict


def generate1DyResidualContainer(matching_variables,nbins,ResidualCutOff):
    output_dict = {}
    for mv in matching_variables:
        output_dict[mv] = {}
        if mv == "glb_rdphi":
            minB = -10
            x_axis_title = "#Deltardphi (cm)"
        else:
            minB = -10
            x_axis_title = "#Deltaphi (rad)"
        for (re,la) in [(1,1),(1,2),(-1,1),(-1,2)]:
            endcap_key = EndcapLayer2label(re,la)
            output_dict[mv][endcap_key] = {}

            for chamber in range(1,37):
                chID = getChamberName(re,chamber,la)
                output_dict[mv][endcap_key][chID] = {}
                for eta in list(range(1,9))+["All"]:
                    titleTH1 = chID+"_eta"+str(eta)+"_"
                    output_dict[mv][endcap_key][chID][eta]={"Residual":ROOT.TH1F(titleTH1+"ResidualY",titleTH1+"ResidualY",nbins,minB,-minB)}
                    output_dict[mv][endcap_key][chID][eta]["Residual"].GetXaxis().SetTitle(x_axis_title)
    return output_dict



def whichBitsAreTrue(value):
    return np.flatnonzero(np.flip(np.where( np.array(list(np.binary_repr(value,width=24))) == '1', 1,0)))
def whichBitsAreFalse(value):
    return np.flatnonzero(np.flip(np.where( np.array(list(np.binary_repr(value,width=24))) == '1', 0,1)))
vec_true = np.vectorize(whichBitsAreTrue,otypes=[list])
vec_false = np.vectorize(whichBitsAreFalse,otypes=[list])
vec_chamberName = np.vectorize(getChamberName,otypes=[list])



def unpackVFATStatus_masked_new(evt,station,region,layer,chamber,vfat):
    VFATMasked_dict = {}
    count=0
    for k,layer in enumerate(evt.gemOHStatus_layer):
        station_OH = evt.gemOHStatus_station[k]
        region_OH = evt.gemOHStatus_region[k]
        chamber_OH = evt.gemOHStatus_chamber[k]
        layer_OH = evt.gemOHStatus_layer[k]
        endcapTag = EndcapLayer2label(region,layer)
        vfat_OH = evt.gemOHStatus_VFATMasked[k]
        chamberID = getChamberName(region,chamber,layer,station)

        if(station_OH == station and region_OH == region and chamber_OH == chamber and layer_OH == layer and ((vfat_OH//2**vfat)%2 == 0)):   
            return True
    return False


def unpackVFATStatus_missing_new(evt,station,region,layer,chamber,vfat):
    VFATMasked_dict = {}
    count=0
    for k,layer in enumerate(evt.gemOHStatus_layer):
        station_OH = evt.gemOHStatus_station[k]
        region_OH = evt.gemOHStatus_region[k]
        chamber_OH = evt.gemOHStatus_chamber[k]
        layer_OH = evt.gemOHStatus_layer[k]
        endcapTag = EndcapLayer2label(region,layer)
        vfat_OH = evt.gemOHStatus_VFATMissing[k]
        chamberID = getChamberName(region,chamber,layer,station)

        if(station_OH == station and region_OH == region and chamber_OH == chamber and layer_OH == layer and ((vfat_OH//2**vfat)%2 == 1)): 
            return True
    return False


def unpackVFATStatus_error_new(evt,station,region,layer,chamber,vfat):
    VFATMasked_dict = {}
    count=0
    for k,layer in enumerate(evt.gemOHStatus_layer):
        station_OH = evt.gemOHStatus_station[k]
        region_OH = evt.gemOHStatus_region[k]
        chamber_OH = evt.gemOHStatus_chamber[k]
        layer_OH = evt.gemOHStatus_layer[k]
        endcapTag = EndcapLayer2label(region,layer)
        vfat_OH = evt.gemOHStatus_VFATMissing[k]
        chamberID = getChamberName(region,chamber,layer,station)
        error = evt.gemOHStatus_errors[k]

        if(station_OH == station and region_OH == region and chamber_OH == chamber and layer_OH == layer and error > 0):
            return True
    return False

def unpackVFATStatus_DAQenabled(evt,station,region,layer,chamber,vfat):
    VFATMasked_dict = {}
    count=0
    for k,layer in enumerate(evt.gemOHStatus_layer):
        station_OH = evt.gemOHStatus_station[k]
        region_OH = evt.gemOHStatus_region[k]
        chamber_OH = evt.gemOHStatus_chamber[k]
        layer_OH = evt.gemOHStatus_layer[k]
        endcapTag = EndcapLayer2label(region,layer)
        vfat_OH = evt.gemOHStatus_VFATMissing[k]
        chamberID = getChamberName(region,chamber,layer,station)
        error = evt.gemOHStatus_errors[k]

        if(station_OH == station and region_OH == region and chamber_OH == chamber and layer_OH == layer):
            return True
        return False


def unpackVFATStatus_masked(evt,VFATMaskBook):
    VFATMasked_dict = {}
    #chambers_withoutGEMOHStatus = GE11_ChamberIDs()
    count=0
    for k,layer in enumerate(evt.gemOHStatus_layer):
        station = evt.gemOHStatus_station[k]
        region = evt.gemOHStatus_region[k]
        chamber = evt.gemOHStatus_chamber[k]
        endcapTag = EndcapLayer2label(region,layer)
        VFATsMasked = whichBitsAreFalse(evt.gemOHStatus_VFATMasked[k])
        VFATsMissing = whichBitsAreTrue(evt.gemOHStatus_VFATMissing[k])
        #print("###################",type(VFATsMasked))
        #print("###################",type(VFATsMasked.shape))
        error = evt.gemOHStatus_errors[k]
        warning = ord(evt.gemOHStatus_warnings[k])
        chamberID = getChamberName(region,chamber,layer,station)
        #print("layer",layer)
        count+=1
        #print("VFATsMasked",VFATsMasked)
        #if chamberID in chambers_withoutGEMOHStatus: chambers_withoutGEMOHStatus.remove(chamberID)

        # if error or warning, mask the entire chamber
        #if (error != 0 or warning != 0) : 
        #    VFATsMasked = list(range(24))
        ## Not ready for GE21, VFAT range to be checked
        #if station != 2:
        #    for v in VFATsMasked: VFATMaskBook[endcapTag].Fill(chamber,v)
        

        if list(VFATsMasked) != []:
            VFATMasked_dict[chamberID] = VFATsMasked if VFATMasked_dict.get(chamberID) is None else  np.append(VFATMasked_dict[chamberID],VFATsMasked)
        #if list(VFATsMissing) != []:
        #    VFATMasked_dict[chamberID] = VFATsMissing if VFATMasked_dict.get(chamberID) is None else  np.append(VFATMasked_dict[chamberID],VFATsMissing)

    #print("count",count)
    ## Masking all VFATs from chambers not listed in GEMOHstatus, assuming they were not included in data taking
    #for chamberID in chambers_withoutGEMOHStatus:
    #    VFATsMasked = list(range(24))
    #    VFATMasked_dict[chamberID] = VFATsMasked
    return VFATMasked_dict,VFATMaskBook



def unpackVFATStatus_missing(evt,VFATMaskBook):
    VFATMasked_dict = {}
    #chambers_withoutGEMOHStatus = GE11_ChamberIDs()
    count=0
    for k,layer in enumerate(evt.gemOHStatus_layer):
        station = evt.gemOHStatus_station[k]
        region = evt.gemOHStatus_region[k]
        chamber = evt.gemOHStatus_chamber[k]
        endcapTag = EndcapLayer2label(region,layer)
        VFATsMasked = whichBitsAreFalse(evt.gemOHStatus_VFATMasked[k])
        VFATsMissing = whichBitsAreTrue(evt.gemOHStatus_VFATMissing[k])
        #print("###################",type(VFATsMasked))
        #print("###################",type(VFATsMasked.shape))
        error = evt.gemOHStatus_errors[k]
        warning = ord(evt.gemOHStatus_warnings[k])
        chamberID = getChamberName(region,chamber,layer,station)
        #print("layer",layer)
        count+=1
        #print("VFATsMasked",VFATsMasked)
        #if chamberID in chambers_withoutGEMOHStatus: chambers_withoutGEMOHStatus.remove(chamberID)

        # if error or warning, mask the entire chamber
        #if (error != 0 or warning != 0) :
        #    VFATsMasked = list(range(24))
        ## Not ready for GE21, VFAT range to be checked
        if station != 2:
            for v in VFATsMasked: VFATMaskBook[endcapTag].Fill(chamber,v)


        #if list(VFATsMasked) != []:                                              
        #    VFATMasked_dict[chamberID] = VFATsMasked if VFATMasked_dict.get(chamberID) is None else  np.append(VFATMasked_dict[chamberID],VFATsMasked)                                                                                             
        if list(VFATsMissing) != []:                                                                                                             VFATMasked_dict[chamberID] = VFATsMissing if VFATMasked_dict.get(chamberID) is None else  np.append(VFATMasked_dict[chamberID],VFATsMissing)

    #print("count",count)
    ## Masking all VFATs from chambers not listed in GEMOHstatus, assuming they were not included in data taking
    #for chamberID in chambers_withoutGEMOHStatus:
    #    VFATsMasked = list(range(24))
    #    VFATMasked_dict[chamberID] = VFATsMasked
    return VFATMasked_dict,VFATMaskBook


def unpackVFATStatus_error(evt,VFATMaskBook):
    VFATMasked_dict = {}
    #chambers_withoutGEMOHStatus = GE11_ChamberIDs()
    count=0
    for k,layer in enumerate(evt.gemOHStatus_layer):
        station = evt.gemOHStatus_station[k]
        region = evt.gemOHStatus_region[k]
        chamber = evt.gemOHStatus_chamber[k]
        endcapTag = EndcapLayer2label(region,layer)
        VFATsMasked = whichBitsAreFalse(evt.gemOHStatus_VFATMasked[k])
        VFATsMissing = whichBitsAreTrue(evt.gemOHStatus_VFATMissing[k])
        #print("###################",type(VFATsMasked))
        #print("###################",type(VFATsMasked.shape))
        error = evt.gemOHStatus_errors[k]
        warning = ord(evt.gemOHStatus_warnings[k])
        chamberID = getChamberName(region,chamber,layer,station)
        #print("layer",layer)
        count+=1
        #print("VFATsMasked",VFATsMasked)
        #if chamberID in chambers_withoutGEMOHStatus: chambers_withoutGEMOHStatus.remove(chamberID)

        # if error or warning, mask the entire chamber
        if (error != 0 or warning != 0) :
            VFATsMasked = list(range(24))
        ## Not ready for GE21, VFAT range to be checked
        if station != 2:
            for v in VFATsMasked: VFATMaskBook[endcapTag].Fill(chamber,v)


        #if list(VFATsMasked) != []:
        #    VFATMasked_dict[chamberID] = VFATsMasked if VFATMasked_dict.get(chamberID) is None else  np.append(VFATMasked_dict[chamberID],VFATsMasked)
        #if list(VFATsMissing) != []:    
        #    VFATMasked_dict[chamberID] = VFATsMissing if VFATMasked_dict.get(chamberID) is None else  np.append(VFATMasked_dict[chamberID],VFATsMissing)

    #print("count",count)
    ## Masking all VFATs from chambers not listed in GEMOHstatus, assuming they were not included in data taking
    #for chamberID in chambers_withoutGEMOHStatus:
    #    VFATsMasked = list(range(24))
    #    VFATMasked_dict[chamberID] = VFATsMasked
    return VFATMasked_dict,VFATMaskBook

def generateVFATDict(matching_variables):
    output_dict = {}
    for mv in matching_variables:
        output_dict[mv] = {}

        for (re,la) in [(1,1),(1,2),(-1,1),(-1,2)]:
            endcap_key = EndcapLayer2label(re,la)
            output_dict[mv][endcap_key] = {}
            for chamber in range(1,37):
                chID = getChamberName(re,chamber,la)
                output_dict[mv][endcap_key][chID] = {}
                for VFAT in range(24):
                    output_dict[mv][endcap_key][chID][VFAT]={"num":0,"den":0}
    return output_dict



def generatePropagationErrorContainer(maxErrOnPropR, maxErrOnPropPhi):
    output_dict = {}

    for which_hit in ["MatchedHits","AllHits"]:
        output_dict[which_hit] = {}
        for (re,la) in [(1,1),(1,2),(-1,1),(-1,2)]:
            endcap_key = EndcapLayer2label(re,la)
            output_dict[which_hit][endcap_key] = {}

            for chamber in range(1,37):
                chID = getChamberName(re,chamber,la)
                output_dict[which_hit][endcap_key][chID] = {}
                for eta in list(range(1,9))+["All"]:
                    titleTH1 = chID+"_eta"+str(eta)+"_"
                    output_dict[which_hit][endcap_key][chID][eta] = {"ErrPhi":ROOT.TH1F(titleTH1+"PropagationError on Phi",titleTH1+"PropagationError on Phi",200,0,2*maxErrOnPropPhi),
                                                          "ErrR":ROOT.TH1F(titleTH1+"PropagationError on R",titleTH1+"PropagationError on R",100,0,2*maxErrOnPropR)}
                    output_dict[which_hit][endcap_key][chID][eta]["ErrPhi"].GetXaxis().SetTitle("ErrPhi (rad)")
                    output_dict[which_hit][endcap_key][chID][eta]["ErrR"].GetXaxis().SetTitle("ErrR (cm)")

    return output_dict
        

def fillPlot2DResidualContainer(TH2Fresidual_collector,matching_variables,nbins):
    
    for key_1 in matching_variables:
        for key_2 in ['all','short','long']:
            for key_3 in ['glb_phi','glb_rdphi']:
                for x_bin in range(1,nbins+1):
                    for y_bin in range(1,nbins+1):
                        AVG_Residual = TH2Fresidual_collector[key_1][key_2][key_3][x_bin][y_bin][1]/TH2Fresidual_collector[key_1][key_2][key_3][x_bin][y_bin][0] if TH2Fresidual_collector[key_1][key_2][key_3][x_bin][y_bin][0]!=0 else 0
                        TH2Fresidual_collector[key_1][key_2][key_3]['TH2F'].SetBinContent(x_bin,y_bin,AVG_Residual)
    return TH2Fresidual_collector

def passCut(PropHitonEta,etaPID,prop_hit_index,maxPropR_Err=0.7,maxPropPhi_Err=0.001,fiducialCutR=1.5,fiducialCutPhi=0.002,minPt=0.,maxChi2=9999999,minME1Hit=0,minME2Hit=0,minME3Hit=0,minME4Hit=0):#,ID=2):
#def passCut(PropHitonEta,etaPID,prop_hit_index,maxPropR_Err=0.7,maxPropPhi_Err=0.001,fiducialCutR=0.5,fiducialCutPhi=0.002,minPt=0.,maxChi2=9999999,minME1Hit=0,minME2Hit=0,minME3Hit=0,minME4Hit=0,ID=2):
    passedCut = True
    
    PropHitPt = PropHitonEta['pt'][prop_hit_index]
    #if PropHitPt < minPt:
    #    passedCut = -4

    if PropHitonEta['err_glb_phi'][prop_hit_index] > maxPropPhi_Err:
        passedCut = -2
    if PropHitonEta['err_glb_r'][prop_hit_index] > maxPropR_Err:
        passedCut = -1

    #if PropHitonEta['STA_Normchi2'][prop_hit_index] > maxChi2: # or PropHitonEta['STA_Normchi2'][prop_hit_index] < 0.5:
    #    passedCut = -5
    #if PropHitonEta['STA_Normchi2'][prop_hit_index] > 0.2: # or PropHitonEta['STA_Normchi2'][prop_hit_index] < 0.5:
    #    passedCut = -5

    if PropHitonEta['nME1Hits'][prop_hit_index] < minME1Hit:
        passedCut = -6
    if PropHitonEta['nME2Hits'][prop_hit_index] < minME2Hit:
        passedCut = -6
    if PropHitonEta['nME3Hits'][prop_hit_index] < minME3Hit:
        passedCut = -6
    if PropHitonEta['nME4Hits'][prop_hit_index] < minME4Hit:
        passedCut = -6

    #print("ID given:",muon_id[ID],"   value PropHitonEta[muon_id[ID]]",PropHitonEta[muon_id[ID]])
    #if int(PropHitonEta[muon_id[ID]][prop_hit_index]) != 1:
    #    passedCut = -9
    
    #if int(PropHitonEta['is_global'][prop_hit_index]) == 1:
    #    passedCut = -9
    

    #PhiMin = boundariesGE11[etaPID][1]
    #PhiMax = boundariesGE11[etaPID][0]
    PhiMin = PropHitonEta['phimin'][prop_hit_index]
    PhiMax = PropHitonEta['phimax'][prop_hit_index]    
    PropHitPhi = PropHitonEta['glb_phi'][prop_hit_index]

    #xMin = PropHitonEta['xmin'][prop_hit_index]
    #xMax = PropHitonEta['xmax'][prop_hit_index] 
    #yMin = PropHitonEta['ymin'][prop_hit_index]
    #yMax = PropHitonEta['ymax'][prop_hit_index] 
    PropHitx = PropHitonEta['glb_x'][prop_hit_index]
    PropHity = PropHitonEta['glb_y'][prop_hit_index]
    rMax = PropHitonEta['rmax'][prop_hit_index]
    rMin = PropHitonEta['rmin'][prop_hit_index]
    #rMax = boundariesGE11[etaPID][2]
    #rMin = boundariesGE11[etaPID][3]
    
    ymin,ymax = get_etaP_boundaries_Y(PropHitonEta['region'][prop_hit_index],PropHitonEta['station'][prop_hit_index],PropHitonEta['chamber'][prop_hit_index],PropHitonEta['etaP'][prop_hit_index])
   
    if (ymin == -999 or ymax==-999):
        passedCut = -3
    else:
        if PropHitonEta['loc_y'][prop_hit_index] > (ymax-fiducialCutR):
            passedCut = -3
        if PropHitonEta['loc_y'][prop_hit_index] < (ymin+fiducialCutR):
            passedCut = -3

    #if PropHitonEta['glb_r'][prop_hit_index] > (rMax-fiducialCutR):
    #    passedCut = -3
    #if PropHitonEta['glb_r'][prop_hit_index] < (rMin+fiducialCutR):
    #    passedCut = -3
    

    if PhiMin > PhiMax: # Happens for chamber 19 cause 181 degrees becomes -179 degrees. So chamber 19 has phiMin = 174 degrees phiMax = -174
        PhiMax = 2*numpy.pi + PhiMax 
        if PropHitPhi<0:
            PropHitPhi = 2*numpy.pi + PropHitPhi


    if PropHitPhi < (PhiMin+fiducialCutPhi) or PropHitPhi > (PhiMax-fiducialCutPhi):
        passedCut = -3

    #if PropHitPt < minPt or PropHitPt>70:
    #    passedCut = -4
    if PropHitPt < minPt:
        #if PropHitPt < 0:
        passedCut = -4
     
     
    ## Fiducial cut on chamber perimeter
    # if PropHitonEta['etaP'][prop_hit_index] == 1 and PropHitonEta['glb_r'][prop_hit_index] > (PropHitonEta['mu_propagated_EtaPartition_rMax'][prop_hit_index]-fiducialCutR):
    #     passedCut = False
    # if PropHitonEta['etaP'][prop_hit_index] == 8 and PropHitonEta['glb_r'][prop_hit_index] < (PropHitonEta['mu_propagated_EtaPartition_rMin'][prop_hit_index]+fiducialCutR):
    #     passedCut = False    

    ## Fiducial cut on etaP perimeter
    ## check it does not exceed rmax
        
    return passedCut

## Generate confidence level limits for value obtained from ratio of Poissonian
## Knowing that MUST BE Num < Den
## Not needed...ROOT can do it automagically with TGraphAsymmErrors::DIVIDE
def generateClopperPeasrsonInterval(num,den):
    confidenceLevel = 0.95
    alpha = 1 - confidenceLevel
    
    if num == 0:
        lowerLimit = 0
    else:
        lowerLimit = round(ROOT.Math.beta_quantile(alpha/2,num,den-num + 1),4)

    if num==den:
        upperLimit=1
    else:
        upperLimit = round(ROOT.Math.beta_quantile(1-alpha/2,num + 1,den-num),4)
    return lowerLimit,upperLimit

def generateEfficiencyPlotbyEta(sourceDict,input_region=1,input_layer=1):
    ## Transforming in list
    reg_tag_string = "All" if isinstance(input_region, list) else "P" if input_region == 1 else "M"
    lay_tag_string = "" if isinstance(input_layer, list) else "L1" if input_layer == 1 else "L2"
    input_region = [input_region] if not isinstance(input_region, list) else input_region
    input_layer = [input_layer] if not isinstance(input_layer, list) else input_layer
    
    TH1F_TempContainer = {}
    Plot_Container = {}
    ColorAssociation = {'All':ROOT.kBlack,'Long':ROOT.kGreen+1,'Short':ROOT.kRed}
    
    title = "GE11"+reg_tag_string+lay_tag_string+"_EffbyEta"
    for chambers in ['All','Long','Short']:   
        Plot_Container[chambers] = ROOT.TGraphAsymmErrors()
        Plot_Container[chambers].GetXaxis().SetTitle("GE11 Eta Partition")
        Plot_Container[chambers].GetYaxis().SetTitle("Efficiency")
        Plot_Container[chambers].SetMaximum(1.1)
        Plot_Container[chambers].SetTitle(chambers+"_"+title)
        Plot_Container[chambers].SetName(chambers+"_"+title)
        Plot_Container[chambers].SetLineColor(ColorAssociation[chambers])
        Plot_Container[chambers].SetMarkerColor(ColorAssociation[chambers])
        Plot_Container[chambers].SetFillColorAlpha(ColorAssociation[chambers],.4)
        Plot_Container[chambers].SetMarkerStyle(20)
        Plot_Container[chambers].SetMarkerSize(.8)

        TH1F_TempContainer.setdefault(chambers,{'num':ROOT.TH1F('num_'+chambers+title, title,8,0.5,8.5),'den':ROOT.TH1F('den_'+chambers+title, title,8,0.5,8.5)})

    for etaPartitionID in sourceDict.keys():
        station,region,chamber,layer,eta = convert_etaPID(etaPartitionID)
    
        if layer not in input_layer or region not in  input_region:
            continue
        
        TH1F_TempContainer['All']['num'].SetBinContent( eta, TH1F_TempContainer['All']['num'].GetBinContent(eta) + sum([sourceDict[etaPartitionID][pt]['num'] for pt in range(0,11)]) )
        TH1F_TempContainer['All']['den'].SetBinContent( eta, TH1F_TempContainer['All']['den'].GetBinContent(eta) + sum([sourceDict[etaPartitionID][pt]['den'] for pt in range(0,11)]) )

        if chamber%2==0:
            TH1F_TempContainer['Long']['num'].SetBinContent( eta, TH1F_TempContainer['Long']['num'].GetBinContent(eta) + sum([sourceDict[etaPartitionID][pt]['num'] for pt in range(0,11)]) )
            TH1F_TempContainer['Long']['den'].SetBinContent( eta, TH1F_TempContainer['Long']['den'].GetBinContent(eta) + sum([sourceDict[etaPartitionID][pt]['den'] for pt in range(0,11)]) )
        else:
            TH1F_TempContainer['Short']['num'].SetBinContent( eta, TH1F_TempContainer['Short']['num'].GetBinContent(eta) + sum([sourceDict[etaPartitionID][pt]['num'] for pt in range(0,11)]) )
            TH1F_TempContainer['Short']['den'].SetBinContent( eta, TH1F_TempContainer['Short']['den'].GetBinContent(eta) + sum([sourceDict[etaPartitionID][pt]['den'] for pt in range(0,11)]) )
    
    
    for chambers in ['All','Long','Short']: 
        Plot_Container[chambers].Divide(TH1F_TempContainer[chambers]['num'],TH1F_TempContainer[chambers]['den'],"B")

    return Plot_Container['Short'],Plot_Container['Long'],Plot_Container['All']

def generate1DEfficiencyPlotbyVFAT(EfficiencyDictVFAT,chamber_ID):
    
    re_ch_la_list = chamberName2ReChLa(chamber_ID)
    re = re_ch_la_list[0]
    ch = re_ch_la_list[1]
    la = re_ch_la_list[2]
    endcapTag = EndcapLayer2label(re,la)

    title = chamber_ID+"_1DEffVFAT"
    Summary = ROOT.TGraphAsymmErrors()
    Summary.GetXaxis().SetTitle("VFAT Number")
    Summary.GetYaxis().SetTitle("Efficiency")
    Summary.SetMaximum(1.1)
    Summary.SetMinimum(0)
    Summary.SetTitle(title)
    Summary.SetName(title)
    Summary.SetMarkerStyle(20)
    Summary.SetMarkerSize(.8)

    TH1Num = ROOT.TH1F(title, title,24,-0.5,23.5)
    TH1Den = ROOT.TH1F(title, title,24,-0.5,23.5)

    for VFATN in range(24):
        TH1Num.SetBinContent(VFATN+1,EfficiencyDictVFAT[endcapTag][chamber_ID][VFATN]['num'])
        TH1Den.SetBinContent(VFATN+1,EfficiencyDictVFAT[endcapTag][chamber_ID][VFATN]['den'])
    
    Summary.Divide(TH1Num,TH1Den,"B")
    Summary.GetXaxis().SetLimits(-0.5,23.5)

    return Summary

def errorOnEfficiency(num,den):
    eff = float(num)/den
    err_num = num**(1./2.)
    err_den = den**(1./2.)
    if num==1 and den==1:
        return 1
    err_eff=eff*( ((err_num/num)**2 + (err_den/den)**2)**(1./2.)  )
    #print("####  err_eff: ",err_num, err_den,eff,err_eff)
    return err_eff

def generate2DEfficiencyPlotbyVFAT(EfficiencyDictVFAT,efficiency_target,extra=0):
    ## GEOMETRY DEFINITION
    VFATGeometryDict = {"Long":dict((VFATN,{}) for VFATN in range(0,24)),"Short":dict((VFATN,{}) for VFATN in range(0,24))}

    VFATGeometryDict['Short'][0]['x'] = np.asarray([-3.83938562003834,-4.14080692494689,-12.4513908518566,-11.5450181214881],dtype=float)
    VFATGeometryDict['Short'][0]['y'] = np.asarray([0,10.206,10.206,0],dtype=float)
    VFATGeometryDict['Short'][1]['x'] = np.asarray([-4.14080692494689,-4.44222822985544,-13.3577635822252,-12.4513908518566],dtype=float)
    VFATGeometryDict['Short'][1]['y'] = np.asarray([10.206,20.412,20.412,10.206],dtype=float)
    VFATGeometryDict['Short'][2]['x'] = np.asarray([-4.44222822985544,-4.79536310569235,-14.4196388259069,-13.3577635822252],dtype=float)
    VFATGeometryDict['Short'][2]['y'] = np.asarray([20.412,32.369,32.369,20.412],dtype=float)
    VFATGeometryDict['Short'][3]['x'] = np.asarray([-4.79536310569235,-5.14849798152926,-15.4815140695887,-14.4196388259069],dtype=float)
    VFATGeometryDict['Short'][3]['y'] = np.asarray([32.369,44.326,44.326,32.369],dtype=float)
    VFATGeometryDict['Short'][4]['x'] = np.asarray([-5.14849798152926,-5.56365370199756,-16.7298857598484,-15.4815140695887],dtype=float)
    VFATGeometryDict['Short'][4]['y'] = np.asarray([44.326,58.383,58.383,44.326],dtype=float)
    VFATGeometryDict['Short'][5]['x'] = np.asarray([-5.56365370199756,-5.97880942246586,-17.9782574501081,-16.7298857598484],dtype=float)
    VFATGeometryDict['Short'][5]['y'] = np.asarray([58.383,72.44,72.44,58.383],dtype=float)
    VFATGeometryDict['Short'][6]['x'] = np.asarray([-5.97880942246586,-6.47069378786385,-19.4573518871341,-17.9782574501081],dtype=float)
    VFATGeometryDict['Short'][6]['y'] = np.asarray([72.4399999999999,89.0949999999999,89.0949999999999,72.4399999999999],dtype=float)
    VFATGeometryDict['Short'][7]['x'] = np.asarray([-6.47069378786385,-6.96257815326184,-20.9364463241602,-19.4573518871341],dtype=float)
    VFATGeometryDict['Short'][7]['y'] = np.asarray([89.0949999999999,105.75,105.75,89.0949999999999],dtype=float)
    VFATGeometryDict['Short'][8]['x'] = np.asarray([-3.83938562003834,-4.14080692494689,4.14080692494689,3.83938562003834],dtype=float)
    VFATGeometryDict['Short'][8]['y'] = np.asarray([0,10.206,10.206,0],dtype=float)
    VFATGeometryDict['Short'][9]['x'] = np.asarray([-4.14080692494689,-4.44222822985544,4.44222822985544,4.14080692494689],dtype=float)
    VFATGeometryDict['Short'][9]['y'] = np.asarray([10.206,20.412,20.412,10.206],dtype=float)
    VFATGeometryDict['Short'][10]['x'] = np.asarray([-4.44222822985544,-4.79536310569235,4.79536310569235,4.44222822985544],dtype=float)
    VFATGeometryDict['Short'][10]['y'] = np.asarray([20.412,32.369,32.369,20.412],dtype=float)
    VFATGeometryDict['Short'][11]['x'] = np.asarray([-4.79536310569235,-5.14849798152926,5.14849798152926,4.79536310569235],dtype=float)
    VFATGeometryDict['Short'][11]['y'] = np.asarray([32.369,44.326,44.326,32.369],dtype=float)
    VFATGeometryDict['Short'][12]['x'] = np.asarray([-5.14849798152926,-5.56365370199756,5.56365370199756,5.14849798152926],dtype=float)
    VFATGeometryDict['Short'][12]['y'] = np.asarray([44.326,58.383,58.383,44.326],dtype=float)
    VFATGeometryDict['Short'][13]['x'] = np.asarray([-5.56365370199756,-5.97880942246586,5.97880942246586,5.56365370199756],dtype=float)
    VFATGeometryDict['Short'][13]['y'] = np.asarray([58.383,72.44,72.44,58.383],dtype=float)
    VFATGeometryDict['Short'][14]['x'] = np.asarray([-5.97880942246586,-6.47069378786385,6.47069378786385,5.97880942246586],dtype=float)
    VFATGeometryDict['Short'][14]['y'] = np.asarray([72.4399999999999,89.0949999999999,89.0949999999999,72.4399999999999],dtype=float)
    VFATGeometryDict['Short'][15]['x'] = np.asarray([-6.47069378786385,-6.96257815326184,6.96257815326184,6.47069378786385],dtype=float)
    VFATGeometryDict['Short'][15]['y'] = np.asarray([89.0949999999999,105.75,105.75,89.0949999999999],dtype=float)
    VFATGeometryDict['Short'][16]['x'] = np.asarray([3.83938562003834,4.14080692494689,12.4513908518566,11.5450181214881],dtype=float)
    VFATGeometryDict['Short'][16]['y'] = np.asarray([0,10.206,10.206,0],dtype=float)
    VFATGeometryDict['Short'][17]['x'] = np.asarray([4.14080692494689,4.44222822985544,13.3577635822252,12.4513908518566],dtype=float)
    VFATGeometryDict['Short'][17]['y'] = np.asarray([10.206,20.412,20.412,10.206],dtype=float)
    VFATGeometryDict['Short'][18]['x'] = np.asarray([4.44222822985544,4.79536310569235,14.4196388259069,13.3577635822252],dtype=float)
    VFATGeometryDict['Short'][18]['y'] = np.asarray([20.412,32.369,32.369,20.412],dtype=float)
    VFATGeometryDict['Short'][19]['x'] = np.asarray([4.79536310569235,5.14849798152926,15.4815140695887,14.4196388259069],dtype=float)
    VFATGeometryDict['Short'][19]['y'] = np.asarray([32.369,44.326,44.326,32.369],dtype=float)
    VFATGeometryDict['Short'][20]['x'] = np.asarray([5.14849798152926,5.56365370199756,16.7298857598484,15.4815140695887],dtype=float)
    VFATGeometryDict['Short'][20]['y'] = np.asarray([44.326,58.383,58.383,44.326],dtype=float)
    VFATGeometryDict['Short'][21]['x'] = np.asarray([5.56365370199756,5.97880942246586,17.9782574501081,16.7298857598484],dtype=float)
    VFATGeometryDict['Short'][21]['y'] = np.asarray([58.383,72.44,72.44,58.383],dtype=float)
    VFATGeometryDict['Short'][22]['x'] = np.asarray([5.97880942246586,6.47069378786385,19.4573518871341,17.9782574501081],dtype=float)
    VFATGeometryDict['Short'][22]['y'] = np.asarray([72.4399999999999,89.0949999999999,89.0949999999999,72.4399999999999],dtype=float)
    VFATGeometryDict['Short'][23]['x'] = np.asarray([6.47069378786385,6.96257815326184,20.9364463241602,19.4573518871341],dtype=float)
    VFATGeometryDict['Short'][23]['y'] = np.asarray([89.0949999999999,105.75,105.75,89.0949999999999],dtype=float)
    VFATGeometryDict['Long'][0]['x'] = np.asarray([-3.83938562003834,-4.17332356777506,-12.5491682745625,-11.5450181214881],dtype=float)
    VFATGeometryDict['Long'][0]['y'] = np.asarray([0,11.307,11.307,0],dtype=float)
    VFATGeometryDict['Long'][1]['x'] = np.asarray([-4.17332356777506,-4.50726151551178,-13.5533184276368,-12.5491682745625],dtype=float)
    VFATGeometryDict['Long'][1]['y'] = np.asarray([11.307,22.614,22.614,11.307],dtype=float)
    VFATGeometryDict['Long'][2]['x'] = np.asarray([-4.50726151551178,-4.90466746092129,-14.7483166110425,-13.5533184276368],dtype=float)
    VFATGeometryDict['Long'][2]['y'] = np.asarray([22.614,36.07,36.07,22.614],dtype=float)
    VFATGeometryDict['Long'][3]['x'] = np.asarray([-4.90466746092129,-5.30207340633079,-15.9433147944483,-14.7483166110425],dtype=float)
    VFATGeometryDict['Long'][3]['y'] = np.asarray([36.07,49.526,49.526,36.07],dtype=float)
    VFATGeometryDict['Long'][4]['x'] = np.asarray([-5.30207340633079,-5.77777328465355,-17.3737425397006,-15.9433147944483],dtype=float)
    VFATGeometryDict['Long'][4]['y'] = np.asarray([49.526,65.633,65.633,49.526],dtype=float)
    VFATGeometryDict['Long'][5]['x'] = np.asarray([-5.77777328465355,-6.2534731629763,-18.804170284953,-17.3737425397006],dtype=float)
    VFATGeometryDict['Long'][5]['y'] = np.asarray([65.633,81.74,81.74,65.633],dtype=float)
    VFATGeometryDict['Long'][6]['x'] = np.asarray([-6.2534731629763,-6.82657530110586,-20.5274862591644,-18.804170284953],dtype=float)
    VFATGeometryDict['Long'][6]['y'] = np.asarray([81.74,101.145,101.145,81.74],dtype=float)
    VFATGeometryDict['Long'][7]['x'] = np.asarray([-6.82657530110586,-7.39967743923543,-22.2508022333757,-20.5274862591644],dtype=float)
    VFATGeometryDict['Long'][7]['y'] = np.asarray([101.145,120.55,120.55,101.145],dtype=float)
    VFATGeometryDict['Long'][8]['x'] = np.asarray([-3.83938562003834,-4.17332356777506,4.17332356777506,3.83938562003834],dtype=float)
    VFATGeometryDict['Long'][8]['y'] = np.asarray([0,11.307,11.307,0],dtype=float)
    VFATGeometryDict['Long'][9]['x'] = np.asarray([-4.17332356777506,-4.50726151551178,4.50726151551178,4.17332356777506],dtype=float)
    VFATGeometryDict['Long'][9]['y'] = np.asarray([11.307,22.614,22.614,11.307],dtype=float)
    VFATGeometryDict['Long'][10]['x'] = np.asarray([-4.50726151551178,-4.90466746092129,4.90466746092129,4.50726151551178],dtype=float)
    VFATGeometryDict['Long'][10]['y'] = np.asarray([22.614,36.07,36.07,22.614],dtype=float)
    VFATGeometryDict['Long'][11]['x'] = np.asarray([-4.90466746092129,-5.30207340633079,5.30207340633079,4.90466746092129],dtype=float)
    VFATGeometryDict['Long'][11]['y'] = np.asarray([36.07,49.526,49.526,36.07],dtype=float)
    VFATGeometryDict['Long'][12]['x'] = np.asarray([-5.30207340633079,-5.77777328465355,5.77777328465355,5.30207340633079],dtype=float)
    VFATGeometryDict['Long'][12]['y'] = np.asarray([49.526,65.633,65.633,49.526],dtype=float)
    VFATGeometryDict['Long'][13]['x'] = np.asarray([-5.77777328465355,-6.2534731629763,6.2534731629763,5.77777328465355],dtype=float)
    VFATGeometryDict['Long'][13]['y'] = np.asarray([65.633,81.74,81.74,65.633],dtype=float)
    VFATGeometryDict['Long'][14]['x'] = np.asarray([-6.2534731629763,-6.82657530110586,6.82657530110586,6.2534731629763],dtype=float)
    VFATGeometryDict['Long'][14]['y'] = np.asarray([81.74,101.145,101.145,81.74],dtype=float)
    VFATGeometryDict['Long'][15]['x'] = np.asarray([-6.82657530110586,-7.39967743923543,7.39967743923543,6.82657530110586],dtype=float)
    VFATGeometryDict['Long'][15]['y'] = np.asarray([101.145,120.55,120.55,101.145],dtype=float)
    VFATGeometryDict['Long'][16]['x'] = np.asarray([3.83938562003834,4.17332356777506,12.5491682745625,11.5450181214881],dtype=float)
    VFATGeometryDict['Long'][16]['y'] = np.asarray([0,11.307,11.307,0],dtype=float)
    VFATGeometryDict['Long'][17]['x'] = np.asarray([4.17332356777506,4.50726151551178,13.5533184276368,12.5491682745625],dtype=float)
    VFATGeometryDict['Long'][17]['y'] = np.asarray([11.307,22.614,22.614,11.307],dtype=float)
    VFATGeometryDict['Long'][18]['x'] = np.asarray([4.50726151551178,4.90466746092129,14.7483166110425,13.5533184276368],dtype=float)
    VFATGeometryDict['Long'][18]['y'] = np.asarray([22.614,36.07,36.07,22.614],dtype=float)
    VFATGeometryDict['Long'][19]['x'] = np.asarray([4.90466746092129,5.30207340633079,15.9433147944483,14.7483166110425],dtype=float)
    VFATGeometryDict['Long'][19]['y'] = np.asarray([36.07,49.526,49.526,36.07],dtype=float)
    VFATGeometryDict['Long'][20]['x'] = np.asarray([5.30207340633079,5.77777328465355,17.3737425397006,15.9433147944483],dtype=float)
    VFATGeometryDict['Long'][20]['y'] = np.asarray([49.526,65.633,65.633,49.526],dtype=float)
    VFATGeometryDict['Long'][21]['x'] = np.asarray([5.77777328465355,6.2534731629763,18.804170284953,17.3737425397006],dtype=float)
    VFATGeometryDict['Long'][21]['y'] = np.asarray([65.633,81.74,81.74,65.633],dtype=float)
    VFATGeometryDict['Long'][22]['x'] = np.asarray([6.2534731629763,6.82657530110586,20.5274862591644,18.804170284953],dtype=float)
    VFATGeometryDict['Long'][22]['y'] = np.asarray([81.74,101.145,101.145,81.74],dtype=float)
    VFATGeometryDict['Long'][23]['x'] = np.asarray([6.82657530110586,7.39967743923543,22.2508022333757,20.5274862591644],dtype=float)
    VFATGeometryDict['Long'][23]['y'] = np.asarray([101.145,120.55,120.55,101.145],dtype=float)

    ## Additional polygon to standardize Axis ranges
    shape_x = np.asarray([-120.55/2,-120.55/2,120.55/2,120.55/2],dtype=float)
    shape_y = np.asarray([125.,125.01,125.01,125.],dtype=float)
    ## END OF GEOMETRY DEFINITION
    h2p = ROOT.TH2Poly()
    h2p_num = ROOT.TH2Poly()
    h2p_den = ROOT.TH2Poly()
    h2p.SetMinimum(0)
    h2p_num.SetMinimum(0)
    h2p_den.SetMinimum(0)
    #h2p.SetTitle(efficiency_target+"_2DEffVFAT")
    #h2p.SetName(efficiency_target+"_2DEffVFAT")
    #h2p.GetYaxis().SetTickLength(0.005)
    #h2p.GetYaxis().SetTitleOffset(1.3)
    #h2p.GetXaxis().SetTickLength(0.005)
    #h2p.GetYaxis().SetLabelSize(0.03)
    #h2p.GetXaxis().SetLabelSize(0.03)
    h2p.SetStats(False)
    h2p_num.SetStats(False)
    h2p_den.SetStats(False)
    if efficiency_target in ["ML1","ML2","PL1","PL2"]:
        endcapTag = efficiency_target
        for chamber_number in range(1,37):
            chamberSize = "Short" if chamber_number%2 == 1 else "Long"
            
            region = 1 if endcapTag[0]=="P" else -1
            chamber_ID = getChamberName(region,chamber_number,endcapTag[-1])          
            #if(chamber_ID in ["GE11-P-33L2-S"]):
            #    continue
            #if(chamber_ID not in ["GE11-M-25L2-S", "GE11-M-26L2-L"]):
            #    continue
            
            
            angle_deg = -90 + (chamber_number - 1)*10## degrees
            angle_rad = np.radians(angle_deg)
            for VFATN in range(24):

                original_x = VFATGeometryDict[chamberSize][VFATN]['x']
                original_y = VFATGeometryDict[chamberSize][VFATN]['y']

                ## fix needed to respect GE11 installation, long SCs have Drift facing the Interaction Point. Short SCs have ReadOut Board facing the IP
                ## Applying reflection along y axis
                if (region == 1 and chamber_number % 2 == 0) or (region == -1 and chamber_number % 2 == 1):
                    original_x = -original_x

                translated_x = original_x
                translated_y = original_y + 130 ## GE11 ~ R = 130 cm                
                rotated_and_translated_x = np.asarray([ translated_x[i]*np.cos(angle_rad) - translated_y[i]*np.sin(angle_rad) for i in range(len(translated_x)) ],dtype=float)
                rotated_and_translated_y = np.asarray([ translated_x[i]*np.sin(angle_rad) + translated_y[i]*np.cos(angle_rad) for i in range(len(translated_x)) ],dtype=float)

                VFAT_den = EfficiencyDictVFAT[endcapTag][chamber_ID][VFATN]['den'] 
                VFAT_num = EfficiencyDictVFAT[endcapTag][chamber_ID][VFATN]['num'] 
                err_eff = 0
                if VFAT_den == 0:
                    efficiency = 0 ## Bin has no text in case of no prop hit
                elif VFAT_num == 0:
                    efficiency = 0.01 ## Avoid empty label when printing it with text
                else:
                    efficiency = 100*round(float(VFAT_num)/VFAT_den,3)
                    err_eff = round(errorOnEfficiency(VFAT_num,VFAT_den),3)
                if(chamber_ID in ["GE11-P-33L2-S"]):
                    efficiency=0

                selected_bin = h2p.AddBin(4,rotated_and_translated_x,rotated_and_translated_y)
                selected_bin_num = h2p_num.AddBin(4,rotated_and_translated_x,rotated_and_translated_y)
                selected_bin_den = h2p_den.AddBin(4,rotated_and_translated_x,rotated_and_translated_y)
                h2p.SetBinContent(selected_bin,efficiency)
                h2p.SetBinError(selected_bin,err_eff)
                h2p.SetMarkerSize(.3) ## Reduces the text size when drawing the option "COLZ TEXT"
                h2p_num.SetMarkerSize(.3)
                h2p_den.SetMarkerSize(.3)
                h2p_num.SetBinContent(selected_bin_num,VFAT_num)
                h2p_den.SetBinContent(selected_bin_den,VFAT_den)
                
                h2p.GetXaxis().SetTitle("Global x [cm]")
                h2p.GetXaxis().SetTitleSize(0.042)
                h2p.GetYaxis().SetTitleSize(0.042)
                h2p.GetXaxis().SetLabelSize(0.03)
                h2p.GetYaxis().SetLabelSize(0.03)
                h2p.GetYaxis().SetTitle("Global y [cm]")
                h2p_num.GetXaxis().SetTitle("Global x [cm]")
                h2p_num.GetYaxis().SetTitle("Global y [cm]")
                h2p_den.GetXaxis().SetTitle("Global x [cm]")
                h2p_den.GetYaxis().SetTitle("Global y [cm]")
    else:
        chamber_ID = efficiency_target
        re_ch_la_list = chamberName2ReChLa(chamber_ID)
        re = re_ch_la_list[0]
        ch = re_ch_la_list[1]
        la = re_ch_la_list[2]
        chamberSize = "Short" if ch%2 == 1 else "Long"
        endcapTag = EndcapLayer2label(re,la)
        

        for VFATN in range(24):
            selected_bin = h2p.AddBin(4,VFATGeometryDict[chamberSize][VFATN]['x'],VFATGeometryDict[chamberSize][VFATN]['y'])
            selected_bin_num = h2p_num.AddBin(4,VFATGeometryDict[chamberSize][VFATN]['x'],VFATGeometryDict[chamberSize][VFATN]['y'])
            selected_bin_den = h2p_den.AddBin(4,VFATGeometryDict[chamberSize][VFATN]['x'],VFATGeometryDict[chamberSize][VFATN]['y'])
                
            VFAT_den = EfficiencyDictVFAT[endcapTag][chamber_ID][VFATN]['den'] 
            VFAT_num = EfficiencyDictVFAT[endcapTag][chamber_ID][VFATN]['num'] 
            err_eff=0    
            if VFAT_den == 0:
                efficiency = 0 ## Bin has no text in case of no prop hit
            elif VFAT_num == 0:
                    efficiency = 0.01 ## Avoid empty label when printing it with text
            else:
                efficiency = round(float(VFAT_num)/VFAT_den,2)
                err_eff = round(errorOnEfficiency(VFAT_num,VFAT_den),2)

            h2p.SetBinContent(selected_bin,efficiency)
            h2p.SetBinError(selected_bin,err_eff)
            h2p_num.SetBinContent(selected_bin_num,VFAT_num)
            h2p_den.SetBinContent(selected_bin_den,VFAT_den)

        h2p.AddBin(4,shape_x,shape_y)
        h2p_num.AddBin(4,shape_x,shape_y)
        h2p_den.AddBin(4,shape_x,shape_y)
        ### by sumit
        h2p_num.GetXaxis().SetTitle("Global x [cm]")
        h2p_num.GetYaxis().SetTitle("Global y [cm]")
        h2p_den.GetXaxis().SetTitle("Global x [cm]")
        h2p_den.GetYaxis().SetTitle("Global y [cm]")
        #h2p_num.SetMarkerSize() ## Reduces the text size when drawing the option "COLZ TEXT"
        #h2p_den.SetMarkerSize(.3) ## Reduces the text size when drawing the option "COLZ TEXT"
        if extra==1:
            return h2p,h2p_num,h2p_den

    # h2p.SetMaximum(1.1)
    if extra==1:
        return h2p,h2p_num,h2p_den
    return h2p


def generateEfficiencyPlotbyPt(sourceDict,input_region=[-1,1],input_layer=[1,2]):
    ## Transforming in list
    reg_tag_string = "All" if isinstance(input_region, list) else "P" if input_region == 1 else "M"
    lay_tag_string = "" if isinstance(input_layer, list) else "L1" if input_layer == 1 else "L2"
    input_region = [input_region] if not isinstance(input_region, list) else input_region
    input_layer = [input_layer] if not isinstance(input_layer, list) else input_layer
    
    TH1F_TempContainer = {}
    Plot_Container = {}
    ColorAssociation = {'All':ROOT.kBlack,'Long':ROOT.kGreen+1,'Short':ROOT.kRed}
    
    title = "GE11"+reg_tag_string+lay_tag_string+"_EffbyPt"
    for chambers in ['All','Long','Short']:   
        Plot_Container[chambers] = ROOT.TGraphAsymmErrors()
        Plot_Container[chambers].GetXaxis().SetTitle("pt (GeV)")
        Plot_Container[chambers].GetYaxis().SetTitle("Efficiency")
        Plot_Container[chambers].SetMaximum(1.1)
        Plot_Container[chambers].SetMinimum(0)
        Plot_Container[chambers].SetTitle(chambers+"_"+title)
        Plot_Container[chambers].SetName(chambers+"_"+title)
        Plot_Container[chambers].SetLineColor(ColorAssociation[chambers])
        Plot_Container[chambers].SetMarkerColor(ColorAssociation[chambers])
        Plot_Container[chambers].SetFillColorAlpha(ColorAssociation[chambers],.4)
        Plot_Container[chambers].SetMarkerStyle(20)
        Plot_Container[chambers].SetMarkerSize(.8)

        TH1F_TempContainer.setdefault(chambers,{'num':ROOT.TH1F('num_'+chambers+title, title,100,0,500),'den':ROOT.TH1F('den_'+chambers+title, title,100,0,500)})

    for pt in range(0,101):
        
        #TH1F_TempContainer['All']['num'].SetBinContent(pt+1, TH1F_TempContainer['All']['num'].GetBinContent(pt) + sum([sourceDict[etaPartitionID][pt]['num'] for etaPartitionID in sourceDict.keys()]))
        #TH1F_TempContainer['All']['den'].SetBinContent(pt+1, TH1F_TempContainer['All']['den'].GetBinContent(pt) + sum([sourceDict[etaPartitionID][pt]['den'] for etaPartitionID in sourceDict.keys()]))
        TH1F_TempContainer['All']['num'].SetBinContent(pt+1, sum([sourceDict[etaPartitionID][pt]['num'] for etaPartitionID in sourceDict.keys()]))
        TH1F_TempContainer['All']['den'].SetBinContent(pt+1, sum([sourceDict[etaPartitionID][pt]['den'] for etaPartitionID in sourceDict.keys()]))
        
        long_chambers_etaPartitionID = [etaPartitionID for etaPartitionID in sourceDict.keys() if convert_etaPID(etaPartitionID)[0] % 2 == 0]
        short_chambers_etaPartitionID = [etaPartitionID for etaPartitionID in sourceDict.keys() if convert_etaPID(etaPartitionID)[0] % 2 == 1]
        
        TH1F_TempContainer['Long']['num'].SetBinContent(pt+1, TH1F_TempContainer['Long']['num'].GetBinContent(pt) + sum([sourceDict[etaPartitionID][pt]['num'] for etaPartitionID in long_chambers_etaPartitionID]))
        TH1F_TempContainer['Long']['den'].SetBinContent(pt+1, TH1F_TempContainer['Long']['den'].GetBinContent(pt) + sum([sourceDict[etaPartitionID][pt]['den'] for etaPartitionID in long_chambers_etaPartitionID]))

        TH1F_TempContainer['Short']['num'].SetBinContent(pt+1, TH1F_TempContainer['Short']['num'].GetBinContent(pt) + sum([sourceDict[etaPartitionID][pt]['num'] for etaPartitionID in short_chambers_etaPartitionID]))
        TH1F_TempContainer['Short']['den'].SetBinContent(pt+1, TH1F_TempContainer['Short']['den'].GetBinContent(pt) + sum([sourceDict[etaPartitionID][pt]['den'] for etaPartitionID in short_chambers_etaPartitionID]))
    
    
    for chambers in ['All','Long','Short']:
        Plot_Container[chambers].Divide(TH1F_TempContainer[chambers]['num'],TH1F_TempContainer[chambers]['den'],"B")

    return Plot_Container['Short'],Plot_Container['Long'],Plot_Container['All'], TH1F_TempContainer['All']


def generateEfficiencyDistribution(sourceDict):
    EfficiencyDistribution = ROOT.TH1F("EfficiencyDistribution","EfficiencyDistribution",50,0.,50.)
    EfficiencyDistribution_neg = ROOT.TH1F("EfficiencyDistribution_neg","EfficiencyDistribution (neg endcap)",100,0.,100.)
    EfficiencyDistribution_pos = ROOT.TH1F("EfficiencyDistribution_pos","EfficiencyDistribution (pos endcap)",100,0.,100.)

    Num = {}
    Den = {}
    Eff = {}
    
    for region in [-1,1]:
        for chamber in range(1,37):
            for layer in [1,2]:
                key = region*(100*chamber+10*layer)
                
                Num[key] = 0
                Den[key] = 0
                Eff[key] = -9

    for etaPartitionID,value in sourceDict.items():
        station,region,chamber,layer,eta = convert_etaPID(etaPartitionID)

        key = region*(100*chamber+10*layer)

        Num[key] += sum([value[k]['num'] for k in value.keys()])
        Den[key] += sum([value[k]['den'] for k in value.keys()])

    for k in Num.keys():
        if (Den[k] != 0):
            Eff[k] = float(Num[k])/float(Den[k])
        ### Sumit ###
        if(k<0):    
           EfficiencyDistribution_neg.Fill(100*Eff[k])
        if(k>0):
           EfficiencyDistribution_pos.Fill(100*Eff[k])
        ######
        EfficiencyDistribution.Fill(100*Eff[k])
    OverFlow = EfficiencyDistribution.GetBinContent(101)
    UnderFlow = EfficiencyDistribution.GetBinContent(0)
    return EfficiencyDistribution,EfficiencyDistribution_neg,EfficiencyDistribution_pos   #Sumit

def shortChambers():
    chamberNames_withShort = []
    #os.system(cmd)
    file1 = open(f'{OUTPUT_PATH}/GE11Short.csv', 'r')
    Lines = file1.readlines()
    for line in Lines:
        line = line.replace("\"","")
        line = line.strip()
        line = line.split(",")
        if "GE" in line[1] and "disappeared" not in line[0] and "anomaly" not in line[0] and line[2].lstrip("-").isdigit():  #check if the short is there, documented in its layer and chamber
            if "-" in line[1]: region_short = -1
            else: region_short = 1
            line[1] = line[1].replace(" ","")
            chamber_short = int(line[1][-2:])
            if line[3] == '':
                chamberNames_withShort.append(getChamberName(region_short,chamber_short,1))
                chamberNames_withShort.append(getChamberName(region_short,chamber_short,2))
            else:
                layer_short = int(line[3])
                chamberNames_withShort.append(getChamberName(region_short,chamber_short,layer_short))
        
    return chamberNames_withShort
"""

def generateEffDistr(filepath,removeShort=False):
    #EfficiencyDistribution = ROOT.TH1F("EfficiencyDistribution","EfficiencyDistribution",50,50.,100.)
    EfficiencyDistribution = ROOT.TH1F("EfficiencyDistribution","EfficiencyDistribution",100,0.,100.)
    chamberNames_withShort =  shortChambers()
    #print("%%%%%%%%% Chambers with shorts %%%%%%%%%%%%%%%%%%\n",chamberNames_withShort)
    for re,la in [(-1,1),(1,1),(-1,2),(1,2)]: 
        for chamber in range(1,37):
            chamberID = getChamberName(re,chamber,la)
            if(chamberID in ["GE11-P-33L2-S"]):
                continue
            #print(chamberID)
            #if(chamberID in ["GE11-P-18L2-L","GE11-M-14L2-L"]):
            #    continue
            n,d,_ = ChamberEfficiencyFromCSV(filepath,chamberID)
            if float(d) !=0.:
                Eff = 100*float(n)/d 
                if removeShort == True and Eff < 85 and chamberID in chamberNames_withShort:
                    print("chamberId: ",chamberID)
                else:
                    EfficiencyDistribution.Fill(Eff)
    return EfficiencyDistribution
"""            
def generateEffDistr(filepath,includeCh=""):
    #EfficiencyDistribution = ROOT.TH1F("EfficiencyDistribution","EfficiencyDistribution",50,50.,100.)
    chamberNames_HV_670 = ['GE11-P-06L1-L', 'GE11-P-10L1-L', 'GE11-P-31L1-S', 'GE11-M-01L1-S', 'GE11-M-02L1-L', 'GE11-M-09L1-S', 'GE11-M-11L1-S', 'GE11-M-15L1-S', 'GE11-P-06L2-L', 'GE11-P-31L2-S', 'GE11-M-01L2-S', 'GE11-M-02L2-L', 'GE11-M-09L2-S', 'GE11-M-11L2-S', 'GE11-M-15L2-S', 'GE11-M-23L2-S']
    chamberNames_HV_675 = ['GE11-P-05L1-S', 'GE11-P-08L1-L', 'GE11-M-10L1-L', 'GE11-M-14L1-L', 'GE11-M-17L1-S', 'GE11-M-22L1-L', 'GE11-M-36L1-L', 'GE11-P-05L2-S', 'GE11-P-08L2-L', 'GE11-P-35L2-S', 'GE11-P-36L2-L', 'GE11-M-14L2-L', 'GE11-M-22L2-L', 'GE11-M-36L2-L']
    #removeChambers = ["GE11-M-25L2-S", "GE11-M-26L2-L"]
    removeChambers = []
    EfficiencyDistribution = ROOT.TH1F("EfficiencyDistribution","EfficiencyDistribution",102,0.,102.)
    EfficiencyDistribution_ML1 = ROOT.TH1F("EfficiencyDistribution_ML1","EfficiencyDistribution_ML1",102,0.,102.)
    EfficiencyDistribution_ML2 = ROOT.TH1F("EfficiencyDistribution_ML2","EfficiencyDistribution_ML2",102,0.,102.)
    EfficiencyDistribution_PL1 = ROOT.TH1F("EfficiencyDistribution_PL1","EfficiencyDistribution_PL1",102,0.,102.)
    EfficiencyDistribution_PL2 = ROOT.TH1F("EfficiencyDistribution_PL2","EfficiencyDistribution_PL2",102,0.,102.)
    chamberNames_withShort =  shortChambers()
    list_of_chambers=[]
    if includeCh != "":
        list_of_chambers = list_of_chambers + CSVtoChamberID(includeCh)
    removeShort = False
    #print("%%%%%%%%% Chambers with shorts %%%%%%%%%%%%%%%%%%\n",chamberNames_withShort)
    for re,la in [(-1,1),(1,1),(-1,2),(1,2)]: 
        for chamber in range(1,37):
            chamberID = getChamberName(re,chamber,la)
            if(chamberID in ["GE11-P-33L2-S"]+removeChambers):
                continue
            #print(chamberID)
            #if chamberID  in chamberNames_HV_670:
            #    continue
            #if chamberID  in chamberNames_HV_675:
            #    continue

            #if(chamberID in ["GE11-P-18L2-L","GE11-M-14L2-L"]):
            #    continue
            #if chamberID not in list_of_chambers and list_of_chambers != []:
            #    continue
            
            n,d,_ = ChamberEfficiencyFromCSV(filepath,chamberID)
            if float(d) !=0.:
                Eff = 100*float(n)/d 
                if removeShort == True and Eff < 85 and chamberID in chamberNames_withShort:
                    print("chamberId: ",chamberID)
                else:
                    if Eff > 10:
                        EfficiencyDistribution.Fill(Eff)
                        with open("efficiency.txt","a") as eff_f:
                            eff_f.write(f"{chamberID}: {Eff}\n")
                            #print(chamberID,end=" ")
                if (re == -1 and la == 1):
                    EfficiencyDistribution_ML1.Fill(Eff)
                elif (re == -1 and la == 2):
                    EfficiencyDistribution_ML2.Fill(Eff)
                elif (re == 1 and la == 1):
                    EfficiencyDistribution_PL1.Fill(Eff)
                elif (re == 1 and la == 2):
                    EfficiencyDistribution_PL2.Fill(Eff)

    return EfficiencyDistribution, EfficiencyDistribution_ML1, EfficiencyDistribution_ML2, EfficiencyDistribution_PL1, EfficiencyDistribution_PL2
    
    
            

def generateEfficiencyPlot2DGE11(sourceDict,input_region=1,input_layer=1,debug=False):
    ## Transforming in list
    reg_tag_string = "All" if isinstance(input_region, list) else "P" if input_region == 1 else "M"
    lay_tag_string = "" if isinstance(input_layer, list) else "L1" if input_layer == 1 else "L2"
    input_region = [input_region] if not isinstance(input_region, list) else input_region
    input_layer = [input_layer] if not isinstance(input_layer, list) else input_layer

    title = "GE11"+reg_tag_string+lay_tag_string
    phi_nbins,phi_min,phi_max = 36,0.5,36.5
    etaP_nbins, etaP_min,etaP_max = 10, -0.5, 9.5    

    EfficiencyTH2D = ROOT.TH2F(title+"_ChambersEfficiency", title+"_ChambersEfficiency", phi_nbins,phi_min,phi_max,etaP_nbins,etaP_min,etaP_max)
    NumTH2D = ROOT.TH2F(title+"_ChambersNum", title+"_ChambersNum", phi_nbins,phi_min,phi_max,etaP_nbins,etaP_min,etaP_max)
    DenTH2D = ROOT.TH2F(title+"_ChambersDen", title+"_ChambersDen", phi_nbins,phi_min,phi_max,etaP_nbins,etaP_min,etaP_max)


    Summary = ROOT.TGraphAsymmErrors()
    Summary.GetXaxis().SetTitle("Chamber Number")
    Summary.GetXaxis().SetTitleSize(0.05)
    Summary.GetYaxis().SetTitle("Efficiency")
    Summary.GetYaxis().SetTitleSize(0.05)
    Summary.SetMaximum(1.1)
    Summary.SetMinimum(0.)
    Summary.SetTitle(title+"_Efficiency")
    Summary.SetName(title+"_Efficiency")
    Summary.SetLineColor(ROOT.kBlack)
    Summary.SetMarkerColor(ROOT.kBlack)
    Summary.SetFillColorAlpha(ROOT.kBlack,.4)
    Summary.SetMarkerStyle(20)
    Summary.SetMarkerSize(.8)

    SummaryNum = ROOT.TH1F("n","n",phi_nbins,phi_min,phi_max)
    SummaryDen = ROOT.TH1F("d","d",phi_nbins,phi_min,phi_max)
    
    N = 1                                                                        
    for etaPartitionID,value in sourceDict.items():
        station,region,chamber,layer,eta = convert_etaPID(etaPartitionID)
        
        if layer not in input_layer or region not in input_region:
            continue

        etaPartitionRecHits = sum([value[k]['num'] for k in value.keys()])
        etaPartitionPropHits = sum([value[k]['den'] for k in value.keys()])
        
        try:
            eta_efficiency = round(float(etaPartitionRecHits)/float(etaPartitionPropHits),2)
        except:
            if debug:print("Warning on Re,Ch,La,etaP = ", region,chamber,layer,eta, "\tDenominator is 0")
            eta_efficiency = 0
            N+=1

        binx = chamber
        biny = eta  + 1
        
        
        existingNumSummary = SummaryNum.GetBinContent(binx)
        existingDenSummary = SummaryDen.GetBinContent(binx)
        existingNumTH2D = NumTH2D.GetBinContent(binx,biny)
        existingDenTH2D = DenTH2D.GetBinContent(binx,biny)
        

        SummaryNum.SetBinContent(binx,existingNumSummary+etaPartitionRecHits)
        SummaryDen.SetBinContent(binx,existingDenSummary+etaPartitionPropHits)

        NumTH2D.SetBinContent(binx,biny,existingNumTH2D+etaPartitionRecHits)
        DenTH2D.SetBinContent(binx,biny,existingDenTH2D+etaPartitionPropHits)
        #EfficiencyTH2D.SetBinContent(binx,biny,eta_efficiency)

    EfficiencyTH2D = NumTH2D.Clone()
    EfficiencyTH2D.SetTitle(title+"_ChambersEfficiency")
    EfficiencyTH2D.SetName(title+"_ChambersEfficiency")
    EfficiencyTH2D.Divide(DenTH2D)
    for x in range(1,phi_nbins+1):
        for y in range(1,etaP_nbins+1):
            EfficiencyTH2D.SetBinContent(x,y,  round(EfficiencyTH2D.GetBinContent(x,y),2) )


    EfficiencyTH2D.SetStats(False)
    EfficiencyTH2D.GetYaxis().SetTickLength(0.005)
    EfficiencyTH2D.GetXaxis().SetTitle("Chamber number")
    EfficiencyTH2D.GetXaxis().SetTitleSize(0.05)
    EfficiencyTH2D.GetYaxis().SetTitle("GEM EtaPartition")
    EfficiencyTH2D.GetYaxis().SetTitleSize(0.05)

    NumTH2D.SetStats(False)
    NumTH2D.GetYaxis().SetTickLength(0.005)
    NumTH2D.GetXaxis().SetTitle("Chamber number")
    NumTH2D.GetXaxis().SetTitleSize(0.05)
    NumTH2D.GetYaxis().SetTitle("GEM EtaPartition")
    NumTH2D.GetYaxis().SetTitleSize(0.05)

    DenTH2D.SetStats(False)
    DenTH2D.GetYaxis().SetTickLength(0.005)
    DenTH2D.GetXaxis().SetTitle("Chamber number")
    DenTH2D.GetXaxis().SetTitleSize(0.05)
    DenTH2D.GetYaxis().SetTitle("GEM EtaPartition")
    DenTH2D.GetYaxis().SetTitleSize(0.05)

    Summary.Divide(SummaryNum,SummaryDen,"B")
    Summary.GetXaxis().SetRangeUser(0.,36.5)
    Summary.GetYaxis().SetTickLength(0.005)
    Summary.GetXaxis().SetTickLength(0.005)
    Summary.SetMarkerColor(ROOT.kBlue)
    Summary.SetLineColor(ROOT.kBlue)
    return EfficiencyTH2D,NumTH2D,DenTH2D,Summary

def pt_index(num):
    if num <10:
        index = 0
    elif num <20:
        index = 1
    elif num <30:
        index = 2
    elif num <40:
        index = 3
    elif num <50:
        index = 4
    elif num <60:
        index = 5
    elif num <70:
        index = 6
    elif num <80:
        index = 7
    elif num <90:
        index = 8
    elif num <100:
        index = 9
    else:
        index = 10
    
    return index

def pt_index_fine(num):
    if num < 0:
        return None  # Handle negative numbers if needed
    # Ensure the index stays between 0 and 100, including for 500.
    return min(num // 5, 100)  # Divide by 5 and cap the index at 100


## returns a dict containing the strip pitch for a given ch,etaP
def GetStripGeometry():
    stripGeometryDict = {}
    df = pd.read_csv("/afs/cern.ch/user/f/fivone/Documents/myLIB/GE11Geometry/GE11StripSpecs.csv")
        
    for ch in range(1,37):
        stripGeometryDict[ch] = {}
        for etaP in range(1,9):
            if ch%2 == 0:
                chID = "Long"
            if ch%2 == 1:
                chID = "Short"
            df_temp = df.loc[df['Chamber'] == chID]
            df_temp = df_temp.loc[df_temp['EtaP'] == etaP]
            firstStripPosition = df_temp["Loc_x"].min()
            lastStripPosition = df_temp["Loc_x"].max()
            stripPitch = (lastStripPosition - firstStripPosition)/float(len(df_temp["Loc_x"]))
            stripGeometryDict[ch].setdefault(etaP,{'stripPitch':stripPitch,'firstStrip':firstStripPosition,"lastStrip":lastStripPosition})

    return stripGeometryDict


def printSummary(sourceDict,matching_variables,ResidualCutOff,matching_variable_units,debug=False):
    for matching_variable in matching_variables:
        print("\n\n#############\nSUMMARY\n#############\nMatchingVariable = "+matching_variable+"\nCutoff = ",ResidualCutOff[matching_variable],matching_variable_units[matching_variable],"\n")
        for eta in range(1,9):
            num = sum([sourceDict[matching_variable][key][k]['num'] for key in sourceDict[matching_variable].keys() for k in range(0,11) if abs(key)%10 == eta])
            den = sum([sourceDict[matching_variable][key][k]['den'] for key in sourceDict[matching_variable].keys() for k in range(0,11) if abs(key)%10 == eta])
            try:
                print("Efficiency GE11 ETA"+str(eta)+"  ==> ", num , "/",den, "\t = ", round(float(num)/float(den),3))
            except:
                print("EtaP = ",str(eta)," has no propagated hits...")

        num = sum([sourceDict[matching_variable][key][k]['num'] for key in sourceDict[matching_variable].keys() for k in range(0,11)])
        den = sum([sourceDict[matching_variable][key][k]['den'] for key in sourceDict[matching_variable].keys() for k in range(0,11)])
        try:
            print("Efficiency GE11       ==> ", num , "/",den, "\t = ", round(float(num)/float(den),3))
        except:
            if debug: print("WARNING")
        print("#############")

def initializeMatchingTree():
    generalMatching = {}
    recHit_Matching = {}
    propHit_Matching = {}
    
    generalMatching['event'] = array('i', [0])
    generalMatching['lumiblock'] = array('i', [0])
    generalMatching['chamber'] = array('i', [0])
    generalMatching['region'] = array('i', [0])
    generalMatching['layer'] = array('i', [0])
    generalMatching['etaPartition'] = array('i', [0])

    recHit_Matching['loc_x'] = array('f', [0.])
    recHit_Matching['glb_x'] = array('f', [0.])
    recHit_Matching['glb_y'] = array('f', [0.])
    recHit_Matching['glb_z'] = array('f', [0.])
    recHit_Matching['glb_r'] = array('f', [0.])
    recHit_Matching['glb_phi'] = array('f', [0.])
    recHit_Matching['firstStrip'] = array('i', [0])
    recHit_Matching['cluster_size'] = array('i', [0])

    propHit_Matching['loc_x'] = array('f', [0.])
    propHit_Matching['loc_y'] = array('f', [0.])
    propHit_Matching['glb_x'] = array('f', [0.])
    propHit_Matching['glb_y'] = array('f', [0.])
    propHit_Matching['glb_z'] = array('f', [0.])
    propHit_Matching['glb_r'] = array('f', [0.])
    propHit_Matching['glb_phi'] = array('f', [0.])
    propHit_Matching['err_glb_r'] = array('f', [0.])
    propHit_Matching['err_glb_phi'] = array('f', [0.])
    propHit_Matching['pt'] = array('f', [0.])
    propHit_Matching['mu_propagated_isGEM'] = array('i', [0])
    propHit_Matching['STA_Normchi2'] = array('f', [0.])
    propHit_Matching['nME1Hits'] = array('i', [0])
    propHit_Matching['nME2Hits'] = array('i', [0])
    propHit_Matching['nME3Hits'] = array('i', [0])
    propHit_Matching['nME4Hits'] = array('i', [0])

    t_out = ROOT.TTree("MatchingTree", "MatchingTree")
    for key in generalMatching.keys():
        if type(generalMatching[key][0]) == float: label="F"
        elif type(generalMatching[key][0]) == int: label="I"
        t_out.Branch(key, generalMatching[key], key+"/"+label)
    for key in recHit_Matching.keys():
        if type(recHit_Matching[key][0]) == float: label="F"
        elif type(recHit_Matching[key][0]) == int: label="I"
        t_out.Branch("recHit_"+key, recHit_Matching[key], "recHit_"+key+"/"+label)
    for key in propHit_Matching.keys():
        if type(propHit_Matching[key][0]) == float: label="F"
        elif type(propHit_Matching[key][0]) == int: label="I"
        t_out.Branch("propHit_"+key, propHit_Matching[key], "propHit_"+key+"/"+label)
    
    return t_out,generalMatching,recHit_Matching,propHit_Matching

def fillMatchingTreeArray(PropHitonEta,prop_hit_index,RecHitonEta,reco_hit_index,propHit_Matching,recHit_Matching):
    for key in recHit_Matching.keys():
        recHit_Matching[key][0] =RecHitonEta[key][reco_hit_index]
    for key in propHit_Matching.keys():
        propHit_Matching[key][0] =PropHitonEta[key][prop_hit_index]
    
    return recHit_Matching,propHit_Matching

def store4evtDspl(name,run,lumi,evt):
    evtDspl_dir = OUTPUT_PATH + "/PFA_Analyzer_Output/EvtDspl/"
    with open(evtDspl_dir+name+".txt", 'a+') as f:
        f.write(str(run)+":"+str(lumi)+":"+str(evt)+"\n")

def printEfficiencyFromCSV(path):
    file_path = path + "/MatchingSummary_glb_rdphi_byVFAT.csv"
    print(file_path,"\n")
    print('{:<8}{:<20}{:<20}{:<20}{:<20}'.format("Region","AnalyzedChambers","Matched","Propagated","Eff. 68%CL"))


    all_prop=0
    all_mat=0
    all_chamb=0
    all_pt=0
    for endcapTag in ["PL1","PL2","ML1","ML2"]:
        df = pd.read_csv(file_path, sep=',')
        mask = df['EndcapTag'] == endcapTag
        df = df[mask]
    
        Matched = df["matchedRecHit"].sum()
        all_mat+=Matched

        Propagated = df["propHit"].sum()
        all_prop+=Propagated

        if "AVG_pt" in df.columns: 
            AVGpt  =  round(df["AVG_pt"].mean(),4)
        else:
            AVGpt = 0
        all_pt+=AVGpt
        analyzed_Chambers = float(len(df))/24
        all_chamb+=analyzed_Chambers
        print('{:<8}{:<20}{:<20}{:<20}{:<20}'.format(endcapTag,int(analyzed_Chambers),int(Matched),int(Propagated),str(generateClopperPeasrsonInterval(Matched,Propagated))))

    print('{:<8}{:<20}{:<20}{:<20}{:<20}'.format("all",all_chamb,int(all_mat),int(all_prop),str(generateClopperPeasrsonInterval(all_mat,all_prop))))
    print("##############\n")


def ChamberEfficiencyFromCSV(path,chamber):
    file_path = path + "/MatchingSummary_glb_rdphi_byVFAT.csv"
    prop=0
    mat=0

    df = pd.read_csv(file_path, sep=',')
    mask = df['chamberID'] == chamber
    df = df[mask]
    if len(df)==0: return 0,0,0

    else:
        
        mat = df["matchedRecHit"].sum()
        prop = df["propHit"].sum()
        if prop == 0:
            return 0,0,0
        return mat,prop,generateClopperPeasrsonInterval(mat,prop)

def CreatEOSFolder(abs_path):
    subprocess.call(["mkdir", "-p", abs_path])
    #subprocess.call(["cp", "/eos/user/f/fivone/www/index.php",abs_path+"/index.php"])
    run_number = GetRunNumber(abs_path)
    if run_number != "000000":
        pass
        
        # command = "source /afs/cern.ch/user/f/fivone/Test/FetchOMS/venv/bin/activate; python3 /afs/cern.ch/user/f/fivone/Test/FetchOMS/RecordedLumi.py  --RunList "+run_number
        # p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True,stderr=subprocess.PIPE)
        # (output, err) = p.communicate()  
        # #Wait until finished...
        # p_status = p.wait()
        # labels = output.splitlines()[0].split(";")
        # values = output.splitlines()[1].split(";")

        # with open (abs_path+"/runInfo.txt",'w+') as f:
        #     for index in range(len(labels)):
        #         line_to_write = '{:<20}  {:<20}'.format(labels[index].replace(" ",""), values[index].replace(" ",""))+"\n"
        #         f.write(line_to_write)

def Convert2png(file_path):
    subprocess.call(["convert","-density", "300", "-trim", file_path, "-quality", "100","-sharpen","0x1,0", file_path.replace("pdf","png")])
def GetRunNumber(input_string):
    run_number = "unknown"
    ## the run number has 6 digis, followed by _
    values = input_string.split("_")
    for v in values:
        onlydigis = regularExpression.sub("[^0-9]", "", v)
        if len(onlydigis) == 6 and int(onlydigis) > 346000:
            run_number=onlydigis
    
    return run_number

def VFATEfficiencyFromCSV(df):
    for index, row in df.iterrows():
        matched = float(row['matchedRecHit'])
        prop = float(row['propHit'])
        Name = row['chamberID']
        VFATN = row['VFATN']
        if prop == 0:
            continue
        else:
            if matched/prop < 0.3:
                print(Name+"\t"+str(VFATN)+"\t"+str(matched/prop))

def BranchesDict(branches):
    b_dic={}
    for b in branches.split(","):
        b_dic[b]=b
    return b_dic


def CSVtoChamberID(file):
    list_of_chambers=[]
    df = pd.read_excel(file)
    for index, row in df.iterrows():
        region = row[1]
        chamber = row[2]
        layer = row[3]
        chamber = getChamberName(region,chamber,layer)
        list_of_chambers.append(chamber)
    return list_of_chambers


def polar2cartesian(r,phi):
    x=r*math.cos(phi)
    y=r*math.sin(phi)
    return (x,y)
def is_point_in_trapezoid(A, B, C, D, P):
    def cross_product(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]
    
    def vector(p1, p2):
        return (p2[0] - p1[0], p2[1] - p1[1])
    
    AB = vector(A, B)
    AP = vector(A, P)
    BC = vector(B, C)
    BP = vector(B, P)
    CD = vector(C, D)
    CP = vector(C, P)
    DA = vector(D, A)
    DP = vector(D, P)
    
    cross_AB_AP = cross_product(AB, AP)
    cross_BC_BP = cross_product(BC, BP)
    cross_CD_CP = cross_product(CD, CP)
    cross_DA_DP = cross_product(DA, DP)
    
    # Check if all cross products have the same sign
    if (cross_AB_AP > 0 and cross_BC_BP > 0 and cross_CD_CP > 0 and cross_DA_DP > 0) or \
       (cross_AB_AP < 0 and cross_BC_BP < 0 and cross_CD_CP < 0 and cross_DA_DP < 0):
        return True
    else:
        return False



if __name__ == '__main__':
    # df = pd.read_csv("/afs/cern.ch/user/f/fivone/Test/Analyzer/Output/PFA_Analyzer_Output/CSV/357333_Express/MatchingSummary_glb_phi_byVFAT.csv")
    # VFATEfficiencyFromCSV(df)
    pass


