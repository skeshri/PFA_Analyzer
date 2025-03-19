import ROOT
import subprocess
import sys
import os
import numpy as np
import time
import argparse
import pandas as pd
from ROOT_Utils import *
from PFA_Analyzer_Utils import *
import logging 
import cProfile,pstats
import re

profiler = cProfile.Profile()

runmaskEvents=True

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(format='[{asctime}] {levelname} - {message}', datefmt='%B %d - %H:%M:%S',level=logging.INFO,style="{")
logging.addLevelName( logging.DEBUG, "\033[1;90m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))
logging.addLevelName( logging.INFO, "\033[1;92m%s\033[1;0m" % logging.getLevelName(logging.INFO))
logging.addLevelName( logging.WARNING, "\033[1;93m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName( logging.ERROR, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.ERROR))


parser = argparse.ArgumentParser(description='PFA Analyzer parser')
parser.add_argument('config', help='Analysis description file')
args = parser.parse_args()

start_time = time.time()

# chamberForEventDisplay = ["GE11-P-28L2-L"]
data_ranges,parameters = load_config(os.path.abspath(args.config))
TH1MetaData = GenerateMetadata(parameters,data_ranges)

#print(data_ranges.keys())

### Sumit njobs
#njobs = parameters['njobs']
#print(f'total number of jobs ==>>> {njobs}')


ROOT.gStyle.SetPaintTextFormat("2.2f")
ROOT.gStyle.SetLineScalePS(1)
ROOT.gROOT.SetBatch(True)
if not parameters["verbose"]: ROOT.gROOT.ProcessLine("gErrorIgnoreLevel=2001;") #suppressed everything less-than-or-equal-to kWarning

files = []

d_run_tag={357802:"357802_ZMu", 357803:"357803_ZMu", 357807: "357807_ZMu", 357813:"357813_ZMu", 357814:"357814_ZMu", 357815:"357815_ZMu"}
d_path_tag={357802:"230216_095512", 357803:"230216_095527", 357807:"230216_095549", 357813:"230216_095607", 357814:"230216_095631", 357815:"230216_095646"}
file_tag="MuDPGNtuple"
runs=[357802, 357803, 357807, 357813, 357814, 357815]

#for run in runs:
#    run_tag  = d_run_tag[run]
#    path_tag = d_path_tag[run]
#    files = files + files_in_folder(run_tag,path_tag=path_tag,filename_tag=file_tag)

#print(data_ranges)

for run in data_ranges:
    run_tag  = data_ranges[run]["tag"]
    path_tags = data_ranges[run]["path_tag"]
    file_tag  = data_ranges[run]["file_tag"]

    if path_tags != "":
        for path_tag in path_tags:    # Sumit
            print("++++++++++++++++ Path Tag :",path_tag)
            files = files + files_in_folder(run_tag,path_tag=path_tag,filename_tag=file_tag)
    else:
        print("+++++++++++++ run_tag",run_tag)
        files = files + files_in_folder(run_tag,path_tag=path_tags,filename_tag=file_tag)
#    files = files +["/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/2022/GEMCommonNtuples/Muon/Fill_8149/MuDPGNtuple_8149.root"] 
#    files = files +["/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/2022/GEMCommonNtuples/Muon/357802_ZMu/230216_095512/0000/MuDPGNtuple_1.root"] 

# sumit added for testing MC
#files = files + ["/afs/cern.ch/work/s/skeshri/GEM_efficiency/Ntuple/lxplus9/CMSSW_14_0_6_patch1/src/MuDPGAnalysis/MuonDPGNtuples/test/MuDPGNtuple.root"]
#files=["/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/cms-gem-automation/prod/prompt-v1/GEMCommonNTuples/371078/output-0.root"]
#print("============= files: \n",files)


#if 'MuDPGNtuple' in file_tag:
#files = ["/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/cms-gem-automation/prod/prompt-v1/GEMCommonNTuples/380310/output-9.root","/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/cms-gem-automation/prod/prompt-v1/GEMCommonNTuples/380310/output-10.root","/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/cms-gem-automation/prod/prompt-v1/GEMCommonNTuples/380310/output-11.root"]
#files = ["/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/cms-gem-automation/prod/prompt-v1/GEMCommonNTuples/380310/output-9.root"]



def strip_versions(trig_names):
    # Remove the version suffix "_v<number>" from each string
    return re.sub(r'_v\d+$', '', str(trig_names))

matching_variables = ['glb_phi','glb_rdphi']
matching_variable_units = {'glb_phi':'rad','glb_rdphi':'cm'}
ResidualCutOff = {'glb_phi':parameters["phi_cut"],'glb_rdphi':parameters["rdphi_cut"]}

TH1nbins = 200
TH2nbins = 200
TH2min = -130

EfficiencyDictGlobal = dict((m,{}) for m in matching_variables)
EfficiencyDictVFAT = generateVFATDict(matching_variables)
EfficiencyDictDirection = dict((m,{}) for m in matching_variables)
EfficiencyDictLayer = dict((m,{}) for m in matching_variables)
#EfficiencyDict_nVtx = {}#dict((m,{}) for m in matching_variables)


## ROOT Object declaration

# Sumit added efficiency wrt chi2
chi2_den = ROOT.TH1F("chi2_den","chi2_den",10,0,10)
chi2_num = ROOT.TH1F("chi2_num","chi2_num",10,0,10)

chi2_inv_den = ROOT.TH1F("chi2_inv_den","chi2_inv_den",10,0,1)
chi2_inv_num = ROOT.TH1F("chi2_inv_num","chi2_inv_num",10,0,1)

nVtx_den = ROOT.TH1F("nVtx_den","nVtx_den",10,0,100)
nVtx_num = ROOT.TH1F("nVtx_num","nVtx_num",10,0,100)
nVtx_ML1_den = ROOT.TH1F("nVtx_ML1_den","nVtx_ML1_den",10,0,100)
nVtx_ML1_num = ROOT.TH1F("nVtx_ML1_num","nVtx_ML1_num",10,0,100)
nVtx_ML2_den = ROOT.TH1F("nVtx_ML2_den","nVtx_ML2_den",10,0,100)
nVtx_ML2_num = ROOT.TH1F("nVtx_ML2_num","nVtx_ML2_num",10,0,100)
nVtx_PL1_den = ROOT.TH1F("nVtx_PL1_den","nVtx_PL1_den",10,0,100)
nVtx_PL1_num = ROOT.TH1F("nVtx_PL1_num","nVtx_PL1_num",10,0,100)
nVtx_PL2_den = ROOT.TH1F("nVtx_PL2_den","nVtx_PL2_den",10,0,100)
nVtx_PL2_num = ROOT.TH1F("nVtx_PL2_num","nVtx_PL2_num",10,0,100)


nVtx_ML1_2D_num = ROOT.TH2F("nVtx_ML1_2D_num","nVtx_ML1_2D_num",10,0,100,36,0.5,36.5);
nVtx_ML1_2D_den = ROOT.TH2F("nVtx_ML1_2D_den","nVtx_ML1_2D_den",10,0,100,36,0.5,36.5);
nVtx_ML2_2D_num = ROOT.TH2F("nVtx_ML2_2D_num","nVtx_ML2_2D_num",10,0,100,36,0.5,36.5);
nVtx_ML2_2D_den = ROOT.TH2F("nVtx_ML2_2D_den","nVtx_ML2_2D_den",10,0,100,36,0.5,36.5);
nVtx_PL1_2D_num = ROOT.TH2F("nVtx_PL1_2D_num","nVtx_PL1_2D_num",10,0,100,36,0.5,36.5);
nVtx_PL1_2D_den = ROOT.TH2F("nVtx_PL1_2D_den","nVtx_PL1_2D_den",10,0,100,36,0.5,36.5);
nVtx_PL2_2D_num = ROOT.TH2F("nVtx_PL2_2D_num","nVtx_PL2_2D_num",10,0,100,36,0.5,36.5);
nVtx_PL2_2D_den = ROOT.TH2F("nVtx_PL2_2D_den","nVtx_PL2_2D_den",10,0,100,36,0.5,36.5);


STAdirX_vs_CLS = ROOT.TH2F("DirX_CLS","DirX_CLS",40,-1,1,128,0,128)
STAdirX_vs_CLS.GetXaxis().SetTitle("dirX")
STAdirX_vs_CLS.GetYaxis().SetTitle("CLS")
nDigis_vs_1l1a = ROOT.TH2F("clusters_vs_1stl1a","",1024,-0.5,1023.5,12000,-0.5,11999.5)
nDigis_vs_1l1a.GetXaxis().SetTitle("1stL1A")
nDigis_vs_1l1a.GetYaxis().SetTitle("nDigis")

nDigis_vs_1l1a_mask_clus = ROOT.TH2F("clusters_vs_1stl1a_mask_clus","",1024,-0.5,1023.5,12000,-0.5,11999.5)
nDigis_vs_1l1a_mask_clus.GetXaxis().SetTitle("1stL1A")
nDigis_vs_1l1a_mask_clus.GetYaxis().SetTitle("nDigis")

nDigis_vs_1l1a_mask_l1A = ROOT.TH2F("clusters_vs_1stl1a_mask_l1A","",1024,-0.5,1023.5,12000,-0.5,11999.5)
nDigis_vs_1l1a_mask_l1A.GetXaxis().SetTitle("1stL1A")
nDigis_vs_1l1a_mask_l1A.GetYaxis().SetTitle("nDigis")

nDigis_vs_1l1a_mask_both = ROOT.TH2F("clusters_vs_1stl1a_mask_both","",1024,-0.5,1023.5,12000,-0.5,11999.5)
nDigis_vs_1l1a_mask_both.GetXaxis().SetTitle("1stL1A")
nDigis_vs_1l1a_mask_both.GetYaxis().SetTitle("nDigis")


nDigis_vs_2l1a = ROOT.TH2F("clusters_vs_2ndl1a","",1024,-0.5,1023.5,12000,-0.5,11999.5)
nDigis_vs_2l1a.GetXaxis().SetTitle("2ndL1A")
nDigis_vs_2l1a.GetYaxis().SetTitle("nDigis")

nDigis_vs_3l1a = ROOT.TH2F("clusters_vs_3rdl1a","",1024,-0.5,1023.5,12000,-0.5,11999.5)
nDigis_vs_3l1a.GetXaxis().SetTitle("3rdL1A")
nDigis_vs_3l1a.GetYaxis().SetTitle("nDigis")

nDigis_vs_4l1a = ROOT.TH2F("clusters_vs_4thl1a","",1024,-0.5,1023.5,12000,-0.5,11999.5)
nDigis_vs_4l1a.GetXaxis().SetTitle("4thLastL1A")
nDigis_vs_4l1a.GetYaxis().SetTitle("nDigis")

DLE_ErrPhi = ROOT.TH1F("DLE_ErrPhi","DLE_ErrPhi",100,0,0.0025)
DLE_ErrR = ROOT.TH1F("DLE_ErrR","DLE_ErrR",100,0,5)
DLE_pt = ROOT.TH1F("pT of STA muons used to probe DLE","pT of STA muons used to probe DLE",200,0,100)

TH1Fresidual_collector_CLS_x = generate1DxResidualContainer_CLS(matching_variables,TH1nbins,ResidualCutOff)
TH1Fresidual_collector_x = generate1DxResidualContainer(matching_variables,TH1nbins,ResidualCutOff)
TH1Fresidual_collector_x_minus = generate1DxResidualContainer(matching_variables,TH1nbins,ResidualCutOff)
TH1Fresidual_collector_x_plus = generate1DxResidualContainer(matching_variables,TH1nbins,ResidualCutOff)
TH1Fresidual_collector_x_ptgt30 = generate1DxResidualContainer(matching_variables,TH1nbins,ResidualCutOff)
TH1Fresidual_collector_x_minus_ptgt30 = generate1DxResidualContainer(matching_variables,TH1nbins,ResidualCutOff)
TH1Fresidual_collector_x_plus_ptgt30 = generate1DxResidualContainer(matching_variables,TH1nbins,ResidualCutOff)
TH1Fresidual_collector_x_ptlt30 = generate1DxResidualContainer(matching_variables,TH1nbins,ResidualCutOff)
TH1Fresidual_collector_x_minus_ptlt30 = generate1DxResidualContainer(matching_variables,TH1nbins,ResidualCutOff)
TH1Fresidual_collector_x_plus_ptlt30 = generate1DxResidualContainer(matching_variables,TH1nbins,ResidualCutOff)
TH1Fresidual_collector_y = generate1DyResidualContainer(matching_variables,TH1nbins,ResidualCutOff)
TH1FpropError_collector = generatePropagationErrorContainer(parameters["maxErrPropR"], parameters["maxErrPropPhi"])
TH2Fresidual_collector = generate2DResidualContainer(matching_variables,TH2nbins,TH2min)  
TH2FexcludedProphits_collector = generate2DMap_ExcludedHits(TH2nbins,TH2min)


TH2Fresidual_collector_x = generate2DxResidualContainer(matching_variables,TH1nbins,ResidualCutOff)
TH2Fresidual_collector_x_minus = generate2DxResidualContainer(matching_variables,TH1nbins,ResidualCutOff)
TH2Fresidual_collector_x_plus = generate2DxResidualContainer(matching_variables,TH1nbins,ResidualCutOff)

VFATMaskBook = {}
THSanityChecks = {'Occupancy':{}, 
                  'NHits':{},
                  'PropagationError':{},
                  'etaP_vs_pt':[],
                  'Residual_Correlation':{},
                  'PropHit_DirLoc_xOnGE11':{'BeforeMatching':{'Long':{},'Short':{}},
                                            'AfterMatching':{'Long':{},'Short':{}}
                                            },
                  'RecHitperStrip':{},
                  'NEvts':ROOT.TH1F("NumberOfAnalyzedEVTs","NumberOfAnalyzedEVTs",100,1,1),
                  'CLS':{'beforematching':{},'aftermatching':{}},
                  'pt':{'glb_phi':{},'glb_rdphi':{},'All':ROOT.TH1F("pT of STA muons used to probe efficiency","pT of STA muons used to probe efficiency",200,0,100), 'All_plus':ROOT.TH1F("pT plus","pT plus",200,0,100), 'All_minus':ROOT.TH1F("pT minus","pT minus",200,0,100)},
                  'nMuons':{'glb_phi':{},'glb_rdphi':{},'All':ROOT.TH1F("nMuons","nMuons",30,0,30)}
                }   

THAll_Residuals = {}
for cls in range(1,6):
    THAll_Residuals[cls] = ROOT.TH1F(f"Residual CLS {cls}",f"Residual CLS {cls}",TH1nbins,-ResidualCutOff["glb_rdphi"],ResidualCutOff["glb_rdphi"])


THSanityChecks['NHits']['BeforeMatching'] = {'PerEVT':{'Reco':ROOT.TH1F("NRecoHitsPerEVT","NRecoHitsPerEVT",200,0,200),'Prop':ROOT.TH1F("NPropHitsPerEVT","NPropHitsPerEVT",20,0,20)},
                                             'ML1':ROOT.TH1F("ML1_NRecoHitsPerEVT","ML1_NRecoHitsPerEVT",200,0,200),
                                             'ML2':ROOT.TH1F("ML2_NRecoHitsPerEVT","ML1_NRecoHitsPerEVT",200,0,200),
                                             'PL1':ROOT.TH1F("PL1_NRecoHitsPerEVT","PL1_NRecoHitsPerEVT",200,0,200),
                                             'PL2':ROOT.TH1F("PL2_NRecoHitsPerEVT","PL2_NRecoHitsPerEVT",200,0,200)
                                         }

THSanityChecks['NHits']['AfterMatching'] = {'All':ROOT.TH1F("N_MatchedRecoHitsPerEVT","N_MatchedRecoHitsPerEVT",20,0,20),
                                             'ML1':ROOT.TH1F("ML1_N_MatchedRecoHitsPerEVT","ML1_N_MatchedRecoHitsPerEVT",20,0,20),
                                             'ML2':ROOT.TH1F("ML2_N_MatchedRecoHitsPerEVT","ML2_N_MatchedRecoHitsPerEVT",20,0,20),
                                             'PL1':ROOT.TH1F("PL1_N_MatchedRecoHitsPerEVT","PL1_N_MatchedRecoHitsPerEVT",20,0,20),
                                             'PL2':ROOT.TH1F("PL2_N_MatchedRecoHitsPerEVT","PL2_N_MatchedRecoHitsPerEVT",20,0,20)
                                         }

THSanityChecks['NHits']['PerEVT_PerEtaPartitionID'] = {'Reco':ROOT.TH1F("NRecoHitsPerEVTPetEtaPartitionID","NRecoHitsPerEVTPetEtaPartitionID",10,0,10),
                                                       'Prop':ROOT.TH1F("NPropHitsPerEVTPetEtaPartitionID","NPropHitsPerEVTPetEtaPartitionID",10,0,10)}

THSanityChecks['Occupancy'].setdefault('BeforeMatching',{'Reco':ROOT.TH2F("RecoHitOccupancyBeforeMatching","RecoHitOccupancyBeforeMatching",200,-300,300,200,-300,300),
                                                         'Prop':ROOT.TH2F("PropHitOccupancyBeforeMatching","PropHitOccupancyBeforeMatching",200,-300,300,200,-300,300),
                                                         'PropLocalLong':ROOT.TH2F("PropLocLongHitBeforeMatching","PropLocLongHitBeforeMatching",TH2nbins,TH2min,-TH2min,TH2nbins,TH2min,-TH2min),
                                                         'PropLocalShort':ROOT.TH2F("PropLocShortHitBeforeMatching","PropLocShortHitBeforeMatching",TH2nbins,TH2min,-TH2min,TH2nbins,TH2min,-TH2min),
                                                         'ML1':{},
                                                         'ML2':{},
                                                         'PL1':{},
                                                         'PL2':{}})
for t_re in [-1,1]:
    for t_la in [1,2]:
        for t_ch in range(1,37):
            ch_id = getChamberName(t_re,t_ch,t_la)
            THSanityChecks['CLS']['beforematching'].setdefault(ch_id,{})
            THSanityChecks['CLS']['aftermatching'].setdefault(ch_id,{})
            THSanityChecks['pt'].setdefault(ch_id,{})
            #THSanityChecks['nMuons'].setdefault(ch_id,{})
            for t_eta in list(range(1,9))+["All"]:
                THSanityChecks['CLS']['beforematching'][ch_id].setdefault(t_eta,ROOT.TH1F(ch_id+"_etaP"+str(t_eta)+" ClusterSize beforeMatch",ch_id+"_etaP"+str(t_eta)+" ClusterSize beforeMatch",20,0,20))
                THSanityChecks['CLS']['aftermatching'][ch_id].setdefault(t_eta,ROOT.TH1F(ch_id+"_etaP"+str(t_eta)+" ClusterSize aftMatch",ch_id+"_etaP"+str(t_eta)+" ClusterSize aftMatch",20,0,20))
                THSanityChecks['pt'][ch_id].setdefault(t_eta,ROOT.TH1F(ch_id+"_etaP"+str(t_eta)+" pT",ch_id+"_etaP"+str(t_eta)+" pT",200,0,100))

for key in ['ML1','ML2','PL1','PL2']:
    THSanityChecks['Occupancy']['BeforeMatching'][key]['RecHits'] = ROOT.TH2F(key+"_RecHitsOccupancy",key+"_RecHitsOccupancy",38,-0.5,37.5,10,-0.5,9.5)
    THSanityChecks['Occupancy']['BeforeMatching'][key]['PropHits'] = ROOT.TH2F(key+"_PropHitsOccupancy",key+"_PropHitsOccupancy",38,-0.5,37.5,10,-0.5,9.5)
    THSanityChecks['Occupancy']['BeforeMatching'][key]['RecHits'].GetXaxis().SetTitle("Chamber Number")
    THSanityChecks['Occupancy']['BeforeMatching'][key]['RecHits'].GetYaxis().SetTitle("EtaPartition")
    THSanityChecks['Occupancy']['BeforeMatching'][key]['PropHits'].GetXaxis().SetTitle("Chamber Number")
    THSanityChecks['Occupancy']['BeforeMatching'][key]['PropHits'].GetYaxis().SetTitle("EtaPartition")

    THSanityChecks['RecHitperStrip'][key] = {}
    for ch in range(1,37):
        size = "S" if ch%2 == 1 else "L"
        chID = 'GE11-'+key[0]+'-%02d' % ch + key[1:]+"-"+size 
        THSanityChecks['RecHitperStrip'][key][ch] = ROOT.TH2F(chID,chID,384,-0.5,383.5,10,-0.5,9.5)
        THSanityChecks['RecHitperStrip'][key][ch].SetStats(0)
        THSanityChecks['RecHitperStrip'][key][ch].SetMaximum(600)
        THSanityChecks['RecHitperStrip'][key][ch].GetXaxis().SetTitle("StripNumber")
        THSanityChecks['RecHitperStrip'][key][ch].GetYaxis().SetTitle("EtaPartition")


THSanityChecks['PropagationError']['glb_phi_error'] = {'all':ROOT.TH1F("All_errProp_glb_phi","All_errProp_glb_phi",100,0,0.0025),
                                                        'long':ROOT.TH1F("Long_errProp_glb_phi","Long_errProp_glb_phi",100,0,0.0025),
                                                        'long_isGEM':ROOT.TH1F("Long_errProp_glb_phi && isGEM==1","Long_errProp_glb_phi && isGEM==1",100,0,0.0025),
                                                        'long_noGEM':ROOT.TH1F("Long_errProp_glb_phi && isGEM==0","Long_errProp_glb_phi && isGEM==0",100,0,0.0025),
                                                        'short':ROOT.TH1F("Short_errProp_glb_phi","Short_errProp_glb_phi",100,0,0.0025),
                                                        'short_isGEM':ROOT.TH1F("Short_errProp_glb_phi && isGEM==1","Short_errProp_glb_phi && isGEM==1",100,0,0.0025),
                                                        'short_noGEM':ROOT.TH1F("Short_errProp_glb_phi && isGEM==0","Short_errProp_glb_phi && isGEM==0",100,0,0.0025),
                                                        'eta1':ROOT.TH1F("eta1_errProp_glb_phi","eta1_errProp_glb_phi",100,0,0.0025),
                                                        'eta2':ROOT.TH1F("eta2_errProp_glb_phi","eta2_errProp_glb_phi",100,0,0.0025),
                                                        'eta3':ROOT.TH1F("eta3_errProp_glb_phi","eta3_errProp_glb_phi",100,0,0.0025),
                                                        'eta4':ROOT.TH1F("eta4_errProp_glb_phi","eta4_errProp_glb_phi",100,0,0.0025),
                                                        'eta5':ROOT.TH1F("eta5_errProp_glb_phi","eta5_errProp_glb_phi",100,0,0.0025),
                                                        'eta6':ROOT.TH1F("eta6_errProp_glb_phi","eta6_errProp_glb_phi",100,0,0.0025),
                                                        'eta7':ROOT.TH1F("eta7_errProp_glb_phi","eta7_errProp_glb_phi",100,0,0.0025),
                                                        'eta8':ROOT.TH1F("eta8_errProp_glb_phi","eta8_errProp_glb_phi",100,0,0.0025)}
THSanityChecks['PropagationError']['glb_r_error'] = {'all':ROOT.TH1F("All_errProp_glb_r","All_errProp_glb_r",100,0,5),
                                                        'long':ROOT.TH1F("Long_errProp_glb_r","Long_errProp_glb_r",100,0,5),
                                                        'long_isGEM':ROOT.TH1F("Long_errProp_glb_r && isGEM==1","Long_errProp_glb_r && isGEM==1",100,0,5),
                                                        'long_noGEM':ROOT.TH1F("Long_errProp_glb_r && isGEM==0","Long_errProp_glb_r && isGEM==0",100,0,5),
                                                        'short':ROOT.TH1F("Short_errProp_glb_r","Short_errProp_glb_r",100,0,5),
                                                        'short_isGEM':ROOT.TH1F("Short_errProp_glb_r && isGEM==1","Short_errProp_glb_r && isGEM==1",100,0,5),
                                                        'short_noGEM':ROOT.TH1F("Short_errProp_glb_r && isGEM==0","Short_errProp_glb_r && isGEM==0",100,0,5),
                                                        'eta1':ROOT.TH1F("eta1_errProp_glb_r","eta1_errProp_glb_r",100,0,5),
                                                        'eta2':ROOT.TH1F("eta2_errProp_glb_r","eta2_errProp_glb_r",100,0,5),
                                                        'eta3':ROOT.TH1F("eta3_errProp_glb_r","eta3_errProp_glb_r",100,0,5),
                                                        'eta4':ROOT.TH1F("eta4_errProp_glb_r","eta4_errProp_glb_r",100,0,5),
                                                        'eta5':ROOT.TH1F("eta5_errProp_glb_r","eta5_errProp_glb_r",100,0,5),
                                                        'eta6':ROOT.TH1F("eta6_errProp_glb_r","eta6_errProp_glb_r",100,0,5),
                                                        'eta7':ROOT.TH1F("eta7_errProp_glb_r","eta7_errProp_glb_r",100,0,5),
                                                        'eta8':ROOT.TH1F("eta8_errProp_glb_r","eta8_errProp_glb_r",100,0,5)}
THSanityChecks['etaP_vs_pt'] = ROOT.TH2F("allChmbrs_etaP_pt","allChmbrs_etaP_pt",8,0,8,11,0,110)
THSanityChecks['etaP_vs_pt'].GetXaxis().SetTitle("i#eta")
THSanityChecks['etaP_vs_pt'].GetYaxis().SetTitle("pt (GeV)")
THSanityChecks['etaP_vs_pt'].SetStats(0)

THSanityChecks['Residual_Correlation']['glb_phi_vs_glb_rdphi'] = ROOT.TH2F("Residual_Correlation #Delta#phi vs R#Delta#phi","Residual_Correlation #Delta#phi vs R#Delta#phi",100,-3*ResidualCutOff['glb_phi'],3*ResidualCutOff['glb_phi'],100,-3*ResidualCutOff['glb_rdphi'],3*ResidualCutOff['glb_rdphi'])
THSanityChecks['Residual_Correlation']['glb_rdphi_dir_x'] = ROOT.TH2F("Residual_Correlation R#Delta#phi vs Dir_x","Residual_Correlation R#Delta#phi vs Dir_x",100,-3*ResidualCutOff['glb_rdphi'],3*ResidualCutOff['glb_rdphi'],100,0,3.1415)
THSanityChecks['Residual_Correlation']['glb_phi_vs_glb_rdphi'].GetXaxis().SetTitle("#Delta#phi (rad)")
THSanityChecks['Residual_Correlation']['glb_phi_vs_glb_rdphi'].GetYaxis().SetTitle("R#Delta#phi (cm)")
THSanityChecks['Residual_Correlation']['glb_rdphi_dir_x'].SetStats(0)
THSanityChecks['Residual_Correlation']['glb_rdphi_dir_x'].GetXaxis().SetTitle("R#Delta#phi (cm)")
THSanityChecks['Residual_Correlation']['glb_rdphi_dir_x'].GetYaxis().SetTitle("Dir_x (as Cos(#alpha) )")

THSanityChecks['charge_beforeSelection'] = ROOT.TH1F("charge_beforeSelection","charge_beforeSelection",4,-2,2)
THSanityChecks['charge_beforeMatching'] = ROOT.TH1F("charge_beforeMatching","charge_beforeMatching",4,-2,2)
THSanityChecks['charge_afterMatching'] = ROOT.TH1F("charge_afterMatching","charge_afterMatching",4,-2,2)
THSanityChecks['STA_Normchi2'] = ROOT.TH1F("STA_NormChi2","STA_NormChi2",200,0,20)
THSanityChecks['nME1Hits'] = ROOT.TH1F("nME1Hits in STA","nME1Hits in STA",20,0,20)
THSanityChecks['nME2Hits'] = ROOT.TH1F("nME2Hits in STA","nME2Hits in STA",20,0,20)
THSanityChecks['nME3Hits'] = ROOT.TH1F("nME3Hits in STA","nME3Hits in STA",20,0,20)
THSanityChecks['nME4Hits'] = ROOT.TH1F("nME4Hits in STA","nME4Hits in STA",20,0,20)
THSanityChecks['nCSCHits'] = ROOT.TH1F("nCSCHits in STA","nCSCHits in STA",40,0,40)

for key_1 in matching_variables:
    THSanityChecks['Occupancy'].setdefault(key_1,{'AfterMatching':{'Reco':ROOT.TH2F("RecoHitAfterMatching_"+key_1,"RecoHitAfterMatching_"+key_1,200,-300,300,200,-300,300),
                                                                   'Prop':ROOT.TH2F("PropHitAfterMatching_"+key_1,"PropHitAfterMatching_"+key_1,200,-300,300,200,-300,300),
                                                                   'PropLocalLong':ROOT.TH2F("PropLocLongHitAfterMatching_"+key_1,"PropLocLongHitAfterMatching_"+key_1,TH2nbins,TH2min,-TH2min,TH2nbins,TH2min,-TH2min),
                                                                   'PropLocalShort':ROOT.TH2F("PropLocShortHitAfterMatching_"+key_1,"PropLocShortHitAfterMatching_"+key_1,TH2nbins,TH2min,-TH2min,TH2nbins,TH2min,-TH2min)}})

for key_1 in ['Long','Short']:
    for key_2 in ["eta"+str(k) for k in range(1,9)]:
        THSanityChecks['PropHit_DirLoc_xOnGE11']['BeforeMatching'][key_1][key_2] = ROOT.TH1F('BeforeMatchPropHit_DirLoc_xOnGE11_'+key_1+key_2,'BeforeMatchPropHit_DirLoc_xOnGE11_'+key_1+key_2,200,-1,1)
        THSanityChecks['PropHit_DirLoc_xOnGE11']['AfterMatching'][key_1][key_2] = ROOT.TH1F('AfterMatchPropHit_DirLoc_xOnGE11_'+key_1+key_2,'AfterMatchPropHit_DirLoc_xOnGE11_'+key_1+key_2,200,-1,1)

print(files)
## Chain files
chain = ROOT.TChain("muNtupleProducer/MuDPGTree")
logging.info(f"TChaining {len(files)} files...")
for fl in files:
    chain.Add(fl)

# Disabling them all branches
chain.SetBranchStatus("*",0)
branchList=["event_trigName","event_trigDecision","event_eventNumber","event_nVtx","event_lumiBlock","event_runNumber","gemOHStatus_station","gemOHStatus_region","gemOHStatus_chamber","gemOHStatus_layer","gemOHStatus_VFATMasked", "gemOHStatus_VFATZS", "gemOHStatus_VFATMissing","gemOHStatus_errors","gemOHStatus_warnings", "gemRecHit_region", "gemRecHit_chamber", "gemRecHit_layer", "gemRecHit_etaPartition", "gemRecHit_g_r", "gemRecHit_loc_x", "gemRecHit_loc_y", "gemRecHit_g_x", "gemRecHit_g_y", "gemRecHit_g_z", "gemRecHit_g_phi", "gemRecHit_firstClusterStrip", "gemRecHit_cluster_size", "mu_propagated_region", "mu_propagated_chamber", "mu_propagated_layer", "mu_propagated_etaP", "mu_propagated_strip","mu_propagated_Outermost_z",  "mu_propagated_isME11", "mu_propagatedGlb_r", "mu_propagatedLoc_x", "mu_propagatedLoc_y", "mu_propagatedGlb_x", "mu_propagatedGlb_y", "mu_propagatedGlb_z", "mu_propagatedGlb_phi", "mu_propagatedGlb_errR", "mu_propagatedGlb_errPhi", "mu_propagatedLoc_dirX", "mu_propagatedLoc_dirY", "mu_propagatedLoc_dirZ", "mu_propagated_pt", "mu_propagated_isGEM", "mu_propagated_TrackNormChi2", "mu_propagated_nME1hits", "mu_propagated_nME2hits", "mu_propagated_nME3hits", "mu_propagated_nME4hits","mu_propagated_station","gemRecHit_station","mu_propagated_isME21","event_1stLast_L1A","event_2ndLast_L1A","event_3rdLast_L1A","event_4thLast_L1A", "mu_propagated_EtaPartition_rMin", "mu_propagated_EtaPartition_rMax","mu_propagated_EtaPartition_phiMin","mu_propagated_EtaPartition_phiMax", "mu_propagated_EtaPartition_xMax","mu_propagated_EtaPartition_xMin","mu_propagated_EtaPartition_yMax","mu_propagated_EtaPartition_yMin","mu_propagated_charge"]#,"mu_propagated_isTight","mu_propagated_isLoose","mu_propagated_isMedium","mu_propagated_isGlobal","mu_propagated_isStandalone","mu_propagated_isTracker"]
# Enabling the useful ones
for b in branchList: chain.SetBranchStatus(b,1)

maxLS = {}
processedEvents = {}
chainEntries = chain.GetEntries()
logging.info("############# Starting #############")
logging.info(f"Analysing run(s): \t { [i for i in data_ranges] }")
logging.info(f"Number of evts \t\t {float(chainEntries)/10**6:.2f} M\n")

THSanityChecks['NEvts'].Fill(chainEntries)

print("total Entries(chainEntries): %d"%chainEntries)
totalEvents=0
totalEventsF=0
filteredEvents=0
filterFlower=0
filterCluster=0
totalFilterClus=0
l1aEvent=0
foundMaskedEvent=False
totalNum=0
totalDen=0
rechit_pid=[]
prophit_pid=[]

for chain_index,evt in enumerate(chain):
    #if chain_index>=5000: break
    fill=True
    EventNumber = evt.event_eventNumber
    LumiSection = evt.event_lumiBlock
    RunNumber = evt.event_runNumber
    totalEvents=totalEvents+1

    VFATMaskBook.setdefault( RunNumber,generateVFATMaskTH2(station=1))
    
     
    if chain_index % 8000 ==0: logging.info(f"{round( float(chain_index)/chainEntries*100,1 )}%\t{chain_index}")

    #if (evt.event_nVtx<50 or evt.event_nVtx>=100):
    #    continue
    n_gemprop = len(evt.mu_propagated_chamber)
    n_gemrec = len(evt.gemRecHit_chamber)
    #print("n_gemprop: ",n_gemprop)
    #print("n_gemrec: ",n_gemrec)
    trigFire = False
    for itrig, trigger in enumerate(evt.event_trigName):
        #print(strip_versions(trigger), evt.event_trigDecision[itrig])
        if(strip_versions(trigger) == "HLT_IsoMu24" and evt.event_trigDecision[itrig] == 1):
            trigFire=True
    if (not trigFire):
        continue
    
    ## Skip on LS
    if not 'MC' in  data_ranges.keys():
        if data_ranges[RunNumber]["lumisection"] != (0,0) and (LumiSection < data_ranges[RunNumber]["lumisection"][0] or LumiSection > data_ranges[RunNumber]["lumisection"][1]):
            logging.debug(f"Skipping, LS is {LumiSection} while range is {data_ranges[RunNumber]['lumisection']}")
            continue
    # If not FullDigis, skip evts with 0 propagations
    if parameters["FD"] == False and n_gemprop==0:
        continue
    #print("Sumit: in propHitFromME11.....")

    maxLS[RunNumber] = LumiSection if maxLS.get(RunNumber) is None else max(maxLS.get(RunNumber),LumiSection)

    THSanityChecks['NHits']['BeforeMatching']['PerEVT']['Prop'].Fill(n_gemprop)
    THSanityChecks['NHits']['BeforeMatching']['PerEVT']['Reco'].Fill(n_gemrec)

    ML1_NGEMRecoHits = 0
    ML2_NGEMRecoHits = 0
    PL1_NGEMRecoHits = 0
    PL2_NGEMRecoHits = 0    
    
    RecHit_Dict = {}
    PropHit_Dict = {}

    processedEvents[RunNumber] = 1 if processedEvents.get(RunNumber) is None else processedEvents.get(RunNumber) + 1
    if not 'MC' in data_ranges.keys():
        maskedVFATs,VFATMaskBook[RunNumber] = unpackVFATStatus_masked(evt,VFATMaskBook[RunNumber]) 
        if processedEvents[RunNumber] > data_ranges[RunNumber]["nevts"] and data_ranges[RunNumber]["nevts"]> 0:
            logging.debug(f"Exiting after reaching max number of events: {data_ranges[RunNumber]['nevts']}")
            break
    
    clusterSize=0
    clusterSize_perChamber={}
    for RecHit_index in range(0,n_gemrec):
        station = evt.gemRecHit_station[RecHit_index]
        region = evt.gemRecHit_region[RecHit_index]
        chamber = evt.gemRecHit_chamber[RecHit_index]
        layer = evt.gemRecHit_layer[RecHit_index]
        etaP = evt.gemRecHit_etaPartition[RecHit_index]
        RecHitEtaPartitionID =  getEtaPID(station,region,chamber,layer,etaP)
        endcapKey = EndcapLayer2label(region,layer)
        chamberID = getChamberName(region,chamber,layer,station)
        #print("chamberID: ",chamberID)
        rechit_pid.append(RecHitEtaPartitionID)

        if not parameters["doGE21"] and station == 2: 
            logging.debug(f"Skipping current rechit cause station = {station}")
            continue

        firstStrip = evt.gemRecHit_firstClusterStrip[RecHit_index]
        CLS = evt.gemRecHit_cluster_size[RecHit_index]
        clusterSize+=CLS
        rec_glb_r = evt.gemRecHit_g_r[RecHit_index]
        rec_loc_x = evt.gemRecHit_loc_x[RecHit_index]
        rec_loc_y = evt.gemRecHit_loc_y[RecHit_index]
        if chamberID in clusterSize_perChamber.keys():
            clusterSize_perChamber[chamberID]+=CLS
        else:
            clusterSize_perChamber[chamberID]=0
        
        ## discard chambers that were kept OFF from the analysis
        if not 'MC' in data_ranges.keys():
            if chamberID in data_ranges[RunNumber]["chamberOFF"].keys() and (LumiSection in data_ranges[RunNumber]["chamberOFF"][chamberID] or -1 in data_ranges[RunNumber]["chamberOFF"][chamberID] ):
                continue

        if region == 1:
            if layer == 1:
                PL1_NGEMRecoHits += 1 
            else:
                PL2_NGEMRecoHits += 1 
        if region == -1:
            if layer == 1:
                ML1_NGEMRecoHits += 1 
            else:
                ML2_NGEMRecoHits += 1 

        RecHit_Dict.setdefault(RecHitEtaPartitionID, {'loc_x':[],'loc_y':[],'glb_x':[],'glb_y':[],'glb_z':[],'glb_r':[],'glb_phi':[],'firstStrip':[],'cluster_size':[]})
        RecHit_Dict[RecHitEtaPartitionID]['loc_x'].append(rec_loc_x)
        RecHit_Dict[RecHitEtaPartitionID]['loc_y'].append(rec_loc_y)
        RecHit_Dict[RecHitEtaPartitionID]['glb_x'].append(evt.gemRecHit_g_x[RecHit_index])
        RecHit_Dict[RecHitEtaPartitionID]['glb_y'].append(evt.gemRecHit_g_y[RecHit_index])
        RecHit_Dict[RecHitEtaPartitionID]['glb_z'].append(evt.gemRecHit_g_z[RecHit_index])
        RecHit_Dict[RecHitEtaPartitionID]['glb_r'].append(rec_glb_r)
        RecHit_Dict[RecHitEtaPartitionID]['glb_phi'].append(evt.gemRecHit_g_phi[RecHit_index])
        RecHit_Dict[RecHitEtaPartitionID]['firstStrip'].append(firstStrip)
        RecHit_Dict[RecHitEtaPartitionID]['cluster_size'].append(CLS)

        THSanityChecks['Occupancy']['BeforeMatching']['Reco'].Fill(evt.gemRecHit_g_x[RecHit_index],evt.gemRecHit_g_y[RecHit_index])
        THSanityChecks['Occupancy']['BeforeMatching'][endcapKey]['RecHits'].Fill(chamber,etaP)
        ## Fill CLS
        THSanityChecks['CLS']["beforematching"][chamberID]["All"].Fill(RecHit_Dict[RecHitEtaPartitionID]['cluster_size'][-1])
        THSanityChecks['CLS']["beforematching"][chamberID][etaP].Fill(RecHit_Dict[RecHitEtaPartitionID]['cluster_size'][-1])
        ## Fill Digis
        for j in range(0,RecHit_Dict[RecHitEtaPartitionID]['cluster_size'][-1]):
            strip = RecHit_Dict[RecHitEtaPartitionID]['firstStrip'][-1] + j
            THSanityChecks['RecHitperStrip'][endcapKey][chamber].Fill(strip,etaP)

    THSanityChecks['NHits']['BeforeMatching']['ML1'].Fill(ML1_NGEMRecoHits)
    THSanityChecks['NHits']['BeforeMatching']['ML2'].Fill(ML2_NGEMRecoHits)
    THSanityChecks['NHits']['BeforeMatching']['PL1'].Fill(PL1_NGEMRecoHits)
    THSanityChecks['NHits']['BeforeMatching']['PL2'].Fill(PL2_NGEMRecoHits)    

    
#### Recipe-1
    nDigis_vs_1l1a.Fill(evt.event_1stLast_L1A,clusterSize)
    
    bigCluster=False
    flowerEvent=False
    
    #if (runmaskEvents): 
    #   if((n_gemrec>650) or (n_gemrec-50)*(2000/600) > clusterSize):
           #print("sumit")
    #       filterFlower+=1
    #       flowerEvent=True
           #continue
    #   for key in clusterSize_perChamber.keys():
    #       if(clusterSize_perChamber[key]>384):
               #print(clusterSize_perChamber[key])
    #           bigCluster=True
       #if (bigCluster):
       #    filterCluster+=1

       #    continue
    #   if(bigCluster or flowerEvent):
    #       totalFilterClus+=1
    #       continue
#           pass
        #   filteredEvents+=1
#       #    print("sumit ",clusterSize_perChamber[key])
#       else:
#           pass
#           #continue
#       totalEventsF+=1
#       #if (totalEvents>30000):
#       #    break
       #if(not (bigCluster and flowerEvent)):
#       
#    
##### Recipe-2:
    nDigis_vs_2l1a.Fill(evt.event_2ndLast_L1A,clusterSize)
    nDigis_vs_3l1a.Fill(evt.event_3rdLast_L1A,clusterSize)
    nDigis_vs_4l1a.Fill(evt.event_4thLast_L1A,clusterSize)
    keep=True
    
    if(evt.event_1stLast_L1A > 150 and evt.event_1stLast_L1A < 200):
        keep=False
        #continue
        #print("L1A ",evt.event_1stLast_L1A)
    elif(evt.event_2ndLast_L1A > 150 and evt.event_2ndLast_L1A < 200):
        keep=False
        #continue
        #print("L1A ",evt.event_2ndLast_L1A)
    elif(evt.event_3rdLast_L1A > 150 and evt.event_3rdLast_L1A < 200):
        keep=False
        #continue
        #print("L1A ",evt.event_3rdLast_L1A)
    elif(evt.event_4thLast_L1A > 150 and evt.event_4thLast_L1A < 200):
        keep=False
        #continue
        #print("L1A ",evt.event_4thLast_L1A)
    else: 
        pass
    
    if(not keep ):
        l1aEvent+=1
        #print("in Keep condition")
        continue
    #print("outside keep condition")
    #if(keep):
    nDigis_vs_1l1a_mask_l1A.Fill(evt.event_1stLast_L1A,clusterSize)
        #continue

    #if(not keep or bigCluster or flowerEvent ):

    #    fill=False
    filteredEvents+=1
#        continue
    #if (fill):
    #  nDigis_vs_1l1a_mask_both.Fill(evt.event_1stLast_L1A,clusterSize)
    
    nDigis_vs_1l1a_mask_clus.Fill(evt.event_1stLast_L1A,clusterSize)
    npropHit_vtx=0
    for PropHit_index in range(0,n_gemprop):
        
        station = evt.mu_propagated_station[PropHit_index]
        region = evt.mu_propagated_region[PropHit_index]
        chamber = evt.mu_propagated_chamber[PropHit_index]
        layer = evt.mu_propagated_layer[PropHit_index]
        etaP = evt.mu_propagated_etaP[PropHit_index]
        chamberID = getChamberName(region,chamber,layer)
        PropHitChamberID = getEtaPID(station,region,chamber,layer,etaP)
        #print("PropHitChamberID",PropHitChamberID)
        endcapKey = EndcapLayer2label(region,layer)
        outermost_z = evt.mu_propagated_Outermost_z[PropHit_index]
        prophit_pid.append(RecHitEtaPartitionID)
        strip = evt.mu_propagated_strip[PropHit_index]
        vfat = iEtaStrip_2_VFAT(etaP,strip,station)

        if not parameters["doGE21"] and station == 2: 
            logging.debug(f"Skipping current prophit cause station = {station}")
            continue       

        if (region == 1 and outermost_z < 0) or (region == -1 and outermost_z > 0):
            logging.debug(f"Skipping prophit due to inconsistent region vs outermost z")
            continue
        #print("Keshri.....................")
        ## discard chambers that were kept OFF from the analysis
        if not 'MC' in data_ranges.keys():
            if chamberID in data_ranges[RunNumber]["chamberOFF"].keys() and (LumiSection in data_ranges[RunNumber]["chamberOFF"][chamberID] or -1 in data_ranges[RunNumber]["chamberOFF"][chamberID] ):
                logging.debug(f"Skipping prophit due to bad chamber HV")
                continue

        propHitFromME11 = bool(evt.mu_propagated_isME11[PropHit_index])
        if propHitFromME11:
            PropHit_Dict.setdefault(PropHitChamberID,{'region':[],'station':[],'chamber':[],'loc_x':[],'loc_y':[],'glb_x':[],'glb_y':[],'glb_z':[],'glb_r':[],'glb_phi':[],'pt':[],'etaP':[],'err_glb_r':[],'err_glb_phi':[],'Loc_dirX':[],'Loc_dirY':[],'Loc_dirZ':[],'mu_propagated_isME11':[],'mu_propagated_isGEM':[],'STA_Normchi2':[],'nME1Hits':[],'nME2Hits':[],'nME3Hits':[],'nME4Hits':[],'rmin':[],'rmax':[],'phimin':[],'phimax':[],'xmin':[],'xmax':[],'ymin':[],'ymax':[], 'charge':[]})#,'tight_id':[],'medium_id':[],'loose_id':[],"is_global":[],"is_sta":[],"is_tracker":[]})
            prop_glb_r = evt.mu_propagatedGlb_r[PropHit_index]
            prop_loc_x = evt.mu_propagatedLoc_x[PropHit_index]
            propVFAT = propHit2VFAT(prop_glb_r,prop_loc_x,etaP)
            if not 'MC' in data_ranges.keys():
                if PropHitChamberID in data_ranges[RunNumber]["VFATOFF"]:
                    if propVFAT in data_ranges[RunNumber]["VFATOFF"][PropHitChamberID]:
                        #pass
                        continue
            
                badVFATs_masked =  unpackVFATStatus_masked_new(evt,station,region,layer,chamber,vfat)            
                badVFATs_missing =  unpackVFATStatus_missing_new(evt,station,region,layer,chamber,vfat)            
                badVFATs_error =  unpackVFATStatus_error_new(evt,station,region,layer,chamber,vfat)            
                DAQenabled =  unpackVFATStatus_DAQenabled(evt,station,region,layer,chamber,vfat)            
                if (badVFATs_masked or badVFATs_missing or badVFATs_error):
                    continue
                if(DAQenabled):

                    continue
            badVFATs =  maskedVFATs.get(chamberID)            
            #if badVFATs is not None:
            #    if propVFAT in badVFATs:
            #        logging.debug(f"On {chamberID} propVFAT {propVFAT} found in MaskedVFATs {badVFATs}. Skipping")
            #        continue
            #    elif VFATcompatibleHit(badVFATs,prop_glb_r,prop_loc_x,etaP):
            #        logging.debug(f"On {chamberID} proVFAT {propVFAT} compatible with MaskedVFATs {badVFATs}. Skipping")
            #        continue

            PropHit_Dict[PropHitChamberID]['region'].append(region)
            PropHit_Dict[PropHitChamberID]['station'].append(station)
            PropHit_Dict[PropHitChamberID]['chamber'].append(chamber)
            PropHit_Dict[PropHitChamberID]['loc_x'].append(prop_loc_x)
            PropHit_Dict[PropHitChamberID]['loc_y'].append(evt.mu_propagatedLoc_y[PropHit_index])
            PropHit_Dict[PropHitChamberID]['glb_x'].append(evt.mu_propagatedGlb_x[PropHit_index])
            PropHit_Dict[PropHitChamberID]['glb_y'].append(evt.mu_propagatedGlb_y[PropHit_index])
            PropHit_Dict[PropHitChamberID]['glb_z'].append(evt.mu_propagatedGlb_z[PropHit_index])
            PropHit_Dict[PropHitChamberID]['glb_r'].append(prop_glb_r)
            PropHit_Dict[PropHitChamberID]['glb_phi'].append(evt.mu_propagatedGlb_phi[PropHit_index])
            PropHit_Dict[PropHitChamberID]['err_glb_r'].append(evt.mu_propagatedGlb_errR[PropHit_index])
            PropHit_Dict[PropHitChamberID]['err_glb_phi'].append(evt.mu_propagatedGlb_errPhi[PropHit_index])
            PropHit_Dict[PropHitChamberID]['Loc_dirX'].append(evt.mu_propagatedLoc_dirX[PropHit_index])
            PropHit_Dict[PropHitChamberID]['Loc_dirY'].append(evt.mu_propagatedLoc_dirY[PropHit_index])
            PropHit_Dict[PropHitChamberID]['Loc_dirZ'].append(evt.mu_propagatedLoc_dirZ[PropHit_index])
            PropHit_Dict[PropHitChamberID]['pt'].append(evt.mu_propagated_pt[PropHit_index])
            PropHit_Dict[PropHitChamberID]['charge'].append(evt.mu_propagated_charge[PropHit_index])
            #PropHit_Dict[PropHitChamberID]['tight_id'].append(evt.mu_propagated_isTight[PropHit_index])
            #PropHit_Dict[PropHitChamberID]['medium_id'].append(evt.mu_propagated_isMedium[PropHit_index])
            #PropHit_Dict[PropHitChamberID]['loose_id'].append(evt.mu_propagated_isLoose[PropHit_index])
            #PropHit_Dict[PropHitChamberID]['is_global'].append(evt.mu_propagated_isGlobal[PropHit_index])
            #PropHit_Dict[PropHitChamberID]['is_sta'].append(evt.mu_propagated_isStandalone[PropHit_index])
            #PropHit_Dict[PropHitChamberID]['is_tracker'].append(evt.mu_propagated_isTracker[PropHit_index])
            #PropHit_Dict[PropHitChamberID]['nMuons'].append(len(evt.mu_propagated_pt))
            PropHit_Dict[PropHitChamberID]['etaP'].append(etaP)
            PropHit_Dict[PropHitChamberID]['mu_propagated_isME11'].append(evt.mu_propagated_isME11[PropHit_index])
            PropHit_Dict[PropHitChamberID]['mu_propagated_isGEM'].append(evt.mu_propagated_isGEM[PropHit_index])
            PropHit_Dict[PropHitChamberID]['STA_Normchi2'].append(evt.mu_propagated_TrackNormChi2[PropHit_index])            
            PropHit_Dict[PropHitChamberID]['nME1Hits'].append(evt.mu_propagated_nME1hits[PropHit_index])
            PropHit_Dict[PropHitChamberID]['nME2Hits'].append(evt.mu_propagated_nME2hits[PropHit_index])
            PropHit_Dict[PropHitChamberID]['nME3Hits'].append(evt.mu_propagated_nME3hits[PropHit_index])
            PropHit_Dict[PropHitChamberID]['nME4Hits'].append(evt.mu_propagated_nME4hits[PropHit_index])
            PropHit_Dict[PropHitChamberID]['rmin'].append(evt.mu_propagated_EtaPartition_rMin[PropHit_index])
            PropHit_Dict[PropHitChamberID]['rmax'].append(evt.mu_propagated_EtaPartition_rMax[PropHit_index])
            PropHit_Dict[PropHitChamberID]['phimin'].append(evt.mu_propagated_EtaPartition_phiMin[PropHit_index])
            PropHit_Dict[PropHitChamberID]['phimax'].append(evt.mu_propagated_EtaPartition_phiMax[PropHit_index])
            
            PropHit_Dict[PropHitChamberID]['xmin'].append(evt.mu_propagated_EtaPartition_xMin[PropHit_index])
            PropHit_Dict[PropHitChamberID]['xmax'].append(evt.mu_propagated_EtaPartition_xMax[PropHit_index])
            PropHit_Dict[PropHitChamberID]['ymin'].append(evt.mu_propagated_EtaPartition_yMin[PropHit_index])
            PropHit_Dict[PropHitChamberID]['ymax'].append(evt.mu_propagated_EtaPartition_yMax[PropHit_index])

            THSanityChecks['Occupancy']['BeforeMatching']['Prop'].Fill(evt.mu_propagatedGlb_x[PropHit_index],evt.mu_propagatedGlb_y[PropHit_index])
            THSanityChecks['Occupancy']['BeforeMatching'][endcapKey]['PropHits'].Fill(chamber,etaP)
            THSanityChecks['etaP_vs_pt'].Fill(PropHit_Dict[PropHitChamberID]['etaP'][-1]-1,10*pt_index_fine(evt.mu_propagated_pt[PropHit_index]))
            THSanityChecks['STA_Normchi2'].Fill(evt.mu_propagated_TrackNormChi2[PropHit_index])
            THSanityChecks['pt']['All'].Fill(evt.mu_propagated_pt[PropHit_index])
            if(evt.mu_propagated_charge[PropHit_index]<0): THSanityChecks['pt']['All_minus'].Fill(evt.mu_propagated_pt[PropHit_index])
            if(evt.mu_propagated_charge[PropHit_index]>0): THSanityChecks['pt']['All_plus'].Fill(evt.mu_propagated_pt[PropHit_index])

            if chamber % 2 == 0:
                THSanityChecks['Occupancy']['BeforeMatching']['PropLocalLong'].Fill(evt.mu_propagatedLoc_x[PropHit_index],evt.mu_propagatedLoc_y[PropHit_index])
                THSanityChecks['PropHit_DirLoc_xOnGE11']['BeforeMatching']['Long']['eta'+str(etaP)].Fill(evt.mu_propagatedLoc_dirX[PropHit_index])
            if chamber % 2 == 1:
                THSanityChecks['Occupancy']['BeforeMatching']['PropLocalShort'].Fill(evt.mu_propagatedLoc_x[PropHit_index],evt.mu_propagatedLoc_y[PropHit_index])
                THSanityChecks['PropHit_DirLoc_xOnGE11']['BeforeMatching']['Short']['eta'+str(etaP)].Fill(evt.mu_propagatedLoc_dirX[PropHit_index])

    ML1_N_MatchedGEMRecoHits = 0
    ML2_N_MatchedGEMRecoHits = 0
    PL1_N_MatchedGEMRecoHits = 0
    PL2_N_MatchedGEMRecoHits = 0

    ML1_N_PropGEMRecoHits = 0
    ML2_N_PropGEMRecoHits = 0
    PL1_N_PropGEMRecoHits = 0
    PL2_N_PropGEMRecoHits = 0
    nVtx_den_dic={}
    nVtx_num_dic={}

    ### Matching criteria between propagated hits from ME11 and RecHits : 
    ##      1.SAME REGION,SC,LAYER,ETA -->SAME etaPartitionID
    ##      When using DLE, only evts w/ exactly 2 PropHit in a SC: 1 hit per Layer with Delta(etaP) < 4

    # if parameters["DLE"] and (len(PropHit_Dict.keys()) != 2 or abs(PropHit_Dict.keys()[0] - PropHit_Dict.keys()[1] ) > 13) :
    #     continue

    layer1Match = False
    layer2Match = False
    layer1PassedCut = False
    layer2PassedCut = False
    layer1pt = 0
    layer1etapID = 0
    layer2pt = 0
    layer2etapID = 0
    nmatched_vtx=0
    #print("etaPartitionID",PropHit_Dict.keys())

    for etaPartitionID in PropHit_Dict.keys():
        
        station,region,chamber,layer,eta = convert_etaPID(etaPartitionID)
        endcapTag = EndcapLayer2label(region,layer)
        current_chamber_ID = getChamberName(region,chamber,layer)
        #key = region*(100*chamber+10*layer)
        #print("chamberID",current_chamber_ID,region,chamber,layer,key)
        
        PropHitonEta = PropHit_Dict[etaPartitionID]

        nPropHitsOnEtaID = len(PropHitonEta['glb_phi'])
        THSanityChecks['NHits']['PerEVT_PerEtaPartitionID']['Prop'].Fill(nPropHitsOnEtaID)
        cos_of_alpha_list = [np.sqrt(PropHitonEta['Loc_dirX'][i]**2 + PropHitonEta['Loc_dirY'][i]**2) for i in range(nPropHitsOnEtaID)]        

        ## Defining Efficiency dict global: [matchingVar][etaPartitionID][pt]
        for mv in matching_variables:
            EfficiencyDictGlobal[mv].setdefault(etaPartitionID,{})
            EfficiencyDictDirection[mv].setdefault(etaPartitionID,{})
            EfficiencyDictLayer[mv].setdefault(etaPartitionID,{})
            for pt in range(0,101):
                EfficiencyDictGlobal[mv][etaPartitionID].setdefault(pt,{'num':0,'den':0})
                EfficiencyDictLayer[mv][etaPartitionID].setdefault(pt,{'num':0,'den':0})
            for j in range(0,20):
                EfficiencyDictDirection[mv][etaPartitionID].setdefault(j,{'num':0,'den':0})

        isGoodTrack = []
        passedCutProp = {key:[] for key in PropHitonEta.keys()}
        #print(PropHitonEta.keys())
        ## Applying cuts on the propagated tracks to be used
        for index in range(nPropHitsOnEtaID):
            cutPassed = passCut(PropHitonEta,etaPartitionID,index,maxPropR_Err=parameters["maxErrPropR"],maxPropPhi_Err=parameters["maxErrPropPhi"],fiducialCutR=parameters["fiducialR"],fiducialCutPhi=parameters["fiducialPhi"],minPt=parameters["minPt"],maxChi2=parameters["chi2cut"],minME1Hit=parameters["minME1"],minME2Hit=parameters["minME2"],minME3Hit=parameters["minME3"],minME4Hit=parameters["minME4"])#,ID = parameters['ID'])
            #print(parameters["fiducialCut"])
            #cutPassed = True
            THSanityChecks['charge_beforeSelection'].Fill(PropHitonEta['charge'][index])

            if parameters["fiducialCut"] and cutPassed != True:
                TH2FexcludedProphits_collector = FillExcludedHits( TH2FexcludedProphits_collector, PropHitonEta, index, current_chamber_ID, cutPassed)
                isGoodTrack.append(False)
            else:
                for mv in matching_variables:
                  if mv == 'glb_phi':  
                      if current_chamber_ID not in nVtx_den_dic.keys():nVtx_den_dic[current_chamber_ID]=1
                      if current_chamber_ID in nVtx_den_dic.keys():nVtx_den_dic[current_chamber_ID]=+1
                      #print("######### Sumit")
                      if region == -1 and layer == 1:
                          ML1_N_PropGEMRecoHits += 1
                      #print("######### Sumit")
                          #print("######### Keshri")
                      if region == -1 and layer == 2:
                          ML2_N_PropGEMRecoHits += 1
                      if region == 1 and layer == 1:
                          PL1_N_PropGEMRecoHits += 1
                      if region == 1 and layer == 2:
                          PL2_N_PropGEMRecoHits += 1
                npropHit_vtx+=1
                EfficiencyDictGlobal['glb_phi'][etaPartitionID][pt_index_fine(PropHitonEta['pt'][index])]['den'] += 1
                EfficiencyDictGlobal['glb_rdphi'][etaPartitionID][pt_index_fine(PropHitonEta['pt'][index])]['den'] += 1
                totalDen+=1
                
                VFAT_propagated = propHit2VFAT(PropHitonEta['glb_r'][index],PropHitonEta['loc_x'][index],eta)
                EfficiencyDictVFAT['glb_phi'][endcapTag][current_chamber_ID][VFAT_propagated]['den'] += 1
                EfficiencyDictVFAT['glb_rdphi'][endcapTag][current_chamber_ID][VFAT_propagated]['den'] += 1

                angle_index = int ( (cos_of_alpha_list[index] * 20 ) )
                EfficiencyDictDirection['glb_phi'][etaPartitionID][angle_index]['den'] += 1 
                EfficiencyDictDirection['glb_rdphi'][etaPartitionID][angle_index]['den'] += 1 
                
                # Sumit add efficinecy wrt chi2
                chi2_den.Fill(PropHitonEta['STA_Normchi2'][index])
                chi2_inv_den.Fill(1./PropHitonEta['STA_Normchi2'][index])

                isGoodTrack.append(True)
                for key in PropHitonEta.keys():
                    passedCutProp[key].append(PropHitonEta[key][index])
        
        #any is the logical or across all elements of a list
        if any(isGoodTrack) == False:
            #print ("No good STA propagation for etaPartitionID =  ",etaPartitionID)
            continue
        totalNum+=1
        #nMuons_good.Fill(evt.mu_nMuons)
    
        ##THSanityChecks['nMuons']["All"].Fill(len(PropHitonEta['pt']))
        
        PropHitonEta = passedCutProp
        nGoodPropagation = len(PropHitonEta['glb_phi'])        
        if parameters["DLE"]:    
            if layer == 1:
                layer1PassedCut = True
                layer1etapID = etaPartitionID
                layer1pt = 5 ## Fake pt value in GeV ... no B field
            if layer == 2:
                layer2PassedCut = True
                layer2etapID = etaPartitionID
                layer2pt = 5 ## Fake pt value in GeV ... no B field

        ## Filling STA properties in histos
        #THSanityChecks['nMuons'][current_chamber_ID]["All"].Fill(len(PropHitonEta['pt']))
        
        for k in range(nGoodPropagation):
            THSanityChecks['PropagationError']['glb_phi_error']['all'].Fill(PropHitonEta['err_glb_phi'][k])
            THSanityChecks['PropagationError']['glb_r_error']['all'].Fill(PropHitonEta['err_glb_r'][k])
            THSanityChecks['nME1Hits'].Fill(PropHitonEta['nME1Hits'][k])
            THSanityChecks['nME2Hits'].Fill(PropHitonEta['nME2Hits'][k])
            THSanityChecks['nME3Hits'].Fill(PropHitonEta['nME3Hits'][k])
            THSanityChecks['nME4Hits'].Fill(PropHitonEta['nME4Hits'][k])
            THSanityChecks['charge_beforeMatching'].Fill(PropHitonEta['charge'][k])
            THSanityChecks['nCSCHits'].Fill( PropHitonEta['nME1Hits'][k] + PropHitonEta['nME2Hits'][k] + PropHitonEta['nME3Hits'][k] + PropHitonEta['nME4Hits'][k] )
           
            TH1FpropError_collector["AllHits"][endcapTag][current_chamber_ID]["All"]["ErrPhi"].Fill(PropHitonEta['err_glb_phi'][k])
            TH1FpropError_collector["AllHits"][endcapTag][current_chamber_ID][eta]["ErrPhi"].Fill(PropHitonEta['err_glb_phi'][k])
            TH1FpropError_collector["AllHits"][endcapTag][current_chamber_ID]["All"]["ErrR"].Fill(PropHitonEta['err_glb_r'][k])
            TH1FpropError_collector["AllHits"][endcapTag][current_chamber_ID][eta]["ErrR"].Fill(PropHitonEta['err_glb_r'][k])

            THSanityChecks['pt'][current_chamber_ID]["All"].Fill(PropHitonEta['pt'][k])
            THSanityChecks['pt'][current_chamber_ID][eta].Fill(PropHitonEta['pt'][k])
            
            
            if chamber%2 == 0:
                THSanityChecks['PropagationError']['glb_phi_error']['long'].Fill(PropHitonEta['err_glb_phi'][k])
                THSanityChecks['PropagationError']['glb_r_error']['long'].Fill(PropHitonEta['err_glb_r'][k])
                if PropHitonEta['mu_propagated_isGEM'][k] == True:
                    THSanityChecks['PropagationError']['glb_phi_error']['long_isGEM'].Fill(PropHitonEta['err_glb_phi'][k])
                    THSanityChecks['PropagationError']['glb_r_error']['long_isGEM'].Fill(PropHitonEta['err_glb_r'][k])
                elif PropHitonEta['mu_propagated_isGEM'][k] == False:
                    THSanityChecks['PropagationError']['glb_phi_error']['long_noGEM'].Fill(PropHitonEta['err_glb_phi'][k])
                    THSanityChecks['PropagationError']['glb_r_error']['long_noGEM'].Fill(PropHitonEta['err_glb_r'][k])
            if chamber % 2 == 1:
                THSanityChecks['PropagationError']['glb_phi_error']['short'].Fill(PropHitonEta['err_glb_phi'][k])
                THSanityChecks['PropagationError']['glb_r_error']['short'].Fill(PropHitonEta['err_glb_r'][k])
                if PropHitonEta['mu_propagated_isGEM'][k] == True:
                    THSanityChecks['PropagationError']['glb_phi_error']['short_isGEM'].Fill(PropHitonEta['err_glb_phi'][k])
                    THSanityChecks['PropagationError']['glb_r_error']['short_isGEM'].Fill(PropHitonEta['err_glb_r'][k])
                elif PropHitonEta['mu_propagated_isGEM'][k] == False:
                    THSanityChecks['PropagationError']['glb_phi_error']['short_noGEM'].Fill(PropHitonEta['err_glb_phi'][k])
                    THSanityChecks['PropagationError']['glb_r_error']['short_noGEM'].Fill(PropHitonEta['err_glb_r'][k])
            
            THSanityChecks['PropagationError']['glb_phi_error']['eta'+str(eta)].Fill(PropHitonEta['err_glb_phi'][k])
            THSanityChecks['PropagationError']['glb_r_error']['eta'+str(eta)].Fill(PropHitonEta['err_glb_r'][k])


        if etaPartitionID not in RecHit_Dict:
            # if current_chamber_ID in chamberForEventDisplay: store4evtDspl(outputname,RunNumber,LumiSection,EventNumber)
            # if current_chamber_ID in chamberForEventDisplay: 
            #     print "event",EventNumber
            #     print "\t\tNothing to match on ", eta,"\t Match looked for phi = ",PropHitonEta['glb_phi']
            #     for s in range(1,9):
            #         sf =  region*(100*chamber+10*layer+s)
            #         if sf in RecHit_Dict.keys(): print sf,"\t",RecHit_Dict[sf]
                
            # print PropHit_Dict[etaPartitionID]['glb_phi']
            #print ("No rechit in etaPartitionID =  ",etaPartitionID)
            continue
        else: 
            RecHitonEta = RecHit_Dict[etaPartitionID]
        THSanityChecks['NHits']['PerEVT_PerEtaPartitionID']['Reco'].Fill(len(RecHitonEta['glb_phi']))


        ## Seek for 1-best match between rec and prop based on Matching Var and Cutoff
        for matchingVar in matching_variables:
                               
            residuals = []
            RecoMatched = []
            PropMatched = []
            if matchingVar == 'glb_rdphi':
                for RecHit_g_phi in RecHitonEta['glb_phi']:
                    deltardphis = [(PropHit_g_phi - RecHit_g_phi)*PropHitonEta['glb_r'][index] for index,PropHit_g_phi in enumerate(PropHitonEta['glb_phi'])]
                    temp_min = min(deltardphis,key=abs)
                    PropMatched.append(PropHitonEta['glb_phi'][deltardphis.index(temp_min)])
                    RecoMatched.append(RecHit_g_phi)
                    residuals.append(temp_min)
                
                min_residual = min(residuals,key=abs)
                #print("min_residual",min_residual)
                min_residual_index = residuals.index(min_residual)
                prop_hit_index = PropHitonEta['glb_phi'].index(PropMatched[min_residual_index])
                reco_hit_index = RecHitonEta['glb_phi'].index(RecoMatched[min_residual_index])




            else:
                for RecHit_var in RecHitonEta[matchingVar]:
                    deltas = [PropHit_var - RecHit_var for PropHit_var in PropHitonEta[matchingVar]]
                    temp_min = min(deltas,key=abs)
                    PropMatched.append(PropHitonEta[matchingVar][deltas.index(temp_min)])
                    RecoMatched.append(RecHit_var)
                    residuals.append(temp_min)
                
                min_residual = min(residuals,key=abs)
                min_residual_index = residuals.index(min_residual)
                prop_hit_index = PropHitonEta[matchingVar].index(PropMatched[min_residual_index])
                reco_hit_index = RecHitonEta[matchingVar].index(RecoMatched[min_residual_index])
            
            glb_phi_residual = PropHitonEta['glb_phi'][prop_hit_index] - RecHitonEta['glb_phi'][reco_hit_index]
            glb_rdphi_residual = (PropHitonEta['glb_phi'][prop_hit_index] - RecHitonEta['glb_phi'][reco_hit_index])*PropHitonEta['glb_r'][prop_hit_index]
            loc_y_residual = PropHitonEta['loc_y'][prop_hit_index] - RecHitonEta['loc_y'][reco_hit_index]
            propagation_error_phi = PropHitonEta['err_glb_phi'][prop_hit_index]
            propagation_error_r = PropHitonEta['err_glb_r'][prop_hit_index]

            if matchingVar == 'glb_phi':
                THSanityChecks['Residual_Correlation']['glb_phi_vs_glb_rdphi'].Fill(glb_phi_residual,glb_rdphi_residual)
                THSanityChecks['Residual_Correlation']['glb_rdphi_dir_x'].Fill(glb_rdphi_residual,np.arccos(PropHitonEta['Loc_dirX'][prop_hit_index]))


            if matchingVar == "glb_rdphi":
                TH1Fresidual_collector_x[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(glb_rdphi_residual)
                TH1Fresidual_collector_x[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(glb_rdphi_residual)
                TH2Fresidual_collector_x[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(PropHitonEta['pt'][prop_hit_index],glb_rdphi_residual)
                TH2Fresidual_collector_x[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(PropHitonEta['pt'][prop_hit_index],glb_rdphi_residual)
                if PropHitonEta['pt'][prop_hit_index] >= 30:
                    TH1Fresidual_collector_x_ptgt30[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(glb_rdphi_residual)
                    TH1Fresidual_collector_x_ptgt30[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(glb_rdphi_residual)
                if PropHitonEta['pt'][prop_hit_index] < 30:
                    TH1Fresidual_collector_x_ptlt30[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(glb_rdphi_residual)
                    TH1Fresidual_collector_x_ptlt30[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(glb_rdphi_residual)
                if PropHitonEta['charge'][prop_hit_index] < 0:
                    TH2Fresidual_collector_x_minus[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(PropHitonEta['pt'][prop_hit_index],glb_rdphi_residual)
                    TH2Fresidual_collector_x_minus[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(PropHitonEta['pt'][prop_hit_index],glb_rdphi_residual)
                    TH1Fresidual_collector_x_minus[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(glb_rdphi_residual)
                    TH1Fresidual_collector_x_minus[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(glb_rdphi_residual)
                    if PropHitonEta['pt'][prop_hit_index] >= 30:
                        TH1Fresidual_collector_x_minus_ptgt30[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(glb_rdphi_residual)
                        TH1Fresidual_collector_x_minus_ptgt30[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(glb_rdphi_residual)
                    if PropHitonEta['pt'][prop_hit_index] < 30:
                        TH1Fresidual_collector_x_minus_ptlt30[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(glb_rdphi_residual)
                        TH1Fresidual_collector_x_minus_ptlt30[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(glb_rdphi_residual)
                if PropHitonEta['charge'][prop_hit_index] > 0:
                    TH1Fresidual_collector_x_plus[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(glb_rdphi_residual)
                    TH1Fresidual_collector_x_plus[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(glb_rdphi_residual)
                    TH2Fresidual_collector_x_plus[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(PropHitonEta['pt'][prop_hit_index],glb_rdphi_residual)
                    TH2Fresidual_collector_x_plus[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(PropHitonEta['pt'][prop_hit_index],glb_rdphi_residual)
                    if PropHitonEta['pt'][prop_hit_index] >= 30:
                        TH1Fresidual_collector_x_plus_ptgt30[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(glb_rdphi_residual)
                        TH1Fresidual_collector_x_plus_ptgt30[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(glb_rdphi_residual)
                    if PropHitonEta['pt'][prop_hit_index] < 30:
                        TH1Fresidual_collector_x_plus_ptlt30[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(glb_rdphi_residual)
                        TH1Fresidual_collector_x_plus_ptlt30[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(glb_rdphi_residual)

            #   TH1Fresidual_collector_x[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(glb_rdphi_residual)
            #    TH1Fresidual_collector_x[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(glb_rdphi_residual)
            #    TH1Fresidual_collector_y[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(loc_y_residual)
            #    TH1Fresidual_collector_y[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(loc_y_residual)
            #else:
            #    TH1Fresidual_collector_x[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(glb_phi_residual)
            #    TH1Fresidual_collector_x[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(glb_phi_residual)
            #    TH1Fresidual_collector_y[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(loc_y_residual)
            #    TH1Fresidual_collector_y[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(loc_y_residual)



            if abs(min_residual) < ResidualCutOff[matchingVar]:

                
                if current_chamber_ID not in nVtx_num_dic.keys():nVtx_num_dic[current_chamber_ID]=1
                if current_chamber_ID in nVtx_num_dic.keys():nVtx_num_dic[current_chamber_ID]=+1

                EfficiencyDictGlobal[matchingVar][etaPartitionID][pt_index_fine(PropHitonEta['pt'][prop_hit_index])]['num'] += 1
                
                VFAT_propagated = propHit2VFAT(PropHitonEta['glb_r'][prop_hit_index],PropHitonEta['loc_x'][prop_hit_index],eta)
                EfficiencyDictVFAT[matchingVar][endcapTag][current_chamber_ID][VFAT_propagated]['num'] += 1

                angle_index = int( np.sqrt(PropHitonEta['Loc_dirX'][prop_hit_index]**2 + PropHitonEta['Loc_dirY'][prop_hit_index]**2) * 20)
                EfficiencyDictDirection[matchingVar][etaPartitionID][angle_index]['num'] += 1

                # Sumit added for efficiency wrt chi2
                chi2_num.Fill(PropHitonEta['STA_Normchi2'][prop_hit_index])
                chi2_inv_num.Fill(1./PropHitonEta['STA_Normchi2'][prop_hit_index])


                cluster_size = RecHitonEta['cluster_size'][reco_hit_index]
                if matchingVar == "glb_rdphi":
                    nmatched_vtx+=1        
                    THSanityChecks['charge_afterMatching'].Fill(PropHitonEta['charge'][prop_hit_index])
                    TH1FpropError_collector["MatchedHits"][endcapTag][current_chamber_ID]["All"]["ErrPhi"].Fill(propagation_error_phi)
                    TH1FpropError_collector["MatchedHits"][endcapTag][current_chamber_ID][eta]["ErrPhi"].Fill(propagation_error_phi)
                    TH1FpropError_collector["MatchedHits"][endcapTag][current_chamber_ID]["All"]["ErrR"].Fill(propagation_error_r)
                    TH1FpropError_collector["MatchedHits"][endcapTag][current_chamber_ID][eta]["ErrR"].Fill(propagation_error_r)
                    if cluster_size>=0:
                        pass
                        #TH1Fresidual_collector_x[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(glb_rdphi_residual)
                        #TH1Fresidual_collector_x[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(glb_rdphi_residual)
                        #if PropHitonEta['charge'][prop_hit_index] < 0:
                        #    TH1Fresidual_collector_x_minus[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(glb_rdphi_residual)
                        #    TH1Fresidual_collector_x_minus[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(glb_rdphi_residual)
                        #if PropHitonEta['charge'][prop_hit_index] > 0:
                        #    TH1Fresidual_collector_x_plus[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(glb_rdphi_residual)
                        #    TH1Fresidual_collector_x_plus[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(glb_rdphi_residual)
                    TH1Fresidual_collector_y[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(loc_y_residual)
                    TH1Fresidual_collector_y[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(loc_y_residual)
                else:
                    TH1Fresidual_collector_x[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(glb_phi_residual)
                    TH1Fresidual_collector_x[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(glb_phi_residual)
                    TH1Fresidual_collector_y[matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(loc_y_residual)
                    TH1Fresidual_collector_y[matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(loc_y_residual)

                binx = int(round((PropHitonEta['loc_x'][prop_hit_index]-TH2min)*(TH2nbins-1)/(-2*TH2min)))+1
                biny = int(round((PropHitonEta['loc_y'][prop_hit_index]-TH2min)*(TH2nbins-1)/(-2*TH2min)))+1
                TH2Fresidual_collector[matchingVar]['all']['glb_phi'][binx][biny][0] += 1
                TH2Fresidual_collector[matchingVar]['all']['glb_rdphi'][binx][biny][0] += 1
                TH2Fresidual_collector[matchingVar]['all']['glb_phi'][binx][biny][1] += abs(glb_phi_residual)
                TH2Fresidual_collector[matchingVar]['all']['glb_rdphi'][binx][biny][1] += abs(glb_rdphi_residual)

                THSanityChecks['Occupancy'][matchingVar]['AfterMatching']['Reco'].Fill(RecHitonEta['glb_x'][reco_hit_index],RecHitonEta['glb_y'][reco_hit_index])
                THSanityChecks['Occupancy'][matchingVar]['AfterMatching']['Prop'].Fill(PropHitonEta['glb_x'][prop_hit_index],PropHitonEta['glb_y'][prop_hit_index])


                if chamber%2 == 0:
                    if matchingVar == 'glb_phi': THSanityChecks['PropHit_DirLoc_xOnGE11']['AfterMatching']['Long']['eta'+str(eta)].Fill(PropHitonEta['Loc_dirX'][prop_hit_index])
                    TH2Fresidual_collector[matchingVar]['long']['glb_phi'][binx][biny][0] += 1
                    TH2Fresidual_collector[matchingVar]['long']['glb_rdphi'][binx][biny][0] += 1
                    TH2Fresidual_collector[matchingVar]['long']['glb_phi'][binx][biny][1] += abs(glb_phi_residual)
                    TH2Fresidual_collector[matchingVar]['long']['glb_rdphi'][binx][biny][1] += abs(glb_rdphi_residual)
                    THSanityChecks['Occupancy'][matchingVar]['AfterMatching']['PropLocalLong'].Fill(PropHitonEta['loc_x'][prop_hit_index],PropHitonEta['loc_y'][prop_hit_index])
                if chamber%2 == 1:
                    if matchingVar == 'glb_phi': THSanityChecks['PropHit_DirLoc_xOnGE11']['AfterMatching']['Short']['eta'+str(eta)].Fill(PropHitonEta['Loc_dirX'][prop_hit_index])
                    TH2Fresidual_collector[matchingVar]['short']['glb_phi'][binx][biny][0] += 1
                    TH2Fresidual_collector[matchingVar]['short']['glb_rdphi'][binx][biny][0] += 1
                    TH2Fresidual_collector[matchingVar]['short']['glb_phi'][binx][biny][1] += abs(glb_phi_residual)
                    TH2Fresidual_collector[matchingVar]['short']['glb_rdphi'][binx][biny][1] += abs(glb_rdphi_residual)                    
                    THSanityChecks['Occupancy'][matchingVar]['AfterMatching']['PropLocalShort'].Fill(PropHitonEta['loc_x'][prop_hit_index],PropHitonEta['loc_y'][prop_hit_index])

                
                if matchingVar == 'glb_phi':                    
                    if region == -1 and layer == 1:
                        ML1_N_MatchedGEMRecoHits += 1
                    if region == -1 and layer == 2:
                        ML2_N_MatchedGEMRecoHits += 1
                    if region == 1 and layer == 1:
                        PL1_N_MatchedGEMRecoHits += 1
                    if region == 1 and layer == 2:
                        PL2_N_MatchedGEMRecoHits += 1
                if matchingVar == 'glb_rdphi':
                    if cluster_size in range(1,20): 
                        TH1Fresidual_collector_CLS_x[cluster_size][matchingVar][endcapTag][current_chamber_ID]["All"]["Residual"].Fill(glb_rdphi_residual)
                        TH1Fresidual_collector_CLS_x[cluster_size][matchingVar][endcapTag][current_chamber_ID][eta]["Residual"].Fill(glb_rdphi_residual)
                    THSanityChecks['CLS']["aftermatching"][current_chamber_ID]["All"].Fill(cluster_size)
                    THSanityChecks['CLS']["aftermatching"][current_chamber_ID][eta].Fill(cluster_size)
                    STAdirX_vs_CLS.Fill(PropHitonEta['Loc_dirX'][prop_hit_index],cluster_size)
                    
                    if layer == 1:
                        layer1Match = True
                        layer1pt = PropHitonEta['pt'][prop_hit_index]
                    if layer == 2:
                        layer2Match = True
                        layer2pt = PropHitonEta['pt'][prop_hit_index]
                    
            else:
                # if current_chamber_ID in chamberForEventDisplay: store4evtDspl(outputname,RunNumber,LumiSection,EventNumber)
                # print "Matching failed for ", etaPartitionID
                # print PropHit_Dict[etaPartitionID]['glb_phi'], RecHit_Dict[etaPartitionID]['glb_phi']
                # raw_input()
                pass

        ## Loop over etaPID 

    ## Double Layer Efficiency (DLE): test layer1(2) with tracks that have matched in layer1(2)
    try:
        evt.event_nVtx>0
        if(nmatched_vtx>0): nVtx_num.Fill(evt.event_nVtx)
        if(npropHit_vtx>0): nVtx_den.Fill(evt.event_nVtx)

        if(ML1_N_MatchedGEMRecoHits>0): nVtx_ML1_num.Fill(evt.event_nVtx)
        if(ML1_N_PropGEMRecoHits>0)   : nVtx_ML1_den.Fill(evt.event_nVtx)
        if(ML2_N_MatchedGEMRecoHits>0): nVtx_ML2_num.Fill(evt.event_nVtx)
        if(ML2_N_PropGEMRecoHits>0)   : nVtx_ML2_den.Fill(evt.event_nVtx)
        if(PL1_N_MatchedGEMRecoHits>0): nVtx_PL1_num.Fill(evt.event_nVtx)
        if(PL1_N_PropGEMRecoHits>0)   : nVtx_PL1_den.Fill(evt.event_nVtx)
        if(PL2_N_MatchedGEMRecoHits>0): nVtx_PL2_num.Fill(evt.event_nVtx)
        if(PL2_N_PropGEMRecoHits>0)   : nVtx_PL2_den.Fill(evt.event_nVtx)

        #print("nVtx_den_dic",nVtx_den_dic,nVtx_den_dic=={})
        #print("nVtx_num_dic",nVtx_num_dic,nVtx_den_dic=={})
        for key in nVtx_den_dic.keys():
            ll=key.split("-")
            if ll[1] == 'P':
                if ll[2][2:]=='L1':
                    # print("evt.event_nVtx",evt.event_nVtx)
                    nVtx_PL1_2D_den.Fill(evt.event_nVtx,int(ll[2][:2]))
                    if key in nVtx_num_dic.keys():
                        nVtx_PL1_2D_num.Fill(evt.event_nVtx,int(ll[2][:2]))
                else:
                    nVtx_PL2_2D_den.Fill(evt.event_nVtx,int(ll[2][:2]))
                    if key in nVtx_num_dic.keys():
                        nVtx_PL2_2D_num.Fill(evt.event_nVtx,int(ll[2][:2]))
            else:
                if ll[2][2:]=='L1':
                    nVtx_ML1_2D_den.Fill(evt.event_nVtx,int(ll[2][:2]))
                    if key in nVtx_num_dic.keys():
                        nVtx_ML1_2D_num.Fill(evt.event_nVtx,int(ll[2][:2]))
                else:
                    nVtx_ML2_2D_den.Fill(evt.event_nVtx,int(ll[2][:2]))
                    if key in nVtx_num_dic.keys():
                        nVtx_ML2_2D_num.Fill(evt.event_nVtx,int(ll[2][:2]))
    except:
        pass



    if parameters["DLE"] and layer2Match and layer1PassedCut and layer2PassedCut:
        EfficiencyDictLayer['glb_rdphi'][layer1etapID][pt_index_fine(layer2pt)]['den'] += 1
        DLE_pt.Fill(layer2pt)
        # DLE_ErrPhi.Fill(PropHitonEta['err_glb_phi'][prop_hit_index])
        # DLE_ErrR.Fill(PropHitonEta['err_glb_r'][prop_hit_index])

        if layer1Match == True:
            EfficiencyDictLayer['glb_rdphi'][layer1etapID][pt_index_fine(layer2pt)]['num'] += 1

    if parameters["DLE"] and layer1Match and layer2PassedCut and layer1PassedCut:
        EfficiencyDictLayer['glb_rdphi'][layer2etapID][pt_index_fine(layer1pt)]['den'] += 1
        DLE_pt.Fill(layer1pt)
        # DLE_ErrPhi.Fill(PropHitonEta['err_glb_phi'][prop_hit_index])
        # DLE_ErrR.Fill(PropHitonEta['err_glb_r'][prop_hit_index])

        if layer2Match:
            EfficiencyDictLayer['glb_rdphi'][layer2etapID][pt_index_fine(layer1pt)]['num'] += 1

    ## Loop over evts

    THSanityChecks['NHits']['AfterMatching']['All'].Fill(ML1_N_MatchedGEMRecoHits + ML2_N_MatchedGEMRecoHits  +  PL1_N_MatchedGEMRecoHits + PL2_N_MatchedGEMRecoHits)
    THSanityChecks['NHits']['AfterMatching']['ML1'].Fill(ML1_N_MatchedGEMRecoHits)
    THSanityChecks['NHits']['AfterMatching']['ML2'].Fill(ML2_N_MatchedGEMRecoHits)
    THSanityChecks['NHits']['AfterMatching']['PL1'].Fill(PL1_N_MatchedGEMRecoHits)
    THSanityChecks['NHits']['AfterMatching']['PL2'].Fill(PL2_N_MatchedGEMRecoHits)
## End of the evts loop
print(f'totalDen:{totalDen}, totalNum:{totalNum}')
print(set(rechit_pid))
print(set(prophit_pid))
print("totalEvents :%d"%totalEvents)
print("totalFilterClus :%d"%totalFilterClus)
print("filteredEvents :%d"%filteredEvents)
print("filterCluster :%d"%filterCluster)
print("filterFlower :%d"%filterFlower)
print("l1aEvent :%d"%l1aEvent)


for run in VFATMaskBook:
    for endcapTag,plot in VFATMaskBook[run].items():
        VFATMaskBook[run][endcapTag].Scale(1/processedEvents[run])
TH2Fresidual_collector = fillPlot2DResidualContainer(TH2Fresidual_collector,matching_variables,TH2nbins)

logging.info(f"--- {round(time.time() - start_time,2)} seconds ---")


## Storing the results
# matchingFile.Write()
OutF = ROOT.TFile(OUTPUT_PATH + "/PFA_Analyzer_Output/ROOT_File/"+parameters["outputname"]+".root","RECREATE")

subprocess.call(["mkdir", "-p", OUTPUT_PATH + "/PFA_Analyzer_Output/CSV/"+parameters["outputname"]])
subprocess.call(["mkdir", "-p", OUTPUT_PATH + "/PFA_Analyzer_Output/CSV/"+parameters["outputname"]])
subprocess.call(["mkdir", "-p", OUTPUT_PATH + "/PFA_Analyzer_Output/Plot/"+parameters["outputname"]+"/"+"glb_phi/"])
subprocess.call(["mkdir", "-p", OUTPUT_PATH + "/PFA_Analyzer_Output/Plot/"+parameters["outputname"]+"/"+"glb_rdphi/"])


### Masking
if not 'MC' in data_ranges.keys(): # sumit 
    for run in data_ranges:
        print("########################## sumit: ",run)
        data_ranges[run]["lumisection"] = (data_ranges[run]["lumisection"][0],maxLS[run])
        masking_dict_key = [f'{run}_chamberOFF',f'{run}_VFATOFF',f'{run}_ExclusionSummary',f"{run}_VFATMaskBook"]
        for k in masking_dict_key:
            TH1MetaData[k]=setUpCanvas(k,2000,2000)
            TH1MetaData[k].Divide(2,2)
    
        OFFChambers_plots = ChambersOFFHisto(data_ranges[run]["chamberOFF"],data_ranges[run]["lumisection"][0],data_ranges[run]["lumisection"][1])
        for counter,plot in enumerate(OFFChambers_plots):
            TH1MetaData[f'{run}_chamberOFF'].cd(counter+1)
            ROOT.gPad.SetGrid()
            plot.Draw()
    
        OFFVFATs_plots = VFATOFFHisto(data_ranges[run]["VFATOFF"])
        for counter,plot in enumerate(OFFVFATs_plots):
            TH1MetaData[f'{run}_VFATOFF'].cd(counter+1)
            plot.Draw("COLZ")
    
        for counter,key in enumerate(VFATMaskBook[RunNumber].keys()):
            pad = TH1MetaData[f'{run}_VFATMaskBook'].cd(counter+1)
            ROOT.gPad.SetGrid()
            VFATMaskBook[RunNumber][key].Draw("COLZ TEXT0")
    
        GE11Discarded_plots = GE11DiscardedSummary(data_ranges[run]["chamberOFF"],data_ranges[run]["lumisection"][0],data_ranges[run]["lumisection"][1],data_ranges[run]["VFATOFF"])
        for counter,plot in enumerate(GE11Discarded_plots):
            TH1MetaData[f'{run}_ExclusionSummary'].cd(counter+1)
            plot.Draw("COLZ")
            TH1MetaData[f'{run}_ExclusionSummary'].cd(counter+1).Update()
            palette = plot.GetListOfFunctions().FindObject("palette")
            palette.SetX2NDC(0.93)
            palette.Draw()
        setUpCanvas("GE11 Masked",1200,1200).cd()
    
        for k in masking_dict_key:
            TH1MetaData[k].Modified()
            TH1MetaData[k].Update()
    
writeToTFile(OutF,THSanityChecks['NEvts'],"SanityChecks/NumberOfEVTs/")
writeToTFile(OutF,THSanityChecks['charge_beforeSelection'],"SanityChecks/charge/")
writeToTFile(OutF,THSanityChecks['charge_beforeMatching'],"SanityChecks/charge/")
writeToTFile(OutF,THSanityChecks['charge_afterMatching'],"SanityChecks/charge/")

writeToTFile(OutF,THSanityChecks['NHits']['BeforeMatching']['PerEVT']['Reco'],"SanityChecks/NHits/BeforeMatching/")
writeToTFile(OutF,THSanityChecks['NHits']['BeforeMatching']['PerEVT']['Prop'],"SanityChecks/NHits/BeforeMatching/")

writeToTFile(OutF,THSanityChecks['NHits']['BeforeMatching']['ML1'],"SanityChecks/NHits/BeforeMatching/")
writeToTFile(OutF,THSanityChecks['NHits']['BeforeMatching']['ML2'],"SanityChecks/NHits/BeforeMatching/")
writeToTFile(OutF,THSanityChecks['NHits']['BeforeMatching']['PL1'],"SanityChecks/NHits/BeforeMatching/")
writeToTFile(OutF,THSanityChecks['NHits']['BeforeMatching']['PL2'],"SanityChecks/NHits/BeforeMatching/")

writeToTFile(OutF,THSanityChecks['NHits']['AfterMatching']['All'],"SanityChecks/NHits/AfterMatching/")
writeToTFile(OutF,THSanityChecks['NHits']['AfterMatching']['ML1'],"SanityChecks/NHits/AfterMatching/")
writeToTFile(OutF,THSanityChecks['NHits']['AfterMatching']['ML2'],"SanityChecks/NHits/AfterMatching/")
writeToTFile(OutF,THSanityChecks['NHits']['AfterMatching']['PL1'],"SanityChecks/NHits/AfterMatching/")
writeToTFile(OutF,THSanityChecks['NHits']['AfterMatching']['PL2'],"SanityChecks/NHits/AfterMatching/")

writeToTFile(OutF,THSanityChecks['NHits']['PerEVT_PerEtaPartitionID']['Reco'],"SanityChecks/NHits/BeforeMatching/")
writeToTFile(OutF,THSanityChecks['NHits']['PerEVT_PerEtaPartitionID']['Prop'],"SanityChecks/NHits/BeforeMatching/")

writeToTFile(OutF,THSanityChecks['Occupancy']['BeforeMatching']['Prop'],"SanityChecks/Occupancy/BeforeMatching")
writeToTFile(OutF,THSanityChecks['Occupancy']['BeforeMatching']['PropLocalLong'],"SanityChecks/Occupancy/BeforeMatching")
writeToTFile(OutF,THSanityChecks['Occupancy']['BeforeMatching']['PropLocalShort'],"SanityChecks/Occupancy/BeforeMatching")
writeToTFile(OutF,THSanityChecks['Occupancy']['BeforeMatching']['Reco'],"SanityChecks/Occupancy/BeforeMatching")

writeToTFile(OutF,STAdirX_vs_CLS,"SanityChecks/CLS_vs_Direction/")
writeToTFile(OutF,nDigis_vs_1l1a,"SanityChecks/CLS_vs_L1A/")
writeToTFile(OutF,nDigis_vs_1l1a_mask_clus,"SanityChecks/CLS_vs_L1A/")
writeToTFile(OutF,nDigis_vs_1l1a_mask_l1A,"SanityChecks/CLS_vs_L1A/")
writeToTFile(OutF,nDigis_vs_1l1a_mask_both,"SanityChecks/CLS_vs_L1A/")
writeToTFile(OutF,nDigis_vs_2l1a,"SanityChecks/CLS_vs_L1A/")
writeToTFile(OutF,nDigis_vs_3l1a,"SanityChecks/CLS_vs_L1A/")
writeToTFile(OutF,nDigis_vs_4l1a,"SanityChecks/CLS_vs_L1A/")
writeToTFile(OutF,THSanityChecks['pt']['All'],"SanityChecks/pt/")
writeToTFile(OutF,THSanityChecks['pt']['All_minus'],"SanityChecks/pt/")
writeToTFile(OutF,THSanityChecks['pt']['All_plus'],"SanityChecks/pt/")
#writeToTFile(OutF,THSanityChecks['nMuons']['All'],"SanityChecks/nMuons/")
if parameters["DLE"]: writeToTFile(OutF,DLE_pt,"SanityChecks/pt/")

for key in ['ML1','ML2','PL1','PL2']:
    writeToTFile(OutF,THSanityChecks['Occupancy']['BeforeMatching'][key]['PropHits'],"SanityChecks/Occupancy/BeforeMatching/"+key)
    writeToTFile(OutF,THSanityChecks['Occupancy']['BeforeMatching'][key]['RecHits'],"SanityChecks/Occupancy/BeforeMatching/"+key)
    for ch in range(1,37):
        writeToTFile(OutF,THSanityChecks['RecHitperStrip'][key][ch],"SanityChecks/Occupancy/RecHitByStrip/"+key)

writeToTFile(OutF,THSanityChecks['PropagationError']['glb_phi_error']['all'],"SanityChecks/PropagationError/glb_phi")
writeToTFile(OutF,THSanityChecks['PropagationError']['glb_phi_error']['long'],"SanityChecks/PropagationError/glb_phi")
writeToTFile(OutF,THSanityChecks['PropagationError']['glb_phi_error']['long_isGEM'],"SanityChecks/PropagationError/glb_phi")
writeToTFile(OutF,THSanityChecks['PropagationError']['glb_phi_error']['long_noGEM'],"SanityChecks/PropagationError/glb_phi")
writeToTFile(OutF,THSanityChecks['PropagationError']['glb_phi_error']['short'],"SanityChecks/PropagationError/glb_phi")
writeToTFile(OutF,THSanityChecks['PropagationError']['glb_phi_error']['short_isGEM'],"SanityChecks/PropagationError/glb_phi")
writeToTFile(OutF,THSanityChecks['PropagationError']['glb_phi_error']['short_noGEM'],"SanityChecks/PropagationError/glb_phi")
writeToTFile(OutF,THSanityChecks['PropagationError']['glb_r_error']['all'],"SanityChecks/PropagationError/glb_r")
writeToTFile(OutF,THSanityChecks['PropagationError']['glb_r_error']['long'],"SanityChecks/PropagationError/glb_r")
writeToTFile(OutF,THSanityChecks['PropagationError']['glb_r_error']['long_isGEM'],"SanityChecks/PropagationError/glb_r")
writeToTFile(OutF,THSanityChecks['PropagationError']['glb_r_error']['long_noGEM'],"SanityChecks/PropagationError/glb_r")
writeToTFile(OutF,THSanityChecks['PropagationError']['glb_r_error']['short'],"SanityChecks/PropagationError/glb_r")
writeToTFile(OutF,THSanityChecks['PropagationError']['glb_r_error']['short_isGEM'],"SanityChecks/PropagationError/glb_r")
writeToTFile(OutF,THSanityChecks['PropagationError']['glb_r_error']['short_noGEM'],"SanityChecks/PropagationError/glb_r")

writeToTFile(OutF,THSanityChecks['etaP_vs_pt'],"SanityChecks/etaP_vs_pt/")

writeToTFile(OutF,THSanityChecks['Residual_Correlation']['glb_phi_vs_glb_rdphi'],"SanityChecks/Residual_Correlation/")
writeToTFile(OutF,THSanityChecks['Residual_Correlation']['glb_rdphi_dir_x'],"SanityChecks/Residual_Correlation/")
writeToTFile(OutF,THSanityChecks['STA_Normchi2'],"SanityChecks/STA_NormChi2/")
writeToTFile(OutF,THSanityChecks['nME1Hits'],"SanityChecks/HitsCSC/")
writeToTFile(OutF,THSanityChecks['nME2Hits'],"SanityChecks/HitsCSC/")
writeToTFile(OutF,THSanityChecks['nME3Hits'],"SanityChecks/HitsCSC/")
writeToTFile(OutF,THSanityChecks['nME4Hits'],"SanityChecks/HitsCSC/")
writeToTFile(OutF,THSanityChecks['nCSCHits'],"SanityChecks/HitsCSC/")


writeToTFile(OutF,THAll_Residuals[1],"Residuals/MatchingOn_glb_rdphi/")
writeToTFile(OutF,THAll_Residuals[2],"Residuals/MatchingOn_glb_rdphi")
writeToTFile(OutF,THAll_Residuals[3],"Residuals/MatchingOn_glb_rdphi/")
writeToTFile(OutF,THAll_Residuals[4],"Residuals/MatchingOn_glb_rdphi/")
writeToTFile(OutF,THAll_Residuals[5],"Residuals/MatchingOn_glb_rdphi/")


for key_1 in ['eta'+str(j) for j in range(1,9)]:
    writeToTFile(OutF,THSanityChecks['PropagationError']['glb_phi_error'][key_1],"SanityChecks/PropagationError/glb_phi/byEta")
    writeToTFile(OutF,THSanityChecks['PropagationError']['glb_r_error'][key_1],"SanityChecks/PropagationError/glb_r/byEta")
    for key_2 in ['Long','Short']:
        writeToTFile(OutF,THSanityChecks['PropHit_DirLoc_xOnGE11']['AfterMatching'][key_2][key_1],"SanityChecks/PropHitDirection/AfterMatching/"+key_2)
        writeToTFile(OutF,THSanityChecks['PropHit_DirLoc_xOnGE11']['BeforeMatching'][key_2][key_1],"SanityChecks/PropHitDirection/BeforeMatching/"+key_2)

for when in ["beforematching","aftermatching"]: 
    for r in [-1,1]:
        for l in [1,2]:
            endcapTag = EndcapLayer2label(r,l)   
            for ch in range(1,37):
                ch_id = getChamberName(r,ch,l)
                for eta in list(range(1,9))+ ["All"]:
                    writeToTFile(OutF,THSanityChecks['CLS'][when][ch_id][eta],f"SanityChecks/CLS/{when}/{endcapTag}/{ch_id}")

for chsize,value in TH2FexcludedProphits_collector.items():
    for exclusion_key, histo in value.items():
        writeToTFile(OutF,histo,"SanityChecks/ExcludedPropagateHits/"+chsize+"/")

for matchingVar in matching_variables:
    for chambers in ['all','long','short']:
        writeToTFile(OutF,TH2Fresidual_collector[matchingVar][chambers]['glb_phi']['TH2F'],"Residuals/MatchingOn_"+matchingVar+"/2D_glb_phi")
        writeToTFile(OutF,TH2Fresidual_collector[matchingVar][chambers]['glb_rdphi']['TH2F'],"Residuals/MatchingOn_"+matchingVar+"/2D_glb_rdphi")

    efficiency2DPlotAll,Num2DAll,Den2DAll,SummaryAll = generateEfficiencyPlot2DGE11(EfficiencyDictGlobal[matchingVar],[-1,1],[1,2],debug=parameters["verbose"])
    EffiDistrAll,EffiDistrNeg,EffiDistrPos = generateEfficiencyDistribution(EfficiencyDictGlobal[matchingVar])
    GE11efficiencyByEta_Short,GE11efficiencyByEta_Long,GE11efficiencyByEta_All = generateEfficiencyPlotbyEta(EfficiencyDictGlobal[matchingVar],[1,-1],[1,2])
    GE11efficiencyByPt_Short,GE11efficiencyByPt_Long,GE11efficiencyByPt_All, GE11HistoByPt_All = generateEfficiencyPlotbyPt(EfficiencyDictGlobal[matchingVar])
    num_angle,den_angle,angle_vs_eff = incidenceAngle_vs_Eff(EfficiencyDictDirection[matchingVar],[-1,1],[1,2])

    writeToTFile(OutF,den_angle,"Efficiency/"+matchingVar+"/Angle/")
    writeToTFile(OutF,num_angle,"Efficiency/"+matchingVar+"/Angle/")
    writeToTFile(OutF,angle_vs_eff,"Efficiency/"+matchingVar+"/Angle/")

    writeToTFile(OutF,efficiency2DPlotAll,"Efficiency/"+matchingVar+"/2DView/")
    writeToTFile(OutF,Num2DAll,"Efficiency/"+matchingVar+"/2DView/")
    writeToTFile(OutF,Den2DAll,"Efficiency/"+matchingVar+"/2DView/")
    writeToTFile(OutF,GE11efficiencyByEta_Short,"Efficiency/"+matchingVar+"/ByEta/")
    writeToTFile(OutF,GE11efficiencyByEta_Long,"Efficiency/"+matchingVar+"/ByEta/")
    writeToTFile(OutF,GE11efficiencyByEta_All,"Efficiency/"+matchingVar+"/ByEta/")
    writeToTFile(OutF,GE11efficiencyByPt_Short,"Efficiency/"+matchingVar+"/ByPt/")
    writeToTFile(OutF,GE11efficiencyByPt_Long,"Efficiency/"+matchingVar+"/ByPt/")
    writeToTFile(OutF,GE11efficiencyByPt_All,"Efficiency/"+matchingVar+"/ByPt/")
    writeToTFile(OutF,GE11HistoByPt_All["num"],"Efficiency/"+matchingVar+"/ByPt/num/")
    writeToTFile(OutF,GE11HistoByPt_All["den"],"Efficiency/"+matchingVar+"/ByPt/den/")

    # Histograms for chi2: Sumit
    writeToTFile(OutF,chi2_den,"Efficiency/chi2/")
    writeToTFile(OutF,chi2_num,"Efficiency/chi2/")
    writeToTFile(OutF,chi2_inv_den,"Efficiency/chi2/")
    writeToTFile(OutF,chi2_inv_num,"Efficiency/chi2/")

    # Histograms for nVtx
    writeToTFile(OutF,nVtx_den,"Efficiency/nVtx/")
    writeToTFile(OutF,nVtx_num,"Efficiency/nVtx/")

    writeToTFile(OutF,nVtx_ML1_den,"Efficiency/nVtx/")
    writeToTFile(OutF,nVtx_ML1_num,"Efficiency/nVtx/")
    writeToTFile(OutF,nVtx_ML2_den,"Efficiency/nVtx/")
    writeToTFile(OutF,nVtx_ML2_num,"Efficiency/nVtx/")
    writeToTFile(OutF,nVtx_PL1_den,"Efficiency/nVtx/")
    writeToTFile(OutF,nVtx_PL1_num,"Efficiency/nVtx/")
    writeToTFile(OutF,nVtx_PL2_den,"Efficiency/nVtx/")
    writeToTFile(OutF,nVtx_PL2_num,"Efficiency/nVtx/")
    # storing 2D histogram Chamber vs nVtx 
    writeToTFile(OutF,nVtx_ML1_2D_den,"Efficiency/nVtx/")
    writeToTFile(OutF,nVtx_ML1_2D_num,"Efficiency/nVtx/")
    writeToTFile(OutF,nVtx_ML2_2D_den,"Efficiency/nVtx/")
    writeToTFile(OutF,nVtx_ML2_2D_num,"Efficiency/nVtx/")
    writeToTFile(OutF,nVtx_PL1_2D_den,"Efficiency/nVtx/")
    writeToTFile(OutF,nVtx_PL1_2D_num,"Efficiency/nVtx/")
    writeToTFile(OutF,nVtx_PL2_2D_den,"Efficiency/nVtx/")
    writeToTFile(OutF,nVtx_PL2_2D_num,"Efficiency/nVtx/")

    if parameters["DLE"] and matchingVar=='glb_rdphi':
        DLE_2DPlotAll,DLE_Num2DAll,DLE_Den2DAll,DLE_SummaryAll = generateEfficiencyPlot2DGE11(EfficiencyDictLayer[matchingVar],[-1,1],[1,2])
        DLE_ByEta_Short,DLE_ByEta_Long,DLE_ByEta_All = generateEfficiencyPlotbyEta(EfficiencyDictLayer[matchingVar],[1,-1],[1,2])
        writeToTFile(OutF,DLE_2DPlotAll,"Efficiency/DLE/2DView/")
        writeToTFile(OutF,DLE_Num2DAll,"Efficiency/DLE/2DView/")
        writeToTFile(OutF,DLE_Den2DAll,"Efficiency/DLE/2DView/")
        writeToTFile(OutF,DLE_SummaryAll,"Efficiency/DLE/")
        writeToTFile(OutF,DLE_ByEta_Short,"Efficiency/DLE/ByEta/")
        writeToTFile(OutF,DLE_ByEta_Long,"Efficiency/DLE/ByEta/")
        writeToTFile(OutF,DLE_ByEta_All,"Efficiency/DLE/ByEta/")
        writeToTFile(OutF,DLE_ErrR,"Efficiency/DLE/PropError/")
        writeToTFile(OutF,DLE_ErrPhi,"Efficiency/DLE/PropError/")

    for r in [-1,1]:
        for l in [1,2]:
            endcapTag = EndcapLayer2label(r,l)

            efficiency2DPlot,Num2D,Den2D,Summary = generateEfficiencyPlot2DGE11(EfficiencyDictGlobal[matchingVar],r,l)
            efficiencyByEta_Short,efficiencyByEta_Long,efficiencyByEta_All =  generateEfficiencyPlotbyEta(EfficiencyDictGlobal[matchingVar],r,l)
                       
            c1,c2 = setUpCanvas(matchingVar+"_"+endcapTag,2400,900),setUpCanvas(matchingVar+"_VFAT2D_"+endcapTag,2000,2000)
            c1.SetLeftMargin(0.07)
            c1.SetRightMargin(0.09)                     
            c2.SetLeftMargin(0.1)
            c2.SetRightMargin(0.1)  
            c1.cd()                   

            writeToTFile(OutF,efficiency2DPlot,"Efficiency/"+matchingVar+"/2DView/"+endcapTag+"/")
            writeToTFile(OutF,Num2D,"Efficiency/"+matchingVar+"/2DView/"+endcapTag+"/")
            writeToTFile(OutF,Den2D,"Efficiency/"+matchingVar+"/2DView/"+endcapTag+"/")
            writeToTFile(OutF,generate2DEfficiencyPlotbyVFAT(EfficiencyDictVFAT[matchingVar],endcapTag),"Efficiency/"+matchingVar+"/ByVFAT/"+endcapTag)
            writeToTFile(OutF,Summary,"Efficiency/"+matchingVar+"/")


            if parameters["DLE"] and matchingVar=='glb_rdphi':
                DLE_2DPlot,DLE_Num2D,DLE_Den2D,DLE_Summary = generateEfficiencyPlot2DGE11(EfficiencyDictLayer[matchingVar],r,l)
                DLE_ByEta_Short,DLE_ByEta_Long,DLE_ByEta_All =  generateEfficiencyPlotbyEta(EfficiencyDictLayer[matchingVar],r,l)
                writeToTFile(OutF,DLE_2DPlot,"Efficiency/DLE/2DView/"+endcapTag+"/")
                writeToTFile(OutF,DLE_Num2D,"Efficiency/DLE/2DView/"+endcapTag+"/")
                writeToTFile(OutF,DLE_Den2D,"Efficiency/DLE/2DView/"+endcapTag+"/")
                writeToTFile(OutF,DLE_Summary,"Efficiency/DLE/"+endcapTag+"/")
                writeToTFile(OutF,DLE_ByEta_Short,"Efficiency/DLE/ByEta/"+endcapTag+"/")
                writeToTFile(OutF,DLE_ByEta_Long,"Efficiency/DLE/ByEta/"+endcapTag+"/")
                writeToTFile(OutF,DLE_ByEta_All,"Efficiency/DLE/ByEta/"+endcapTag+"/")
            
            ## Save plots
            for plot_obj in [efficiency2DPlot,Num2D,Den2D]:
                plot_obj.Draw("COLZ TEXT45")
                c1.Modified()
                c1.Update()
                c1.SaveAs(OUTPUT_PATH + "/PFA_Analyzer_Output/Plot/"+parameters["outputname"]+"/"+matchingVar+"/"+plot_obj.GetTitle()+".pdf")
            
            Summary.Draw("APE")
            c1.Modified()
            c1.Update()
            c1.SaveAs(OUTPUT_PATH + "/PFA_Analyzer_Output/Plot/"+parameters["outputname"]+"/"+matchingVar+"/"+Summary.GetTitle()+".pdf")

            c2.cd()
            endcapVFAT2D = generate2DEfficiencyPlotbyVFAT(EfficiencyDictVFAT[matchingVar],endcapTag)
            endcapVFAT2D.Draw("COLZ TEXT")
            c2.Modified()
            c2.Update()
            c2.SaveAs(OUTPUT_PATH + "/PFA_Analyzer_Output/Plot/"+parameters["outputname"]+"/"+matchingVar+"/"+endcapVFAT2D.GetTitle()+".pdf")

            writeToTFile(OutF,efficiencyByEta_Short,"Efficiency/"+matchingVar+"/ByEta/"+endcapTag+"/")
            writeToTFile(OutF,efficiencyByEta_Long,"Efficiency/"+matchingVar+"/ByEta/"+endcapTag+"/")
            writeToTFile(OutF,efficiencyByEta_All,"Efficiency/"+matchingVar+"/ByEta/"+endcapTag+"/")

            for t_ch in range(1,37):
                current_chamber_ID = getChamberName(r,t_ch,l)
                writeToTFile(OutF,generate2DEfficiencyPlotbyVFAT(EfficiencyDictVFAT[matchingVar],current_chamber_ID),"Efficiency/"+matchingVar+"/ByVFAT/"+endcapTag+"/"+current_chamber_ID)
                writeToTFile(OutF,generate1DEfficiencyPlotbyVFAT(EfficiencyDictVFAT[matchingVar],current_chamber_ID),"Efficiency/"+matchingVar+"/ByVFAT/"+endcapTag+"/"+current_chamber_ID)
                #writeToTFile(OutF,THSanityChecks['nMuons'][current_chamber_ID]["All"],"SanityChecks/nMuons/"+endcapTag+"/"+current_chamber_ID)
                for t_eta in list(range(1,9)) + ["All"]:
                    writeToTFile(OutF,THSanityChecks['pt'][current_chamber_ID][t_eta],"SanityChecks/pt/"+endcapTag+"/"+current_chamber_ID)
                    writeToTFile(OutF,TH1Fresidual_collector_x[matchingVar][endcapTag][current_chamber_ID][t_eta]["Residual"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/Residualx")
                    writeToTFile(OutF,TH1Fresidual_collector_x_minus[matchingVar][endcapTag][current_chamber_ID][t_eta]["Residual"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/Residualx_minus")
                    writeToTFile(OutF,TH1Fresidual_collector_x_plus[matchingVar][endcapTag][current_chamber_ID][t_eta]["Residual"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/Residualx_plus")
                   

                    writeToTFile(OutF,TH2Fresidual_collector_x[matchingVar][endcapTag][current_chamber_ID][t_eta]["Residual"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/Residualx2D")
                    writeToTFile(OutF,TH2Fresidual_collector_x_minus[matchingVar][endcapTag][current_chamber_ID][t_eta]["Residual"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/Residualx2D_minus")
                    writeToTFile(OutF,TH2Fresidual_collector_x_plus[matchingVar][endcapTag][current_chamber_ID][t_eta]["Residual"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/Residualx2D_plus")

                    writeToTFile(OutF,TH1Fresidual_collector_x_ptgt30[matchingVar][endcapTag][current_chamber_ID][t_eta]["Residual"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/Residualx_ptgt30")
                    writeToTFile(OutF,TH1Fresidual_collector_x_minus_ptgt30[matchingVar][endcapTag][current_chamber_ID][t_eta]["Residual"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/Residualx_minus_ptgt30")
                    writeToTFile(OutF,TH1Fresidual_collector_x_plus_ptgt30[matchingVar][endcapTag][current_chamber_ID][t_eta]["Residual"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/Residualx_plus_ptgt30")
                    
                    writeToTFile(OutF,TH1Fresidual_collector_x_ptlt30[matchingVar][endcapTag][current_chamber_ID][t_eta]["Residual"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/Residualx_ptlt30")
                    writeToTFile(OutF,TH1Fresidual_collector_x_minus_ptlt30[matchingVar][endcapTag][current_chamber_ID][t_eta]["Residual"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/Residualx_minus_ptlt30")
                    writeToTFile(OutF,TH1Fresidual_collector_x_plus_ptlt30[matchingVar][endcapTag][current_chamber_ID][t_eta]["Residual"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/Residualx_plus_ptlt30")
                    for cls in range(1,20): 
                        writeToTFile(OutF,TH1Fresidual_collector_CLS_x[cls][matchingVar][endcapTag][current_chamber_ID][t_eta]["Residual"],"Residuals/MatchingOn_"+matchingVar+"/"+"cls"+str(cls)+"/"+endcapTag+"/"+current_chamber_ID+"/Residualx")
                    writeToTFile(OutF,TH1Fresidual_collector_y[matchingVar][endcapTag][current_chamber_ID][t_eta]["Residual"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/Residualy")
                    writeToTFile(OutF,TH1FpropError_collector["AllHits"][endcapTag][current_chamber_ID][t_eta]["ErrPhi"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/PropagationError/AllHits")
                    writeToTFile(OutF,TH1FpropError_collector["AllHits"][endcapTag][current_chamber_ID][t_eta]["ErrR"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/PropagationError/AllHits")

                    ## feature unavailable for glb_phi
                    if matchingVar == "glb_rdphi":
                        writeToTFile(OutF,TH1FpropError_collector["MatchedHits"][endcapTag][current_chamber_ID][t_eta]["ErrR"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/PropagationError/MatchedHits")
                        writeToTFile(OutF,TH1FpropError_collector["MatchedHits"][endcapTag][current_chamber_ID][t_eta]["ErrPhi"],"Residuals/MatchingOn_"+matchingVar+"/"+endcapTag+"/"+current_chamber_ID+"/PropagationError/MatchedHits")

    writeToTFile(OutF,SummaryAll,"Efficiency/"+matchingVar+"/")
    writeToTFile(OutF,EffiDistrAll,"Efficiency/"+matchingVar+"/")
    writeToTFile(OutF,EffiDistrNeg,"Efficiency/"+matchingVar+"/")
    writeToTFile(OutF,EffiDistrPos,"Efficiency/"+matchingVar+"/")


    writeToTFile(OutF,THSanityChecks['Occupancy'][matchingVar]['AfterMatching']['Reco'],"SanityChecks/Occupancy/AfterMatching_"+matchingVar+"/")
    writeToTFile(OutF,THSanityChecks['Occupancy'][matchingVar]['AfterMatching']['Prop'],"SanityChecks/Occupancy/AfterMatching_"+matchingVar+"/") 
    writeToTFile(OutF,THSanityChecks['Occupancy'][matchingVar]['AfterMatching']['PropLocalLong'],"SanityChecks/Occupancy/AfterMatching_"+matchingVar+"/") 
    writeToTFile(OutF,THSanityChecks['Occupancy'][matchingVar]['AfterMatching']['PropLocalShort'],"SanityChecks/Occupancy/AfterMatching_"+matchingVar+"/") 


for key in TH1MetaData.keys():
    writeToTFile(OutF,TH1MetaData[key],"Metadata/")

printSummary(EfficiencyDictGlobal,matching_variables,ResidualCutOff,matching_variable_units,debug=parameters["verbose"])

## CSV
for matchingVar in matching_variables:
    tempList = []
    tempList_byVFAT = []
    tempList_byDLE = []
    for etaPID,subDict in EfficiencyDictGlobal[matchingVar].items():
        station,region,chamber,layer,eta = convert_etaPID(etaPID)
        
        matchedRecHit = sum([subDict[k]['num'] for k in subDict.keys()])
        propHit = sum([subDict[k]['den'] for k in subDict.keys()])
        chID = getChamberName(region,chamber,layer)

        AVG_CLS = THSanityChecks['CLS']["aftermatching"][chID][eta].GetMean()
        AVG_pt = THSanityChecks['pt'][chID][eta].GetMean()
        tempList.append([chID,region,chamber,layer,eta,matchedRecHit,propHit,AVG_CLS,AVG_pt])
            
    for (re,la) in [(1,1),(1,2),(-1,1),(-1,2)]:
        endcap_key = EndcapLayer2label(re,la)
        for chID,subDict in EfficiencyDictVFAT[matchingVar][endcap_key].items():
            for VFATN in subDict.keys():
                tempList_byVFAT.append([chID,endcap_key,VFATN,subDict[VFATN]['num'],subDict[VFATN]['den']])

    data = pd.DataFrame(tempList,columns=['chamberID',"region","chamber","layer","etaPartition","matchedRecHit","propHit","AVG_CLS","AVG_pt"])
    data.to_csv(OUTPUT_PATH + '/PFA_Analyzer_Output/CSV/'+parameters["outputname"]+'/MatchingSummary_'+matchingVar+'.csv', index=False)
    
    data_byVFAT = pd.DataFrame(tempList_byVFAT,columns=['chamberID',"EndcapTag","VFATN","matchedRecHit","propHit"])
    data_byVFAT.to_csv(OUTPUT_PATH + '/PFA_Analyzer_Output/CSV/'+parameters["outputname"]+'/MatchingSummary_'+matchingVar+'_byVFAT.csv', index=False)
if parameters["DLE"]:
    for etaPID,subDict in EfficiencyDictLayer['glb_rdphi'].items():
        station,region,chamber,layer,eta = convert_etaPID(etaPID)

        matchedRecHit = sum([subDict[k]['num'] for k in subDict.keys()])
        propHit = sum([subDict[k]['den'] for k in subDict.keys()])
        chID = getChamberName(region,chamber,layer)
        tempList_byDLE.append([chID,region,chamber,layer,eta,matchedRecHit,propHit])
    data_byDLE = pd.DataFrame(tempList_byDLE,columns=['chamberID',"region","chamber","layer","etaPartition","matchedRecHit","propHit"])
    data_byDLE.to_csv(OUTPUT_PATH + '/PFA_Analyzer_Output/CSV/'+parameters["outputname"]+'/MatchingSummary_glb_rdphi_byDLE.csv', index=False)

OutF.Close()
print(f"\n#############\nOUTPUT\n#############")
print(f"\tCSVs in \t {OUTPUT_PATH}/PFA_Analyzer_Output/CSV/{parameters['outputname']}")
print(f"\tROOT_File \t {OUTPUT_PATH}/PFA_Analyzer_Output/ROOT_File/{parameters['outputname']}.root")
# print "\tMatchingTTree File \t"+"./Output/PFA_Analyzer_Output/ROOT_File/MatchingTree_"+outputname+".root"
print(f"\tPlots in\t{OUTPUT_PATH}/PFA_Analyzer_Output/Plot/{parameters['outputname']}")
