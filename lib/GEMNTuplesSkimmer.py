import ROOT
import time
# from PFA_Analyzer_Utils import *
# from ROOT_Utils import *
import argparse
from argparse import RawTextHelpFormatter
import subprocess
ROOT.gROOT.SetBatch(True)


parser = argparse.ArgumentParser(
        description='''Scripts that: \n\t-Reads the GEMMuonNtuple\n\t-Skims them to take only the useful branches''',
        epilog="""Typical exectuion\n\t python GEMNTuplesSkimmer.py  --dataset 2Jul2021""",
        formatter_class=RawTextHelpFormatter
)
parser.add_argument('--dataset','-ds', type=str,help="TAG to the folder containing the NTuples to be analyzed",required=True,nargs='*')

args = parser.parse_args()

start_time = time.time()


## Input data
files = []

for folder in args.dataset:
    files = files + files_in_folder(folder)


n = len(files)
#print (files)
chain = ROOT.TChain("muNtupleProducer/MuDPGTree")
print(f"Chaining {n} files")
for index,fl in enumerate(files):
    chain.Add(fl)
    print(f"Chained {index+1}/{n} files")


## Prepping output file
OutF = ROOT.TFile("/eos/user/f/fivone/GEMNTuples/"+args.dataset[0]+".root","RECREATE")

## Selecting useful branches, skim the others
chain.SetBranchStatus("*", 0)
chain.SetBranchStatus("gemRecHit_cluster_size")
chain.SetBranchStatus("gemRecHit_nRecHits")
## PROPAGATED HITS
# chain.SetBranchStatus("mu_propagated_region")
# chain.SetBranchStatus("mu_propagated_chamber")
# chain.SetBranchStatus("mu_propagated_layer")
# chain.SetBranchStatus("mu_propagated_etaP")
# chain.SetBranchStatus("mu_propagatedGlb_r")
# chain.SetBranchStatus("mu_propagatedLoc_x")
# chain.SetBranchStatus("mu_propagatedLoc_x")
# chain.SetBranchStatus("mu_propagated_pt")


#### FLOWER EVENTS
# chain.SetBranchStatus("event_1stLast_L1A",1)
# chain.SetBranchStatus("event_2ndLast_L1A",1)
# chain.SetBranchStatus("event_3rdLast_L1A",1)
# chain.SetBranchStatus("event_4thLast_L1A",1)
# chain.SetBranchStatus("gemRecHit_region",1)
# chain.SetBranchStatus("gemRecHit_chamber",1)
# chain.SetBranchStatus("gemRecHit_layer",1)
# chain.SetBranchStatus("gemRecHit_etaPartition",1)
# chain.SetBranchStatus("gemRecHit_firstClusterStrip",1)




OutF.cd()
tree = ROOT.TTree()
print(f"Created output TTree")
tree = chain.CloneTree()
OutF.Write()
OutF.Close()

## Summary Plots
print(f"--- {(time.time() - start_time)} seconds ---")
print(f"\n/eos/user/f/fivone/GEMNTuples/{args.dataset[0]}.root")
