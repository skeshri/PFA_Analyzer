import ROOT
import glob
import os
import numpy as np
import math
import fnmatch
import CMS_lumi

def GetROOTType(obj):
    type_found = type(obj)
    if type_found == ROOT.TDirectoryFile:
        return "(TDirectory)"
    elif type_found == ROOT.TFile:
        return "(TFile)"
    elif type_found == ROOT.TTree:
        return "(TTree)"
    elif type_found == ROOT.TBranchElement:
        return "(TBranchElement)"
    elif type_found == ROOT.TBranch:
        return "(TBranch)"
    elif type_found == ROOT.TCanvas:
        return "(TCanvas)"
    elif type_found == ROOT.TH1D:
        return "(TH1D)"
    elif type_found == ROOT.TFitResult:
        return ("(TFitResult)")
    elif type_found == ROOT.TGraph2D:
        return ("(TFitResult)")
    elif type_found == ROOT.TGraph2D:
        return ("(TFitResult)")
    elif type_found == ROOT.TGraphErrors:
        return ("(TFitResult)")
    elif type_found == ROOT.TH1F:
        return ("(TH1F)")
    elif type_found == ROOT.TH2F:
        return ("(TH2F)")
    elif type_found == ROOT.TGraphAsymmErrors:
        return ("(TGraphAsymmErrors)")
    elif type_found == ROOT.TH2Poly:
        return("(TH2Poly)")
        

    else:
        print(type_found)
        return "(unknown)"

def FillTTree(inputFile,branchDescriptor, outputFile):
    f = ROOT.TFile(outputFile,"RECREATE")
    T = ROOT.TTree("t1","t1")
    nlines = T.ReadFile(inputFile,branchDescriptor)

    print(" found ", nlines, " points")
    T.Write()
    f.Close()

def SaveToFile(fileName, obj, recreate=False):
    if recreate:
        writeMode = "RECREATE"
    else:
        writeMode = "UPDATE"
    f = ROOT.TFile(fileName,writeMode)
    obj.Write(obj.GetTitle())
    f.Close()

    directory=os.path.dirname(os.path.abspath(fileName))
    objType = obj.IsA()
    if "TCanvas" in str(objType):
        name = str(obj.GetTitle()).replace(" ","_")
        realtive_filename =directory+"/"+name+".pdf"
        absolute_filename = os.path.abspath(realtive_filename)
        #absolute_filename = absolute_filename.replace("'","")
        obj.SaveAs(absolute_filename)



####### Maps functions from here https://root-forum.cern.ch/t/loop-over-all-objects-in-a-root-file/10807/5
def Map(tf, browsable_to, tpath=None):
    """
    Maps objets as dict[obj_name][0] using a TFile (tf) and TObject to browse.
    """
    m = {}
    for k in browsable_to.GetListOfKeys():
        n = k.GetName()
        if tpath == None:
            m[n] = [tf.Get(n)]
        else:
            m[n] = [tf.Get(tpath + "/" + n)]
    return m

def Expand_deep_TDirs(tf, to_map, tpath=None):
    """
    A recursive deep-mapping function that expands into TDirectory(ies)
    """
    names = sorted(to_map.keys())
    for n in names:
        if len(to_map[n]) != 1:
            continue
        if tpath == None:
            tpath_ = n
        else:
            tpath_ = tpath + "/" + n

        tobject = to_map[n][0]
        if type(tobject) is ROOT.TDirectoryFile:
            m = Map(tf, tobject, tpath_)
            to_map[n].append(m)
            Expand_deep_TDirs(tf, m, tpath_)

def Map_TFile(filename, deep_maps=None):
    """
    Maps an input file as TFile into a dictionary(ies) of objects and names.
    Structure: dict[name] =list(object, deeper dict)
    """
    if deep_maps == None:
        deep_maps = {}
    if not type(deep_maps) is dict:
        return deep_maps

    f = ROOT.TFile(filename)
    m = Map(f, f)
    Expand_deep_TDirs(f, m)

    deep_maps[filename] = [f]
    deep_maps[filename].append(m)

    return deep_maps

####### DEMO
#filename = "myroot.root"
#mp = Map_TFile(filename)
# Now mp is a dict that has this struct
#mp{filename} is a list
#mp{filename}[0] this level of folder name
#mp{filename}[1] are the  dict(s) that pooints to the sublevel
#mp{filename}[1] has as keys the sublevel object names
#mp{filename}[1] has as values  the sub-sublevel dict


def printMap (dictMap, lvl=0):
    for key0, value0 in dictMap.items():
        
        print('\t'*lvl,key0,GetROOTType(value0[0]))
        if len(value0)>1 and type(value0[1]) is dict:  ### There is a sublevel
            printMap(value0[1],lvl+1)
        else:
            for i in range(0,len(value0)):
                if type(value0[i]) == ROOT.TTree:
                    for branch in value0[i].GetListOfBranches():
                        print('\t'*(lvl+1),branch.GetTitle(),GetROOTType(branch))
                    


## Returns a list of the objects found that contain searched_string
## accepts wildcards
def getObjectNamesThatContains(dictMap, searched_string, lvl=0):
    outputList = []
    no_wildcard = searched_string.replace("*","")
    searched_chunks= searched_string.split('*')
    if '' in searched_chunks: searched_chunks.remove('')
    

    ## Check if the key of the dict contains any of the searched chunks
    ## If so, add to the output list
    for key0, value0 in dictMap.items():
        if any(x in key0 for x in searched_chunks):
            outputList.append(key0)
        if len(value0)>1 and type(value0[1]) is dict:  ### There is a sublevel
            outputList = outputList + getObjectNamesThatContains(value0[1], searched_string, lvl=lvl+1)
    
    ## Filter the output list by taking only those strings that match the searched_string with wildcard
    return fnmatch.filter(outputList,searched_string)

## Returns  object that has objName. If not found returns false
## accepts wildcards
def getObjects(dictMap, objectName,verbose=False):
    valid_names = getObjectNamesThatContains(dictMap, objectName)
    output = {}
    if verbose: print(f"Found {len(valid_names)} objects that match {objectName}")
    for v in valid_names:
        output[v] = getTFileObject(dictMap,v)
    return output


## Returns  object that has objName. If not found returns false
def getTFileObject(dictMap, objectName):
    ObjToReturn = False
    for key0, value0 in dictMap.items():
        if key0==objectName:
            ObjToReturn = value0[0]
            return ObjToReturn
        if len(value0)>1 and type(value0[1]) is dict:  ### There is a sublevel
           ObjToReturn= getTFileObject(value0[1],objectName)
           ## Object has been found during the recursive search. Exiting loop
           if ObjToReturn!= False:
               break
    return ObjToReturn

def getTFileFitResult(dictMap, fitobjName):
    Param1 = False
    Param2 = False
    
    for key0, value0 in dictMap.items():
        if key0==fitobjName:
            Param1 = value0[0].Parameter(1)
            Param2 = value0[0].Parameter(2)
            return Param1,Param2
        if len(value0)>1 and type(value0[1]) is dict:  ### There is a sublevel
            Param1,Param2 = getTFileFitResult(value0[1],fitobjName)

    return Param1,Param2


def setUpCanvas(canvasName="canvas",W_input=None,H_input=None):
    CMS_lumi.writeExtraText = 1
    CMS_lumi.extraText = "CMS Preliminary"


    iPos = 0
    if( iPos==0 ): CMS_lumi.relPosX = 0.12

    if W_input!= None:
        W_ref = W_input
    else:
        W_ref = 800

    if H_input!= None:
        H_ref = H_input
    else:
        H_ref = 600
    
    
    W = W_ref
    H  = H_ref

    iPeriod = 0
    # references for T, B, L, R
    T = 0.08*H_ref
    B = 0.12*H_ref
    L = 0.12*W_ref
    R = 0.04*W_ref


    ROOT.gStyle.SetEndErrorSize(0)
    ROOT.gStyle.SetLegendBorderSize(0)
    ROOT.gStyle.SetOptFit(0)  # Remove Fit stats from plot
    ROOT.gStyle.SetLabelSize(.04, "XY")

    canvas = ROOT.TCanvas(canvasName,canvasName,50,50,W,H)
    canvas.SetFillColor(0)
    canvas.SetBorderMode(0)
    canvas.SetFrameFillStyle(0)
    canvas.SetFrameBorderMode(0)
    canvas.SetLeftMargin( L/W )
    canvas.SetRightMargin( R/W )
    canvas.SetTopMargin( T/H )
    canvas.SetBottomMargin( B/H )
    canvas.SetTickx(0)
    canvas.SetTicky(0)

    #canvas.SetGridy()
    canvas.SetFillStyle(4000)
    return canvas


def fit_and_return_function(TGraph,xstart,xstop,seed_param_0=-1,seed_param_1=-1):

    if seed_param_0 != -1. and seed_param_1 != -1:
        fit_param_0 = seed_param_0
        fit_param_1 = seed_param_1
    else:
        fit_param_0 = 10**(-6)
        fit_param_1 = 0.032

    
    f1 = ROOT.TF1("f1", "[0]*exp(x*[1])",xstart,xstop)
    f1.SetParNames("Constant", "Exponent")
    f1.SetParameters(fit_param_0,fit_param_1)
    f1.SetLineColor(ROOT.kRed)
    f1.SetLineStyle(1)

    TGraph.Fit(f1,"RMBQ")
    TGraph.GetListOfFunctions().FindObject("f1").SetRange(xstart, xstop)
    return f1

def fit(TGraphError,xstart,xstop,color,linestyle=1,Normalized_Data=False,seed_param_0=-1.,seed_param_1=-1.):

    if seed_param_0 != -1. and seed_param_1 != -1:
        fit_param_0 = seed_param_0
        fit_param_1 = seed_param_1
    elif Normalized_Data :
        fit_param_0 = 1**(-10)
        fit_param_1 = 7*(10**(-3))
    else:
        fit_param_0 = 1**(-7)
        fit_param_1 = 7*(10**(-3))

    
    f1 = ROOT.TF1("f1", "[0]*exp(x*[1])",xstart,xstop)
    f1.SetParNames("Constant", "Exponent")
    f1.SetParameters(fit_param_0,fit_param_1)
    f1.SetLineColor(color)
    f1.SetLineStyle(linestyle)
    if linestyle != 1:
        f1.SetLineWidth(2)

    TGraphError.Fit(f1,"RME","SAME")
    TGraphError.GetListOfFunctions().FindObject("f1").SetRange(xstart, xstop)
    return f1.GetParameter(0),f1.GetParameter(1)

def refit(fit_loops,tgraph,x_min,x_max,fit_color,linestyle=6,Normalized_Data=False):
    constant, exponent = fit(tgraph,x_min,x_max,fit_color,linestyle,Normalized_Data)
    for i in range(1,fit_loops):
        constant, exponent = fit(tgraph,x_min,x_max,fit_color,linestyle,Normalized_Data,constant,exponent)

    return constant,exponent

def Generate_Uncertainity_Plot(dict_of_fit,reference_geometry):
    multigraph = ROOT.TMultiGraph()

    x_ref = dict_of_fit[reference_geometry][4]
    ey_ref = dict_of_fit[reference_geometry][6]
    n_ref = len(x_ref)
    y_ref = dict_of_fit[reference_geometry][5]
    ey_ref = np.asarray([ey_ref[i]/y_ref[i] for i in range(0,n_ref)],dtype=float)
    y = np.ones(n_ref)
    ex_ref = np.zeros(n_ref,dtype=float)
    constant_ref = dict_of_fit[reference_geometry][0]
    exponent_ref = dict_of_fit[reference_geometry][1]
    
    gerror = ROOT.TGraphErrors(n_ref,x_ref,y,ex_ref,ey_ref)
    gerror.SetMarkerSize(.4)
    myColor = dict_of_fit[reference_geometry][2]
    myMarker = dict_of_fit[reference_geometry][3]
    gerror.SetMarkerColor(myColor)
    gerror.SetLineColor(myColor)
    gerror.SetMarkerStyle(myMarker)
    gerror.SetLineStyle(6)
    gerror.SetLineWidth(3)
    multigraph.Add(gerror)
    for geometry,fit_list in dict_of_fit.items():
        if geometry == reference_geometry:
            continue
        x = dict_of_fit[geometry][4]
        n = len(x)
        y = dict_of_fit[geometry][5]
        constant = dict_of_fit[geometry][0]
        exponent = dict_of_fit[geometry][1]
        y = [(constant/constant_ref)*math.exp((exponent-exponent_ref)*x[i]) for i in range(0,n)]
        ey = dict_of_fit[geometry][6]
        ey = [ey[i]/y_ref[i] for i in range(0,n)]
        y = np.asarray(y,dtype=float)
        ey = np.asarray(ey,dtype=float)
        ex = np.zeros(n,dtype=float)
        gerror = ROOT.TGraphErrors(n,x,y,ex,ey)
        gerror.SetMarkerSize(0.4)
        myColor = dict_of_fit[geometry][2]
        myMarker = dict_of_fit[geometry][3]
        gerror.SetMarkerColor(myColor)
        gerror.SetLineColor(myColor)
        gerror.SetMarkerStyle(myMarker)
        gerror.SetLineStyle(6)
        gerror.SetLineWidth(3)
        multigraph.Add(gerror)

    return multigraph

## Save the ROOT obj in the TFile according to the path specified in directory. If a ROOT obj w/ same name exists, skip
def writeToTFile(file,obj,directory=None,forceUpdate=False):
    if directory != None:
        if bool(file.GetDirectory(directory)):
            Tdir = file.GetDirectory(directory)
        else:
            file.mkdir(directory)
            Tdir = file.GetDirectory(directory)
        Tdir.cd()
        
        # Force duplicate
        if forceUpdate:
            pass
        # Avoid duplicate
        else:
            if Tdir.GetListOfKeys().Contains(obj.GetName()): return

    else:
        file.cd()
    
    obj.Write()

def files_in_folder(folder,path_tag=None,filename_tag=None):
    files  = []
    for root, dirnames, filenames in os.walk('/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/'):
        if folder in root.split("/") and "GEMCommonNtuples" in root:
            for filename in fnmatch.filter(filenames,'*root'):
                if path_tag != "":
                    if path_tag is not None and path_tag not in root: continue
                if filename_tag is None or filename_tag in filename:
                    files.append(root+"/"+filename)
                else:
                    pass
        elif folder in root.split("/") and "GEMCommonNTuples" in root:
            for filename in fnmatch.filter(filenames,'*root'):
                if path_tag != "":
                    if path_tag is not None and path_tag not in root: continue
                if filename_tag is None or filename_tag in filename:
                    files.append(root+"/"+filename)
                else:
                    pass
    return files

# bool values are handled with custom type in the ntuples.
# pythonizing
def ROOTBitReferenceVector_to_BoolList(input):
    output = []
    for j in input:
        output.append(bool(j))
    return output


def HistoGausFit(h):
    rms = h.GetRMS()
    mean = h.GetMean()
    y_max = h.GetMaximum()
    x_range = ( mean - float(rms*4)/2 , mean + float(rms*4)/2 )
    
    g = ROOT.TF1("GausFit "+h.GetTitle(),"gaus",x_range[0],x_range[1])

    # g.SetParameter(0,y_max)
    # g.SetParameter(1,mean)
    # g.SetParameter(2,rms)
    g.SetLineColor(ROOT.kBlack)#h.GetLineColor()+1)
    g.SetLineStyle(7)
    g.SetLineStyle(h.GetLineWidth())

    h.Fit(g,"R")

    A,mu,sigma = g.GetParameter(0),g.GetParameter(1),g.GetParameter(2)
    #print(f"{h.GetTitle()} fit resutl:\n\tA = {A}\n\tmu = {mu}\n\tsigma = {sigma}")
    return g,A,mu,sigma


if __name__ == '__main__':
    ROOT.gROOT.SetBatch(True)
    mp = Map_TFile("/eos/cms/store/group/dpg_gem/comm_gem/P5_Commissioning/2022/GEMCommonNtuples/Muon/360018_ZMu/221023_204649/0000/MuDPGNtuple_1.root")
    printMap(mp)
    # fn = "Aggregate.root"
    # openFile = ROOT.TFile("Aggregate.root","update")
    # mp=Map_TFile(fn)
    # for s in range(1,6):
    #     h = mp[fn][1]["ResidualCLS"][1][f"Residuals ClusterSize{s}"][0]
    #     g,_,_,_ = HistoGausFit(h)
    
    #     writeToTFile(openFile,g,"ResidualCLS",True)    
    #     openFile.Write("",ROOT.TObject.kOverwrite)
#    input()
    
