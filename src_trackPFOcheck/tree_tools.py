import sys
import math
import ROOT
from array import array
from ROOT import TFile, TTree, TLorentzVector
import numpy as np
from podio import root_io
import edm4hep
import ctypes

def which_H_process(filename: str) -> dict:
    """Uses the file name to determine which Higgs process is being simulated."""
    particle_map = {
        "Huu": {"recojet_isU": 1, "recojet_isD": 0, "recojet_isS": 0, "recojet_isC": 0, "recojet_isB": 0, "recojet_isTAU": 0, "recojet_isG": 0},
        "Hdd": {"recojet_isU": 0, "recojet_isD": 1, "recojet_isS": 0, "recojet_isC": 0, "recojet_isB": 0, "recojet_isTAU": 0, "recojet_isG": 0},
        "Hss": {"recojet_isU": 0, "recojet_isD": 0, "recojet_isS": 1, "recojet_isC": 0, "recojet_isB": 0, "recojet_isTAU": 0, "recojet_isG": 0},
        "Hcc": {"recojet_isU": 0, "recojet_isD": 0, "recojet_isS": 0, "recojet_isC": 1, "recojet_isB": 0, "recojet_isTAU": 0, "recojet_isG": 0},
        "Hbb": {"recojet_isU": 0, "recojet_isD": 0, "recojet_isS": 0, "recojet_isC": 0, "recojet_isB": 1, "recojet_isTAU": 0, "recojet_isG": 0},
        "Htautau": {"recojet_isU": 0, "recojet_isD": 0, "recojet_isS": 0, "recojet_isC": 0, "recojet_isB": 0, "recojet_isTAU": 1, "recojet_isG": 0},
        "Hgg": {"recojet_isU": 0, "recojet_isD": 0, "recojet_isS": 0, "recojet_isC": 0, "recojet_isB": 0, "recojet_isTAU": 0, "recojet_isG": 1},
    }
    return particle_map.get(filename, {"recojet_isU": 0, "recojet_isD": 0, "recojet_isS": 0, "recojet_isC": 0, "recojet_isB": 0, "recojet_isTAU": 0, "recojet_isG": 0})

def initialize(t):
    # MC truth jet IDs
    recojet_isG = array("B", [0])
    t.Branch("recojet_isG", recojet_isG, "recojet_isG/B")
    recojet_isU = array("B", [0])
    t.Branch("recojet_isU", recojet_isU, "recojet_isU/B")
    recojet_isD = array("B", [0])
    t.Branch("recojet_isD", recojet_isD, "recojet_isD/B")
    recojet_isS = array("B", [0])
    t.Branch("recojet_isS", recojet_isS, "recojet_isS/B")
    recojet_isC = array("B", [0])
    t.Branch("recojet_isC", recojet_isC, "recojet_isC/B")
    recojet_isB = array("B", [0])
    t.Branch("recojet_isB", recojet_isB, "recojet_isB/B")
    recojet_isTAU = array("B", [0])
    t.Branch("recojet_isTAU", recojet_isTAU, "recojet_isTAU/B")
    # charged particles
    mcpid = ROOT.std.vector("int")()
    t.Branch("mcpid", mcpid)
    mc_pfo_type = ROOT.std.vector("int")() # 0=lost, 1=neutral, 2=track
    t.Branch("mc_pfo_type", mc_pfo_type)
    recopid = ROOT.std.vector("int")()
    t.Branch("recopid", recopid)
    momentum = ROOT.std.vector("float")()
    t.Branch("momentum", momentum)
    theta = ROOT.std.vector("float")()
    t.Branch("theta", theta)

    

    dic = {
        # MC truth jet IDs
        "recojet_isG": recojet_isG,
        "recojet_isU": recojet_isU,
        "recojet_isD": recojet_isD,
        "recojet_isS": recojet_isS,
        "recojet_isC": recojet_isC,
        "recojet_isB": recojet_isB,
        "recojet_isTAU": recojet_isTAU,
        # charged particles
        "mcpid": mcpid,
        "mc_pfo_type": mc_pfo_type,
        "recopid": recopid,
        "momentum": momentum,
        "theta": theta
        
    }
    return (dic, t)

def clear_dic(dic):
    scalars = ["jet_p", "jet_e", "jet_mass", "jet_nconst", "jet_npfcand", "jet_theta", "jet_phi", \
        "recojet_isG", "recojet_isU", "recojet_isD", "recojet_isS", "recojet_isC", "recojet_isB", "recojet_isTAU", \
        "jet_nmu", "jet_nel", "jet_ngamma", "jet_nnhad", "jet_nchad"]
    for key in dic:
        if key in scalars:
            dic[key][0] = 0
        else:
            dic[key].clear()
    return dic

def store_event(event, dic, t, H_to_xx):
    # loop over all MC particles
    for p, MCpart in enumerate(event.get("MCParticles")):
        '''
        print(dir(MCpart))
        'addToDaughters', 'addToParents', 'clone', 'colorFlow', 'daughters_begin', 'daughters_end', 'daughters_size', 'endpoint', 'getCharge', 'getColorFlow', 'getDaughters', 'getEndpoint', 'getEnergy', 'getGeneratorStatus', 'getMass', 'getMomentum', 'getMomentumAtEndpoint', 'getObjectID', 'getPDG', 'getParents', 'getSimulatorStatus', 'getSpin', 'getTime', 'getVertex', 'hasLeftDetector', 'id', 'isAvailable', 'isBackscatter', 'isCreatedInSimulation', 'isDecayedInCalorimeter', 'isDecayedInTracker', 'isOverlay', 'isStopped', 'momentum', 'momentumAtEndpoint', 'operator MCParticle', 'parents_begin', 'parents_end', 'parents_size', 'setBackscatter', 'setCharge', 'setColorFlow', 'setCreatedInSimulation', 'setDecayedInCalorimeter', 'setDecayedInTracker', 'setEndpoint', 'setGeneratorStatus', 'setHasLeftDetector', 'setMass', 'setMomentum', 'setMomentumAtEndpoint', 'setOverlay', 'setPDG', 'setSimulatorStatus', 'setSpin', 'setStopped', 'setTime', 'setVertex', 'setVertexIsNotEndpointOfParent', 'set_bit', 'spin', 'unlink', 'vertex', 'vertexIsNotEndpointOfParent'
        '''
        # find chad, e, mu
        mcpid = MCpart.getPDG()
        if abs(mcpid) == 11 or abs(mcpid) == 13 or abs(mcpid) == 211: # found a charged particle
            if MCpart.getGeneratorStatus() == 1: #  undecayed particle, stable in the generator. From https://ilcsoft.desy.de/LCIO/current/doc/doxygen_api/html/classEVENT_1_1MCParticle.html 
                
                # requirements fullfilled, save attributes
                jet_type = which_H_process(H_to_xx) # use file name to determine which Higgs process is being simulated
                for key in jet_type:
                    dic[key][0] = jet_type[key]
                dic["mcpid"].push_back(mcpid) # save particle mc pid
                part_p = MCpart.getMomentum()
                dic["momentum"].push_back(np.sqrt(part_p.x**2 +part_p.y**2 + part_p.z**2)) # save particle momentum
                tlv_p = TLorentzVector() # initialize new TLorentzVector for particle
                tlv_p.SetXYZM(part_p.x, part_p.y, part_p.z, MCpart.getMass())
                dic["theta"].push_back(tlv_p.Theta()) # save theta of particle

                index = MCpart.getObjectID().index
                # loop over links to reco particles
                count = 0
                track_weights = []
                link_index = []
                
                for l, link in enumerate(event.get("MCTruthRecoLink")): # weights express what fraction of MC particle hits are used in this reco particle
                    mc = link.getSim()
                    mc_index_link = mc.getObjectID().index
                    if mc_index_link == index: # found link to MC particle that is connected to reco
                        wgt = link.getWeight()
                        trackwgt = (int(wgt)%10000)/1000
                        clusterwgt = (int(wgt)/10000)/1000
                        if trackwgt > 0.5 or clusterwgt > 0.5: # at least half of the MC hits are associated to the reco particle
                            track_weights.append(trackwgt)
                            link_index.append(l)
                            count += 1
                #print("# of links to MC particle: ", count) # 0,1,2
                track_weights = np.array(track_weights)
                link_index = np.array(link_index)
                # check if PF0 track, neutral oder lost is associated to the particle
                if count==0: # lost
                    dic["mc_pfo_type"].push_back(0)
                    dic["recopid"].push_back(-999)
                elif count>0: # reconstructed
                    if np.max(track_weights) > 0.5: # if half of MC track hits are reconstruced
                        dic["mc_pfo_type"].push_back(2)
                    else: # neutral particle
                        dic["mc_pfo_type"].push_back(1)
                    # extract reco PID
                    if count==1:
                        link = event.get("MCTruthRecoLink")[int(link_index[0])]
                        reco = link.getRec()
                        dic["recopid"].push_back(reco.getType())
                    else:
                        max_index = np.argmax(track_weights)
                        link = event.get("MCTruthRecoLink")[int(link_index[max_index])]
                        reco = link.getRec()
                        dic["recopid"].push_back(reco.getType())

                t.Fill()
    return dic, t