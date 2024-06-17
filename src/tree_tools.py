import sys
import math
import ROOT
from array import array
from ROOT import TFile, TTree, TLorentzVector
import numpy as np
from podio import root_io
import edm4hep
import ctypes

cSpeed = 2.99792458e8 * 1.0e-9

def arccot(x, epsilon=1e-10):
    if abs(x) < epsilon:
        return np.pi / 2
    elif x > 0:
        return np.arctan(1 / x)
    else:  # x < 0
        return np.arctan(1 / x) + np.pi

def PDG_ID_to_bool(number: int) -> dict:
    """Maps the PDG ID to the particle type for jets using https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf """
    particle_map = {
        1: {"recojet_isU": True, "recojet_isD": False, "recojet_isS": False, "recojet_isC": False, "recojet_isB": False, "recojet_isTAU": False, "recojet_isG": False},
        2: {"recojet_isU": False, "recojet_isD": True, "recojet_isS": False, "recojet_isC": False, "recojet_isB": False, "recojet_isTAU": False, "recojet_isG": False},
        3: {"recojet_isU": False, "recojet_isD": False, "recojet_isS": True, "recojet_isC": False, "recojet_isB": False, "recojet_isTAU": False, "recojet_isG": False},
        4: {"recojet_isU": False, "recojet_isD": False, "recojet_isS": False, "recojet_isC": True, "recojet_isB": False, "recojet_isTAU": False, "recojet_isG": False},
        5: {"recojet_isU": False, "recojet_isD": False, "recojet_isS": False, "recojet_isC": False, "recojet_isB": True, "recojet_isTAU": False, "recojet_isG": False},
        15: {"recojet_isU": False, "recojet_isD": False, "recojet_isS": False, "recojet_isC": False, "recojet_isB": False, "recojet_isTAU": True, "recojet_isG": False},
        21: {"recojet_isU": False, "recojet_isD": False, "recojet_isS": False, "recojet_isC": False, "recojet_isB": False, "recojet_isTAU": False, "recojet_isG": True},
    }
    return particle_map.get(number, {"recojet_isU": False, "recojet_isD": False, "recojet_isS": False, "recojet_isC": False, "recojet_isB": False, "recojet_isTAU": False, "recojet_isG": False})

def which_H_process(filename: str) -> dict:
    """Uses the file name to determine which Higgs process is being simulated."""
    particle_map = {
        "Huu": {"recojet_isU": True, "recojet_isD": False, "recojet_isS": False, "recojet_isC": False, "recojet_isB": False, "recojet_isTAU": False, "recojet_isG": False},
        "Hdd": {"recojet_isU": False, "recojet_isD": True, "recojet_isS": False, "recojet_isC": False, "recojet_isB": False, "recojet_isTAU": False, "recojet_isG": False},
        "Hss": {"recojet_isU": False, "recojet_isD": False, "recojet_isS": True, "recojet_isC": False, "recojet_isB": False, "recojet_isTAU": False, "recojet_isG": False},
        "Hcc": {"recojet_isU": False, "recojet_isD": False, "recojet_isS": False, "recojet_isC": True, "recojet_isB": False, "recojet_isTAU": False, "recojet_isG": False},
        "Hbb": {"recojet_isU": False, "recojet_isD": False, "recojet_isS": False, "recojet_isC": False, "recojet_isB": True, "recojet_isTAU": False, "recojet_isG": False},
        "Htautau": {"recojet_isU": False, "recojet_isD": False, "recojet_isS": False, "recojet_isC": False, "recojet_isB": False, "recojet_isTAU": True, "recojet_isG": False},
        "Hgg": {"recojet_isU": False, "recojet_isD": False, "recojet_isS": False, "recojet_isC": False, "recojet_isB": False, "recojet_isTAU": False, "recojet_isG": True},
    }
    return particle_map.get(filename, {"recojet_isU": False, "recojet_isD": False, "recojet_isS": False, "recojet_isC": False, "recojet_isB": False, "recojet_isTAU": False, "recojet_isG": False})

def PDG_ID_to_bool_particles(number: int, ntracks: int) -> dict:
    """Maps the PDG ID to the particle type for particles in jet"""
    particle_map = {
        11: {"pfcand_isEl": True, "pfcand_isMu": False, "pfcand_isGamma": False, "pfcand_isNeutralHad": False, "pfcand_isChargedHad": False},
        -11: {"pfcand_isEl": True, "pfcand_isMu": False, "pfcand_isGamma": False, "pfcand_isNeutralHad": False, "pfcand_isChargedHad": False},
        13: {"pfcand_isEl": False, "pfcand_isMu": True, "pfcand_isGamma": False, "pfcand_isNeutralHad": False, "pfcand_isChargedHad": False},
        -13: {"pfcand_isEl": False, "pfcand_isMu": True, "pfcand_isGamma": False, "pfcand_isNeutralHad": False, "pfcand_isChargedHad": False},
        22: {"pfcand_isEl": False, "pfcand_isMu": False, "pfcand_isGamma": True, "pfcand_isNeutralHad": False, "pfcand_isChargedHad": False},
        #2112: {"pfcand_isEl": False, "pfcand_isMu": False, "pfcand_isGamma": False, "pfcand_isNeutralHad": True, "pfcand_isChargedHad": False},
        #211: {"pfcand_isEl": False, "pfcand_isMu": False, "pfcand_isGamma": False, "pfcand_isNeutralHad": False, "pfcand_isChargedHad": True},
        #-211: {"pfcand_isEl": False, "pfcand_isMu": False, "pfcand_isGamma": False, "pfcand_isNeutralHad": False, "pfcand_isChargedHad": True}
    } 
    # {211: 883, -211: 896, 22: 1921, 2112: 543, 11: 66, -11: 73, 13: 24, -13: 22}
    # Mappings for charged and neutral hadrons
    if number not in [-11, 11, -13, 13, 22]:
        if ntracks == 0:
            return {"pfcand_isEl": False, "pfcand_isMu": False, "pfcand_isGamma": False, "pfcand_isNeutralHad": True, "pfcand_isChargedHad": False}
        else:
            return {"pfcand_isEl": False, "pfcand_isMu": False, "pfcand_isGamma": False, "pfcand_isNeutralHad": False, "pfcand_isChargedHad": True}
    # mapping for leptons and photon
    return particle_map.get(number, {"pfcand_isEl": False, "pfcand_isMu": False, "pfcand_isGamma": False, "pfcand_isNeutralHad": False, "pfcand_isChargedHad": False})

def get_MCparticle_ID(event, dic, reco_collection_id, reco_index):
     # get MC particle ID - loop over all MC-Reco-Pairs
    count = 0
    track_weights = []
    cluster_weights = []
    link_index = []
    for l, link in enumerate(event.get("RecoMCTruthLink")):
        #print(dir(link)) #'clone', 'getObjectID', 'getRec', 'getSim', 'getWeight', 'id', 'isAvailable', 'operator MCRecoParticleAssociation', 'setRec', 'setSim', 'setWeight', 'unlink'
        reco_part = link.getRec()
        reco_collection_id_link = reco_part.getObjectID().collectionID
        reco_index_link = reco_part.getObjectID().index
        if (reco_collection_id_link == reco_collection_id) and (reco_index_link == reco_index): # if right collection & index
            MC_part = link.getSim() # Sim refers to MC
            #print(dir(MC_part)) # print all available bindings for MC_part
            """
            'clone', 'daughters_begin', 'daughters_end', 'daughters_size', 'getCharge', 'getColorFlow', 'getDaughters', 
            'getEndpoint', 'getEnergy', 'getGeneratorStatus', 'getMass', 'getMomentum', 'getMomentumAtEndpoint', 'getObjectID', 
            'getPDG', 'getParents', 'getSimulatorStatus', 'getSpin', 'getTime', 'getVertex', 'hasLeftDetector', 'id', 'isAvailable', 
            'isBackscatter', 'isCreatedInSimulation', 'isDecayedInCalorimeter', 'isDecayedInTracker', 'isOverlay', 'isStopped', 
            'makeEmpty', 'parents_begin', 'parents_end', 'parents_size', 'unlink', 'vertexIsNotEndpointOfParent'
            """
            wgt = link.getWeight()
            trackwgt = (int(wgt)%10000)/1000
            clusterwgt = (int(wgt)/10000)/1000
            track_weights.append(trackwgt)
            cluster_weights.append(clusterwgt)
            link_index.append(l)

            count += 1

    #print("count MC particles: ", count)
    track_weights = np.array(track_weights)
    cluster_weights = np.array(cluster_weights)
    link_index = np.array(link_index)
    # find best matching particle
    if np.any(track_weights>0.8):
        best_match = np.argmax(track_weights)
        best_link_index = link_index[best_match]
        best_link = event.get("RecoMCTruthLink").at(int(best_link_index))
        MC_part = best_link.getSim()
        dic["pfcand_MCPID"].push_back(MC_part.getPDG()) # save MC particle ID
    elif np.any(cluster_weights>0.8):
        best_match = np.argmax(cluster_weights)
        best_link_index = link_index[best_match]
        best_link = event.get("RecoMCTruthLink").at(int(best_link_index))
        MC_part = best_link.getSim()
        dic["pfcand_MCPID"].push_back(MC_part.getPDG())
    else:
        dic["pfcand_MCPID"].push_back(-999)
    return dic

def calculate_params_wrt_PV(track, primaryVertex, particle, Bz=2):
    """
    Recalculate d0. Before it was calculated with respect to (0,0,0), now we will update it with respect to the primary vertex.
    Do it the same as in https://github.com/HEP-FCC/FCCAnalyses/blob/63d346103159c4fc88cdee7884e09b3966cfeca4/analyzers/dataframe/src/ReconstructedParticle2Track.cc#L64 (ReconstrctedParticle2Track.cc XPtoPar_dxy)
    I don't understand the maths behind this though...

    Returns d0, z0, phi, c, ct with respect to PV
    """ 
    pv = ROOT.TVector3(primaryVertex.x, primaryVertex.y, primaryVertex.z) # primary vertex
    point_on_track = ROOT.TVector3(- track.D0 * np.sin(track.phi), track.D0 * np.cos(track.phi), track.Z0) # point on particle track
    x = point_on_track - pv # vector from primary vertex to point on track
    pt = np.sqrt(particle.getMomentum().x**2 + particle.getMomentum().y **2) # transverse momentum of particle
    a = - particle.getCharge() * Bz * cSpeed # Lorentz force on particle in magnetic field
    r2 = x.x()**2 + x.y()**2
    cross = x.x() * particle.getMomentum().y - x.y() * particle.getMomentum().x
    discrim = pt*pt - 2 * a * cross + a*a * r2

    # calculate d0
    if discrim>0:
        if pt<10.0:
            d0 = (np.sqrt(discrim) - pt) / a
        else:
            d0 = (-2 * cross + a * r2) / (np.sqrt(discrim) + pt)
    else:
        d0 = -9

    # calculate z0
    c = a/ (2 * pt)
    b = c * np.sqrt(np.max([r2 - d0*d0, 0])/ (1 + 2 * c * d0))
    if abs(b)>1:
        b = np.sign(b)
    st = np.arcsin(b) / c
    ct = particle.getMomentum().z / pt
    dot = x.x() * particle.getMomentum().x + x.y() * particle.getMomentum().y
    if dot>0:
        z0 = x.z() - st * ct
    else:
        z0 = x.z() + st * ct

    # calculate phi
    phi = np.arctan2((particle.getMomentum().y - a * x.x())/np.sqrt(discrim), (particle.getMomentum().x + a * x.y())/np.sqrt(discrim))

    return d0, z0, phi, c, ct

def calculate_covariance_matrix(dic, track):
    """
    Covariance matrix of helix parameters - use indices like used in https://github.com/HEP-FCC/FCCAnalyses/blob/d39a711a703244ee2902f5d2191ad1e2367363ac/analyzers/dataframe/src/ReconstructedParticle2Track.cc#L355 
    NOTE: the calculation is in principle not fully correct because the cov matrix is calculated in the point of closest approach to the origin (0,0,0) and not to the primary vertex.
    """
    #print(track.covMatrix.shape) # 21 values: 6 dimensional covariance matrix with values stored in lower triangular form
    # diagonal: d0 = xy, phi, omega = pt, z0, tanLambda = eta
    dic["pfcand_dxydxy"].push_back(track.covMatrix[0]) 
    dic["pfcand_dphidphi"].push_back(track.covMatrix[2]) 
    dic["pfcand_dptdpt"].push_back(track.covMatrix[5]) # omega 
    dic["pfcand_dzdz"].push_back(track.covMatrix[9]) 
    dic["pfcand_detadeta"].push_back(track.covMatrix[14]) # tanLambda 
    
    dic["pfcand_dxydz"].push_back(track.covMatrix[6]) 
    dic["pfcand_dphidxy"].push_back(track.covMatrix[1])  
    dic["pfcand_dlambdadz"].push_back(track.covMatrix[13]) 
    dic["pfcand_dxyc"].push_back(track.covMatrix[3]) 
    dic["pfcand_dxyctgtheta"].push_back(track.covMatrix[10]) 
    
    dic["pfcand_phic"].push_back(track.covMatrix[4]) 
    dic["pfcand_phidz"].push_back(track.covMatrix[7])  
    dic["pfcand_phictgtheta"].push_back(track.covMatrix[11]) 
    dic["pfcand_cdz"].push_back(track.covMatrix[8])  
    dic["pfcand_cctgtheta"].push_back(track.covMatrix[12]) 
    return dic

        
def initialize(t):
    event_number = array("i", [0])
    n_hit = array("i", [0])
    n_part = array("i", [0])
    t.Branch("event_number", event_number, "event_number/I")
    t.Branch("n_hit", n_hit, "n_hit/I")
    t.Branch("n_part", n_part, "n_part/I")
    
    # Parameters from fast sim to be reproduced in full sim
    
    # Jet parameters
    jet_p = array("f", [0])
    t.Branch("jet_p", jet_p, "jet_p/F")
    jet_e = array("f", [0])
    t.Branch("jet_e", jet_e, "jet_e/F")
    jet_mass = array("f", [0])
    t.Branch("jet_mass", jet_mass, "jet_mass/F")
    jet_nconst = array("i", [0])
    t.Branch("jet_nconst", jet_nconst, "jet_nconst/I")
    jet_npfcand = array("i", [0])
    t.Branch("jet_npfcand", jet_npfcand, "jet_npfcand/I")
    jet_theta = array("f", [0])
    t.Branch("jet_theta", jet_theta, "jet_theta/F")
    jet_phi = array("f", [0])
    t.Branch("jet_phi", jet_phi, "jet_phi/F")
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
    # particles in jet
    pfcand_e = ROOT.std.vector("float")()
    t.Branch("pfcand_e", pfcand_e)
    pfcand_p = ROOT.std.vector("float")()
    t.Branch("pfcand_p", pfcand_p)
    pfcand_theta = ROOT.std.vector("float")()
    t.Branch("pfcand_theta", pfcand_theta)
    pfcand_phi = ROOT.std.vector("float")()
    t.Branch("pfcand_phi", pfcand_phi)
    pfcand_type = ROOT.std.vector("int")()
    t.Branch("pfcand_type", pfcand_type)
    pfcand_charge = ROOT.std.vector("float")()
    t.Branch("pfcand_charge", pfcand_charge)
    pfcand_isEl = ROOT.std.vector("bool")()
    t.Branch("pfcand_isEl", pfcand_isEl)
    pfcand_isMu = ROOT.std.vector("bool")()
    t.Branch("pfcand_isMu", pfcand_isMu)
    pfcand_isGamma = ROOT.std.vector("bool")()
    t.Branch("pfcand_isGamma", pfcand_isGamma)
    pfcand_isNeutralHad = ROOT.std.vector("bool")()
    t.Branch("pfcand_isNeutralHad", pfcand_isNeutralHad)
    pfcand_isChargedHad = ROOT.std.vector("bool")()
    t.Branch("pfcand_isChargedHad", pfcand_isChargedHad)
    #count number of particles in jet
    jet_nmu = array("i", [0])
    t.Branch("jet_nmu", jet_nmu, "jet_nmu/I")
    jet_nel = array("i", [0])
    t.Branch("jet_nel", jet_nel, "jet_nel/I")
    jet_ngamma = array("i", [0])
    t.Branch("jet_ngamma", jet_ngamma, "jet_ngamma/I")
    jet_nnhad = array("i", [0])
    t.Branch("jet_nnhad", jet_nnhad, "jet_nnhad/I")
    jet_nchad = array("i", [0])
    t.Branch("jet_nchad", jet_nchad, "jet_nchad/I")
    pfcand_erel_log = ROOT.std.vector("float")()
    t.Branch("pfcand_erel_log", pfcand_erel_log)
    pfcand_phirel = ROOT.std.vector("float")()
    t.Branch("pfcand_phirel", pfcand_phirel)
    pfcand_thetarel = ROOT.std.vector("float")()
    t.Branch("pfcand_thetarel", pfcand_thetarel)
    # Covariance parameters of helix!
    pfcand_dptdpt = ROOT.std.vector("float")()
    t.Branch("pfcand_dptdpt", pfcand_dptdpt)
    pfcand_detadeta = ROOT.std.vector("float")()
    t.Branch("pfcand_detadeta", pfcand_detadeta)
    pfcand_dphidphi = ROOT.std.vector("float")()
    t.Branch("pfcand_dphidphi", pfcand_dphidphi)
    pfcand_dxydxy = ROOT.std.vector("float")()
    t.Branch("pfcand_dxydxy", pfcand_dxydxy)
    pfcand_dzdz = ROOT.std.vector("float")()
    t.Branch("pfcand_dzdz", pfcand_dzdz)
    pfcand_dxydz = ROOT.std.vector("float")()
    t.Branch("pfcand_dxydz", pfcand_dxydz)
    pfcand_dphidxy = ROOT.std.vector("float")()
    t.Branch("pfcand_dphidxy", pfcand_dphidxy)
    pfcand_dlambdadz = ROOT.std.vector("float")()
    t.Branch("pfcand_dlambdadz", pfcand_dlambdadz)
    pfcand_dxyc = ROOT.std.vector("float")()
    t.Branch("pfcand_dxyc", pfcand_dxyc)
    pfcand_dxyctgtheta = ROOT.std.vector("float")()
    t.Branch("pfcand_dxyctgtheta", pfcand_dxyctgtheta)
    pfcand_phic = ROOT.std.vector("float")()
    t.Branch("pfcand_phic", pfcand_phic) 
    pfcand_phidz = ROOT.std.vector("float")()
    t.Branch("pfcand_phidz", pfcand_phidz)   
    pfcand_phictgtheta = ROOT.std.vector("float")()
    t.Branch("pfcand_phictgtheta", pfcand_phictgtheta)  
    pfcand_cdz = ROOT.std.vector("float")()
    t.Branch("pfcand_cdz", pfcand_cdz) 
    pfcand_cctgtheta = ROOT.std.vector("float")()
    t.Branch("pfcand_cctgtheta", pfcand_cctgtheta) 
    # discplacement paramters of the track  
    pfcand_dxy = ROOT.std.vector("float")()
    t.Branch("pfcand_dxy", pfcand_dxy)
    pfcand_dz = ROOT.std.vector("float")()
    t.Branch("pfcand_dz", pfcand_dz)
    pfcand_btagSip2dVal = ROOT.std.vector("float")()
    t.Branch("pfcand_btagSip2dVal", pfcand_btagSip2dVal)
    pfcand_btagSip2dSig = ROOT.std.vector("float")()
    t.Branch("pfcand_btagSip2dSig", pfcand_btagSip2dSig)
    pfcand_btagSip3dVal = ROOT.std.vector("float")()
    t.Branch("pfcand_btagSip3dVal", pfcand_btagSip3dVal)
    pfcand_btagSip3dSig = ROOT.std.vector("float")()
    t.Branch("pfcand_btagSip3dSig", pfcand_btagSip3dSig)
    pfcand_btagJetDistVal = ROOT.std.vector("float")()
    t.Branch("pfcand_btagJetDistVal", pfcand_btagJetDistVal)
    pfcand_btagJetDistSig = ROOT.std.vector("float")()
    t.Branch("pfcand_btagJetDistSig", pfcand_btagJetDistSig)
    pfcand_mtof = ROOT.std.vector("float")()
    t.Branch("pfcand_mtof", pfcand_mtof)
    pfcand_dndx = ROOT.std.vector("float")()
    t.Branch("pfcand_dndx", pfcand_dndx)
    pfcand_MCPID = ROOT.std.vector("int")()
    t.Branch("pfcand_MCPID", pfcand_MCPID)
   
    
    

    dic = {
        "jet_p": jet_p,
        "jet_e": jet_e,
        "jet_mass": jet_mass,
        "jet_nconst": jet_nconst,
        "jet_npfcand": jet_npfcand,
        "jet_theta": jet_theta,
        "jet_phi": jet_phi,
        "recojet_isG": recojet_isG,
        "recojet_isU": recojet_isU,
        "recojet_isD": recojet_isD,
        "recojet_isS": recojet_isS,
        "recojet_isC": recojet_isC,
        "recojet_isB": recojet_isB,
        "recojet_isTAU": recojet_isTAU,
        "pfcand_e": pfcand_e,
        "pfcand_p": pfcand_p,
        "pfcand_theta": pfcand_theta,
        "pfcand_phi": pfcand_phi,
        "pfcand_type": pfcand_type,
        "pfcand_charge": pfcand_charge,
        "pfcand_isEl": pfcand_isEl,
        "pfcand_isMu": pfcand_isMu,
        "pfcand_isGamma": pfcand_isGamma,
        "pfcand_isNeutralHad": pfcand_isNeutralHad,
        "pfcand_isChargedHad": pfcand_isChargedHad,
        "jet_nmu": jet_nmu,
        "jet_nel": jet_nel,
        "jet_ngamma": jet_ngamma,
        "jet_nnhad": jet_nnhad,
        "jet_nchad": jet_nchad,
        "pfcand_erel_log": pfcand_erel_log,
        "pfcand_phirel": pfcand_phirel,
        "pfcand_thetarel": pfcand_thetarel,
        "pfcand_dptdpt": pfcand_dptdpt,
        "pfcand_detadeta": pfcand_detadeta,
        "pfcand_dphidphi": pfcand_dphidphi,
        "pfcand_dxydxy": pfcand_dxydxy,
        "pfcand_dzdz": pfcand_dzdz,
        "pfcand_dxydz": pfcand_dxydz,
        "pfcand_dphidxy": pfcand_dphidxy,
        "pfcand_dlambdadz": pfcand_dlambdadz,
        "pfcand_dxyc": pfcand_dxyc,
        "pfcand_dxyctgtheta": pfcand_dxyctgtheta,
        "pfcand_phic": pfcand_phic,
        "pfcand_phidz": pfcand_phidz,
        "pfcand_phictgtheta": pfcand_phictgtheta,
        "pfcand_cdz": pfcand_cdz,
        "pfcand_cctgtheta": pfcand_cctgtheta,
        "pfcand_dxy": pfcand_dxy, # transverse impact parameter
        "pfcand_dz": pfcand_dz, # longitudinal impact parameter
        "pfcand_btagSip2dVal": pfcand_btagSip2dVal,
        "pfcand_btagSip2dSig": pfcand_btagSip2dSig,
        "pfcand_btagSip3dVal": pfcand_btagSip3dVal,
        "pfcand_btagSip3dSig": pfcand_btagSip3dSig,
        "pfcand_btagJetDistVal": pfcand_btagJetDistVal,
        "pfcand_btagJetDistSig": pfcand_btagJetDistSig,
        "pfcand_mtof": pfcand_mtof,
        "pfcand_dndx": pfcand_dndx,
        "pfcand_MCPID": pfcand_MCPID
        
    }
    return (event_number, n_hit, n_part, dic, t)

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


def store_jet(event, debug, dic, event_number, t, H_to_xx):
    """The jets have the following args that can be accessed with dir(jets)
    ['__add__', '__assign__', '__bool__', '__class__', '__delattr__', '__destruct__',
    '__dict__', '__dir__', '__dispatch__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__',
    '__gt__', '__hash__', '__init__', '__init_subclass__', '__invert__', '__le__', '__lt__', '__module__',
    '__mul__', '__ne__', '__neg__', '__new__', '__pos__', '__python_owns__', '__radd__', '__reduce__',
    '__reduce_ex__', '__repr__', '__rmul__', '__rsub__', '__rtruediv__', '__setattr__', '__sizeof__',
    '__smartptr__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__weakref__',
    'addToClusters', 'addToParticleIDs', 'addToParticles', 'addToTracks', 'clone', 'clusters_begin',
    'clusters_end', 'clusters_size', 'covMatrix', 'getCharge', 'getClusters', 'getCovMatrix', 'getEnergy',
    'getGoodnessOfPID', 'getMass', 'getMomentum', 'getObjectID', 'getParticleIDUsed', 'getParticleIDs',
    'getParticles', 'getReferencePoint', 'getStartVertex', 'getTracks', 'getType', 'id', 'isAvailable'
    , 'isCompound', 'momentum', 'operator ReconstructedParticle', 'particleIDs_begin', 'particleIDs_end'
    , 'particleIDs_size', 'particles_begin', 'particles_end', 'particles_size', 'referencePoint',
    'setCharge', 'setCovMatrix', 'setEnergy', 'setGoodnessOfPID', 'setMass', 'setMomentum',
    'setParticleIDUsed', 'setReferencePoint', 'setStartVertex', 'setType', 'tracks_begin',
    'tracks_end', 'tracks_size', 'unlink']

    Args:
        event (_type_): single event from the input rootfile
        debug (_type_): debug flat
        dic (_type_): dic with tree information for output root file

    Returns:
        _type_: _description_
    """

    # calculate PV
    for v, vertex in enumerate(event.get("PrimaryVertices")):
        if v>0:
            raise ValueError("More than one primary vertex")
        primaryVertex = vertex.getPosition()

    RefinedVertexJets = "RefinedVertexJets"

    if debug:
        print("")
    for j, jet in enumerate(event.get(RefinedVertexJets)): # loop over the two jets
        clear_dic(dic) # clear the dictionary for new jet

        # Extract the jet momentum
        jet_momentum = jet.getMomentum()
        #print(jet_momentum.x, jet_momentum.y)
        particles_jet = jet.getParticles()
        #print(dir(particles_jet)) # print all available bindings for particles_jet
        """['__add__', '__assign__', '__bool__', '__class__', '__delattr__', '__destruct__', '__dict__', '__dir__', '__dispatch__', 
        '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', 
        '__init_subclass__', '__invert__', '__iter__', '__le__', '__len__', '__lt__', '__module__', '__mul__', '__ne__', '__neg__', 
        '__new__', '__pos__', '__python_owns__', '__radd__', '__reduce__', '__reduce_ex__', '__repr__', '__rmul__', '__rsub__', 
        '__rtruediv__', '__setattr__', '__sizeof__', '__smartptr__', '__str__', '__sub__', '__subclasshook__', '__truediv__', 
        '__weakref__', 'at', 'begin', 'empty', 'end', 'size']
        """
        
        # calculate angles
        tlv = TLorentzVector()
        tlv.SetXYZM(jet_momentum.x, jet_momentum.y, jet_momentum.z, jet.getMass())
        jet_phi = tlv.Phi() # done like in ReconstructedParticle.cc in get_phi ( https://github.com/HEP-FCC/FCCAnalyses/blob/d39a711a703244ee2902f5d2191ad1e2367363ac/analyzers/dataframe/src/ReconstructedParticle.cc#L3 )
        jet_theta = tlv.Theta()
        
        # Fill branches of the tree
        dic["jet_p"][0] = np.sqrt(jet_momentum.x**2 + jet_momentum.y**2 + jet_momentum.z**2)
        dic["jet_e"][0] = jet.getEnergy()
        dic["jet_mass"][0] = jet.getMass()
        dic["jet_nconst"][0] = particles_jet.size()
        dic["jet_npfcand"][0] = particles_jet.size()
        dic["jet_theta"][0] = jet_theta
        dic["jet_phi"][0] = jet_phi
        # reco jet particle IDs
        #jet_type = PDG_ID_to_bool(jet.getType()) # returns a dictionary with the particle type
        jet_type = which_H_process(H_to_xx) # use file name to determine which Higgs process is being simulated
        for key in jet_type:
            dic[key][0] = jet_type[key]
            
        for i, part in enumerate(particles_jet):
            particle = particles_jet.at(i)
            #print("-------- new reco particle ------------")
            #print(dir(particle)) # print all available bindings for particle
            """
            ['clone', 'clusters_begin', 
            'clusters_end', 'clusters_size', 'getCharge', 'getClusters', 'getCovMatrix', 'getEnergy', 'getGoodnessOfPID', 
            'getMass', 'getMomentum', 'getObjectID', 'getParticleIDUsed', 'getParticleIDs', 'getParticles', 'getReferencePoint', 
            'getStartVertex', 'getTracks', 'getType', 'id', 'isAvailable', 'isCompound', 'makeEmpty', 'particleIDs_begin', 
            'particleIDs_end', 'particleIDs_size', 'particles_begin', 'particles_end', 'particles_size', 'tracks_begin', 'tracks_end', 
            'tracks_size', 'unlink']
            """
            #print("ref point particle: ", particle.getReferencePoint().x, particle.getReferencePoint().y, particle.getReferencePoint().z)



            particle_momentum = particle.getMomentum()
            dic["pfcand_e"].push_back(particle.getEnergy())
            tlv_p = TLorentzVector()
            tlv_p.SetXYZM(particle_momentum.x, particle_momentum.y, particle_momentum.z, particle.getMass())
            dic["pfcand_p"].push_back(np.sqrt(particle_momentum.x**2 + particle_momentum.y**2 + particle_momentum.z**2))
            #print("reco PID: ", particle.getType())
            dic["pfcand_theta"].push_back(tlv_p.Theta())
            dic["pfcand_phi"].push_back(tlv_p.Phi())
            dic["pfcand_type"].push_back(particle.getType())
            dic["pfcand_charge"].push_back(particle.getCharge()) 
            MC_particle_type = PDG_ID_to_bool_particles(particle.getType(), particle.getTracks().size()) # dictionary with the particle type (bool)
            for key in MC_particle_type:
                dic[key].push_back(MC_particle_type[key])
            # calculate relative values
            dic["pfcand_erel_log"].push_back(np.log10(dic["pfcand_e"][-1]/dic["jet_e"][0])) # like in JetConstituentsUtils.cc in get_erel_log_cluster ( https://github.com/HEP-FCC/FCCAnalyses/blob/d0abc8d76e37630ea157f9d5c48e7867a86be2e2/analyzers/dataframe/src/JetConstituentsUtils.cc#L4 line 877)
            # to calculate the angle correctly in 3D, roate the angle by the jet angle first
            tlv_p.RotateZ(-jet_phi) # rotate the particle by the jet angle in the xy-plane
            tlv_p.RotateY(-jet_theta) # rotate the particle by the jet angle in the xz-plane
            dic["pfcand_phirel"].push_back(tlv_p.Phi()) # same as in  rv::RVec<FCCAnalysesJetConstituentsData> get_phirel_cluster in https://github.com/HEP-FCC/FCCAnalyses/blob/d39a711a703244ee2902f5d2191ad1e2367363ac/analyzers/dataframe/src/JetConstituentsUtils.cc#L2 
            dic["pfcand_thetarel"].push_back(tlv_p.Theta()) # rel theta should be positive!  

            # calculate true PID of particles in jet
            reco_obj = particle.getObjectID()
            reco_collection_id = reco_obj.collectionID # 4196981182
            reco_index =  reco_obj.index
            dic = get_MCparticle_ID(event, dic, reco_collection_id, reco_index)

            
            # get tracks of each particle (should be only one track)
            tracks = particle.getTracks()
            #print(dir(tracks))
            """
            ['at', 'begin', 'empty', 'end', 'size']
            """
            #print(dir(tracks.at(0)))
            """
            ['clone', 'dxQuantities_begin', 'dxQuantities_end', 'dxQuantities_size', 
            'getChi2', 'getDEdx', 'getDEdxError', 'getDxQuantities', 'getNdf', 'getObjectID', 'getRadiusOfInnermostHit', 'getSubdetectorHitNumbers', 
            'getTrackStates', 'getTrackerHits', 'getTracks', 'getType', 'id', 'isAvailable', 'makeEmpty', 'subdetectorHitNumbers_begin', 
            'subdetectorHitNumbers_end', 'subdetectorHitNumbers_size', 'trackStates_begin', 'trackStates_end', 'trackStates_size', 'trackerHits_begin', 
            'trackerHits_end', 'trackerHits_size', 'tracks_begin', 'tracks_end', 'tracks_size', 'unlink']
            """
            if tracks.size() == 1: # charged particle
                #print(dir(tracks.at(0).getTrackStates().at(0))) # -> 'covMatrix', 'location', 'omega', 'phi', 'referencePoint', 'tanLambda', 'time'
                #print(track.getTrackStates().size()) 
                """static const int AtOther = 0 ; // any location other than the ones defined below\n
                static const int AtIP = 1 ;\n
                static const int AtFirstHit = 2 ;\n
                static const int AtLastHit = 3 ;\n
                static const int AtCalorimeter = 4 ;\n
                static const int AtVertex = 5 ;\n """
                #print("---- new charged track")
    
                track = tracks.at(0).getTrackStates().at(0) # info at IP
                """dir(track)
                'covMatrix', 'location', 'omega', 'phi', 'referencePoint', 'tanLambda', 'time'
                but look at https://github.com/key4hep/EDM4hep/blob/997ab32b886899253c9bc61adea9a21b57bc5a21/edm4hep.yaml#L192 for details of the TrackState object"""
                #print("track ref point: ", track.referencePoint.x, track.referencePoint.y, track.referencePoint.z) # (0,0,0) for IP


                # calculate impact parameter with respect to primary vertex
                d0, z0, phi, c, ct = calculate_params_wrt_PV(track, primaryVertex, particle, Bz=2)
                dic["pfcand_dxy"].push_back(d0) # transverse impact parameter
                dic["pfcand_dz"].push_back(z0) # longitudinal impact parameter
                # correct for phi
                dic["pfcand_phi"].pop_back() # remove the phi calculated with respect to (0,0,0)
                dic["pfcand_phi"].push_back(phi) # update phi with respect to PV
                # correct for theta
                dic["pfcand_theta"].pop_back() # remove the theta calculated with respect to (0,0,0)
                dic["pfcand_theta"].push_back(arccot(ct)) # update theta with respect to PV

                
                part_p = ROOT.TVector3(particle_momentum.x, particle_momentum.y, particle_momentum.z)
                jet_p = ROOT.TVector3(jet_momentum.x, jet_momentum.y, jet_momentum.z)
                
                # calculate d_3d as in FCCAnalysis, JetConstituentsUtils.cc in get_JetDistVal() https://github.com/HEP-FCC/FCCAnalyses/blob/d39a711a703244ee2902f5d2191ad1e2367363ac/analyzers/dataframe/src/JetConstituentsUtils.cc#L2
                n = part_p.Cross(jet_p).Unit() # distance of closest approach always in direction perpendicular to both (particle and jet). Problem: What if they are parallel?
                pt_ct = ROOT.TVector3(- d0 * np.sin(phi), d0 * np.cos(phi), z0) # point on particle track
                pt_jet = ROOT.TVector3(0,0,0) # point on jet
                d_3d = n.Dot(pt_ct - pt_jet) # distance of closest approach
                dic["pfcand_btagJetDistVal"].push_back(d_3d)
                # NOTE: error of distance of closest approach is calculated in the point of closest approach to the origin (0,0,0) and not to the primary vertex - this is in principle wrong!!!
                err3d = np.sqrt(track.covMatrix[0] + track.covMatrix[9]) # error of distance of closest approach
                dic["pfcand_btagJetDistSig"].push_back(d_3d/err3d) # significance of distance of closest approach
                
                
                # calculate signed 2D impact parameter - like in get_Sip2dVal_clusterV() in JetConstituentsUtils.cc (https://github.com/HEP-FCC/FCCAnalyses/blob/d39a711a703244ee2902f5d2191ad1e2367363ac/analyzers/dataframe/src/JetConstituentsUtils.cc#L450 )
                p_jet_2d = ROOT.TVector2(jet_momentum.x, jet_momentum.y)
                pt_ct_2d = ROOT.TVector2(pt_ct.x(), pt_ct.y())
                sip2d = np.sign(pt_ct_2d * p_jet_2d) * abs(d0) # if angle between track and jet greater 90 deg -> negative sign; if smaller 90 deg -> positive sign
                dic["pfcand_btagSip2dVal"].push_back(sip2d)
                dic["pfcand_btagSip2dSig"].push_back(sip2d/np.sqrt(track.covMatrix[0]))
                
                # calculate signed 3D impact parameter - like in get_Sip3Val() in JetConstituentsUtils.cc 
                IP_3d = np.sqrt(d0**2 + z0**2)
                sip3d = np.sign(pt_ct * jet_p) * abs(IP_3d) 
                dic["pfcand_btagSip3dVal"].push_back(sip3d)
                dic["pfcand_btagSip3dSig"].push_back(sip3d/np.sqrt(track.covMatrix[0] + track.covMatrix[9]) )
                
                dic = calculate_covariance_matrix(dic, track)
                
                
            elif tracks.size() == 0: # neutral particle -> no track -> set them to -9!
                dic["pfcand_dptdpt"].push_back(-9) # like in ReconstructedParticle2Track.cc line 336 ( https://github.com/HEP-FCC/FCCAnalyses/blob/d39a711a703244ee2902f5d2191ad1e2367363ac/analyzers/dataframe/src/ReconstructedParticle2Track.cc#L335 )
                dic["pfcand_detadeta"].push_back(-9)
                dic["pfcand_dphidphi"].push_back(-9)
                dic["pfcand_dxydxy"].push_back(-9)
                dic["pfcand_dzdz"].push_back(-9)
                dic["pfcand_dxydz"].push_back(-9)
                dic["pfcand_dphidxy"].push_back(-9)
                dic["pfcand_dlambdadz"].push_back(-9)
                dic["pfcand_dxyc"].push_back(-9)
                dic["pfcand_dxyctgtheta"].push_back(-9)
                dic["pfcand_phic"].push_back(-9)
                dic["pfcand_phidz"].push_back(-9)
                dic["pfcand_phictgtheta"].push_back(-9)
                dic["pfcand_cdz"].push_back(-9)
                dic["pfcand_cctgtheta"].push_back(-9)
                dic["pfcand_dxy"].push_back(-9)
                dic["pfcand_dz"].push_back(-9)
                dic["pfcand_btagSip2dVal"].push_back(-9)
                dic["pfcand_btagSip2dSig"].push_back(-200) # set significance to -200 so it's outside of the distribution (FCCAnalyses: -9)
                dic["pfcand_btagSip3dVal"].push_back(-9)
                dic["pfcand_btagSip3dSig"].push_back(-200)
                dic["pfcand_btagJetDistVal"].push_back(-9) # line 641 in https://github.com/HEP-FCC/FCCAnalyses/blob/d39a711a703244ee2902f5d2191ad1e2367363ac/analyzers/dataframe/src/JetConstituentsUtils.cc#L2 
                dic["pfcand_btagJetDistSig"].push_back(-200)
            else:
                raise ValueError("Particle has more than one track")
            
            # ignore these because CLD as no drift chamber
            dic["pfcand_mtof"].push_back(0)
            dic["pfcand_dndx"].push_back(0)
            
            # count number of particles in jet
            if MC_particle_type["pfcand_isMu"]:
                dic["jet_nmu"][0] += 1
            elif MC_particle_type["pfcand_isEl"]:
                dic["jet_nel"][0] += 1
            elif MC_particle_type["pfcand_isGamma"]:
                dic["jet_ngamma"][0] += 1
            elif MC_particle_type["pfcand_isNeutralHad"]:
                dic["jet_nnhad"][0] += 1
            elif MC_particle_type["pfcand_isChargedHad"]:
                dic["jet_nchad"][0] += 1   

            
            """ROOT.gInterpreter.Declare('#include <marlinutil/HelixClassT.h>')
            h = ROOT.HelixClassT["float"]()
            h.Initialize_Canonical(track.phi, track.D0, track.Z0, track.omega, track.tanLambda, 2.0) # 2 is B field strength in T
            distance_before = ROOT.std.vector("float")(3)
            d0 = dic["pfcand_dxy"][-1]
            z0 = dic["pfcand_dz"][-1]
            d3d = dic["pfcand_btagJetDistVal"][-1]
            distance_before[0] = d0 # d0
            distance_before[1] = z0 # z0
            distance_before[2] = d3d # 3D distance of closest approach
            #primaryVertex = [primaryVertex.x, primaryVertex.y, primaryVertex.z] 
            # Convert primaryVertex to the required format (array of floats)
            primaryVertex_array = (ctypes.c_float * 3)(*[primaryVertex.x, primaryVertex.y, primaryVertex.z] )

            # Convert distance_before to an array of floats
            distance_before_array = (ctypes.c_float * 3)(*distance_before)

            # Call getDistanceToPoint with the proper arguments
            distance = h.getDistanceToPoint(primaryVertex_array, distance_before_array)
            #distance = h.getDistanceToPoint(primaryVertex, distance_before)
            #print("distance: ", distance) #float...
            """
            
        # this needs to be updates to fill the tree with the info as in the fastsim rootfile
        t.Fill()
        # because we want to go from an event based tree to a jet based tree for each jet we add an event
        event_number[0] += 1

    return dic, event_number, t
