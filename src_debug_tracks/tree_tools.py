import sys
import math
import ROOT
from array import array
from ROOT import TFile, TTree, TLorentzVector
import numpy as np
from podio import root_io
import edm4hep
import ctypes


class MomentumVector:
    def __init__(self, px, py, pz):
        self.vector = ROOT.TVector3(px, py, pz)

    @property
    def x(self):
        return self.vector.X()

    @property
    def y(self):
        return self.vector.Y()

    @property
    def z(self):
        return self.vector.Z()

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
        1: {"recojet_isU": 1, "recojet_isD": 0, "recojet_isS": 0, "recojet_isC": 0, "recojet_isB": 0, "recojet_isTAU": 0, "recojet_isG": 0},
        2: {"recojet_isU": 0, "recojet_isD": 1, "recojet_isS": 0, "recojet_isC": 0, "recojet_isB": 0, "recojet_isTAU": 0, "recojet_isG": 0},
        3: {"recojet_isU": 0, "recojet_isD": 0, "recojet_isS": 1, "recojet_isC": 0, "recojet_isB": 0, "recojet_isTAU": 0, "recojet_isG": 0},
        4: {"recojet_isU": 0, "recojet_isD": 0, "recojet_isS": 0, "recojet_isC": 1, "recojet_isB": 0, "recojet_isTAU": 0, "recojet_isG": 0},
        5: {"recojet_isU": 0, "recojet_isD": 0, "recojet_isS": 0, "recojet_isC": 0, "recojet_isB": 1, "recojet_isTAU": 0, "recojet_isG": 0},
        15: {"recojet_isU": 0, "recojet_isD": 0, "recojet_isS": 0, "recojet_isC": 0, "recojet_isB": 0, "recojet_isTAU": 1, "recojet_isG": 0},
        21: {"recojet_isU": 0, "recojet_isD": 0, "recojet_isS": 0, "recojet_isC": 0, "recojet_isB": 0, "recojet_isTAU": 0, "recojet_isG": 1},
    }
    return particle_map.get(number, {"recojet_isU": 0, "recojet_isD": 0, "recojet_isS": 0, "recojet_isC": 0, "recojet_isB": 0, "recojet_isTAU": 0, "recojet_isG": 0})

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

# MC info helper

def get_MCparticle_ID(event, reco_collection_id, reco_index, min_frac=0.8):
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
    if np.any(track_weights>min_frac):
        best_match = np.argmax(track_weights)
        best_link_index = link_index[best_match]
        best_link = event.get("RecoMCTruthLink").at(int(best_link_index))
        MC_part = best_link.getSim()
        return MC_part
    elif np.any(cluster_weights>min_frac):
        best_match = np.argmax(cluster_weights)
        best_link_index = link_index[best_match]
        best_link = event.get("RecoMCTruthLink").at(int(best_link_index))
        MC_part = best_link.getSim()
        return MC_part
    else:
        return None

def mcpid_to_reco(ptype, pmom):
    """converts MC ptype pid to reco type pid"""
    num_chad = [3334, 3312, 3222, 3112, 2212, 411, 321, 211, 521, 1000010020, 1000010030, 1000020040] # change to 211
    num_nhad = [3322, 2112, 3122, 130, 310] # change to 2112
    if abs(ptype) in num_chad:
        new_ptype = np.sign(ptype) * 211
    elif abs(ptype) in num_nhad:
        new_ptype = 2112
    elif abs(ptype) in [11, 13]:
        if pmom<2: # low momenta can't be resolved, so set to 211 (Michele said so)
            new_ptype = np.sign(ptype) * 211
        else:
            new_ptype = ptype
    else:
        new_ptype = ptype
    return int(new_ptype)

def store_MC_info(event, dic, MC_part, particle):
    correct_matching = False
    if MC_part!=None: # if MC particle is found
        dic["pfcand_MCPID"].push_back(MC_part.getPDG()) # MC PID of particle
        dic["pfcand_nMCtrackerhits"].push_back(count_tracker_hits(event, MC_part)) # save number of tracker hits

        # artificial correction for track-cluster matching
        if (particle.getType() in [22, 2112]) and (abs(MC_part.getPDG()) in [3334, 3312, 3222, 3112, 2212, 411, 321, 211, 521, 1000010020, 1000010030, 1000020040]): # if reco neutral but MC charged
            correct_matching = True

        # debug info
        tlv_MC = TLorentzVector()
        tlv_MC.SetXYZM(MC_part.getMomentum().x, MC_part.getMomentum().y, MC_part.getMomentum().z, particle.getMass())
        dic["pfcand_MC_phi"].push_back(tlv_MC.Phi())


        # parents
        parents = MC_part.getParents()
        if parents.size() > 1:
            #print("# const in jet: ", particles_jet.size())
            #print("MC PID: ", MC_part.getPDG())
            #print(f"Particle has {parents.size()} parents.")
            #for p in range(parents.size()):
            #    print("ID: ", parents.at(p).getPDG())
            #    print("index: ", parents.at(p).getObjectID().index)
            if parents.size()==2 and MC_part.getPDG()==22:
                print("Photon with two parents from inital state...") # initial state radiation? 
                dic["pfcand_parent_ID"].push_back(-444)
                dic["pfcand_parent_index"].push_back(-444)
            else: 
                raise ValueError("Particle has more than one parent")
        elif parents.size() == 0:
            #raise ValueError("Particle has no parent. Particle ID: ", MC_part.getPDG()) # 22
            dic["pfcand_parent_ID"].push_back(-999)
            dic["pfcand_parent_index"].push_back(-999)
        else:
            parent = parents.at(0)
            #print(dir(parent)) # same as particle above
            parent_ID = parent.getPDG() # MC PID of parent
            dic["pfcand_parent_index"].push_back(parent.getObjectID().index)
            dic["pfcand_parent_ID"].push_back(parent_ID)

    else: # no MC particle found
        dic["pfcand_MCPID"].push_back(-999)
        dic["pfcand_parent_ID"].push_back(-999)
        dic["pfcand_parent_index"].push_back(-999)
        dic["pfcand_MC_phi"].push_back(-9)
        dic["pfcand_nMCtrackerhits"].push_back(-9)
    return dic, correct_matching

def get_MC_quark(event):
    """checks Higgs -> qq and return one of the MC quarks"""
    quarks = []
    higgs_pid = 25
    expected_flavors = {1, 2, 3, 4, 5, 15, 21}  # u, d, s, c, b, tau, g
    for MC_part in event.get("MCParticles"):
        if MC_part.getPDG() == higgs_pid:
            daughters = MC_part.getDaughters()
            if daughters.size() != 2:
                raise ValueError("Higgs has {} daughters. Expected 2.".format(daughters.size()))
            for daughter in daughters:
                if abs(daughter.getPDG()) in expected_flavors:
                    quarks.append(daughter)
                else:
                    raise ValueError("Higgs daughter has unexpected PID: {}".format(daughter.getPDG()))
    if len(quarks) != 2:
        raise ValueError("Expected 2 quarks from Higgs. Found {}.".format(len(quarks)))
    return quarks[0] # return one of the quarks, because PV is the same for both

# track params helper

def get_charge(omega, Bz=2):
    # from https://flc.desy.de/lcnotes/notes/localfsExplorer_read?currentPath=/afs/desy.de/group/flc/lcnotes/LC-DET-2006-004.pdf 
    return np.sign(Bz/omega)

def omega_to_pt(omega, Bz):
    # from Dolores
    c_light = 2.99792458e8
    a = c_light * 1e3 * 1e-15
    return a * Bz / abs(omega)

def get_p_from_track(trackstate, Bz = 2.0):
    # from Dolores
    mchp = 0.139570
    pt = omega_to_pt(trackstate.omega, Bz)
    phi = trackstate.phi
    pz = trackstate.tanLambda * pt
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    p = math.sqrt(px * px + py * py + pz * pz)
    energy = math.sqrt(p * p + mchp * mchp)
    theta = math.acos(pz / p)
    # print(p, theta, phi, energy)
    particle_momentum = MomentumVector(px, py, pz)
    return particle_momentum

def calculate_params_wrt_PV(track, primaryVertex, particle_p, particle_q, Bz=2):
    """
    Recalculate d0. Before it was calculated with respect to (0,0,0), now we will update it with respect to the primary vertex.
    Do it the same as in https://github.com/HEP-FCC/FCCAnalyses/blob/63d346103159c4fc88cdee7884e09b3966cfeca4/analyzers/dataframe/src/ReconstructedParticle2Track.cc#L64 (ReconstrctedParticle2Track.cc XPtoPar_dxy)
    I don't understand the maths behind this though...

    Returns d0, z0, phi, c, ct with respect to PV
    """ 
    cSpeed = 2.99792458e8 * 1.0e-9
    
    pv = ROOT.TVector3(primaryVertex.x, primaryVertex.y, primaryVertex.z) # primary vertex
    point_on_track = ROOT.TVector3(- track.D0 * np.sin(track.phi), track.D0 * np.cos(track.phi), track.Z0) # point on particle track
    x = point_on_track - pv # vector from primary vertex to point on track
    pt = np.sqrt(particle_p.x**2 + particle_p.y **2) # transverse momentum of particle
    a = - particle_q * Bz * cSpeed # Lorentz force on particle in magnetic field
    r2 = x.x()**2 + x.y()**2
    cross = x.x() * particle_p.y - x.y() * particle_p.x
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
    ct = particle_p.z / pt
    dot = x.x() * particle_p.x + x.y() * particle_p.y
    if dot>0:
        z0 = x.z() - st * ct
    else:
        z0 = x.z() + st * ct

    # calculate phi
    phi = np.arctan2((particle_p.y - a * x.x())/np.sqrt(discrim), (particle_p.x + a * x.y())/np.sqrt(discrim))

    return d0, z0, phi, c, ct

def caluclate_charged_track_params(dic, d0, z0, phi, ct, particle_momentum, jet_momentum, track):
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

    # significance
    in_sqrt = track.covMatrix[0] + track.covMatrix[9]
    if in_sqrt>0: # can caluclate sqrt?
        err3d = np.sqrt(in_sqrt) # error of distance of closest approach
        dic["pfcand_btagJetDistSig"].push_back(d_3d/err3d) # significance of distance of closest approach
        dic["pfcand_btagSip3dSig"].push_back(sip3d/np.sqrt(in_sqrt))
    else: # handle error -> dummy value
        dic["pfcand_btagJetDistSig"].push_back(-999)
        dic["pfcand_btagSip3dSig"].push_back(-999)
    return dic

def fill_neutrals_track_params(dic):
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
    dic["pfcand_omega"].push_back(-9)
    dic["pfcand_C"].push_back(-9)
    dic["pfcand_ct"].push_back(-9)
    dic["pfcand_btagSip2dVal"].push_back(-9)
    dic["pfcand_btagSip2dSig"].push_back(-200) # set significance to -200 so it's outside of the distribution (FCCAnalyses: -9)
    dic["pfcand_btagSip3dVal"].push_back(-9)
    dic["pfcand_btagSip3dSig"].push_back(-200)
    dic["pfcand_btagJetDistVal"].push_back(-9) # line 641 in https://github.com/HEP-FCC/FCCAnalyses/blob/d39a711a703244ee2902f5d2191ad1e2367363ac/analyzers/dataframe/src/JetConstituentsUtils.cc#L2 
    dic["pfcand_btagJetDistSig"].push_back(-200)
    return dic

def calculate_covariance_matrix(dic, track):
    """
    Covariance matrix of helix parameters - use indices like used in https://github.com/HEP-FCC/FCCAnalyses/blob/d39a711a703244ee2902f5d2191ad1e2367363ac/analyzers/dataframe/src/ReconstructedParticle2Track.cc#L355 
    NOTE: the calculation is in principle not fully correct because the cov matrix is calculated in the point of closest approach to the origin (0,0,0) and not to the primary vertex.
    """
    #print(track.covMatrix.shape) # 21 values: 6 dimensional covariance matrix with values stored in lower triangular form
    # see https://github.com/key4hep/EDM4hep/blob/997ab32b886899253c9bc61adea9a21b57bc5a21/edm4hep.yaml#L195C9-L200 
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

def count_tracker_hits(event, MCpart):
    count_hits = 0
    for collection in ["InnerTrackerBarrelCollection", "InnerTrackerEndcapCollection", "OuterTrackerBarrelCollection", "OuterTrackerEndcapCollection", "VertexBarrelCollection", "VertexEndcapCollection"]:
        for hit in event.get(collection):
            if hit.getMCParticle().getObjectID().index == MCpart.getObjectID().index:
                count_hits += 1
    return count_hits

def reco_track_ass_to_MC(event, MCpart):
    """
    Correct track-cluster matching. Sometimes the pfo is neutral although is was MC charged. 
    Add track info by hand
    """

    # find corresponding track to MC particle
    collection_id = MCpart.getObjectID().collectionID
    index = MCpart.getObjectID().index
    count = 0
    track_weights = []
    link_index = []
    for l, link in enumerate(event.get("MCTruthSiTracksLink")):
        """print(dir(link))
        'clone', 'getObjectID', 'getRec', 'getSim', 'getWeight', 'id', 'isAvailable', 'operator MCRecoTrackParticleAssociation', 'setRec', 'setSim', 'setWeight', 'unlink'"""
        mc = link.getSim()
        if mc.getObjectID().collectionID == collection_id and mc.getObjectID().index == index: # found link from MC particle to reco track
            wgt = link.getWeight()
            #print("found link with weight ", wgt)
            # if wgt > 0.1: # NO! As long as there is any track! 
            track_weights.append(wgt)
            link_index.append(l)
            count += 1

    track_weights = np.array(track_weights)
    link_index = np.array(link_index)
    # check if PF0 track, neutral oder lost is associated to the particle
    if count==0: # no reco track associated to MC particle
        return None
    elif count>0: # reco track found
        best_match = np.argmax(track_weights)
        best_link_index = link_index[best_match]
        best_link = event.get("MCTruthSiTracksLink").at(int(best_link_index))
        reco_track = best_link.getRec() # get reco track
        track = reco_track.getTrackStates().at(0)
        return track

# vertex info helper

def save_PV_info(dic, primaryVertex, event_number):
    dic["jet_PV_x"][0] = primaryVertex.x
    dic["jet_PV_y"][0] = primaryVertex.y
    dic["jet_PV_z"][0] = primaryVertex.z
    dic["jet_PV_id"][0] = event_number
    return dic

def V_info_dic(event, ev_num, collection):
    """
    Retrieve vertex information from event and store it in a dictionary.
    Calculating invariant mass: https://opendata-education.github.io/en_Physics/Exercises-with-open-data/Warming-up/Calculate-invariant-mass.html 
    """
    dic = {
        "V_id": [],
        "V_M": [],
        "V_x": [],
        "V_y": [],
        "V_z": []
        }
    for v, vertex in enumerate(event.get(collection)):
        ass_part = vertex.getAssociatedParticle()
        part = ass_part.getParticles()
        energies = np.array([p.getEnergy() for p in part])
        momenta = np.array([[p.getMomentum().x, p.getMomentum().y, p.getMomentum().z] for p in part])

        # Sum the energies and momenta
        E = np.sum(energies)
        p_x, p_y, p_z = np.sum(momenta, axis=0)
        M2 = E**2 - (p_x**2 + p_y**2 + p_z**2)
        if M2<0:
            dic["V_M"].append(-9)
        else:
            dic["V_M"].append(np.sqrt(M2))
        dic["V_id"].append(ev_num*100 + v+1) # set id for this vertex
        dic["V_x"].append(vertex.getPosition().x)
        dic["V_y"].append(vertex.getPosition().y)
        dic["V_z"].append(vertex.getPosition().z)
    return dic

def V_info(event, dic, p_index, j, V_dic, ev_num, collection):
    """find collection in CLD steering file: https://github.com/key4hep/CLDConfig/blob/main/CLDConfig/CLDReconstruction.xml#L1364 
    - BuildUpVerticies:https://github.com/lcfiplus/LCFIPlus/blob/39cf1736f3f05345dc67553bca0fcc0cf64be43e/src/process.cc#L150C6-L150C19 
    """
    if collection == "BuildUpVertices":
        t = "SV"
    elif collection == "BuildUpVertices_V0":
        t = "V0"
    else:
        raise ValueError(f"{collection} not a supported collection. Choose 'BuildUpVertices' or 'BuildUpVertices_V0'.")
    ismatch = 0
    v_match = 0
    for v, vertex in enumerate(event.get(collection)): # loop over all vertices
        
        ass_part = vertex.getAssociatedParticle()
        part = ass_part.getParticles()

        for i in range(part.size()):
            p = part.at(i)
            #print("I'm a ", p.getType())
            #print(p.getObjectID().collectionID) # 4196981182 -> PandoraPFOs -> this is my reco particle!
            index = p.getObjectID().index
            if index == p_index: # found particle as part of V0 vertex
                ismatch += 1
                v_match = v
    if ismatch == 1: 
        # find out v0 index
        if j==0:
            v0_ind = ev_num*100 + v_match + 1
        else:
            v0_ind = (ev_num-1)*100 + v_match + 1
        ind = np.where(np.array(V_dic["V_id"]) == v0_ind) # get correct vertex
        if ind[0].size>1 or ind[0].size==0:
            raise ValueError(f"Found {ind[0].size} indices instead of one.")
        dic[f"pfcand_{t}_x"].push_back(V_dic["V_x"][ind[0][0]])
        dic[f"pfcand_{t}_y"].push_back(V_dic["V_y"][ind[0][0]])
        dic[f"pfcand_{t}_z"].push_back(V_dic["V_z"][ind[0][0]]) 
        dic[f"pfcand_{t}_M"].push_back(V_dic["V_M"][ind[0][0]]) 
        dic[f"pfcand_{t}_id"].push_back(v0_ind)
        #print(f"FOUND {t} at: ", V_dic["V_x"][ind[0][0]], V_dic["V_y"][ind[0][0]], V_dic["V_z"][ind[0][0]])
    elif ismatch == 0: # if no V0, fill with -200/0
        dic[f"pfcand_{t}_x"].push_back(-200)
        dic[f"pfcand_{t}_y"].push_back(-200)
        dic[f"pfcand_{t}_z"].push_back(-200)
        dic[f"pfcand_{t}_M"].push_back(-200)
        dic[f"pfcand_{t}_id"].push_back(0)
    elif ismatch>1:
        raise ValueError(f"Found {ismatch} (more than 1) V0/secondary vertex (collection: {collection}) assosiated with one particle/PFO in jet")
    return dic

def save_MCPV_info(event, dic):
    MCquark = get_MC_quark(event)
    vertex = MCquark.getVertex()
    dic["jet_MCPV_x"][0] = vertex.x
    dic["jet_MCPV_y"][0] = vertex.y
    dic["jet_MCPV_z"][0] = vertex.z
    return dic

# set up dic

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
    # primary vertex info - once per jet
    jet_PV_x = array("f", [0])
    t.Branch("jet_PV_x", jet_PV_x, "jet_PV_x/F")
    jet_PV_y = array("f", [0])
    t.Branch("jet_PV_y", jet_PV_y, "jet_PV_y/F")
    jet_PV_z = array("f", [0])
    t.Branch("jet_PV_z", jet_PV_z, "jet_PV_z/F")
    jet_PV_id = array("i", [0])
    t.Branch("jet_PV_id", jet_PV_id, "jet_PV_id/I")
    jet_MCPV_x = array("f", [0])
    t.Branch("jet_MCPV_x", jet_MCPV_x, "jet_MCPV_x/F")
    jet_MCPV_y = array("f", [0])
    t.Branch("jet_MCPV_y", jet_MCPV_y, "jet_MCPV_y/F")
    jet_MCPV_z = array("f", [0])
    t.Branch("jet_MCPV_z", jet_MCPV_z, "jet_MCPV_z/F")
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
    pfcand_isEl = ROOT.std.vector("int")() #problem with NN: store as int not as bool
    t.Branch("pfcand_isEl", pfcand_isEl)
    pfcand_isMu = ROOT.std.vector("int")()
    t.Branch("pfcand_isMu", pfcand_isMu)
    pfcand_isGamma = ROOT.std.vector("int")()
    t.Branch("pfcand_isGamma", pfcand_isGamma)
    pfcand_isNeutralHad = ROOT.std.vector("int")()
    t.Branch("pfcand_isNeutralHad", pfcand_isNeutralHad)
    pfcand_isChargedHad = ROOT.std.vector("int")()
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
    pfcand_C = ROOT.std.vector("float")()
    t.Branch("pfcand_C", pfcand_C)
    pfcand_ct = ROOT.std.vector("float")()
    t.Branch("pfcand_ct", pfcand_ct)
    pfcand_omega = ROOT.std.vector("float")()
    t.Branch("pfcand_omega", pfcand_omega)
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
    # MC info
    pfcand_MCPID = ROOT.std.vector("int")()
    t.Branch("pfcand_MCPID", pfcand_MCPID)
    # secondary vertex info
    pfcand_V0_x = ROOT.std.vector("float")()
    t.Branch("pfcand_V0_x", pfcand_V0_x)
    pfcand_V0_y = ROOT.std.vector("float")()
    t.Branch("pfcand_V0_y", pfcand_V0_y)
    pfcand_V0_z = ROOT.std.vector("float")()
    t.Branch("pfcand_V0_z", pfcand_V0_z)
    pfcand_V0_M = ROOT.std.vector("float")()
    t.Branch("pfcand_V0_M", pfcand_V0_M)
    pfcand_V0_id = ROOT.std.vector("float")()
    t.Branch("pfcand_V0_id", pfcand_V0_id)
    pfcand_SV_x = ROOT.std.vector("float")()
    t.Branch("pfcand_SV_x", pfcand_SV_x)
    pfcand_SV_y = ROOT.std.vector("float")()
    t.Branch("pfcand_SV_y", pfcand_SV_y)
    pfcand_SV_z = ROOT.std.vector("float")()
    t.Branch("pfcand_SV_z", pfcand_SV_z)
    pfcand_SV_M = ROOT.std.vector("float")()
    t.Branch("pfcand_SV_M", pfcand_SV_M)
    pfcand_SV_id = ROOT.std.vector("float")()
    t.Branch("pfcand_SV_id", pfcand_SV_id)
    # debug phi peaks 
    pfcand_MC_phi = ROOT.std.vector("float")()
    t.Branch("pfcand_MC_phi", pfcand_MC_phi)
    pfcand_MC_phirel = ROOT.std.vector("float")()
    t.Branch("pfcand_MC_phirel", pfcand_MC_phirel)
    pfcand_parent_ID = ROOT.std.vector("int")()
    t.Branch("pfcand_parent_ID", pfcand_parent_ID)
    pfcand_parent_index = ROOT.std.vector("int")()
    t.Branch("pfcand_parent_index", pfcand_parent_index)
    # track-cluster matching for neutral pfos 
    pfcand_track_cluster_matching = ROOT.std.vector("int")()
    t.Branch("pfcand_track_cluster_matching", pfcand_track_cluster_matching) # 0: no correction (neutral), 1: correction (neutral), 2: couldn't apply correction (neutral), 3: charged (not applicable)
    pfcand_nMCtrackerhits = ROOT.std.vector("int")()
    t.Branch("pfcand_nMCtrackerhits", pfcand_nMCtrackerhits)

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
        "jet_PV_x": jet_PV_x,
        "jet_PV_y": jet_PV_y,
        "jet_PV_z": jet_PV_z,
        "jet_PV_id": jet_PV_id,
        "jet_MCPV_x": jet_MCPV_x,
        "jet_MCPV_y": jet_MCPV_y,
        "jet_MCPV_z": jet_MCPV_z,
        # pfcand parameters
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
        "pfcand_C": pfcand_C, 
        "pfcand_ct": pfcand_ct,
        "pfcand_omega": pfcand_omega,
        "pfcand_btagSip2dVal": pfcand_btagSip2dVal,
        "pfcand_btagSip2dSig": pfcand_btagSip2dSig,
        "pfcand_btagSip3dVal": pfcand_btagSip3dVal,
        "pfcand_btagSip3dSig": pfcand_btagSip3dSig,
        "pfcand_btagJetDistVal": pfcand_btagJetDistVal,
        "pfcand_btagJetDistSig": pfcand_btagJetDistSig,
        "pfcand_mtof": pfcand_mtof,
        "pfcand_dndx": pfcand_dndx,
        "pfcand_MCPID": pfcand_MCPID,
        "pfcand_V0_x": pfcand_V0_x,
        "pfcand_V0_y": pfcand_V0_y,
        "pfcand_V0_z": pfcand_V0_z,
        "pfcand_V0_M": pfcand_V0_M,
        "pfcand_V0_id": pfcand_V0_id,
        "pfcand_SV_x": pfcand_SV_x,
        "pfcand_SV_y": pfcand_SV_y,
        "pfcand_SV_z": pfcand_SV_z,
        "pfcand_SV_M": pfcand_SV_M,
        "pfcand_SV_id": pfcand_SV_id,
        #debug
        "pfcand_MC_phi": pfcand_MC_phi,
        "pfcand_MC_phirel": pfcand_MC_phirel,
        "pfcand_parent_ID": pfcand_parent_ID,
        "pfcand_parent_index": pfcand_parent_index,
        # track-cluster matching
        "pfcand_track_cluster_matching": pfcand_track_cluster_matching,
        "pfcand_nMCtrackerhits": pfcand_nMCtrackerhits
        
    }
    return (event_number, n_hit, n_part, dic, t)

def clear_dic(dic):
    scalars = ["jet_p", "jet_e", "jet_mass", "jet_nconst", "jet_npfcand", "jet_theta", "jet_phi", \
        "recojet_isG", "recojet_isU", "recojet_isD", "recojet_isS", "recojet_isC", "recojet_isB", "recojet_isTAU", \
        "jet_nmu", "jet_nel", "jet_ngamma", "jet_nnhad", "jet_nchad", "jet_PV_x", "jet_PV_y", "jet_PV_z", "jet_PV_id", "jet_MCPV_x", "jet_MCPV_y", "jet_MCPV_z"]
    for key in dic:
        if key in scalars:
            dic[key][0] = 0
        else:
            dic[key].clear()
    return dic

# main function:

def store_jet(event, debug, dic, event_number, t, H_to_xx):
    """The yets have the following args that can be accessed with dir(jets)
    ['addToClusters', 'addToParticleIDs', 'addToParticles', 'addToTracks', 'clone', 'clusters_begin',
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
    Bz = 2.0 # T
    

    # calculate PV
    for v, vertex in enumerate(event.get("PrimaryVertices")):
        if v>0:
            raise ValueError("More than one primary vertex")
        primaryVertex = vertex.getPosition()
        #print("Primary vertex: ", primaryVertex.x, primaryVertex.y, primaryVertex.z)

    V0_dic = V_info_dic(event, event_number[0], "BuildUpVertices_V0") # only calculate once
    SV_dic = V_info_dic(event, event_number[0], "BuildUpVertices")

    RefinedVertexJets = "RefinedVertexJets"

    flag_save_pv = True
    for j, jet in enumerate(event.get(RefinedVertexJets)): # loop over the two jets
        clear_dic(dic) # clear the dictionary for new jet

        # save primary vertex info
        if flag_save_pv:
            dic = save_PV_info(dic, primaryVertex, event_number[0])
            dic = save_MCPV_info(event, dic)
            flag_save_pv = False
        else: 
            dic = save_PV_info(dic, primaryVertex, event_number[0]-1) # save info with the same event number as the first jet
            dic = save_MCPV_info(event, dic) # MC PV is the same for both jets
        # save MC primary vertex info

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
            ['clone', 'clusters_begin', 'clusters_end', 'clusters_size', 'getCharge', 'getClusters', 'getCovMatrix', 'getEnergy', 'getGoodnessOfPID', 'getMass', 'getMomentum', 'getObjectID', 'getParticleIDUsed', 'getParticleIDs', 'getParticles', 'getReferencePoint', 'getStartVertex', 'getTracks', 'getType', 'id', 'isAvailable', 'isCompound', 'makeEmpty', 'particleIDs_begin', 'particleIDs_end', 'particleIDs_size', 'particles_begin', 'particles_end', 'particles_size', 'tracks_begin', 'tracks_end', 'tracks_size', 'unlink']
            """

            particle_momentum = particle.getMomentum()
            dic["pfcand_e"].push_back(particle.getEnergy())
            tlv_p = TLorentzVector() # initialize new TLorentzVector for particle
            tlv_p.SetXYZM(particle_momentum.x, particle_momentum.y, particle_momentum.z, particle.getMass())
            dic["pfcand_p"].push_back(np.sqrt(particle_momentum.x**2 + particle_momentum.y**2 + particle_momentum.z**2))
            #print("reco PID: ", particle.getType())
            dic["pfcand_theta"].push_back(tlv_p.Theta())
            dic["pfcand_phi"].push_back(tlv_p.Phi())
            dic["pfcand_type"].push_back(particle.getType())
            dic["pfcand_charge"].push_back(particle.getCharge()) 
            reco_particle_type = PDG_ID_to_bool_particles(particle.getType(), particle.getTracks().size()) # dictionary with the particle type (bool)
            for key in reco_particle_type:
                dic[key].push_back(reco_particle_type[key])
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
            MC_part = get_MCparticle_ID(event, reco_collection_id, reco_index)
            # print(dir(MC_part))'clone', 'daughters_begin', 'daughters_end', 'daughters_size', 'getCharge', 'getColorFlow', 'getDaughters', 'getEndpoint', 'getEnergy', 'getGeneratorStatus', 'getMass', 'getMomentum', 'getMomentumAtEndpoint', 'getObjectID', 'getPDG', 'getParents', 'getSimulatorStatus', 'getSpin', 'getTime', 'getVertex', 'hasLeftDetector', 'id', 'isAvailable', 'isBackscatter', 'isCreatedInSimulation', 'isDecayedInCalorimeter', 'isDecayedInTracker', 'isOverlay', 'isStopped', 'makeEmpty', 'parents_begin', 'parents_end', 'parents_size', 'unlink', 'vertexIsNotEndpointOfParent']
            dic, correct_matching = store_MC_info(event, dic, MC_part, particle)

            # Vertex info
            dic = V_info(event, dic, reco_index, j, V0_dic, event_number[0], "BuildUpVertices_V0")
            dic = V_info(event, dic, reco_index, j, SV_dic, event_number[0], "BuildUpVertices")

            # get tracks of each particle (should be only one track)
            tracks = particle.getTracks()
            #print(dir(tracks)) #['at', 'begin', 'empty', 'end', 'size']
            #print(dir(tracks.at(0))) ['clone', 'dxQuantities_begin', 'dxQuantities_end', 'dxQuantities_size', 'getChi2', 'getDEdx', 'getDEdxError', 'getDxQuantities', 'getNdf', 'getObjectID', 'getRadiusOfInnermostHit', 'getSubdetectorHitNumbers', 'getTrackStates', 'getTrackerHits', 'getTracks', 'getType', 'id', 'isAvailable', 'makeEmpty', 'subdetectorHitNumbers_begin', 'subdetectorHitNumbers_end', 'subdetectorHitNumbers_size', 'trackStates_begin', 'trackStates_end', 'trackStates_size', 'trackerHits_begin', 'trackerHits_end', 'trackerHits_size', 'tracks_begin', 'tracks_end', 'tracks_size', 'unlink']
            
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
                dic["pfcand_omega"].push_back(track.omega)
                d0, z0, phi, c, ct = calculate_params_wrt_PV(track, primaryVertex, particle_momentum, particle.getCharge(),  Bz=Bz)
                dic["pfcand_C"].push_back(c)
                dic["pfcand_ct"].push_back(ct)
                dic = caluclate_charged_track_params(dic, d0, z0, phi, ct, particle_momentum, jet_momentum, track)
                dic = calculate_covariance_matrix(dic, track)
                
                dic["pfcand_track_cluster_matching"].push_back(3) # charged particle
                
            elif tracks.size() == 0: # neutral particle -> no track -> set track variables to dummy value! 
                dic = fill_neutrals_track_params(dic)
                dic["pfcand_track_cluster_matching"].push_back(0) 
            else:
                raise ValueError("Particle has more than one track")
            
            # ignore these because CLD as no drift chamber
            dic["pfcand_mtof"].push_back(0)
            dic["pfcand_dndx"].push_back(0)
            
            # count number of particles in jet
            if reco_particle_type["pfcand_isMu"]:
                dic["jet_nmu"][0] += 1
            elif reco_particle_type["pfcand_isEl"]:
                dic["jet_nel"][0] += 1
            elif reco_particle_type["pfcand_isGamma"]:
                dic["jet_ngamma"][0] += 1
            elif reco_particle_type["pfcand_isNeutralHad"]:
                dic["jet_nnhad"][0] += 1
            elif reco_particle_type["pfcand_isChargedHad"]:
                dic["jet_nchad"][0] += 1  


        # this needs to be updates to fill the tree with the info as in the fastsim rootfile
        t.Fill()
        # because we want to go from an event based tree to a jet based tree for each jet we add an event
        event_number[0] += 1

    return dic, event_number, t
