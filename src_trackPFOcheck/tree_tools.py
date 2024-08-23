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

def get_MCparticle_ID(event, reco_collection_id, reco_index, minfrac = 0.7):
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
    if np.any(track_weights>minfrac):
        best_match = np.argmax(track_weights)
        best_link_index = link_index[best_match]
        best_link = event.get("RecoMCTruthLink").at(int(best_link_index))
        MC_part = best_link.getSim()
        return MC_part
    elif np.any(cluster_weights>minfrac):
        best_match = np.argmax(cluster_weights)
        best_link_index = link_index[best_match]
        best_link = event.get("RecoMCTruthLink").at(int(best_link_index))
        MC_part = best_link.getSim()
        return MC_part
    else:
        return None

def PFO_track_efficiency(event, dic, MCpart, find_curlers, i, min_frac = 0.5):
    dic["mcpid"].push_back(MCpart.getPDG()) # save particle mc pid
    part_p = MCpart.getMomentum()
    dic["momentum"].push_back(np.sqrt(part_p.x**2 +part_p.y**2 + part_p.z**2)) # save particle momentum
    tlv_p = TLorentzVector() # initialize new TLorentzVector for particle
    tlv_p.SetXYZM(part_p.x, part_p.y, part_p.z, MCpart.getMass())
    dic["theta"].push_back(tlv_p.Theta()) # save theta of particle
    dic["energy"].push_back(MCpart.getEnergy()) # save energy of particle
    

    index = MCpart.getObjectID().index
    # loop over links to reco particles
    count = 0
    track_weights = []
    link_index = []
    #print("---- new particle ----")
    for l, link in enumerate(event.get("MCTruthRecoLink")): # weights express what fraction of MC particle hits are used in this reco particle
        mc = link.getSim()
        mc_index_link = mc.getObjectID().index
        if mc_index_link == index: # found link to MC particle that is connected to reco
            wgt = link.getWeight()
            trackwgt = (int(wgt)%10000)/1000
            clusterwgt = (int(wgt)/10000)/1000
            #print(trackwgt)
            if trackwgt > min_frac or clusterwgt > min_frac: # at least half of the MC hits are associated to the reco particle
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
        if find_curlers:
            if np.sqrt(part_p.x**2 +part_p.y**2 + part_p.z**2)< 0.6 and tlv_p.Theta() < 1.67 and tlv_p.Theta() > 1.47:
                print(f"Curler found! (event {i})")
    elif count>0: # reconstructed
        if np.max(track_weights) > min_frac: # if half of MC track hits are reconstruced
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
    return dic

def track_efficiency(event, dic, MCpart):
    """'mc_track_found' can be 0,1,2 where 0 means not found, 1 means found, 2 means not reconstrucable bc #hits < 4"""

    # only consider MC particles that leave more than 4 hits in the detector
    # loop over these 6 collections: InnerTrackerBarrelCollection, InnerTrackerEndcapCollection, OuterTrackerBarrelCollection, OuterTrackerEndcapCollection, VertexBarrelCollection, VertexEndcapCollection
    # for hit in these collection -> getParticle() -> that's a MC -> check if same as mine, if yes, increase count!
    count_hits = 0
    for collection in ["InnerTrackerBarrelCollection", "InnerTrackerEndcapCollection", "OuterTrackerBarrelCollection", "OuterTrackerEndcapCollection", "VertexBarrelCollection", "VertexEndcapCollection"]:
        for hit in event.get(collection):
            if hit.getMCParticle().getObjectID().index == MCpart.getObjectID().index:
                count_hits += 1
    dic["n_trackerhits"].push_back(count_hits)
    if count_hits < 4:
        dic["mc_track_found"].push_back(2)
    else:

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
                reco_track = link.getRec()
                wgt = link.getWeight()
                if wgt > 0.5: # at least half of the MC hits are associated to the reco particle
                    track_weights.append(wgt)
                    link_index.append(l)
                    count += 1

                """print(dir(reco_track))
                'clone', 'dxQuantities_begin', 'dxQuantities_end', 'dxQuantities_size', 'getChi2', 'getDEdx', 'getDEdxError', 'getDxQuantities', 'getNdf', 'getObjectID', 'getRadiusOfInnermostHit', 'getSubdetectorHitNumbers', 'getTrackStates', 'getTrackerHits', 'getTracks', 'getType', 'id', 'isAvailable', 'makeEmpty', 'subdetectorHitNumbers_begin', 'subdetectorHitNumbers_end', 'subdetectorHitNumbers_size', 'trackStates_begin', 'trackStates_end', 'trackStates_size', 'trackerHits_begin', 'trackerHits_end', 'trackerHits_size', 'tracks_begin', 'tracks_end', 'tracks_size', 'unlink'"""
                #print(reco_track.getTracks().size()) # 0
                #print(reco_track.getType()) # ??? whats that? 122, 106, 26, 90, 104, 42 ...
                #t = reco_track.getTrackStates().at(0)
                #print(dir(t)) # 'covMatrix', 'location', 'omega', 'phi', 'referencePoint', 'tanLambda', 'time'
                #print(t.phi) # filled
        track_weights = np.array(track_weights)
        link_index = np.array(link_index)
        # check if PF0 track, neutral oder lost is associated to the particle
        if count==0: # lost
            dic["mc_track_found"].push_back(0)
        elif count>0: # reconstructed
            if np.max(track_weights) > 0.5: # if half of MC track hits are reconstruced, doubled here, see above
                dic["mc_track_found"].push_back(1)
            else: # track not truely reconstructed but splitted to different reco tracks
                dic["mc_track_found"].push_back(0)
    return dic

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
    # MC charged particles
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
    energy = ROOT.std.vector("float")()
    t.Branch("energy", energy)
    mc_track_found = ROOT.std.vector("int")()
    t.Branch("mc_track_found", mc_track_found)
    n_trackerhits = ROOT.std.vector("int")()
    t.Branch("n_trackerhits", n_trackerhits)
    # reco (pfo) charged particles
    pfo_recopid = ROOT.std.vector("int")()
    t.Branch("pfo_recopid", pfo_recopid)
    pfo_momentum = ROOT.std.vector("float")()
    t.Branch("pfo_momentum", pfo_momentum)
    pfo_theta = ROOT.std.vector("float")()
    t.Branch("pfo_theta", pfo_theta)
    pfo_energy = ROOT.std.vector("float")()
    t.Branch("pfo_energy", pfo_energy)
    pfo_MCpid = ROOT.std.vector("int")()
    t.Branch("pfo_MCpid", pfo_MCpid)
    pfo_track = ROOT.std.vector("int")()
    t.Branch("pfo_track", pfo_track)

    dic = {
        # MC truth jet IDs
        "recojet_isG": recojet_isG,
        "recojet_isU": recojet_isU,
        "recojet_isD": recojet_isD,
        "recojet_isS": recojet_isS,
        "recojet_isC": recojet_isC,
        "recojet_isB": recojet_isB,
        "recojet_isTAU": recojet_isTAU,
        # MC charged particles
        "mcpid": mcpid,
        "mc_pfo_type": mc_pfo_type,
        "recopid": recopid,
        "momentum": momentum,
        "theta": theta,
        "energy": energy,
        "mc_track_found": mc_track_found,
        "n_trackerhits": n_trackerhits,
        # reco charged particles
        "pfo_recopid": pfo_recopid,
        "pfo_momentum": pfo_momentum,
        "pfo_theta": pfo_theta,
        "pfo_energy": pfo_energy,
        "pfo_MCpid": pfo_MCpid,
        "pfo_track": pfo_track,
        
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

def store_event(event, dic, t, H_to_xx, i):
    find_curlers = False
    jet_type = which_H_process(H_to_xx) # use file name to determine which Higgs process is being simulated
    for key in jet_type:
        dic[key][0] = jet_type[key]

    # loop over all MC particles for efficiency
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


                # PFO track efficiency
                dic = PFO_track_efficiency(event, dic, MCpart, find_curlers, i, min_frac=0.1)

                # track efficiency
                dic = track_efficiency(event, dic, MCpart)
                #t.Fill() 


    # loop over reco particles for fake rate
    for p, recopart in enumerate(event.get("PandoraPFOs")):
        dic["pfo_recopid"].push_back(recopart.getType()) # save particle pfo reco pid
        part_p = recopart.getMomentum()
        dic["pfo_momentum"].push_back(np.sqrt(part_p.x**2 +part_p.y**2 + part_p.z**2)) # save particle momentum
        tlv_p = TLorentzVector() # initialize new TLorentzVector for particle
        tlv_p.SetXYZM(part_p.x, part_p.y, part_p.z, recopart.getMass())
        dic["pfo_theta"].push_back(tlv_p.Theta()) # save theta of particle
        dic["pfo_energy"].push_back(recopart.getEnergy()) # save energy of particle
        """ print(dir(recopart))
        'addToClusters', 'addToParticleIDs', 'addToParticles', 'addToTracks', 'clone', 'clusters_begin', 'clusters_end', 'clusters_size', 'covMatrix', 'getCharge', 'getClusters', 'getCovMatrix', 'getEnergy', 'getGoodnessOfPID', 'getMass', 'getMomentum', 'getObjectID', 'getParticleIDUsed', 'getParticleIDs', 'getParticles', 'getReferencePoint', 'getStartVertex', 'getTracks', 'getType', 'id', 'isAvailable', 'isCompound', 'momentum', 'operator ReconstructedParticle', 'particleIDs_begin', 'particleIDs_end', 'particleIDs_size', 'particles_begin', 'particles_end', 'particles_size', 'referencePoint', 'setCharge', 'setCovMatrix', 'setEnergy', 'setGoodnessOfPID', 'setMass', 'setMomentum', 'setParticleIDUsed', 'setReferencePoint', 'setStartVertex', 'setType', 'tracks_begin', 'tracks_end', 'tracks_size', 'unlink' """
        reco_collection_id = recopart.getObjectID().collectionID
        reco_index = recopart.getObjectID().index
        MC_pfo_part = get_MCparticle_ID(event, reco_collection_id, reco_index, minfrac = 0.7)
        if MC_pfo_part is not None:
            dic["pfo_MCpid"].push_back(MC_pfo_part.getPDG())
        else:
            dic["pfo_MCpid"].push_back(-999)
        # check if pfo has track assigned or not
        tracks = recopart.getTracks()
        if tracks.size() == 0:
            dic["pfo_track"].push_back(0)
        else:
            dic["pfo_track"].push_back(1)


    t.Fill() # fill once for each event

    return dic, t