import sys
import math
import ROOT
from array import array
from ROOT import TFile, TTree
import numpy as np
from podio import root_io
import edm4hep


def PDG_ID_to_bool(number: int) -> dict:
    """Following https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf """
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

def initialize(t):
    event_number = array("i", [0])
    n_hit = array("i", [0])
    n_part = array("i", [0])

    hit_chis = ROOT.std.vector("float")()
    hit_x = ROOT.std.vector("float")()
    hit_y = ROOT.std.vector("float")()
    hit_z = ROOT.std.vector("float")()
    hit_px = ROOT.std.vector("float")()
    hit_py = ROOT.std.vector("float")()
    hit_pz = ROOT.std.vector("float")()

    t.Branch("event_number", event_number, "event_number/I")
    t.Branch("n_hit", n_hit, "n_hit/I")
    t.Branch("n_part", n_part, "n_part/I")

    t.Branch("hit_chis", hit_chis)
    t.Branch("hit_x", hit_x)
    t.Branch("hit_y", hit_y)
    t.Branch("hit_z", hit_z)
    t.Branch("hit_px", hit_px)
    t.Branch("hit_py", hit_py)
    t.Branch("hit_pz", hit_pz)
    
    # Parameters from fast sim to be reproduced in full sim
    
    # Jet parameters
    jet_p = array("f", [0])
    t.Branch("jet_p", jet_p, "jet_p/F")
    jet_E = array("f", [0])
    t.Branch("jet_E", jet_E, "jet_E/F")
    jet_mass = array("f", [0])
    t.Branch("jet_mass", jet_mass, "jet_mass/F")
    jet_nconst = array("i", [0])
    t.Branch("jet_nconst", jet_nconst, "jet_nconst/I")
    jet_theta = array("f", [0])
    t.Branch("jet_theta", jet_theta, "jet_theta/F")
    jet_phi = array("f", [0])
    t.Branch("jet_phi", jet_phi, "jet_phi/F")
    # MC truth particle IDs
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
    

    dic = {
        "hit_chis": hit_chis,
        "hit_x": hit_x,
        "hit_y": hit_y,
        "hit_z": hit_z,
        "hit_px": hit_px,
        "hit_py": hit_py,
        "hit_pz": hit_pz,
        "jet_p": jet_p,
        "jet_E": jet_E,
        "jet_mass": jet_mass,
        "jet_nconst": jet_nconst,
        "jet_theta": jet_theta,
        "jet_phi": jet_phi,
        "recojet_isG": recojet_isG,
        "recojet_isU": recojet_isU,
        "recojet_isD": recojet_isD,
        "recojet_isS": recojet_isS,
        "recojet_isC": recojet_isC,
        "recojet_isB": recojet_isB,
        "recojet_isTAU": recojet_isTAU
    }
    return (event_number, n_hit, n_part, dic, t)

def clear_dic(dic):
    scalars = ["jet_p", "jet_E", "jet_mass", "jet_nconst", "jet_theta", "jet_phi", "recojet_isG", "recojet_isU", "recojet_isD", "recojet_isS", "recojet_isC", "recojet_isB", "recojet_isTAU"]
    for key in dic:
        if key in scalars:
            dic[key][0] = 0
        else:
            dic[key].clear()
    return dic


def store_jet(event, debug, dic, event_number, t):
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

    RefinedVertexJets = "RefinedVertexJets"

    if debug:
        print("")
    for j, jet in enumerate(event.get(RefinedVertexJets)): # loop over the two jets
        clear_dic(dic) # clear the dictionary for new jet

        # Use dir(jet) to print all available bindings
        # print(dir(jet))

        # break
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
        # because we want to go from an event based tree to a jet based tree for each jet we add an event
        event_number[0] += 1
        
        
        #print(jet.getCovMatrix()) # Covariance matrix but only filled with zeros 

        MC_type = PDG_ID_to_bool(jet.getType()) # returns a dictionary with the particle type
        for key in MC_type:
            dic[key][0] = MC_type[key]
        
        
        # calculate angles
        jet_theta = np.arcsin(np.sqrt(jet_momentum.x**2 + jet_momentum.y**2)/np.sqrt(jet_momentum.x**2 + jet_momentum.y**2 + jet_momentum.z**2))
        jet_phi = np.arccos(jet_momentum.x/np.sqrt(jet_momentum.x**2 + jet_momentum.y**2))
        
        # Fill branches of the tree
        dic["jet_p"][0] = np.sqrt(jet_momentum.x**2 + jet_momentum.y**2 + jet_momentum.z**2)
        dic["jet_E"][0] = jet.getEnergy()
        dic["jet_mass"][0] = jet.getMass()
        dic["jet_nconst"][0] = particles_jet.size()
        dic["jet_theta"][0] = jet_theta
        dic["jet_phi"][0] = jet_phi
        
        # this needs to be updates to fill the tree with the info as in the fastsim rootfile
        t.Fill()

    return dic, event_number, t
