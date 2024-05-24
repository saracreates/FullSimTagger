import sys
import math
import ROOT
from array import array
from ROOT import TFile, TTree
import numpy as np
from podio import root_io
import edm4hep


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
    
    # parameters from fast sim to be reproduced in full sim
    jet_p = ROOT.std.vector("float")()
    t.Branch("jet_p", jet_p, "jet_p/F")
    jet_E = ROOT.std.vector("float")()
    t.Branch("jet_E", jet_E, "jet_E/F")
    jet_mass = ROOT.std.vector("float")()
    t.Branch("jet_mass", jet_mass, "jet_mass/F")
    jet_nconst = ROOT.std.vector("int")()
    t.Branch("jet_nconst", jet_nconst, "jet_nconst/I")

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
        "jet_nconst": jet_nconst
    }
    return (event_number, n_hit, n_part, dic, t)


def clear_dic(dic):
    for key in dic:
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
    for j, jet in enumerate(event.get(RefinedVertexJets)):

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
        
        # fill branches of tree
        dic["jet_p"].push_back(np.sqrt(jet_momentum.x**2 + jet_momentum.y**2 + jet_momentum.z**2))
        dic["jet_E"].push_back(jet.getEnergy())
        dic["jet_mass"].push_back(jet.getMass())
        dic["jet_nconst"].push_back(particles_jet.size())
        
        # this needs to be updates to fill the tree with the info as in the fastsim rootfile
        t.Fill()

    return dic, event_number, t
