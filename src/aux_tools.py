import sys
import math
import ROOT
from array import array
from ROOT import TFile, TTree
import numpy as np
from podio import root_io
import edm4hep


c_light = 2.99792458e8
Bz_clic = 4.0
Bz_cld = 2.0
mchp = 0.139570


def omega_to_pt(omega, isclic):
    if isclic:
        Bz = Bz_clic
    else:
        Bz = Bz_cld
    a = c_light * 1e3 * 1e-15
    return a * Bz / abs(omega)


def track_momentum(trackstate, isclic=True):
    pt = omega_to_pt(trackstate.omega, isclic)
    phi = trackstate.phi
    pz = trackstate.tanLambda * pt
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    p = math.sqrt(px * px + py * py + pz * pz)
    energy = math.sqrt(p * p + mchp * mchp)
    theta = math.acos(pz / p)
    # print(p, theta, phi, energy)
    return p, theta, phi, energy, px, py, pz


def get_genparticle_daughters(i, mcparts):

    p = mcparts[i]
    daughters = p.getDaughters()
    daughter_positions = []
    # for j in range(p.daughters_begin, p.daughters_end):
    #     # print(j, daughters[j].index)
    #     daughter_positions.append(daughters[j].index)
    #     # break
    for daughter in daughters:
        daughter_positions.append(daughter.getObjectID().index)

    return daughter_positions


def find_pandora_cluster_of_hit(hit_index, hit_collection, cluster_collection):
    cluster_energy_found = 0
    for index_c, cluster in enumerate(cluster_collection):
        cluster_hits = cluster.getHits()
        cluster_energy = cluster.getEnergy()
        for index_h, hit in enumerate(cluster_hits):
            object_id_hit = hit.getObjectID()
            if (
                hit_index == object_id_hit.index
                and object_id_hit.collectionID == hit_collection
            ):
                pandora_cluster_index = index_c
                cluster_energy_found = cluster_energy
                break
            else:
                pandora_cluster_index = -1
                cluster_energy_found = 0
        if pandora_cluster_index >= 0:
            break
    return pandora_cluster_index, cluster_energy_found


def check_pandora_pfos(event):
    pandora_pfo = "PandoraPFOs"
    pandora_pfos_event = event.get(pandora_pfo)
    for index, pfo in enumerate(pandora_pfos_event):
        clusters_pfo = pfo.getClusters()
        for index_c, cluster in enumerate(clusters_pfo):
            cluster_hits = cluster.getHits()
            for index_h, hit in enumerate(cluster_hits):
        # clusters = pfo.getClusters()
        # for cluster in clusters:
        #     print("clusters", dir(cluster))
        #     print("id", cluster.getObjectID().index)
        #     cluster_energy = cluster.getEnergy()
        #     print("cluster energy", cluster_energy)
        break


def find_pandora_pfo_and_cluster_of_hit(
    hit_index, hit_collection, cluster_collection, pfo_collection
):
    pandora_cluster_index = -1
    pfo_index = -1
    cluster_energy_found = 0
    pfo_energy_found = 0
    reference_point_found = None
    momentum_found = None
    for index_pfo, pfo in enumerate(pfo_collection):
        # print("index pfo ", index_pfo)
        clusters_pfo = pfo.getClusters()
        pfo_energy = pfo.getEnergy()
        reference_point = pfo.getReferencePoint()
        momentum = pfo.getMomentum()
        for index_c, cluster in enumerate(clusters_pfo):
            # print("index cluster ", index_c)
            cluster_hits = cluster.getHits()
            cluster_energy = cluster.getEnergy()
            cluster_id = cluster.getObjectID().index
            for index_h, hit in enumerate(cluster_hits):
                object_id_hit = hit.getObjectID()
                if (
                    hit_index == object_id_hit.index
                    and object_id_hit.collectionID == hit_collection
                ):
                    pandora_cluster_index = cluster_id
                    cluster_energy_found = cluster_energy
                    pfo_energy_found = pfo_energy
                    reference_point_found = reference_point
                    momentum_found = momentum
                    pfo_index = index_pfo
                    break
                else:
                    pandora_cluster_index = -1
                    cluster_energy_found = 0
                    pfo_energy_found = 0
                    pfo_index = -1
            if pandora_cluster_index >= 0:
                break
        if pandora_cluster_index >= 0:
            break
    # print("PFO", pfo_index, pfo_energy_found)
    return (
        pandora_cluster_index,
        cluster_energy_found,
        pfo_energy_found,
        pfo_index,
        reference_point_found,
        momentum_found,
    )


def find_pandora_pfo_track(hit_index, hit_collection, pfo_collection):
    pandora_cluster_index = -1
    pandora_pfo_index = -1
    pfo_energy_found = 0
    pfo_momentum_found = None
    pfo_reference_point_found = None
    for index_pfo, pfo in enumerate(pfo_collection):
        tracks_pfo = pfo.getTracks()
        pfo_energy = pfo.getEnergy()
        pfo_momentum = pfo.getMomentum()
        pfo_reference_point = pfo.getReferencePoint()
        for index_t, track in enumerate(tracks_pfo):
            # print("index cluster ", index_c)
            track_index = track.getObjectID().index
            track_collection_id = track.getObjectID().collectionID

            if hit_index == track_index and track_collection_id == hit_collection:
                pandora_pfo_index = index_pfo
                pfo_energy_found = pfo_energy
                pfo_momentum_found = pfo_momentum
                pfo_reference_point_found = pfo_reference_point
                break
            else:
                pandora_pfo_index = -1
                pfo_energy_found = 0
                pfo_momentum_found = None
                pfo_reference_point_found = None
        if pandora_pfo_index >= 0:
            break
    # print(pandora_cluster_index, pfo_energy_found, pandora_pfo_index)
    return (
        pandora_cluster_index,
        pfo_energy_found,
        pandora_pfo_index,
        pfo_reference_point_found,
        pfo_momentum_found,
    )


def get_genparticle_parents(i, mcparts):

    p = mcparts[i]
    parents = p.getParents()
    # print(p.parents_begin(), p.parents_end())
    parent_positions = []
    # for j in range(p.parents_begin(), p.parents_end()):
    #     # print(j, daughters[j].index)
    #     parent_positions.append(parents[j].index)
    #     # break
    for parent in parents:
        parent_positions.append(parent.getObjectID().index)

    return parent_positions


def find_mother_particle(j, gen_part_coll):
    parent_p = j
    counter = 0
    while len(np.reshape(np.array(parent_p), -1)) < 1.5:
        if type(parent_p) == list:
            parent_p = parent_p[0]
        parent_p_r = get_genparticle_parents(
            parent_p,
            gen_part_coll,
        )
        pp_old = parent_p
        counter = counter + 1
        # if len(np.reshape(np.array(parent_p_r), -1)) < 1.5:
        #     print(parent_p, parent_p_r)
        parent_p = parent_p_r
    # if j != pp_old:
    #     print("old parent and new parent", j, pp_old)
    return pp_old


def find_gen_link(
    j,
    id,
    SiTracksMCTruthLink,
    # gen_link_indexmc,
    # gen_link_weight,
    genpart_indexes,
    calo=False,
    gen_part_coll=None,
):
    # print(id)
    gen_positions = []
    gen_weights = []
    for i, l in enumerate(SiTracksMCTruthLink):
        rec_ = l.getRec()
        object_id = rec_.getObjectID()
        index = object_id.index
        collectionID = object_id.collectionID
        # print(index, j, collectionID, id)
        if index == j and collectionID == id:
            # print(j, "found match")
            gen_positions.append(l.getSim().getObjectID().index)
            weight = l.getWeight()
            gen_weights.append(weight)

    indices = []
    for i, pos in enumerate(gen_positions):
        if pos in genpart_indexes:
            if calo:
                mother = find_mother_particle(genpart_indexes[pos], gen_part_coll)
                indices.append(mother)
                indices.append(genpart_indexes[pos])
            else:
                indices.append(genpart_indexes[pos])

    indices += [-1] * (5 - len(indices))
    gen_weights += [-1] * (5 - len(gen_weights))

    return indices, gen_weights
