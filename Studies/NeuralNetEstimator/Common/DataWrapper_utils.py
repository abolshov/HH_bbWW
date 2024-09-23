import awkward as ak
import vector
import numpy as np


def Px(obj_p4):
    return obj_p4.px


def Py(obj_p4):
    return obj_p4.py


def Pz(obj_p4):
    return obj_p4.pz


def E(obj_p4):
    return obj_p4.E


def GetNumPyArray(awk_arr, tot_length, i):
    return ak.to_numpy(ak.fill_none(ak.pad_none(awk_arr[:, :tot_length], tot_length), 0.0))[:, i]


def EventTopology(branches, bb_res, qq_res):
    bb_resolved = ak.Array(np.ones(len(branches)) == 1)
    if bb_res:
        b1_p4 = vector.zip({'pt': branches['genb1_pt'], 
                            'eta': branches['genb1_eta'], 
                            'phi': branches['genb1_phi'], 
                            'mass': branches['genb1_mass']})

        b2_p4 = vector.zip({'pt': branches['genb2_pt'], 
                            'eta': branches['genb2_eta'], 
                            'phi': branches['genb2_phi'], 
                            'mass': branches['genb2_mass']})

        bb_dr = b1_p4.deltaR(b2_p4)
        bb_resolved = bb_dr >= 0.4


    qq_resolved = ak.Array(np.ones(len(branches)) == 1)
    if qq_res:
        q1_p4 = vector.zip({'pt': branches['genV2prod1_pt'], 
                            'eta': branches['genV2prod1_eta'], 
                            'phi': branches['genV2prod1_phi'], 
                            'mass': branches['genV2prod1_mass']})

        q2_p4 = vector.zip({'pt': branches['genV2prod2_pt'], 
                            'eta': branches['genV2prod2_eta'], 
                            'phi': branches['genV2prod2_phi'], 
                            'mass': branches['genV2prod2_mass']})

        qq_dr = q1_p4.deltaR(q2_p4)
        qq_resolved = qq_dr >= 0.4

    event_topology = qq_resolved & bb_resolved
    return event_topology
    