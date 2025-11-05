import numpy as np
import Analysis.hh_bbww as analysis
import FLAF.Common.Utilities as Utilities
import awkward as ak
import os

class BtagShapeProducer:
    def __init__(self, cfg, payload_name):
        self.payload_name = payload_name
        self.cfg = cfg
        self.jet_multiplicities = np.unique([int(col.split('_')[-1]) for col in self.cfg['columns']]) 
        if len(self.jet_multiplicities) == 0:
            raise RuntimeError(f'Some error message')
        self.vars_to_save = ['ncentralJet', 'weight_bTagShape_Central'] # probably also should come from cfg

    def run(self, array, keep_all_columns=False):
        ncentralJet = array['ncentralJet']
        btag_weights = array['weight_bTagShape_Central']
        res = {}
        for jet_multiplicity in self.jet_multiplicities:
            # calculate number of events and total btag shape weight
            events_with_jet_multiplicity_mask = array['ncentralJet'] == jet_multiplicity
            res[f'numEvents_ncentralJet_{jet_multiplicity}'] = int(ak.count_nonzero(events_with_jet_multiplicity_mask))
            res[f'totBtagWeight_ncentralJet_{jet_multiplicity}'] = float(np.sum(btag_weights[events_with_jet_multiplicity_mask]))

        return res