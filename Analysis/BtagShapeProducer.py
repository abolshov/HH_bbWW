import numpy as np
import Analysis.hh_bbww as analysis
import FLAF.Common.Utilities as Utilities
import awkward as ak
import os
import Analysis.hh_bbww as analysis

class BtagShapeProducer:
    def __init__(self, cfg, payload_name):
        self.payload_name = payload_name
        self.cfg = cfg
        self.jet_multiplicities = np.unique([int(col.split('_')[-1]) for col in self.cfg['columns']]) 
        if len(self.jet_multiplicities) == 0:
            raise RuntimeError(f'Some error message')
        self.vars_to_save = ['ncentralJet', 
                             'weight_noBtag', 
                             'weight_total',
                             'e',
                             'mu',
                             'eE',
                             'eMu',
                             'muMu'] # probably also should come from cfg

    def prepare_dfw(self, dfw):
        # 1. what arguments do I pass to GetWeight?
        # it's a string representing a formula for the total weight 
        # in terms of branches of the tree
        total_weight = analysis.GetWeight(None, None, None)
        dfw.df = dfw.df.Define('weight_noBtag', f'return {total_weight}')
        dfw.df = dfw.df.Define('weight_total', f'return {total_weight} * weight_bTagShape_Central')
        dfw.df = dfw.df.Define('e', f'return channelId == 1;')
        dfw.df = dfw.df.Define('mu', f'return channelId == 2;')
        dfw.df = dfw.df.Define('eE', f'return channelId == 11;')
        dfw.df = dfw.df.Define('eMu', f'return channelId == 12;')
        dfw.df = dfw.df.Define('muMu', f'return channelId == 22;')
        return dfw

    def run(self, array, keep_all_columns=False):
        ncentralJet = array['ncentralJet']
        weights_noBtag = array['weight_noBtag'] 
        weights_total = array['weight_total'] 
        res = {}
        # here there also should be a loop over categories: [eE, eMu, muMu] for DL and [e, mu] for SL
        for cat in ['e', 'mu', 'eE', 'eMu', 'muMu']:
            category_mask = array[cat]
            category_dict = {}
            for jet_multiplicity in self.jet_multiplicities:
                # calculate number of events and total btag shape weight
                events_with_jet_multiplicity_mask = array['ncentralJet'] == jet_multiplicity
                mask = np.logical_and(category_mask, events_with_jet_multiplicity_mask)
                category_dict[f'weight_noBtag_ncentralJet_{jet_multiplicity}'] = float(np.sum(weights_noBtag[mask]))
                category_dict[f'weight_total_ncentralJet_{jet_multiplicity}'] = float(np.sum(weights_total[mask]))
            res[cat] = category_dict
        return res