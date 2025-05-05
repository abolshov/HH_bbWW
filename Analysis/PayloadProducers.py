from Studies.HME.new.hmeVariables import GetHMEVariables

class HMEProducer:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, dfw):
        df_selected = dfw
        df_not_selected = dfw
        if self.cfg['channel'] == "DL":
            df_selected = df_selected.Filter(f"ncentralJet >= 2 && lep1_pt > 0.0 && lep2_pt > 0.0", "selected_for_HME")
            df_not_selected = df_selected.Filter(f"!(ncentralJet >= 2 && lep1_pt > 0.0 && lep2_pt > 0.0)", "selected_for_HME")
        elif self.cfg['channel'] == "SL":
            df_selected = df_selected.Filter(f"ncentralJet >= 4 && lep1_pt > 0.0", "selected_for_HME")
            df_not_selected = df_selected.Filter(f"!(ncentralJet >= 4 && lep1_pt > 0.0)", "selected_for_HME")
        
        df_not_selected = df_not_selected.Define("hme_mass", "return -1.0f;")
        df_selected = GetHMEVariables(df_selected, self.cfg['channel'])

        # need to merge df_selected and df_not_selected into single dataframe
        # how??

        return dfw