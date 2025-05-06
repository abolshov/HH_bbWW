from Studies.HME.new.hmeVariables import GetHMEVariables

class HMEProducer:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, dfw):
        if self.cfg['channel'] == "DL":
            dfw.Define("has_necessary_inputs", "ncentralJet >= 2 && lep1_pt > 0.0 && lep2_pt > 0.0")
        elif self.cfg['channel'] == "SL":
            dfw.Define("has_necessary_inputs", "ncentralJet >= 4")
        
        dfw.df = GetHMEVariables(dfw.df, self.cfg['channel'])
        for col in self.cfg['columns']:
            dfw.DefineAndAppend(f"HME_{col}", f"static_cast<size_t>(HME::EstimOut::{col})")
        return dfw
