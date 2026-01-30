import importlib
from FLAF.Common.Utilities import *
from FLAF.Common.HistHelper import *
from Corrections.Corrections import Corrections
from Corrections.CorrectionsCore import getSystName, central
from FLAF.Common.Setup import Setup

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

initialized = False
analysis = None


def Initialize():
    global initialized
    if not initialized:
        headers_dir = os.path.dirname(os.path.abspath(__file__))
        ROOT.gROOT.ProcessLine(f".include {os.environ['ANALYSIS_PATH']}")
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/HistHelper.h"')
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/Utilities.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/MT2.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/Lester_mt2_bisect.cpp"')
        ROOT.gROOT.ProcessLine('#include "FLAF/include/AnalysisTools.h"')
        ROOT.gROOT.ProcessLine('#include "FLAF/include/AnalysisMath.h"')
        initialized = True


def analysis_setup(setup):
    global analysis
    analysis_import = setup.global_params["analysis_import"]
    analysis = importlib.import_module(f"{analysis_import}")


def GetDfw(df, setup, dataset_name):
    global_params = setup.global_params
    isData = dataset_name == "data"
    period = global_params["era"]
    dfw = analysis.DataFrameBuilderForHistograms(df, global_params, period)
    new_dfw = analysis.PrepareDfForHistograms(dfw, isData)
    return new_dfw


central_df_weights_computed = False
btag_shape_weight_corrected = False

cat_to_channelId = {"e": 1, "mu": 2, "eE": 11, "eMu": 12, "muMu": 22}


class BtagShapeWeightCorrector:
    def __init__(self, btag_integral_ratios):
        self.exisiting_srcScale_combs = [key for key in btag_integral_ratios.keys()]
        # if the btag_integral_ratios dictionary is not empty, do stuff
        if self.exisiting_srcScale_combs:
            ROOT.gInterpreter.Declare("#include <map>")

            for key in btag_integral_ratios.keys():
                # key in btag_integral_ratios has form f"{source}_{scale}", so function expects that
                # and creates a map and function to rescale btag weights for each f"{source}_{scale}" value
                self._declare_cpp_map_and_resc_func(btag_integral_ratios, key)

    def _declare_cpp_map_and_resc_func(self, btag_integral_ratios, unc_src_scale):
        correction_factors = btag_integral_ratios[unc_src_scale]

        # init c++ map
        cpp_map_entries = []
        for cat, multipl_dict in correction_factors.items():
            channelId = cat_to_channelId[cat]
            for key, ratio in multipl_dict.items():
                # key has structure f"ratio_ncetnralJet_{number}""
                num_jet = int(key.split("_")[-1])
                cpp_map_entries.append(f"{{{{{channelId}, {num_jet}}}, {ratio}}}")
        cpp_init = ", ".join(cpp_map_entries)

        ROOT.gInterpreter.Declare(
            f"""
            static const std::map<std::pair<int, int>, float> ratios_{unc_src_scale} = {{
                {cpp_init}
            }};

            float integral_correction_ratio_{unc_src_scale}(int ncentralJet, int channelId) {{
                std::pair<int, int> key{{channelId, ncentralJet}};
                try 
                {{
                    float ratio = ratios_{unc_src_scale}.at(key);
                    return ratio;
                }}
                catch (...)
                {{
                    return 1.0f;
                }}
            }}"""
        )

    def UpdateBtagWeight(self, dfw, unc_src="Central", unc_scale=None):
        # return original dfw if empty dict was passed to constructor
        if not self.exisiting_srcScale_combs:
            return dfw

        if unc_scale is None:
            unc_src_scale = unc_src
        else:
            unc_src_scale = f"{unc_src}_{unc_scale}"

        if unc_src_scale not in self.exisiting_srcScale_combs:
            raise RuntimeError(
                f"`BtagShapeWeightCorrection.json` does not contain key `{unc_src_scale}`."
            )

        dfw.df = dfw.df.Redefine(
            "weight_bTagShape_Central",
            f"""if (ncentralJet >= 2 && ncentralJet <= 8) 
                    return integral_correction_ratio_{unc_src_scale}(ncentralJet, channelId)*weight_bTagShape_Central;
                return weight_bTagShape_Central;""",
        )

        return dfw

def DefineWeightForHistograms(
    *,
    dfw,
    isData,
    uncName,
    uncScale,
    unc_cfg_dict,
    hist_cfg_dict,
    global_params,
    final_weight_name,
    df_is_central,
    btag_integral_ratios,
):
    global central_df_weights_computed
    is_central = uncName == central
    if not isData and (not central_df_weights_computed or not df_is_central):
        corrections = Corrections.getGlobal()
        lepton_legs = ["lep1", "lep2"]
        offline_legs = ["lep1", "lep2"]
        triggers_to_use = set()
        channels = global_params["channelSelection"]
        for channel in channels:
            trigger_list = global_params.get("triggers", {}).get(channel, [])
            for trigger in trigger_list:
                if trigger not in corrections.trigger_dict.keys():
                    raise RuntimeError(
                        f"Trigger does not exist in triggers.yaml, {trigger}"
                    )
                triggers_to_use.add(trigger)

        dfw.df, all_weights = corrections.getNormalisationCorrections(
            dfw.df,
            lepton_legs=lepton_legs,
            offline_legs=offline_legs,
            trigger_names=triggers_to_use,
            unc_source=uncName,
            unc_scale=uncScale,
            ana_caches=None,
            return_variations=is_central and global_params["compute_unc_histograms"],
            use_genWeight_sign_only=True,
        )
        if df_is_central:
            central_df_weights_computed = True

    # btag shape weight column appears here
    correct_btagShape_weights = global_params.get("correct_btagShape_weights", False)
    global btag_shape_weight_corrected
    if correct_btagShape_weights and not btag_shape_weight_corrected and btag_integral_ratios:
        isMC = not isData
        if is_central and isMC:
            weight_corrector = BtagShapeWeightCorrector(btag_integral_ratios)
            print(f"Calling weight_corrector.UpdateBtagWeight for unc_source={uncName} unc_scale={uncScale}")
            weight_corrector.UpdateBtagWeight(dfw, unc_src=uncName)
            btag_shape_weight_corrected = True

    categories = global_params["categories"]
    boosted_categories = global_params.get("boosted_categories", [])
    process_group = global_params["process_group"]
    total_weight_expression = (
        # channel, cat, boosted_categories --> these are not needed in the GetWeight function therefore I just put some placeholders
        # if btag shape weight was corrected => must be applied, else no
        analysis.GetWeight("", "", boosted_categories, apply_btag_shape_weights=btag_shape_weight_corrected)
        if process_group != "data"
        else "1"
    )  # are we sure?
    weight_name = "final_weight"
    if weight_name not in dfw.df.GetColumnNames():
        dfw.df = dfw.df.Define(weight_name, total_weight_expression)
    if not is_central and type(unc_cfg_dict) == dict:
        if (
            uncName in unc_cfg_dict["norm"].keys()
            and "expression" in unc_cfg_dict["norm"][uncName].keys()
        ):
            weight_name = unc_cfg_dict["norm"][uncName]["expression"].format(
                scale=uncScale
            )
    dfw.df = dfw.df.Define(final_weight_name, weight_name)
