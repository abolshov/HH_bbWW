import ROOT
import argparse
import numpy as np
import time
from hmeVariables import GetHMEVariables


def main():
	parser = argparse.ArgumentParser(prog='HME', description='Runs HME')
	parser.add_argument('file', type=str, help="Input file")
	parser.add_argument('mod', type=int, help="Value modulo which event should be selected")
	parser.add_argument('val', type=int, help="Value for event selection")

	args = parser.parse_args()
	input_file = args.file
	mod = args.mod
	val = args.val

	ROOT.gROOT.SetBatch(True)
	ROOT.EnableImplicitMT(8)

	# ROOT.gROOT.ProcessLine('#include "include/EstimatorLTWrapper.hpp"')

	start = time.perf_counter()
	df = ROOT.RDataFrame("Events", input_file)
	print(f"Total events: {df.Count().GetValue()}")
	df = df.Filter(f"event % {mod} == {val}", "Evaluation selection")
	df = df.Filter(f"nJet >= 2", "jets")
	df = df.Filter(f"lep1_pt > 0.0 && lep2_pt > 0.0", "leptons")
	hme_events = df.Count().GetValue()
	print(f"HME events: {hme_events}")

	# df = df.Define("jets", """HME::VecLVF_t res;
	# 							for (size_t i = 0; i < nJet; ++i)
	# 							{{
	# 								res.emplace_back(centralJet_pt[i], centralJet_eta[i], centralJet_phi[i], centralJet_mass[i]);  
	# 							}}
	# 							return res;""")

	# df = df.Define("leptons", """HME::VecLVF_t res;
	# 							 res.emplace_back(lep1_pt, lep1_eta, lep1_phi, lep1_mass);
	# 							 res.emplace_back(lep2_pt, lep2_eta, lep2_phi, lep2_mass);
	# 							 return res;""")

	# df = df.Define("met", """HME::LorentzVectorF_t res(PuppiMET_pt, 0.0, PuppiMET_phi, 0.0);    
	# 						 return res;""")
		
	# df = df.Define("hme_mass", """auto hme = HME::EstimatorLTWrapper::Instance().GetEstimator().EstimateMass(jets, leptons, met, event, HME::Channel::DL);
	# 							  Float_t mass = -1.0f;
	# 							  if (hme.has_value())
	# 							  {{
	# 							  	auto const& result_array = hme.value();
	# 								mass = result_array[static_cast<size_t>(HME::EstimOut::mass)];
	# 							  }}
	# 							  return mass;""")
	
	
	df = GetHMEVariables(df)

	c1 = ROOT.TCanvas("c1", "c1")
	c1.SetGrid()
	hist = df.Histo1D(("hme_mass", "HME X->HH mass", 100, -10, 2000), "hme_mass")
	hist.GetXaxis().SetTitle("mass, [GeV]")
	hist.GetXaxis().SetTitle("Count")
	hist.Draw()
	c1.SaveAs("hme.png")
	end = time.perf_counter()

	# df = df.Filter("hme_mass < 0.0")
	# hme_failed_events = df.Count().GetValue()
	# print(f"HME success rate: {(1.0 - hme_failed_events/hme_events)*100.0:.2f}%")
	print(f"Execution time: {(end - start):.2f}s")


if __name__ == '__main__':
    main()