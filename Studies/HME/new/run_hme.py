import ROOT
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(prog='train_net', description='Runs HME')
    parser.add_argument('file', type=str, help="Input file")
    parser.add_argument('mod', type=int, help="Value modulo which event should be selected")
    parser.add_argument('val', type=int, help="Value for event selection")

    args = parser.parse_args()
    input_file = args.file
    mod = args.mod
    val = args.val

    ROOT.gROOT.SetBatch(True)
    ROOT.EnableImplicitMT(8)

    ROOT.gROOT.ProcessLine('#include "include/Constants.hpp"')
    ROOT.gROOT.ProcessLine('#include "include/Definitions.hpp"')

    ROOT.gInterpreter.Declare("""auto file_pdf_dl = std::make_unique<TFile>("pdf_dl.root", "READ");""")
    ROOT.gInterpreter.Declare("""TRandom3 rg; rg.SetSeed(42);""")

    df = ROOT.RDataFrame("Events", input_file)
    print(f"Total events: {df.Count().GetValue()}")
    df = df.Filter(f"event % {mod} == {val}", "Evaluation selection")
    # add code here
    
    print("\nCutflow report:")
    df.Report().Print()

if __name__ == '__main__':
    main()