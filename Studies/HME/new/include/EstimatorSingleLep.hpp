#ifndef ESTIM_SL
#define ESTIM_SL

#include "EstimatorBase.hpp"
#include "Constants.hpp"
#include "Definitions.hpp"

class EstimatorSingleLep final : public EstimatorBase
{
    public:
    EstimatorSingleLep(TString const& pdf_file_name, TString const& dbg_file_name);
    ~EstimatorSingleLep() override = default;

    ArrF_t<ESTIM_OUT_SZ> EstimateCombination(VecLVF_t const& particles, ULong64_t evt_id, TString const& comb_label) override;
    OptArrF_t<ESTIM_OUT_SZ> EstimateMass(VecLVF_t const& jets, VecLVF_t const& leptons, LorentzVectorF_t const& met, ULong64_t evt_id) override;

    private: 
    struct IterData;
    std::unique_ptr<IterData> m_iter_data;
    std::unique_ptr<TTree> MakeTree(TString const& tree_name) override;
};

ArrF_t<ESTIM_OUT_SZ> EstimatorSingleLep::EstimateCombination(VecLVF_t const& particles, ULong64_t evt_id, TString const& comb_label)
{
    ArrF_t<ESTIM_OUT_SZ> res{};
    return res;
}

OptArrF_t<ESTIM_OUT_SZ> EstimateMass(VecLVF_t const& jets, VecLVF_t const& leptons, LorentzVectorF_t const& met, ULong64_t evt_id)
{
    return std::nullopt;
}

std::unique_ptr<TTree> EstimatorSingleLep::MakeTree(TString const& tree_name)
{
    auto tree = std::make_unique<TTree>(tree_name, "SL channel debug tree");
    tree->SetDirectory(nullptr);

    tree->Branch("light_jet_1_pt", m_iter_data->j1_pt, "light_jet_1_pt[4]/F");
    tree->Branch("light_jet_1_eta", m_iter_data->j1_eta, "light_jet_1_eta[4]/F");
    tree->Branch("light_jet_1_phi", m_iter_data->j1_phi, "light_jet_1_phi[4]/F");
    tree->Branch("light_jet_1_mass", m_iter_data->j1_mass, "light_jet_1_mass[4]/F");

    tree->Branch("light_jet_2_pt", m_iter_data->j2_pt, "light_jet_2_pt[4]/F");
    tree->Branch("light_jet_2_eta", m_iter_data->j2_eta, "light_jet_2_eta[4]/F");
    tree->Branch("light_jet_2_phi", m_iter_data->j2_phi, "light_jet_2_phi[4]/F");
    tree->Branch("light_jet_2_mass", m_iter_data->j2_mass, "light_jet_2_mass[4]/F");

    tree->Branch("met_corr_pt", m_iter_data->met_corr_pt, "met_corr_pt[4]/F");
    tree->Branch("met_corr_phi", m_iter_data->met_corr_phi, "met_corr_phi[4]/F");

    tree->Branch("nu_pt", m_iter_data->nu_pt, "nu_pt[4]/F");
    tree->Branch("nu_eta", m_iter_data->nu_eta, "nu_eta[4]/F");
    tree->Branch("nu_phi", m_iter_data->nu_phi, "nu_phi[4]/F");

    tree->Branch("lepW_pt", m_iter_data->lepW_pt, "lepW_pt[4]/F");
    tree->Branch("lepW_eta", m_iter_data->lepW_eta, "lepW_eta[4]/F");
    tree->Branch("lepW_phi", m_iter_data->lepW_phi, "lepW_phi[4]/F");
    tree->Branch("lepW_mass", m_iter_data->lepW_mass, "lepW_mass[4]/F");

    tree->Branch("hadW_pt", m_iter_data->hadW_pt, "hadW_pt[4]/F");
    tree->Branch("hadW_eta", m_iter_data->hadW_eta, "hadW_eta[4]/F");
    tree->Branch("hadW_phi", m_iter_data->hadW_phi, "hadW_phi[4]/F");
    tree->Branch("hadW_mass", m_iter_data->hadW_mass, "hadW_mass[4]/F");

    tree->Branch("Hww_pt", m_iter_data->Hww_pt, "Hww_pt[4]/F");
    tree->Branch("Hww_eta", m_iter_data->Hww_eta, "Hww_eta[4]/F");
    tree->Branch("Hww_phi", m_iter_data->Hww_phi, "Hww_phi[4]/F");
    tree->Branch("Hww_mass", m_iter_data->Hww_mass, "Hww_mass[4]/F");

    tree->Branch("Xhh_pt", m_iter_data->Xhh_pt, "Xhh_pt[4]/F");
    tree->Branch("Xhh_eta", m_iter_data->Xhh_eta, "Xhh_eta[4]/F");
    tree->Branch("Xhh_phi", m_iter_data->Xhh_phi, "Xhh_phi[4]/F");
    tree->Branch("Xhh_mass", m_iter_data->Xhh_mass, "Xhh_mass[4]/F");

    tree->Branch("bjet_1_pt", &m_iter_data->b1_pt, "bjet_1_pt/F");
    tree->Branch("bjet_1_eta", &m_iter_data->b1_eta, "bjet_1_eta/F");
    tree->Branch("bjet_1_phi", &m_iter_data->b1_phi, "bjet_1_phi/F");
    tree->Branch("bjet_1_mass", &m_iter_data->b1_mass, "bjet_1_mass/F");

    tree->Branch("bjet_2_pt", &m_iter_data->b2_pt, "bjet_2_pt/F");
    tree->Branch("bjet_2_eta", &m_iter_data->b2_eta, "bjet_2_eta/F");
    tree->Branch("bjet_2_phi", &m_iter_data->b2_phi, "bjet_2_phi/F");
    tree->Branch("bjet_2_mass", &m_iter_data->b2_mass, "bjet_2_mass/F");

    tree->Branch("Hbb_pt", &m_iter_data->Hbb_pt, "Hbb_pt/F");
    tree->Branch("Hbb_eta", &m_iter_data->Hbb_eta, "Hbb_eta/F");
    tree->Branch("Hbb_phi", &m_iter_data->Hbb_phi, "Hbb_phi/F");
    tree->Branch("Hbb_mass", &m_iter_data->Hbb_mass, "Hbb_mass/F");
    
    tree->Branch("bjet_resc_fact_1", &m_iter_data->bjet_resc_fact_1, "bjet_resc_fact_1/F");
    tree->Branch("bjet_resc_fact_2", &m_iter_data->bjet_resc_fact_2, "bjet_resc_fact_2/F");
    tree->Branch("mw1", &m_iter_data->mw1, "mw1/F");
    tree->Branch("mw2", &m_iter_data->mw2, "mw2/F");
    tree->Branch("smear_dpx", &m_iter_data->smear_dpx, "smear_dpx/F");
    tree->Branch("smear_dpy", &m_iter_data->smear_dpy, "smear_dpy/F");
    tree->Branch("bjet_resc_dpx", &m_iter_data->bjet_resc_dpx, "bjet_resc_dpx/F");
    tree->Branch("bjet_resc_dpy", &m_iter_data->bjet_resc_dpy, "bjet_resc_dpy/F");
    tree->Branch("ljet_resc_fact_1", m_iter_data->ljet_resc_fact_1, "ljet_resc_fact_1[4]/F");
    tree->Branch("ljet_resc_fact_2", m_iter_data->ljet_resc_fact_2, "ljet_resc_fact_2[4]/F");
    tree->Branch("ljet_resc_dpx", m_iter_data->ljet_resc_dpx, "ljet_resc_dpx[4]/F");
    tree->Branch("ljet_resc_dpy", m_iter_data->ljet_resc_dpy, "ljet_resc_dpy[4]/F");
    tree->Branch("mass", m_iter_data->mass, "mass[4]/F");
    tree->Branch("weight", &m_iter_data->weight, "weight/F");
    tree->Branch("num_sol", &m_iter_data->num_sol, "num_sol/I");
    tree->Branch("correct_hww_mass", m_iter_data->correct_hww_mass, "correct_hww_mass[4]/B");
    return tree;
}

struct EstimatorSingleLep::IterData
{
    Float_t j1_pt[CONTROL] = {};
    Float_t j1_eta[CONTROL] = {};
    Float_t j1_phi[CONTROL] = {};
    Float_t j1_mass[CONTROL] = {};

    Float_t j2_pt[CONTROL] = {};
    Float_t j2_eta[CONTROL] = {};
    Float_t j2_phi[CONTROL] = {};
    Float_t j2_mass[CONTROL] = {};

    Float_t lepW_pt[CONTROL] = {};
    Float_t lepW_eta[CONTROL] = {};
    Float_t lepW_phi[CONTROL] = {};
    Float_t lepW_mass[CONTROL] = {};

    Float_t hadW_pt[CONTROL] = {};
    Float_t hadW_eta[CONTROL] = {};
    Float_t hadW_phi[CONTROL] = {};
    Float_t hadW_mass[CONTROL] = {};

    Float_t Hww_pt[CONTROL] = {};
    Float_t Hww_eta[CONTROL] = {};
    Float_t Hww_phi[CONTROL] = {};
    Float_t Hww_mass[CONTROL] = {};

    Float_t Xhh_pt[CONTROL] = {};
    Float_t Xhh_eta[CONTROL] = {};
    Float_t Xhh_phi[CONTROL] = {};
    Float_t Xhh_mass[CONTROL] = {};

    Float_t nu_pt[CONTROL] = {};
    Float_t nu_eta[CONTROL] = {};
    Float_t nu_phi[CONTROL] = {};

    Float_t mass[CONTROL] = {};

    Float_t met_corr_pt[CONTROL] = {};
    Float_t met_corr_phi[CONTROL] = {};

    Float_t ljet_resc_fact_1[CONTROL] = {};
    Float_t ljet_resc_fact_2[CONTROL] = {};

    Float_t ljet_resc_dpx[CONTROL] = {};
    Float_t ljet_resc_dpy[CONTROL] = {};

    Float_t b1_pt{0.0};
    Float_t b1_eta{0.0};
    Float_t b1_phi{0.0};
    Float_t b1_mass{0.0};

    Float_t b2_pt{0.0};
    Float_t b2_eta{0.0};
    Float_t b2_phi{0.0};
    Float_t b2_mass{0.0};

    Float_t Hbb_pt{0.0};
    Float_t Hbb_eta{0.0};
    Float_t Hbb_phi{0.0};
    Float_t Hbb_mass{0.0};

    Float_t bjet_resc_fact_1{0.0};
    Float_t bjet_resc_fact_2{0.0};
    Float_t mh{0.0};
    Float_t mw1{0.0};
    Float_t mw2{0.0};
    Float_t smear_dpx{0.0};
    Float_t smear_dpy{0.0};
    Float_t bjet_resc_dpx{0.0};
    Float_t bjet_resc_dpy{0.0};
    Float_t weight{0.0};
    Int_t num_sol{0};
    Bool_t correct_hww_mass[CONTROL] = {};
};