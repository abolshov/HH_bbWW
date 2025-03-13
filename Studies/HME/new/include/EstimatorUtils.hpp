#ifndef ESTIMATOR_UTILS_HPP
#define ESTIMATOR_UTILS_HPP

#include <vector>
#include <type_traits>

#include "Definitions.hpp"
#include "Storage.hpp"
#include "Constants.hpp"

#include "TFile.h"
#include "TH1.h"

template <typename T>
UHist_t<T> ReadHist(TFile* file, TString hist_name)
{
    UHist_t<T> res(static_cast<T*>(file->Get<T>(hist_name)));
    res->SetDirectory(nullptr);
    return res;
}

template <typename T>
UHist_t<T> Copy(UHist_t<T> const& hist)
{
    UHist_t<T> res(static_cast<T*>(hist->Clone()));
    res->SetDirectory(nullptr);
    return res;
}

template <typename T>
void ResetHist(UHist_t<T>& hist)
{
    hist->Reset("ICES");
}

inline LorentzVectorF_t GetRecoMET(Storage const& s) { return LorentzVectorF_t(s.reco_met_pt, 0.0, s.reco_met_phi, 0.0); }

VecLVF_t GetRecoJetP4(Storage const& s)
{
    VecLVF_t res;
    for (int i = 0; i < s.n_reco_jet; ++i)
    {
        res.emplace_back(s.reco_jet_pt[i], s.reco_jet_eta[i], s.reco_jet_phi[i], s.reco_jet_mass[i]);
    }
    return res;
}

VecLVF_t GetRecoLepP4(Storage const& s, Channel ch)
{
    VecLVF_t res;
    res.emplace_back(s.reco_lep_pt[static_cast<size_t>(Lep::lep1)], 
                     s.reco_lep_eta[static_cast<size_t>(Lep::lep1)], 
                     s.reco_lep_phi[static_cast<size_t>(Lep::lep1)], 
                     s.reco_lep_mass[static_cast<size_t>(Lep::lep1)]);
    if (ch == Channel::DL)
    {
        res.emplace_back(s.reco_lep_pt[static_cast<size_t>(Lep::lep2)], 
                         s.reco_lep_eta[static_cast<size_t>(Lep::lep2)], 
                         s.reco_lep_phi[static_cast<size_t>(Lep::lep2)], 
                         s.reco_lep_mass[static_cast<size_t>(Lep::lep2)]);
    }
    return res;
}

std::vector<Float_t> GetPNetRes(Storage const& s)
{
    std::vector<Float_t> res;
    for (int i = 0; i < s.n_reco_jet; ++i)
    {   
        Float_t mult = s.reco_jet_corr[i]*s.reco_jet_res[i] == 0 ? DEFAULT_JET_RES : s.reco_jet_corr[i]*s.reco_jet_res[i];
        res.push_back(s.reco_jet_pt[i]*mult);
    }
    return res;
}

void Get1dPDFs(TFile* fptr, HistVec_t<TH1F>& pdfs, Channel ch)
{
    if (ch == Channel::SL)
    {
        for (auto const& [pdf, name]: pdf1d_sl_names)
        {
            pdfs[static_cast<size_t>(pdf)] = ReadHist<TH1F>(fptr, name);
        }
    }
    else if (ch == Channel::DL)
    {
        for (auto const& [pdf, name]: pdf1d_dl_names)
        {
            pdfs[static_cast<size_t>(pdf)] = ReadHist<TH1F>(fptr, name);
        }
    }
    else 
    {
        throw std::runtime_error("Get1dPDFs: attempting to read PDFs for unnkown channel");
    }
}

void Get2dPDFs(TFile* fptr, HistVec_t<TH2F>& pdfs, Channel ch)
{
    if (ch == Channel::SL)
    {
        for (auto const& [pdf, name]: pdf2d_sl_names)
        {
            pdfs[static_cast<size_t>(pdf)] = ReadHist<TH2F>(fptr, name);
        }
    }
    else if (ch == Channel::DL)
    {
        for (auto const& [pdf, name]: pdf2d_dl_names)
        {
            pdfs[static_cast<size_t>(pdf)] = ReadHist<TH2F>(fptr, name);
        }
    }
    else 
    {
        throw std::runtime_error("Get2dPDFs: attempting to read PDFs for unnkown channel");
    }
}

Float_t ComputeWidth(UHist_t<TH1F> const& h, unsigned l, unsigned r)
{
    int const nq = 100;
    Double_t xq[nq];  // position where to compute the quantiles in [0,1]
    Double_t yq[nq];  // array to contain the quantiles
    for (int i = 0; i < nq; ++i) 
    {
        xq[i] = static_cast<Double_t>(i+1)/nq;
    }
    h->GetQuantiles(nq, yq, xq);
    return static_cast<Float_t>(yq[r - 1] - yq[l - 1]);
}

template <typename It, std::enable_if_t<std::is_floating_point_v<typename It::value_type>, bool> = true>
void ZScoreTransform(It begin, It end)
{
    if (begin == end)
    {
        return;
    }

    typename It::value_type sum = 0.0;
    typename It::value_type sum_sqr = 0.0;

    It it = begin;
    while (it != end)
    {
        typename It::value_type val = *it;
        sum += val;
        sum_sqr += val*val;
        ++it;
    }

    size_t n = end - begin;
    typename It::value_type mean = sum/n;
    typename It::value_type mean_sqr = sum_sqr/n;
    typename It::value_type std = std::sqrt(mean_sqr - mean*mean);

    it = begin;
    while (it != end)
    {
        *it -= mean;
        *it /= std;
        ++it;
    }
}

template <typename It, std::enable_if_t<std::is_floating_point_v<typename It::value_type>, bool> = true>
void MinMaxTransform(It begin, It end)
{
    if (begin == end)
    {
        return;
    }

    auto [min_it, max_it] = std::minmax_element(begin, end);
    typename It::value_type diff = *max_it - *min_it;
    typename It::value_type min = *min_it;

    auto Func = [&min, &diff](typename It::value_type const& val)
    { 
        typename It::value_type ret = val;
        ret -= min; 
        ret /= diff;
        return ret;
    };
    std::transform(begin, end, begin, Func);
}

#endif