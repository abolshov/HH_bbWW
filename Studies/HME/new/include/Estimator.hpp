#ifndef ESTIMATOR_HPP
#define ESTIMATOR_HPP

#include "EstimatorSingleLep.hpp"
#include "EstimatorDoubleLep.hpp"

class Estimator
{
    public:
    Estimator(TString const& pdf_file_name_sl, TString const& pdf_file_name_dl, TString const& dbg_file_name);
    OptArrF_t<ESTIM_OUT_SZ> EstimateMass(VecLVF_t const& jets, VecLVF_t const& leptons, LorentzVectorF_t const& met, ULong64_t evt_id, Channel ch);

    private:
    EstimatorSingleLep m_estimator_sl;
    EstimatorDoubleLep m_estimator_dl;
};

Estimator::Estimator(TString const& pdf_file_name_sl, TString const& pdf_file_name_dl, TString const& dbg_file_name)
:   m_estimator_sl(pdf_file_name_sl, dbg_file_name)
,   m_estimator_dl(pdf_file_name_dl, dbg_file_name)
{}

OptArrF_t<ESTIM_OUT_SZ> Estimator::EstimateMass(VecLVF_t const& jets, VecLVF_t const& leptons, LorentzVectorF_t const& met, ULong64_t evt_id, Channel ch)
{
    if (ch == Channel::SL)
    {
        throw std::runtime_error("EstimatorSigleLep is not yet implemented");
    }
    else if (ch == Channel::DL)
    {
        return m_estimator_dl.EstimateMass(jets, leptons, met, evt_id);
    }
    else 
    {
        throw std::runtime_error("Attempting to process data in unknown channel");
    }
}

#endif