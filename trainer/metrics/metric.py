import torch
import py_sod_metrics

# F-measure ↑ - 09CPVR
def maxfm(pred, gt):
    FM = py_sod_metrics.Fmeasure()
    FM.step(pred, gt)
    fm = FM.get_results()["fm"]
    return fm["curve"]

# Mean Absolute Error ↓ - 12CVPR
def mae(pred, gt):
    MAE = py_sod_metrics.MAE()
    MAE.step(pred, gt)
    mae = MAE.get_results()["mae"]
    return mae 

# Weighted F-measure ↑ - 14CVPR
def wfm(pred, gt):
    WFM = py_sod_metrics.WeightedFmeasure()
    WFM.step(pred, gt)
    wfm = WFM.get_results()["wfm"]
    return wfm 

# Structure-measure ↑ - 17ICCV
def sm(pred, gt):
    SM = py_sod_metrics.Smeasure()
    SM.step(pred, gt)
    sm = SM.get_results()["sm"]
    return sm

# Enhanced-measure ↑ - 18IJCAI
def em(pred, gt):
    EM = py_sod_metrics.Emeasure()
    EM.step(pred, gt)
    em = EM.get_results()["em"]
    # mean 
    return em["curve"].mean()

