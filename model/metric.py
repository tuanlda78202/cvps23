import py_sod_metrics

FM = py_sod_metrics.Fmeasure()
WFM = py_sod_metrics.WeightedFmeasure()
SM = py_sod_metrics.Smeasure()
EM = py_sod_metrics.Emeasure()
MAE = py_sod_metrics.MAE()

# F-measure
def maxfm(pred, gt):
    FM = py_sod_metrics.Fmeasure()
    FM.step(pred, gt)
    fm = FM.get_results()["fm"]
    return fm["curve"].max()

# Mean Absolute Error
def mae(pred, gt):
    MAE = py_sod_metrics.MAE()
    MAE.step(pred, gt)
    mae = MAE.get_results()["mae"]
    return mae 

# Weighted F-measure
def wfm(pred, gt):
    WFM = py_sod_metrics.WeightedFmeasure()
    WFM.step(pred, gt)
    wfm = WFM.get_results()["wfm"]
    return wfm 

# S-measure
def sm(pred, gt):
    SM = py_sod_metrics.Smeasure()
    SM.step(pred, gt)
    sm = SM.get_results()["sm"]
    return sm