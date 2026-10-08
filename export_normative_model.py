"""
Export the normative model (pickle) into normative_model.js for the brain normative atlas explorer.

run:  python3 export_normative_model.py to write normative_model.js in the app folder, replacing the demo file.

The curves are computed as:
    mean curve  = model.predict(DataFrame(age=..., sex="female"/"male"))[:, roi_idx]
    band        = mean +/- model.stds_[roi_idx]
Both gray matter (_GM_Vol) and CSF (_CSF_Vol) ROIs are exported. The training volumes were
scaled to a total intracranial volume of 1500 ml (volume x 1500 / TIV); this is recorded in the
output so that the app can show it and scale new participants the same way. The exported file contains these
curves only: no participant data and no model weights.
"""

import os, sys, json, pickle, datetime
import numpy as np
import pandas as pd

# --- settings: files in the app folder (the folder of this script) ---
HERE = os.path.dirname(os.path.abspath(__file__))
# normative_models.py must be in this folder! --> pickle needs the NormativeBLR class to load the model
sys.path.insert(0, HERE)
from normative_models import NormativeBLR  # noqa: F401  (import needed for unpickling)

MODEL_NAME = "normative_BLR_BigHC_janv2026_280rois"
MODEL_PATH = os.path.join(HERE, MODEL_NAME + ".pkl")
# ROI order (= model output columns): the list of the 280 CAT12 12.7 Neuromorphometrics ROIs
# (140 GM then 140 CSF), one name per row. A CSV whose header contains the ROI columns in the
# model's order also works. The order is checked against the model's predictions before export.
ROI_ORDER_CSV = os.path.join(HERE, "regions_neuromorphometrics_cat12_7.csv")
N_TRAIN = 53856       # number of healthy training participants (Supplementary Table 1), shown in the app
OUT = os.path.join(HERE, "normative_model.js")   # replaces the demo file used by the app
N_AGES = 137          # number of points on each curve
# ----------------------------------------------------------------------------------

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def read_roi_order(path):
    """ROI names in the model's output order, from either a one-column list of names
    (e.g. regions_neuromorphometrics_cat12_7.csv) or a CSV header containing the ROI columns."""
    df = pd.read_csv(path)
    is_roi = lambda c: str(c).endswith("_GM_Vol") or str(c).endswith("_CSF_Vol")
    header = [c for c in df.columns if is_roi(c)]
    if header:
        return header
    if df.shape[1] == 1:
        return [str(v).strip() for v in df.iloc[:, 0] if is_roi(str(v).strip())]
    raise ValueError(f"no ROI names found in {path}")


# Approximate relative size of each ROI (ml): sum of the SPM12 tissue probability (GM or CSF) over the
# region's voxels in the Neuromorphometrics atlas. Only used to check the ROI order (size fingerprint).
REFERENCE_VOLUMES = {
    "l3thVen_GM_Vol": 0.0221,
    "r3thVen_GM_Vol": 0.143,
    "l4thVen_GM_Vol": 0.0916,
    "r4thVen_GM_Vol": 0.1947,
    "lAcc_GM_Vol": 0.4613,
    "rAcc_GM_Vol": 0.4037,
    "lAmy_GM_Vol": 0.8874,
    "rAmy_GM_Vol": 0.8648,
    "lBst_GM_Vol": 1.128,
    "rBst_GM_Vol": 1.7013,
    "lCau_GM_Vol": 2.393,
    "rCau_GM_Vol": 2.3887,
    "lExtCbe_GM_Vol": 41.701,
    "rExtCbe_GM_Vol": 41.6495,
    "lCbeWM_GM_Vol": 2.6312,
    "rCbeWM_GM_Vol": 2.4959,
    "lCbrWM_GM_Vol": 56.3564,
    "rCbrWM_GM_Vol": 57.5599,
    "lCSF_GM_Vol": 0.0334,
    "rCSF_GM_Vol": 0.0952,
    "lHip_GM_Vol": 2.881,
    "rHip_GM_Vol": 3.0256,
    "rInfLatVen_GM_Vol": 0.0979,
    "lLatVen_GM_Vol": 0.8697,
    "rLatVen_GM_Vol": 0.7362,
    "lPal_GM_Vol": 0.2444,
    "rPal_GM_Vol": 0.2132,
    "lPut_GM_Vol": 3.355,
    "rPut_GM_Vol": 3.2018,
    "lThaPro_GM_Vol": 3.1407,
    "rThaPro_GM_Vol": 3.143,
    "lVenVen_GM_Vol": 1.0323,
    "rVenVen_GM_Vol": 0.9225,
    "rOC_GM_Vol": 0.0041,
    "lCbeLoCbe1-5_GM_Vol": 1.415,
    "rCbeLoCbe1-5_GM_Vol": 2.1627,
    "lCbeLoCbe6-7_GM_Vol": 0.5515,
    "rCbeLoCbe6-7_GM_Vol": 1.1553,
    "lCbeLoCbe8-10_GM_Vol": 0.843,
    "rCbeLoCbe8-10_GM_Vol": 1.5171,
    "lBasCbr+FobBr_GM_Vol": 0.2744,
    "rBasCbr+FobBr_GM_Vol": 0.2515,
    "lAntCinGy_GM_Vol": 4.4315,
    "rAntCinGy_GM_Vol": 3.4288,
    "lAntIns_GM_Vol": 3.657,
    "rAntIns_GM_Vol": 3.6504,
    "lAntOrbGy_GM_Vol": 1.238,
    "rAntOrbGy_GM_Vol": 1.385,
    "lAngGy_GM_Vol": 6.2837,
    "rAngGy_GM_Vol": 7.543,
    "lCal+Cbr_GM_Vol": 2.5857,
    "rCal+Cbr_GM_Vol": 2.6087,
    "lCenOpe_GM_Vol": 3.1355,
    "rCenOpe_GM_Vol": 3.1856,
    "lCun_GM_Vol": 2.9897,
    "rCun_GM_Vol": 3.324,
    "lEnt_GM_Vol": 1.5047,
    "rEnt_GM_Vol": 1.4601,
    "lFroOpe_GM_Vol": 1.6582,
    "rFroOpe_GM_Vol": 1.6558,
    "lFroPo_GM_Vol": 1.8411,
    "rFroPo_GM_Vol": 2.1272,
    "lFusGy_GM_Vol": 6.5348,
    "rFusGy_GM_Vol": 6.601,
    "lRecGy_GM_Vol": 1.8836,
    "rRecGy_GM_Vol": 1.6771,
    "lInfOccGy_GM_Vol": 4.5255,
    "rInfOccGy_GM_Vol": 4.3049,
    "lInfTemGy_GM_Vol": 9.1747,
    "rInfTemGy_GM_Vol": 9.4133,
    "lLinGy_GM_Vol": 5.2766,
    "rLinGy_GM_Vol": 5.9653,
    "lLatOrbGy_GM_Vol": 1.4317,
    "rLatOrbGy_GM_Vol": 1.4455,
    "lMidCinGy_GM_Vol": 4.055,
    "rMidCinGy_GM_Vol": 3.9666,
    "lMedFroCbr_GM_Vol": 1.5653,
    "rMedFroCbr_GM_Vol": 1.5656,
    "lMidFroGy_GM_Vol": 15.052,
    "rMidFroGy_GM_Vol": 15.0727,
    "lMidOccGy_GM_Vol": 4.3862,
    "rMidOccGy_GM_Vol": 3.7966,
    "lMedOrbGy_GM_Vol": 3.249,
    "rMedOrbGy_GM_Vol": 3.2767,
    "lMedPoCGy_GM_Vol": 0.4422,
    "rMedPoCGy_GM_Vol": 0.3906,
    "lMedPrcGy_GM_Vol": 1.8171,
    "rMedPrcGy_GM_Vol": 1.7909,
    "lSupMedFroGy_GM_Vol": 5.2005,
    "rSupMedFroGy_GM_Vol": 6.1458,
    "lMidTemGy_GM_Vol": 12.0276,
    "rMidTemGy_GM_Vol": 12.3556,
    "lOccPo_GM_Vol": 2.1033,
    "rOccPo_GM_Vol": 1.8174,
    "lOccFusGy_GM_Vol": 2.8039,
    "rOccFusGy_GM_Vol": 2.6381,
    "lInfFroGy_GM_Vol": 2.3984,
    "rInfFroGy_GM_Vol": 2.5114,
    "lInfFroOrbGy_GM_Vol": 1.0876,
    "rInfFroOrbGy_GM_Vol": 1.132,
    "lPosCinGy_GM_Vol": 3.439,
    "rPosCinGy_GM_Vol": 3.1587,
    "lPCu_GM_Vol": 8.4822,
    "rPCu_GM_Vol": 8.8011,
    "lParHipGy_GM_Vol": 2.0877,
    "rParHipGy_GM_Vol": 1.957,
    "lPosIns_GM_Vol": 1.7965,
    "rPosIns_GM_Vol": 1.8782,
    "lParOpe_GM_Vol": 2.0488,
    "rParOpe_GM_Vol": 1.7459,
    "lPoCGy_GM_Vol": 6.5242,
    "rPoCGy_GM_Vol": 5.7121,
    "lPosOrbGy_GM_Vol": 1.9484,
    "rPosOrbGy_GM_Vol": 1.9228,
    "lPla_GM_Vol": 1.4344,
    "rPla_GM_Vol": 1.2187,
    "lPrcGy_GM_Vol": 8.1111,
    "rPrcGy_GM_Vol": 8.0932,
    "lTem_GM_Vol": 1.6219,
    "rTem_GM_Vol": 1.3656,
    "lSCA_GM_Vol": 0.8929,
    "rSCA_GM_Vol": 0.8735,
    "lSupFroGy_GM_Vol": 10.4452,
    "rSupFroGy_GM_Vol": 10.4324,
    "lCbr+Mot_GM_Vol": 4.0912,
    "rCbr+Mot_GM_Vol": 3.9642,
    "lSupMarGy_GM_Vol": 6.2579,
    "rSupMarGy_GM_Vol": 5.9858,
    "lSupOccGy_GM_Vol": 2.0364,
    "rSupOccGy_GM_Vol": 2.3871,
    "lSupParLo_GM_Vol": 6.5347,
    "rSupParLo_GM_Vol": 6.2896,
    "lSupTemGy_GM_Vol": 5.4204,
    "rSupTemGy_GM_Vol": 5.7706,
    "lTemPo_GM_Vol": 6.6796,
    "rTemPo_GM_Vol": 6.6983,
    "lInfFroAngGy_GM_Vol": 2.5353,
    "rInfFroAngGy_GM_Vol": 2.3906,
    "lTemTraGy_GM_Vol": 1.1537,
    "rTemTraGy_GM_Vol": 0.9974,
    "l3thVen_CSF_Vol": 0.0482,
    "r3thVen_CSF_Vol": 0.4373,
    "l4thVen_CSF_Vol": 0.4387,
    "r4thVen_CSF_Vol": 0.6923,
    "lAcc_CSF_Vol": 0.0302,
    "rAcc_CSF_Vol": 0.0311,
    "lAmy_CSF_Vol": 0.1098,
    "rAmy_CSF_Vol": 0.1253,
    "lBst_CSF_Vol": 1.1339,
    "rBst_CSF_Vol": 1.4175,
    "lCau_CSF_Vol": 0.601,
    "rCau_CSF_Vol": 0.6393,
    "lExtCbe_CSF_Vol": 8.3958,
    "rExtCbe_CSF_Vol": 7.925,
    "lCbeWM_CSF_Vol": 0.8395,
    "rCbeWM_CSF_Vol": 0.8242,
    "lCbrWM_CSF_Vol": 11.3758,
    "rCbrWM_CSF_Vol": 11.5711,
    "lCSF_CSF_Vol": 0.1918,
    "rCSF_CSF_Vol": 0.5786,
    "lHip_CSF_Vol": 0.484,
    "rHip_CSF_Vol": 0.5481,
    "rInfLatVen_CSF_Vol": 0.1053,
    "lLatVen_CSF_Vol": 7.0702,
    "rLatVen_CSF_Vol": 5.9727,
    "lPal_CSF_Vol": 0.0678,
    "rPal_CSF_Vol": 0.0687,
    "lPut_CSF_Vol": 0.1715,
    "rPut_CSF_Vol": 0.1655,
    "lThaPro_CSF_Vol": 0.9749,
    "rThaPro_CSF_Vol": 0.9494,
    "lVenVen_CSF_Vol": 0.8586,
    "rVenVen_CSF_Vol": 0.8161,
    "rOC_CSF_Vol": 0.0232,
    "lCbeLoCbe1-5_CSF_Vol": 0.4289,
    "rCbeLoCbe1-5_CSF_Vol": 0.6815,
    "lCbeLoCbe6-7_CSF_Vol": 0.1453,
    "rCbeLoCbe6-7_CSF_Vol": 0.3329,
    "lCbeLoCbe8-10_CSF_Vol": 0.1613,
    "rCbeLoCbe8-10_CSF_Vol": 0.2736,
    "lBasCbr+FobBr_CSF_Vol": 0.1001,
    "rBasCbr+FobBr_CSF_Vol": 0.0899,
    "lAntCinGy_CSF_Vol": 1.0599,
    "rAntCinGy_CSF_Vol": 0.9479,
    "lAntIns_CSF_Vol": 0.963,
    "rAntIns_CSF_Vol": 0.8881,
    "lAntOrbGy_CSF_Vol": 0.2571,
    "rAntOrbGy_CSF_Vol": 0.284,
    "lAngGy_CSF_Vol": 2.0465,
    "rAngGy_CSF_Vol": 2.0172,
    "lCal+Cbr_CSF_Vol": 0.5032,
    "rCal+Cbr_CSF_Vol": 0.4407,
    "lCenOpe_CSF_Vol": 0.9585,
    "rCenOpe_CSF_Vol": 0.9644,
    "lCun_CSF_Vol": 1.149,
    "rCun_CSF_Vol": 1.0727,
    "lEnt_CSF_Vol": 0.2689,
    "rEnt_CSF_Vol": 0.2551,
    "lFroOpe_CSF_Vol": 0.4359,
    "rFroOpe_CSF_Vol": 0.4216,
    "lFroPo_CSF_Vol": 1.3589,
    "rFroPo_CSF_Vol": 1.5244,
    "lFusGy_CSF_Vol": 0.8406,
    "rFusGy_CSF_Vol": 0.9129,
    "lRecGy_CSF_Vol": 0.543,
    "rRecGy_CSF_Vol": 0.3943,
    "lInfOccGy_CSF_Vol": 0.6815,
    "rInfOccGy_CSF_Vol": 0.6643,
    "lInfTemGy_CSF_Vol": 1.3534,
    "rInfTemGy_CSF_Vol": 1.3916,
    "lLinGy_CSF_Vol": 1.5279,
    "rLinGy_CSF_Vol": 1.5513,
    "lLatOrbGy_CSF_Vol": 0.4273,
    "rLatOrbGy_CSF_Vol": 0.4087,
    "lMidCinGy_CSF_Vol": 0.9189,
    "rMidCinGy_CSF_Vol": 0.8272,
    "lMedFroCbr_CSF_Vol": 0.3521,
    "rMedFroCbr_CSF_Vol": 0.4054,
    "lMidFroGy_CSF_Vol": 5.4333,
    "rMidFroGy_CSF_Vol": 5.2493,
    "lMidOccGy_CSF_Vol": 1.0552,
    "rMidOccGy_CSF_Vol": 0.7993,
    "lMedOrbGy_CSF_Vol": 0.645,
    "rMedOrbGy_CSF_Vol": 0.6114,
    "lMedPoCGy_CSF_Vol": 0.2954,
    "rMedPoCGy_CSF_Vol": 0.226,
    "lMedPrcGy_CSF_Vol": 0.7069,
    "rMedPrcGy_CSF_Vol": 0.6573,
    "lSupMedFroGy_CSF_Vol": 1.8924,
    "rSupMedFroGy_CSF_Vol": 2.2176,
    "lMidTemGy_CSF_Vol": 1.9102,
    "rMidTemGy_CSF_Vol": 1.9191,
    "lOccPo_CSF_Vol": 0.6127,
    "rOccPo_CSF_Vol": 0.5646,
    "lOccFusGy_CSF_Vol": 0.4806,
    "rOccFusGy_CSF_Vol": 0.448,
    "lInfFroGy_CSF_Vol": 0.794,
    "rInfFroGy_CSF_Vol": 0.7641,
    "lInfFroOrbGy_CSF_Vol": 0.2455,
    "rInfFroOrbGy_CSF_Vol": 0.2412,
    "lPosCinGy_CSF_Vol": 0.8953,
    "rPosCinGy_CSF_Vol": 0.6934,
    "lPCu_CSF_Vol": 2.7537,
    "rPCu_CSF_Vol": 2.7223,
    "lParHipGy_CSF_Vol": 0.6841,
    "rParHipGy_CSF_Vol": 0.5491,
    "lPosIns_CSF_Vol": 0.5626,
    "rPosIns_CSF_Vol": 0.5387,
    "lParOpe_CSF_Vol": 0.4509,
    "rParOpe_CSF_Vol": 0.4174,
    "lPoCGy_CSF_Vol": 3.7899,
    "rPoCGy_CSF_Vol": 3.3155,
    "lPosOrbGy_CSF_Vol": 0.5484,
    "rPosOrbGy_CSF_Vol": 0.4518,
    "lPla_CSF_Vol": 0.61,
    "rPla_CSF_Vol": 0.5523,
    "lPrcGy_CSF_Vol": 3.6068,
    "rPrcGy_CSF_Vol": 3.6334,
    "lTem_CSF_Vol": 0.4278,
    "rTem_CSF_Vol": 0.4432,
    "lSCA_CSF_Vol": 0.3032,
    "rSCA_CSF_Vol": 0.3245,
    "lSupFroGy_CSF_Vol": 4.7062,
    "rSupFroGy_CSF_Vol": 4.8716,
    "lCbr+Mot_CSF_Vol": 1.3581,
    "rCbr+Mot_CSF_Vol": 1.3153,
    "lSupMarGy_CSF_Vol": 2.1798,
    "rSupMarGy_CSF_Vol": 2.0757,
    "lSupOccGy_CSF_Vol": 0.7728,
    "rSupOccGy_CSF_Vol": 0.8723,
    "lSupParLo_CSF_Vol": 3.9071,
    "rSupParLo_CSF_Vol": 4.0203,
    "lSupTemGy_CSF_Vol": 1.1592,
    "rSupTemGy_CSF_Vol": 1.2277,
    "lTemPo_CSF_Vol": 1.7943,
    "rTemPo_CSF_Vol": 1.7967,
    "lInfFroAngGy_CSF_Vol": 0.8058,
    "rInfFroAngGy_CSF_Vol": 0.6808,
    "lTemTraGy_CSF_Vol": 0.3919,
    "rTemTraGy_CSF_Vol": 0.4381,
}


def _shift_within_tissue(rois, k):
    gm = [r for r in rois if r.endswith("_GM_Vol")]; csf = [r for r in rois if r.endswith("_CSF_Vol")]
    return gm[k:] + gm[:k] + csf[k:] + csf[:k]


def size_fingerprint(rois, mean_pred):
    """Rank correlation between the model's predicted volumes and the reference sizes, per tissue
    (the lower of the GM and CSF values)."""
    from scipy.stats import spearmanr
    rhos = []
    for t in ("_GM_Vol", "_CSF_Vol"):
        idx = [i for i, r in enumerate(rois) if r.endswith(t) and r in REFERENCE_VOLUMES]
        rhos.append(spearmanr([REFERENCE_VOLUMES[rois[i]] for i in idx], mean_pred[idx]).statistic)
    return min(rhos)


def check_roi_order(model, rois, age_range):
    """Sanity checks that the ROI names match the model's outputs, using the model's own predictions:
    - size fingerprint: predicted volumes must rank like the reference region sizes, and better than
      the same list shifted by 1 or 2 positions (catches shifted or shuffled orders);
    - most GM volumes should decrease with age and most CSF volumes increase (catches GM/CSF swaps);
    - left/right pairs should have similar volumes.
    A left/right swap cannot be detected this way (both hemispheres have similar volumes)."""
    a0, a1 = age_range
    young, old = a0 + 0.2 * (a1 - a0), a0 + 0.8 * (a1 - a0)
    X = pd.DataFrame({"age": [young, young, old, old], "sex": ["female", "male", "female", "male"]})
    p = np.asarray(model.predict(X))
    mid, change = p.mean(axis=0), (p[2] + p[3]) - (p[0] + p[1])
    idx = {r: i for i, r in enumerate(rois)}
    ratios = [mid[idx[r]] / mid[idx["r" + r[1:]]] for r in rois if r.startswith("l") and "r" + r[1:] in idx]
    pairs_ok = np.mean([0.67 < x < 1.5 for x in ratios])
    gm = [i for i, r in enumerate(rois) if r.endswith("_GM_Vol")]
    csf = [i for i, r in enumerate(rois) if r.endswith("_CSF_Vol")]
    gm_down, csf_up = np.mean(change[gm] < 0), np.mean(change[csf] > 0)
    rho = size_fingerprint(rois, mid)
    rho_shift = max(size_fingerprint(_shift_within_tissue(rois, k), mid) for k in (-2, -1, 1, 2))
    print(f"ROI order check: size fingerprint rho = {rho:.2f} (best shifted order: {rho_shift:.2f}); "
          f"{pairs_ok:.0%} of left/right pairs have similar volumes; "
          f"{gm_down:.0%} of GM regions decrease with age, {csf_up:.0%} of CSF regions increase")
    if rho < 0.5 or rho <= rho_shift + 0.05 or pairs_ok < 0.8 or gm_down < 0.6 or csf_up < 0.6:
        sys.exit("The ROI names do not seem to match the model outputs (wrong order?). Check the ROI order file.")


rois_HC = read_roi_order(ROI_ORDER_CSV)
assert len(rois_HC) == len(set(rois_HC)), "duplicate ROI names"
assert len(rois_HC) == 280, f"expected 280 ROIs, found {len(rois_HC)}"
assert len(model.stds_) == len(rois_HC), "model.stds_ does not match the ROI list"


def training_age_range(model):
    """Training age range, read from the fitted age splines: SplineTransformer places its
    boundary knots at the minimum and maximum training age."""
    assert model.bsplines, "age range can only be read from a model fitted with bsplines=True"
    ranges = set()
    for m in model.models_:
        spl = m.named_steps["columntransformer"].named_transformers_["spline_age"]
        t, d = spl.bsplines_[0].t, spl.degree
        ranges.add((round(float(t[d]), 6), round(float(t[-d - 1]), 6)))
    assert len(ranges) == 1, f"different age ranges across ROI models: {ranges}"
    return ranges.pop()


# curves only within the training age range (no extrapolation)
age_min, age_max = training_age_range(model)
check_roi_order(model, rois_HC, (age_min, age_max))
ages = np.round(np.linspace(age_min, age_max, N_AGES), 2)

preds = {}
for sex in ["female", "male"]:
    X = pd.DataFrame({"age": ages, "sex": [sex] * len(ages)})
    p = np.asarray(model.predict(X))
    assert p.shape == (len(ages), len(rois_HC)), p.shape
    assert np.isfinite(p).all(), f"non-finite predictions for {sex}"
    preds[sex] = p

rois, skipped = {}, []
for i, roi in enumerate(rois_HC):
    if not roi.endswith(("_GM_Vol", "_CSF_Vol")):
        continue
    sd = float(model.stds_[i])
    if not np.isfinite(sd) or sd <= 0:   # e.g. a region with constant volumes: no usable chart
        skipped.append(roi)
        continue
    rois[roi] = {
        "f": [round(float(v), 4) for v in preds["female"][:, i]],
        "m": [round(float(v), 4) for v in preds["male"][:, i]],
        "sd": round(sd, 6),
    }

out = {
    "name": MODEL_NAME,
    "demo": False,
    "ages": [float(a) for a in ages],
    "age_range": [round(age_min, 1), round(age_max, 1)],
    "n_train": N_TRAIN,          # whole healthy cohort (train + test split), if set above
    "tiv_target": 1500.0,        # training volumes were scaled as volume x 1500 / TIV (see the data preparation script)
    "exported": datetime.date.today().isoformat(),
    "rois": rois,
}
with open(OUT, "w") as f:
    f.write("// Normative curves exported by export_normative_model.py\n")
    f.write("window.NORMATIVE_MODEL = " + json.dumps(out, separators=(",", ":")) + ";\n")

n_gm = sum(k.endswith("_GM_Vol") for k in rois); n_csf = sum(k.endswith("_CSF_Vol") for k in rois)
assert n_gm + n_csf + len(skipped) == 280, f"expected 280 ROIs, found {n_gm} GM + {n_csf} CSF + {len(skipped)} skipped"
if skipped:
    print(f"WARNING: {len(skipped)} ROI(s) skipped because their SD is zero or not finite: {', '.join(skipped)}")
print(f"Wrote {OUT}: {n_gm} GM + {n_csf} CSF ROIs, ages {age_min:.1f}-{age_max:.1f}")

# prints to have a glimpse at how well things turned out
for roi in ["lHip_GM_Vol", "rAmy_GM_Vol", "lLatVen_CSF_Vol"]:
    if roi in rois:
        print(f"  {roi}: female mean at {ages[0]} y = {rois[roi]['f'][0]}, male = {rois[roi]['m'][0]}, sd = {rois[roi]['sd']}")
