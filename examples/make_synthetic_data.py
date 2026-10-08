"""
Generate **SYNTHETIC** CAT12-style ROI volumes (Neuromorphometrics atlas: 140 GM + 140 CSF columns)
to test the brain normative atlas explorer. These are not real participants.

How the values are made
-----------------------
- Volumes are generated in TIV-scaled space, like the normative model's training data: each
  participant's volumes correspond to volume x 1500 / TIV (total intracranial volume, ml).
  This minimizes the head-size (TIV) heterogeneity within each sex in the normative curves.
- Mean volume for each region, age and sex: taken from the normative model itself, by loading the
  pickle and calling model.predict() at each participant's exact age and sex. Each synthetic volume
  is mean + z x SD, where SD is the model's residual standard deviation for that region
  (model.stds_), and z is the z-score in SD units, so the synthetic data are on exactly the scale of the model.
- Mean volume by age and sex: because TIV scaling doesn't take into account different proportions of GM, WM, and CSF
  for varying head sizes, females can have slightly larger gray matter volumes than males.
  (females have smaller heads than males)
  Part of this difference could be real morphological differences, but so far the literature is mixed on that question.
  --> see Sanchis-Segura et al, "Sex differences in gray matter volume: how many and how large are they really?", 
  Biol Sex Differ. 2019 Jul 1, doi: 10.1186/s13293-019-0245-7.
- The general trend: gray matter declines with age and cerebrospinal fluid (CSF) increases.
- Each participant also has a raw TIV (about 1400 ml for females, 1600 ml for males). With --raw,
  the file contains raw (unscaled) volumes plus a TIV column, to test the app's automatic scaling.

Deviations from the norm (z, in SD units)
-----------------------------------------
- Gray matter: z = 0.6 x (participant offset) + 0.6 x (random noise per region).
- The offset is arbitrarily chosen for the participant examples created to test the display.
- Group 2 participants have lower offsets than group 1 participants to create a difference between the two groups.
- CSF: the participant offset has the opposite sign (lower gray matter goes with more CSF).
- Some participants are given a lower gray matter volume (and more CSF) in the hippocampus, amygdala
  and entorhinal cortex, so that something stands out in the plots.
- Volumes are kept positive (at least 5% of the normative mean) for very small regions.

Usage
-----
Edit the default paths below (same as in export_normative_model.py) or pass them as options:
  python make_synthetic_data.py                                  # 4 participants (2 F, 2 M) (scaled by TIV)
  python make_synthetic_data.py --preset groups                  # 8 participants (4 F, 4 M), 2 groups (scaled by TIV)
  python make_synthetic_data.py --preset groups --raw            # same, raw volumes + TIV column (to test TIV scaling)
  python make_synthetic_data.py --pkl model.pkl --roi-order-csv regions_neuromorphometrics_cat12_7.csv --module-dir path/to/folder
Needs numpy, pandas and the packages used by NormativeBLR (scikit-learn, scipy).
"""
import argparse, csv, os, pickle, random, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# --- default paths: files in the app folder (the parent of examples/); same as in export_normative_model.py ---
APP_DIR = os.path.normpath(os.path.join(HERE, ".."))
MODULE_DIR = APP_DIR                                                         # folder containing normative_models.py
MODEL_PATH = os.path.join(APP_DIR, "normative_BLR_BigHC_janv2026_280rois.pkl")
ROI_ORDER_CSV = os.path.join(APP_DIR, "regions_neuromorphometrics_cat12_7.csv")   # ROI names in the model's output order
# -----------------------------------------------------------------------------------------------------

# (participant_id, sex, age, group, participant offset in SD, medial temporal offset in SD or None, raw TIV in ml)
PRESETS = {
    'basic': [('sub-01', 'female', 27.4, None, 0.3, None, 1386.0), ('sub-02', 'female', 61.8, None, -0.4, None, 1452.0),
              ('sub-03', 'male', 34.9, None, 0.5, None, 1618.0), ('sub-04', 'male', 70.2, None, -0.8, -2.3, 1561.0)],
    'groups': [('sub-01', 'female', 24.6, 'group_one', 0.4, None, 1411.0), ('sub-02', 'female', 38.1, 'group_two', -0.3, -1.2, 1347.0),
               ('sub-03', 'female', 52.7, 'group_one', 0.1, None, 1478.0), ('sub-04', 'female', 66.3, 'group_two', -0.5, -1.6, 1395.0),
               ('sub-05', 'male', 29.2, 'group_two', -0.2, -1.0, 1642.0), ('sub-06', 'male', 43.8, 'group_one', 0.5, None, 1575.0),
               ('sub-07', 'male', 58.4, 'group_two', -0.6, -1.5, 1689.0), ('sub-08', 'male', 71.0, 'group_one', 0.2, None, 1534.0)],
}
TIV_TARGET = 1500.0
MEDIAL_TEMPORAL = ('Hip', 'Amy', 'Ent')


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


def check_roi_order(model, rois, age_range):
    """Sanity check that the ROI names match the model's outputs, using the model's own predictions:
    left/right pairs should have similar volumes, most GM volumes should decrease with age and most
    CSF volumes increase. A shifted, shuffled or GM/CSF-swapped order fails these checks."""
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
    print(f"ROI order check: {pairs_ok:.0%} of left/right pairs have similar volumes, "
          f"{gm_down:.0%} of GM regions decrease with age, {csf_up:.0%} of CSF regions increase")
    if pairs_ok < 0.8 or gm_down < 0.6 or csf_up < 0.6:
        sys.exit("The ROI names do not seem to match the model outputs (wrong order?). Check the ROI order file.")


def training_age_range(model):
    """Training age range, read from the fitted age splines (boundary knots = min and max training age)."""
    spl = model.models_[0].named_steps['columntransformer'].named_transformers_['spline_age']
    t, d = spl.bsplines_[0].t, spl.degree
    return float(t[d]), float(t[-d - 1])


def means_from_pkl(pkl, roi_order_csv, module_dir, people):
    """Normative means and SDs from the pickle, with model.predict() at each participant's exact age and sex."""
    sys.path.insert(0, module_dir)
    from normative_models import NormativeBLR  # noqa: F401  (needed to unpickle)
    with open(pkl, 'rb') as f:
        model = pickle.load(f)
    rois = read_roi_order(roi_order_csv)   # = model output order
    assert len(rois) == len(model.stds_), 'ROI list does not match model.stds_'
    age_min, age_max = training_age_range(model)
    check_roi_order(model, rois, (age_min, age_max))
    for pid, sex, age, *_ in people:   # no extrapolation outside the training age range
        if not age_min <= age <= age_max:
            sys.exit(f'{pid}: age {age} is outside the training age range {age_min}-{age_max}')
    X = pd.DataFrame({'age': [p[2] for p in people], 'sex': [p[1] for p in people]})
    means = np.asarray(model.predict(X))
    assert means.shape == (len(people), len(rois)) and np.isfinite(means).all()
    return rois, means, np.asarray(model.stds_, dtype=float)


def main():
    ap = argparse.ArgumentParser(description='Synthetic CAT12-style participants drawn from the normative model.')
    ap.add_argument('--pkl', default=MODEL_PATH, help='normative model pickle')
    ap.add_argument('--roi-order-csv', default=ROI_ORDER_CSV, help='ROI names in the model output order (one per row, or as CSV header)')
    ap.add_argument('--module-dir', default=MODULE_DIR, help='folder containing normative_models.py')
    ap.add_argument('--preset', choices=PRESETS, default='basic')
    ap.add_argument('--raw', action='store_true', help='write raw volumes (scaled volume x TIV / 1500) and a TIV column')
    ap.add_argument('--out', default=None, help='output CSV (default depends on preset and --raw)')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    people = PRESETS[args.preset]
    rois, means, sds = means_from_pkl(args.pkl, args.roi_order_csv, args.module_dir, people)
    sds = np.where(np.isfinite(sds), sds, 0.0)   # a region with no usable SD gets its normative mean
    source = args.pkl

    rng = np.random.default_rng(args.seed)
    with_group = any(p[3] is not None for p in people)
    rows = []
    for k, (pid, sex, age, group, glob, mt, tiv) in enumerate(people):
        row = {'participant_id': pid, 'age': age, 'sex': sex}
        if with_group: row['group'] = group
        if args.raw: row['TIV'] = tiv
        for roi in sorted(rois):                        # draws in ROI-name order: same output from the .js or the .pkl
            j = rois.index(roi)
            code, tissue = roi.rsplit('_', 2)[0], roi.rsplit('_', 2)[1]
            sign = 1 if tissue == 'GM' else -1          # lower GM goes with more CSF
            if mt is not None and code.endswith(MEDIAL_TEMPORAL):
                z = sign * mt + 0.3 * rng.standard_normal()
            else:
                z = sign * 0.6 * glob + 0.6 * rng.standard_normal()
            v = round(float(max(means[k, j] + z * sds[j], 0.05 * means[k, j])), 4)   # TIV-scaled volume
            # raw volume (undo the TIV scaling), with enough decimals that x 1500 / TIV gives back exactly v
            row[roi] = round(v * tiv / TIV_TARGET, 7) if args.raw else v
        rows.append(row)

    roi_cols = list(rois)
    random.Random(3).shuffle(roi_cols)  # columns in random order, to show the app does not depend on it
    out = args.out or os.path.join(HERE, 'fake_participants_cat12' + ('' if args.preset == 'basic' else f'_{args.preset}')
                                   + ('_raw_with_TIV' if args.raw else '') + '.csv')
    head = ['participant_id', 'age', 'sex'] + (['group'] if with_group else []) + (['TIV'] if args.raw else [])
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, head + roi_cols); w.writeheader(); w.writerows(rows)
    n_gm = sum(r.endswith('_GM_Vol') for r in rois); n_csf = sum(r.endswith('_CSF_Vol') for r in rois)
    print(f'Wrote {out}: {len(rows)} synthetic participants, {n_gm} GM + {n_csf} CSF columns, '
          + ('raw volumes + TIV column' if args.raw else f'volumes scaled to TIV = {TIV_TARGET:.0f} ml')
          + f' (from {os.path.basename(source)})')


if __name__ == '__main__':
    main()
