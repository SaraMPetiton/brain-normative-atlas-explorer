# Synthetic test files

**These are not real participants.** They are drawn from the normative model (mean for each participant's age and sex, plus a deviation in SD units), to test the app.

| File | Content |
|---|---|
| `fake_participants_cat12.csv` | 4 participants (2 females, 2 males), volumes scaled to TIV = 1500 ml; sub-04 has low hippocampus, amygdala and entorhinal gray matter volumes |
| `fake_participants_cat12_groups.csv` | 8 participants (4 females, 4 males) in two groups (`group_one`, `group_two`), volumes scaled to TIV = 1500 ml; `group_two` has lower medial temporal gray matter volumes |
| `fake_participants_cat12_groups_raw_with_TIV.csv` | the same 8 participants with raw (unscaled) volumes and a `TIV` column, to test the automatic scaling: loading it gives exactly the same charts as the scaled file |
| `make_synthetic_data.py` | the script that generates them (method described at the top of the file) |

All files have the 140 GM and 140 CSF columns of the model, in random order.

## Regenerate them from your normative model

The script loads the model pickle with the `NormativeBLR` class and calls `model.predict()` at each participant's age and sex. By default, the script uses the model pickle, `normative_models.py` and `regions_neuromorphometrics_cat12_7.csv` in the app folder (the parent of `examples/`), like `export_normative_model.py`. Run:

    python3 make_synthetic_data.py
    python3 make_synthetic_data.py --preset groups
    python3 make_synthetic_data.py --preset groups --raw

or give the paths as options:

    python3 make_synthetic_data.py --pkl model.pkl --roi-order-csv ../regions_neuromorphometrics_cat12_7.csv --module-dir path/to/folder_with_normative_models

The ROI order comes from `../regions_neuromorphometrics_cat12_7.csv` by default and is checked against the model's predictions. The training age range is read from the model itself.

Needs `numpy`, `pandas`, and the packages used by `NormativeBLR` (`scikit-learn`, `scipy`).

The files currently in this folder were generated from the demo curves shipped with the app. 
Regenerate them from your model as described above.
