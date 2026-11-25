# data/
* `raw/<dataset>/` original files (never modified, git-ignored, may be licensed: do not redistribute)
* `interim/` scratch
* `processed/<dataset>/` canonical CSVs, predictions, validation reports (generated)
* `synthetic/` generated synthetic data (`python scripts/generate_synthetic_data.py`)
* `external/` optional resources
See docs/DATASET_CREATION_GUIDE.md.
