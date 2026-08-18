# Similarity-Normalized Conformal Prediction for NCI-60

Research code for reliable anticancer drug-response prediction across the NCI-60 panel under chemical shift. The repository implements data curation, leakage-controlled chemical splits, point-prediction models, similarity-normalized conformal prediction, comparator methods, external HTS384 evaluation, sensitivity analyses, figures, and regression tests.

## Repository contents

- `scripts/`: preprocessing, split generation, model training, calibration, comparator, statistical-analysis, and figure scripts.
- `experiments/`: shared-scale theory simulation used by the theoretical validation.
- `tests/`: unit and regression tests based on synthetic or temporary fixtures.
- `splits/phase1a/`: frozen compound identifiers for the five random, scaffold, and leader-cluster seeds.
- `requirements/`: pinned CPU/base and CUDA-enabled dependency specifications.
- `DATA.md`: source, expected layout, and redistribution notes.

Raw CellMiner files, processed matrices, learned models, and generated result directories are intentionally excluded. They are covered by `.gitignore` and must not be committed accidentally.

## Environment

Python 3.11 is recommended. From the repository root:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` selects the CUDA 12.1 PyTorch build retained from the validated experiment environment and installs the test runner. For a CPU-only installation, first install the appropriate PyTorch wheel for the target platform and then install `requirements/requirements-phase0.txt` plus `pytest`.

## Data preparation

Download the public CellMiner exports described in [`DATA.md`](DATA.md), preserve the expected directory structure, and run:

```bash
python scripts/preprocess_cellminer.py --root .
python scripts/make_basic_splits.py --root . --seeds 1,2,3,4,5
python scripts/make_chemical_splits.py --root . --seeds 1,2,3,4,5
python scripts/prepare_phase1a.py --root . --seeds 1,2,3,4,5
```

The frozen identifiers in `splits/phase1a/` are provided for audit and comparison. Generated working splits are written below `data/splits/`.

## Core experiment sequence

The principal development and confirmation commands are:

```bash
python scripts/run_phase1a_baselines.py --root .
python scripts/evaluate_phase1a_conformal.py --root .
python scripts/evaluate_phase1b_similarity.py --root .
python scripts/select_dad_k_development.py --root .
python scripts/evaluate_dad_locked_confirmation.py --root .
python scripts/prepare_hts384_lockbox.py --root .
python scripts/evaluate_hts384_comparators.py --root . --families scaffold,leader_cluster --seeds 2,3,4,5 --ensemble-size 100 --bootstrap-replicates 5000
python scripts/correct_endpoint_boundary_labels.py --root .
python scripts/build_submission_results_20260817.py --root .
```

Individual scripts expose additional options through `python scripts/<name>.py --help`. Training and large bootstrap jobs can require substantial compute and memory.

## Tests

Run the automated checks from the repository root:

```bash
python -m pytest -q
```

Tests requiring processed CellMiner features are skipped until the public data-preparation steps have been completed.

## Reproducibility and data leakage controls

- Exact Morgan-fingerprint duplicates remain grouped during split construction.
- Scaffold and leader-cluster families assess chemical novelty beyond random holdout.
- Development seed 1 is used for comparator selection; seeds 2--5 remain confirmation-only.
- External HTS384 outcomes are not used when constructing prediction intervals or selecting hyperparameters.
- Scripts accept `--root` so the repository is portable across operating systems and local paths.

## Citation and license

Please cite the associated manuscript when using this code. A formal software license has not been selected in this upload package; the repository owner should add the intended license before inviting reuse or redistribution.
