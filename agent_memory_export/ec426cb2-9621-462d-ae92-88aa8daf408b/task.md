# Tasks: Production-Readiness Remediation

## 🔴 Phase 0 — Critical Fixes
- `[x]` **0.1** Fix data leakage — fit scaler on training data only (`data_loader.py`)
- `[x]` **0.2** Set random seeds for reproducibility (`train.py`)
- `[x]` **0.3** Create `requirements.txt` with pinned versions
- `[x]` **0.4** Replace `print()` with `logging` module (all files) — console + file output
- `[x]` **0.5** Add input validation on public functions (`arima_model.py`, `data_loader.py`)

## 🟠 Phase 1 — High Priority
- `[x]` **1.1** Externalize configuration (YAML config + argparse)
- `[x]` **1.2** Break up `main()` in `train.py` into distinct functions
- `[x]` **1.3** Add comprehensive tests (80%+ coverage target) — 24 tests passing
- `[ ]` **1.4** Create `pyproject.toml` with packaging config
- `[ ]` **1.5** Set up CI/CD (GitHub Actions)

## 🟡 Phase 2 — Medium Priority
- `[ ]` **2.1** Add data validation (schema checks)
- `[ ]` **2.2** Remove hardcoded absolute paths from SLURM scripts
- `[ ]` **2.3** Add Makefile for common tasks

## 🟢 Phase 3 — Polish
- `[ ]` **3.1** Add pre-commit hooks (ruff, black)
- `[ ]` **3.2** Add CONTRIBUTING.md, CHANGELOG.md
- `[ ]` **3.3** Architecture diagram in README

## 🚀 Final
- `[ ]` Re-run training on cluster with fixed pipeline
- `[ ]` Update metrics in README and model_comparison.md
- `[ ]` Git commit and push all changes
