# Production-Readiness Remediation Plan

We'll work through the P0 → P1 → P2 → P3 roadmap from the [enterprise audit](file:///home/alika/.gemini/antigravity-ide/brain/ec426cb2-9621-462d-ae92-88aa8daf408b/enterprise_audit.md), one step at a time. Each non-trivial step will follow the **doubt-driven development** cycle (CLAIM → EXTRACT → DOUBT → RECONCILE → STOP) so you can verify every engineering decision and learn the rationale.

---

## Approach: Doubt-Driven Development (DDD)

For **every non-trivial change**, I will:
1. **CLAIM** — State what I believe the correct change is and why it matters
2. **EXTRACT** — Show you the exact artifact (diff/code) + contract (what it must satisfy)
3. **DOUBT** — Invoke a fresh-context adversarial reviewer to find flaws in my proposal
4. **RECONCILE** — Classify each reviewer finding (actionable / trade-off / noise) and fix what matters
5. **STOP** — Present the final version for your approval

Mechanical changes (file renames, formatting) skip DDD per the skill's own rules.

---

## Step Sequence (Following Audit Priority Order)

### 🔴 Phase 0 — Critical Fixes

| Step | Item | DDD? | Why |
|------|------|:----:|-----|
| **0.1** | Fix data leakage (scaler fit on train only) | ✅ Yes | Correctness of all reported metrics depends on this — highest blast radius |
| **0.2** | Set random seeds for reproducibility | ✅ Yes | Affects whether results are reproducible; seed placement strategy matters |
| **0.3** | Create `requirements.txt` with pinned versions | ❌ Mechanical | Captures `pip freeze` — no design decisions |
| **0.4** | Replace `print()` with `logging` module | ✅ Yes | Touches every file; logging architecture (levels, formatters, handlers) is a design decision |
| **0.5** | Add input validation on public functions | ✅ Yes | Which parameters to validate and how to fail involves judgment |

### 🟠 Phase 1 — High Priority

| Step | Item | DDD? | Why |
|------|------|:----:|-----|
| **1.1** | Externalize configuration (YAML config + argparse) | ✅ Yes | Architecture decision: config format, hierarchy, override semantics |
| **1.2** | Break up `main()` in `train.py` | ✅ Yes | Function boundaries are a design decision |
| **1.3** | Add comprehensive tests (80%+ coverage target) | ✅ Yes | Test strategy, fixtures, edge cases are design decisions |
| **1.4** | Create `pyproject.toml` with packaging config | ✅ Yes | Package structure and tool configuration |
| **1.5** | Set up CI/CD (GitHub Actions) | ✅ Yes | Pipeline stages, when to gate, what to test |

### 🟡 Phase 2 — Medium Priority

| Step | Item | DDD? | Why |
|------|------|:----:|-----|
| **2.1** | Add data validation (schema checks) | ✅ Yes | What to validate, how to fail, tolerance thresholds |
| **2.2** | Remove hardcoded absolute paths from SLURM scripts | ❌ Mechanical | Simple env var substitution |
| **2.3** | Add Makefile for common tasks | ❌ Mechanical | Convenience targets, no logic |

### 🟢 Phase 3 — Polish

| Step | Item | DDD? | Why |
|------|------|:----:|-----|
| **3.1** | Add pre-commit hooks (ruff, black) | ❌ Mechanical | Standard config |
| **3.2** | Add CONTRIBUTING.md, CHANGELOG.md | ❌ Mechanical | Template docs |
| **3.3** | Architecture diagram in README | ❌ Mechanical | Documentation |

---

## Proposed Changes

### Phase 0 — Critical Fixes

---

#### Step 0.1: Fix Data Leakage

##### [MODIFY] [data_loader.py](file:///home/alika/projects/def-csimo/alika/LSTM-Electericity-Load-Forecast/data_loader.py)

**Current problem:** `scaler.fit_transform(df)` on line 91 fits the MinMaxScaler on the **entire dataset** including val/test data. This means the model indirectly "sees" the range of future data during training.

**Proposed fix:** 
1. Compute train/val/test index boundaries **before** scaling
2. Call `scaler.fit()` on the training partition only
3. Call `scaler.transform()` on the full dataset using training-fitted parameters

This is a non-trivial change because it alters the data pipeline and potentially affects all downstream metric values.

---

#### Step 0.2: Set Random Seeds

##### [MODIFY] [train.py](file:///home/alika/projects/def-csimo/alika/LSTM-Electericity-Load-Forecast/train.py)

**Proposed fix:** Add a `set_seeds(seed: int)` function that sets:
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- `np.random.seed(seed)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

Called at the top of `main()` before any data loading or model initialization.

---

#### Step 0.3: Create `requirements.txt`

##### [NEW] [requirements.txt](file:///home/alika/projects/def-csimo/alika/LSTM-Electericity-Load-Forecast/requirements.txt)

Capture current environment with `pip freeze` and curate to project dependencies only.

---

#### Step 0.4: Replace `print()` with `logging`

##### [MODIFY] All Python files

Replace all `print()` calls with proper `logging.info()`, `logging.warning()`, `logging.error()` calls. Add a logging configuration module.

---

#### Step 0.5: Input Validation

##### [MODIFY] [arima_model.py](file:///home/alika/projects/def-csimo/alika/LSTM-Electericity-Load-Forecast/arima_model.py), [data_loader.py](file:///home/alika/projects/def-csimo/alika/LSTM-Electericity-Load-Forecast/data_loader.py)

Add parameter validation (non-negative orders, valid history hours, non-empty dataframes, etc.) with descriptive `ValueError` messages.

---

## Open Questions

> [!IMPORTANT]
> **Q1:** The data leakage fix (Step 0.1) will likely change the reported LSTM/ARIMA metrics slightly. Should I re-run the full training pipeline after the fix so we have updated, honest metrics? Or is the current codebase purely for demonstration and you don't plan to re-train?

> [!IMPORTANT]
> **Q2:** For logging (Step 0.4), do you want logs to go to **both** console and a file, or console only? On the HPC cluster, SLURM already captures stdout/stderr to `.log`/`.err` files, so file-based logging may be redundant there.

> [!IMPORTANT]
> **Q3:** For configuration (Step 1.1), do you prefer **YAML files** (common in ML projects, used by Hydra) or **TOML** (Python-native, used by `pyproject.toml`)? Both are valid — YAML is more common in the ML ecosystem.

---

## Verification Plan

### After Each Step
- Run existing tests (`pytest tests/`) to ensure no regressions
- Run a quick smoke test of the training pipeline (first 2 epochs) to verify the data pipeline still works

### After Phase 0 Complete
- All tests pass
- `python train.py` runs to completion on the cluster (or first few epochs locally)
- `python test_arima_only.py --mode evaluate --stride 24` still produces valid results

### After Phase 1 Complete
- Test coverage ≥ 80% (`pytest --cov`)
- CI pipeline passes on GitHub Actions
- `pyproject.toml` allows `pip install -e .`
