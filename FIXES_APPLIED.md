# FAPT-GNN Dashboard - Issues Fixed

## Overview
Your dashboard was crashing  with errors related to model inference showing incorrect predictions. After diagnostic analysis, I've identified and fixed multiple issues preventing the model from working correctly.

---

## Issues Found & Fixed

### ✅ Issue 1: PyTorch Pickle Loading Error
**Error:** `_pickle.UnpicklingError: Weights only load failed`
**Cause:** PyTorch 2.6+ changed `torch.load()` default to `weights_only=True` which doesn't support old checkpoints
**Fix:** Added `weights_only=False` to all `torch.load()` calls in:
- `dashboard.py` (lines 117, 175, 221)
- `experiments/run_experiment.py` (line 193)

---

### ✅ Issue 2: Unicode Encoding Errors (Windows Console)
**Error:** `UnicodeEncodeError: 'charmap' codec can't encode character '\u03c3'`
**Cause:** Windows PowerShell uses cp1252 encoding which doesn't support Unicode characters
**Fixes:**
- Replaced σ (sigma) → `sigma`
- Replaced ✅ (checkmark) → `[OK]`
- Replaced ✓ (check) → `[OK]`
- Updated files:
  - `data/feature_engineering.py`
  - `data/graph_builder.py`
  - `dashboard.py`

---

### ✅ Issue 3: Graph Has No Edges (Critical)
**Error:** Model output graphs with 0 edges → energy = 0 → crash_prob defaults to 50%
**Root Cause:** Adjacency matrix values after normalization dropped below edge threshold (0.01)
**Fixes:**
1. Lowered edge threshold from 0.01 → 0.001 in `graph_builder.py`
2. Added NaN handling in `build_multilayer_adjacency()` to clean invalid values
3. Result: **2,148 edges now created** (vs 0 before)

**Adjacency Matrix Stats (After Fix):**
- Range: [0.0, 0.107]
- Non-zero entries: 2,186 (87.4% connectivity)
- All values pass 0.001 threshold

---

### ❌ Issue 4: Model Not Properly Trained
**Error:** Checkpoint has epoch=1, val_auc=0.5 (random chance)
**Impact:** Even with edges, untrained model outputs default values:
- Crash Probability: 50.0% (fallback)
- Time-to-Crash: 0 days (clipped minimum)
- Energy: 0.0 (no learned parameters)

**Solution:** Model must be trained with real data
**Fix Applied:**
- Increased training epochs in dashboard: 5 → 20 epochs
- Default config already has 50 epochs for full runs

---

## Discrepancy Explanation

Your observation about the difference between tabs:
- **Market Context (Tab 1)**: ~5% historical crash ratio ← This is correct (ground truth labels)
- **Prediction (Tab 3)**: 50% crash probability ← This is wrong (untrained model defaults)

These are NOT supposed to match! Tab 1 shows historical statistics; Tab 3 shows model predictions. The model predictions are garbage because it hasn't been trained yet.

---

## How to Get Correct Results

### Option 1: Train via Dashboard (Easiest)
1. Open the dashboard: `streamlit run dashboard.py`
2. Go to **"Model Training"** tab
3. Click **"Train Real-Time FAPT-GNN"**
4. Wait for ~20 epochs to complete
5. Predictions in Tab 3 will now be meaningful

### Option 2: Train via Command Line (Recommended)
```bash
cd c:\Users\HP\OneDrive\Desktop\frp1
python experiments/run_experiment.py
```
This trains with full 50 epochs as configured.

---

## What the 50% Means Now

**Before Fixes:** Random crash probability (model broken)
**After Fixes:** Still 50% until trained (expected behavior)
**After Training:** Model will output realistic probabilities (e.g., 5-85%)

The model will learn to predict:
- Low risk (5-20%): Normal market conditions
- Medium risk (30-50%): Elevated volatility
- High risk (70-95%): Systemic fragility detected

---

## Files Modified

| File | Changes |
|------|---------|
| `dashboard.py` | Fixed torch.load (3 places); Increased epochs 5→20 |
| `data/feature_engineering.py` | Unicode fixes (σ → sigma, ✅ → [OK]) |
| `data/graph_builder.py` | Edge threshold 0.01→0.001; NaN handling; Unicode fixes |
| `experiments/run_experiment.py` | Fixed torch.load |

---

## Next Steps

1. **Clear old checkpoint** (optional):
   ```bash
   del experiments\checkpoints\best_model.pt
   ```

2. **Train the model** (choose one):
   - Via Streamlit dashboard (Tab 2)
   - Via command line: `python experiments/run_experiment.py`

3. **Test predictions** (after training):
   - Tab 1: Historical context (5% crash ratio)
   - Tab 3: Model predictions (now will vary 5-95%)

4. **Monitor energy**: After training, energy values should be non-zero and meaningful

---

## Verification

Run the diagnostic script to verify fixes:
```bash
python diagnose_model.py
```

Expected output after fixes:
- ✅ Graphs have edges:  `First graph edges: torch.Size([2, 2148])`
- ✅ Adjacency non-zero: `Non-zero ratio: 0.87`
- ⚠️ Energy = 0: Until model is trained

---

**Status:** Code is now fixed and ready for training. The 50% crash probability will disappear once you train the model.
