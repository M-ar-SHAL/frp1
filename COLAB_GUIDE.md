# Google Colab Training Guide - FAPT-GNN

## Quick Start (3 steps)

### 1. Upload to Google Drive

```
UPLOAD these to your Google Drive:
  📁 frp1/  (entire project folder)
     ├─ data/
     ├─ models/
     ├─ training/
     ├─ experiments/
     ├─ dashboard.py
     └─ ... (all files)

Result: /MyDrive/frp1/
```

### 2. Open Notebook in Colab

- Download: `FAPT_GNN_Colab_Training.ipynb` from your project
- Go to: [Google Colab](https://colab.research.google.com)
- Upload the notebook
- Or: Right-click notebook on Drive → Open with → Google Colaboratory

### 3. Run All Cells

```
Menu → Runtime → Run all (Ctrl+F9)
```

**Training time**: ~5-10 minutes for 10 epochs on T4 GPU

---

## What Each Section Does

| Cell   | Purpose                                | Time      |
| ------ | -------------------------------------- | --------- |
| 1-2    | GPU check & install packages           | 30s       |
| 3      | Mount Google Drive                     | 5s        |
| 4-5    | Copy project from Drive                | 10s       |
| 6      | Import modules                         | 10s       |
| 7      | Load config (fast settings)            | 1s        |
| 8-9    | Download market data                   | 2-3m      |
| 10     | Build features (centrality bottleneck) | 3-5m      |
| 11     | Calculate crash labels                 | 1m        |
| 12     | Build graph snapshots                  | 1-2m      |
| 13     | Prepare dataset                        | 10s       |
| 14     | Initialize model                       | 5s        |
| **15** | **TRAIN MODEL**                        | **5-10m** |
| 16     | Save model to Drive                    | 30s       |
| 17     | Copy to Drive folder                   | 10s       |
| 18     | Test predictions                       | 20s       |

**Total time**: ~20-30 minutes (most is data prep and feature engineering)

---

## Where to Put the Trained Model

### After Colab Training Finishes

1. **In Google Drive**:
   - Colab automatically saves to: `/MyDrive/frp1_trained_model/best_model.pt`
2. **Download to your computer**:

   ```
   Right-click folder → Download
   ```

3. **Place locally**:
   ```
   c:\Users\HP\OneDrive\Desktop\frp1\experiments\checkpoints\best_model.pt
   ```

### File Structure After Download

```
c:\Users\HP\OneDrive\Desktop\frp1\
├─ experiments/
│  └─ checkpoints/
│     └─ best_model.pt  ← PUT FILE HERE
├─ dashboard.py
├─ train_fast.py
└─ ...
```

### Verify the Model Works

```bash
cd c:\Users\HP\OneDrive\Desktop\frp1

# Check model loads
python diagnose_model.py
```

Expected output:

```
✓ Checkpoint loaded
✓ Model initialized
✓ Predictions: Crash Prob: 0.XXXX, Energy: X.XXXXX
```

---

## Using the Trained Model in Dashboard

Once placed in correct location:

```bash
streamlit run dashboard.py
```

Dashboard will **automatically**:

- ✅ Detect `best_model.pt`
- ✅ Load trained weights
- ✅ Show real predictions (not 50%)
- ✅ Display meaningful energy values

### Expected Results

**Before training** (with random checkpoint):

- Tab 3 Crash Probability: 50.0%
- System Energy: 0.0000
- Time-to-Crash: 0 days

**After training** (with Colab checkpoint):

- Tab 3 Crash Probability: 5-85% (realistic range)
- System Energy: 0.001-0.5 (meaningful values)
- Time-to-Crash: 1-60 days (realistic predictions)

---

## Troubleshooting

### Model not loading in dashboard?

Check directory exists:

```bash
# Windows
dir c:\Users\HP\OneDrive\Desktop\frp1\experiments\checkpoints\

# Should show:
# best_model.pt
```

If missing, move file from Google Drive download.

### "Checkpoint not found" error?

Check the file is in the right place:

```bash
cd c:\Users\HP\OneDrive\Desktop\frp1
python -c "import os; print(os.path.exists('experiments/checkpoints/best_model.pt'))"
```

Should print: `True`

### Model loads but predictions are still 50%?

Colab might not have fully trained. Check:

1. How many epochs completed? (Last few cells show epoch numbers)
2. Val AUC improved? (Should go from ~0.5 to 0.6+)
3. Try retraining with more epochs in Colab

---

## Advanced: Modify Colab Training

### Train for MORE epochs (better model)

Find this cell:

```yaml
training:
  epochs: 10
```

Change to:

```yaml
training:
  epochs: 20 # or 50 for full training
```

⚠️ **Warning**: 50 epochs = ~30-50 minutes, uses more GPU quota

### Use MORE data (better results, slower)

Find this cell:

```yaml
data:
  start_date: "2020-01-01" # 6.5 years
```

Change to:

```yaml
data:
  start_date: "2010-01-01" # 16 years (slower feature engineering)
```

### Reduce stride for more training samples

Find this cell:

```yaml
training:
  stride: 5 # sample every 5th window
```

Change to:

```yaml
training:
  stride: 1 # use all windows (10x more samples, slower)
```

---

## File Locations Reference

| File           | Location                                            | Purpose                |
| -------------- | --------------------------------------------------- | ---------------------- |
| Colab Notebook | `FAPT_GNN_Colab_Training.ipynb`                     | Training script        |
| Trained Model  | `c:\...\frp1\experiments\checkpoints\best_model.pt` | Saved weights          |
| Config File    | `experiments\config_fast.yaml`                      | Fast training settings |
| Dashboard      | `dashboard.py`                                      | Load predictions       |
| Diagnostic     | `diagnose_model.py`                                 | Verify setup           |

---

## Summary

✅ **Steps**:

1. Upload `frp1/` to Google Drive
2. Run Colab notebook (20-30 min)
3. Download `best_model.pt`
4. Place in `c:\...\frp1\experiments\checkpoints\`
5. Run `streamlit run dashboard.py`

✅ **Result**: Dashboard shows real crash predictions (not 50%)

**That's it!** 🚀
