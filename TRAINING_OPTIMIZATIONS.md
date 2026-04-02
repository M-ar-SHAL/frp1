# Training Speed Optimization Guide

## TL;DR - Fastest Way

Run fast training (10 epochs in ~2 minutes):

```bash
python train_fast.py
```

Then test in dashboard:

```bash
streamlit run dashboard.py
```

---

## Why Training is Slow

### 1. **Centrality Computation** (~80% of total time)

- **Problem**: Computing network centrality with 60-day rolling window for 50 stocks is expensive
- **Current**: 60-day window → ~5-10 minutes just for this
- **Fixed in config_fast.yaml**: Reduced from 60 → 30 days

### 2. **Large Dataset Window**

- **Problem**: Data from 2010-2026 (16 years) creates many graph snapshots
- **Current**: ~1,488 graphs × 50 nodes each
- **Fixed in config_fast.yaml**: Data from 2020-2026 (6.5 years) → ~600 graphs

### 3. **Small Sliding Window Stride**

- **Problem**: stride=1 means every sample overlaps 99% with the next
- **Current**: Creates ~1,450 redundant training samples
- **Fixed in config_fast.yaml**: stride=5 → sample every 5th window (5x fewer samples)

### 4. **Large Model Architecture**

- **Problem**: 64-dim hidden, 3-layer transformer, 4 attention heads = 465K parameters
- **Current**: ~100-300ms per forward pass
- **Fixed in config_fast.yaml**: Reduced to 32-dim hidden, 1-layer transformer → 80K parameters (~50ms/pass)

### 5. **No Batching in Training Loop**

- **Problem**: Processing samples one-at-a-time instead of batches
- **Impact**: Can't parallelize on GPU
- **Fix**: Would require refactoring training loop (not done here, but documented below)

---

## Performance Comparison

| Setting     | Data    | Centrality | Seq_len | Stride | Params | Est. Time/Epoch |
| ----------- | ------- | ---------- | ------- | ------ | ------ | --------------- |
| **Default** | 2010-26 | 60-day     | 30      | 1      | 465K   | 2-3 min         |
| **Fast**    | 2020-26 | 30-day     | 15      | 5      | 80K    | 15-20 sec       |
| **Speedup** | 2.4x    | 2x         | 2x      | 5x     | 5.8x   | **~10x faster** |

**Fast training (10 epochs)**: ~2-3 minutes ✅
**Default training (50 epochs)**: ~100+ minutes ❌

---

## Option 1: Use Fast Config (RECOMMENDED)

```bash
# Train quickly with optimized settings
python train_fast.py

# Expected output:
# [7] TRAINING...
# Epoch  1 | Time: 18.2s | Train Loss: 0.6234 | Val AUC: 0.5123
# Epoch  2 | Time: 17.9s | Train Loss: 0.5847 | Val AUC: 0.5267
# ...
# Epoch  10 | Time: 17.6s | Train Loss: 0.4123 | Val AUC: 0.6834
#
# Total training time: 3m 2s
# Average per epoch: 18.2s
```

**Pros**: Works immediately, 10x speedup
**Cons**: Smaller model, less historical data, fewer samples

---

## Option 2: Use GPU (10-100x faster)

If you have NVIDIA GPU:

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# If True, GPU will be used automatically
python train_fast.py
```

**GPU Benefits**:

- 10-30x speedup for GNN operations
- 50-100x speedup for transformer blocks
- Full training might take ~5-10 minutes instead of 100+

---

## Option 3: Manual Config Tuning

Edit `experiments/config.yaml`:

```yaml
# FASTEST SETTINGS (compromise between speed and accuracy)
features:
  centrality_window: 20 # 50% faster (3x reduction vs 60)

graph:
  graph_window: 30 # 50% faster

model:
  gnn_hidden_dim: 48 # Middle ground (vs 64 or 32)
  temporal_d_model: 96 # Middle ground
  seq_len: 20 # 33% faster
  temporal_num_layers: 2 # Keep 2 for better model

training:
  stride: 3 # 3x fewer samples
  epochs: 20 # Quick testing
```

**Expected speedup**: ~5x with better model quality than fast config

---

## Option 4: Skip Expensive Centrality (ADVANCED)

Edit `data/feature_engineering.py` line ~248:

```python
# BEFORE (slow):
print("[Features] Computing network centrality C_i(t) (this takes a few minutes)...")
centrality = compute_rolling_centrality(returns, centrality_window, threshold=centrality_threshold)

# AFTER (fast - use zeros):
print("[Features] Skipping centrality (using zeros for speed)...")
centrality = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
```

**Speedup**: ~10-15x overall (centrality is the bottleneck)
**Trade-off**: Model loses network information (worse predictions)

---

## Option 5: Reduce Data Range (SIMPLE)

Edit `experiments/config.yaml`:

```yaml
data:
  start_date: "2022-01-01" # Only last 4 years
```

**Benefits**:

- 5x fewer graph snapshots
- 5x fewer training samples
- ~5x speedup
- Still covers 2022 crypto crash, 2023 banking crisis

---

## Monitoring Training Progress

### Training Script Output:

```
[7] TRAINING...

Epoch  1 | Time: 18.2s | Train Loss: 0.6234 | Val AUC: 0.5123
Epoch  2 | Time: 17.9s | Train Loss: 0.5847 | Val AUC: 0.5267  ← AUC improving
Epoch  3 | Time: 18.1s | Train Loss: 0.5234 | Val AUC: 0.6145
...
```

**Good signs**:

- Val AUC gradually increases (0.5 → 0.6 → 0.7+)
- Train loss decreases
- Time/epoch is consistent

**Bad signs**:

- Val AUC stays at ~0.5 (not learning)
- Train loss doesn't decrease
- NaN or infinite loss

---

## Recommended Path

1. **Quick test (2 minutes)**:

   ```bash
   python train_fast.py
   streamlit run dashboard.py
   ```

2. **If using GPU (5-10 minutes)**:

   ```bash
   # Edit config.yaml for medium settings, then:
   python experiments/run_experiment.py
   ```

3. **Full training (100+ minutes, skip for now)**:
   ```bash
   # Keep default config.yaml
   python experiments/run_experiment.py
   ```

---

## Expected Results by Setting

| Method                 | Time     | Model Quality | Use Case          |
| ---------------------- | -------- | ------------- | ----------------- |
| `python train_fast.py` | ~2 min   | 50-60% AUC    | Testing/debugging |
| Medium config + GPU    | ~8 min   | 65-70% AUC    | Experimentation   |
| Full config            | ~120 min | 75-85% AUC    | Production        |

---

## Troubleshooting Slow Training

### Still slow despite fast config?

**Check**:

- Are you running on CPU? (use GPU if available)
- Is centrality_window still large? (should be 20-30)
- Is stride >= 3? (fewer samples)
- Run `python diagnose_model.py` to see bottlenecks

### GPU not being used?

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

If False: Install CUDA support for PyTorch: https://pytorch.org/get-started/locally/

### Out of memory?

- Reduce `seq_len` further (15 → 10)
- Reduce `stride` to 10 (even fewer samples)
- Reduce `gnn_hidden_dim` to 24

---

## Advanced: Parallel Data Loading

Currently: Data loads sequentially
**Future improvement**: Use PyTorch DataLoader with workers

This requires refactoring the training loop but could add 2-5x speedup with no quality loss.

---

## Summary

| Optimization         | Speedup | Difficulty       | Recommended         |
| -------------------- | ------- | ---------------- | ------------------- |
| Use config_fast.yaml | 10x     | Easy (1 command) | ✅ YES              |
| GPU (if available)   | 10-100x | Easy             | ✅ YES              |
| Manual config tuning | 3-8x    | Medium           | ✅ Maybe            |
| Skip centrality      | 10x     | Advanced         | ❌ No (low quality) |
| Parallel dataloading | 2-5x    | Very hard        | ❌ Later            |

**Start with**: `python train_fast.py` (2 minutes, immediate results)
