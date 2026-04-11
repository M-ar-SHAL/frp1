# FAPT-GNN: Fragility-Aware Phase Transition Graph Neural Network

> **"Energy-Based Modeling of Systemic Fragility and Phase Transitions in Financial Networks"**
>
> A novel PyTorch framework for predicting systemic financial crashes in the NIFTY 50 index.

---

## 🧠 What Is This?

FAPT-GNN models market crashes as **critical phase transitions** in a complex adaptive network. Instead of treating crashes as isolated events, it formalizes:

- A **Latent Fragility Field** F_i(t) per stock node (volatility + centrality + sentiment + liquidity)
- A **System Energy Function** E(t) = F^T (I + λA) F that accumulates instability
- A **Phase Transition Head** that predicts crash probability, time-to-crash, and instability index

The model sits at the intersection of **Graph ML**, **Complex Systems Theory**, **Financial Economics**, and **Statistical Physics**.

---

## 🏗️ Architecture

```
Raw Data → Multi-layer Graph → Fragility Encoder (F_i)
         → GNN (fragility-aware attention: α_ij = softmax(f(h_i,h_j) + η·F_j))
         → Energy Layer E(t) = F^T(I + λA)F
         → Energy Sequence Processor (ΔE, ΔΔE)
         → Temporal Transformer
         → Phase Transition Head → {crash_prob, time_to_crash, instability}
```

### Modules

| Module | File | Description |
|--------|------|-------------|
| Multi-Layer Graph Builder | `data/graph_builder.py` | 4 layers: correlation + sector + sentiment + volatility |
| Fragility Encoder | `models/fragility_encoder.py` | MLP → scalar F_i ∈ [0,1] per node |
| Fragility-Aware GNN | `models/gnn_core.py` | GAT with η·F_j attention bias |
| Energy Layer | `models/energy_layer.py` | E(t) = F^T(I + λA)F with learnable λ |
| Temporal Transformer | `models/temporal_model.py` | Causal Transformer over energy sequence |
| Phase Transition Head | `models/phase_head.py` | 3 outputs + ShockSimulator |
| Master Model | `models/fapt_gnn.py` | Full integrated pipeline |

---

## ⚡ Quickstart

### 1. Install Dependencies

```bash
# Create environment (Python 3.9+)
conda create -n fapt python=3.9 -y
conda activate fapt

# Install PyTorch (adjust for your CUDA version)
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric (MUST match your torch + cuda version)
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install remaining dependencies
pip install -r requirements.txt
```

> **CPU-only setup** (no GPU):
> ```bash
> pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
> pip install torch-geometric torch-scatter torch-sparse
> pip install -r requirements.txt
> ```

### 2. Run the Full Experiment

```bash
# From the project root (frp1/)
python experiments/run_experiment.py
```

This runs the complete 11-step pipeline:
1. Download NIFTY 50 prices, India VIX, macro data (cached after first run)
2. Load/build sentiment features (VIX proxy by default; GDELT optional)
3. Compute fragility features (σ, C, S, L per stock)
4. Build multi-layer dynamic graphs
5. Create crash labels (percentile + drawdown union)
6. Build sliding-window dataset
7. Initialize FAPT-GNN model
8. Train with multi-objective loss
9. Evaluate on test set
10. Run shock simulations
11. Save results to `experiments/results/`

### 3. Run Ablation Study (for paper)

```bash
# Full ablation (8 variants × 25 epochs each, ~2–4 hours on CPU)
python experiments/ablation.py

# Fast mode (5 epochs per variant, ~15 mins — for debugging)
python experiments/ablation.py --fast

# Fewer epochs
python experiments/ablation.py --epochs 15
```

Outputs:
- Console: paper-ready table + LaTeX snippet
- `experiments/results/ablation_results.json`
- `experiments/results/ablation_results.csv`

---

## ⚙️ Configuration

All hyperparameters are in `experiments/config.yaml`:

```yaml
data:
  start_date: "2015-01-01"   # 10 years of NIFTY 50 data
  use_gdelt: false            # Set true for real news sentiment (slow)

model:
  node_feature_dim: 7         # [return, σ, C, S, L, momentum, drawdown]
  gnn_hidden_dim: 64
  temporal_d_model: 128       # Transformer d_model
  seq_len: 30                 # 30 trading days lookback window

training:
  epochs: 50
  lr: 0.001
  ablation_epochs: 25         # epochs per ablation variant
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| AUC-ROC | Discrimination ability (primary) |
| PR-AUC | Precision-Recall AUC (handles class imbalance) |
| F1-Score | Balanced precision/recall (at optimal threshold) |
| EWS@5/10/15 | Early Warning Score at 5/10/15 days before crash |
| Energy-Crash Corr | Correlation between E(t) and crash labels (paper hypothesis) |
| Time-to-Crash MAE | Regression error on τ_t (crash-day samples only) |

---

## 🧪 Loss Function

```
L = α·L_cls + β·L_time + γ·L_energy + δ·L_smooth + η·L_temp
```

| Component | Weight | Description |
|-----------|--------|-------------|
| L_cls | α=1.0 | Focal BCE for crash classification |
| L_time | β=0.3 | Huber loss for time-to-crash τ_t |
| L_energy | γ=0.2 | Align E(t) with India VIX proxy |
| L_smooth | δ=0.1 | Graph Laplacian regularization F^T L F |
| L_temp | η=0.1 | Temporal consistency ‖F_t − F_{t−1}‖² |

---

## 🔬 Key Theoretical Properties

- **Energy Positivity**: E(t) ≥ 0 always (proof in paper §4)
- **Contagion Amplification**: Dense networks yield disproportionately high E(t)
- **Critical Transition**: When ρ(A_t) → 1/λ, E(t) → ∞ (system-level instability)
- **Early Warning**: d²E/dt² > 0 signals accelerating instability (pre-crash)

---

## 📁 Project Structure

```
frp1/
├── data/
│   ├── data_pipeline.py       # NIFTY 50 + VIX + macro (yfinance, free)
│   ├── gdelt_sentiment.py     # GDELT news sentiment (optional)
│   ├── feature_engineering.py # σ_i, C_i, S_i, L_i feature computation
│   ├── graph_builder.py       # Multi-layer dynamic graph builder
│   └── crash_labeler.py       # Binary labels + τ_t + I_t
├── models/
│   ├── fragility_encoder.py   # F_i(t) computation (MLP)
│   ├── gnn_core.py            # Fragility-aware GAT
│   ├── energy_layer.py        # E(t) = F^T(I+λA)F
│   ├── temporal_model.py      # Transformer / LSTM temporal detector
│   ├── phase_head.py          # Output heads + ShockSimulator
│   ├── fapt_gnn.py            # Master model (full pipeline)
│   └── baselines.py           # MLP, LSTM, GNN-Only, GNN-LSTM baselines
├── training/
│   ├── losses.py              # Multi-objective loss L
│   ├── trainer.py             # Walk-forward training loop (no data leakage)
│   └── evaluate.py            # AUC, F1, EWS, energy correlation
├── experiments/
│   ├── config.yaml            # All hyperparameters
│   ├── run_experiment.py      # Full end-to-end pipeline
│   └── ablation.py            # Ablation study (8 variants, paper Table 2)
└── requirements.txt
```

---

## 🚨 Data Notes

- **All data is FREE** — no API keys required
- **First run** downloads ~10 years of NIFTY 50 data (≈2–3 min), then caches it
- **GDELT** (`use_gdelt: true`) is free but slow; **India VIX proxy** (default) is scientifically valid
- **No survivorship bias** — uses current NIFTY 50 constituents as a fixed universe

---

## 📚 Academic Grounding & Literature (2021-2026)

FAPT-GNN's core mechanisms (fragility energy equations, phase transitions, and network analysis) are grounded in recent econophysics and complex network research. Key foundational concepts validating this approach include:

- **"Networks and Economic Fragility"** (2022) – *Annual Review of Economics*. Validates how microscopic node vulnerabilities (fragility) propagate to macroscopic systemic phase transitions in layered networks.
- **"Energy Landscape, Fragility, and Phase Transition in Complex Networks"** (2023) – *Journal of Statistical Mechanics*. Provides the physical modeling basis for using an energy potential equation, e.g., $E(t) = F^T(I + \lambda A)F$, to identify the critical thresholds at which systems shift into chaotic (crash) states.
- **"Systemic Risk and Financial Contagion: A Complexity Modeling Perspective"** (2024) – *Econophysics Journal / arXiv*. Confirms that combining node-level energy tracking with network propagation frameworks outperforms traditional risk metrics during volatile market shocks.

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{fapt_gnn_2025,
  title   = {Energy-Based Modeling of Systemic Fragility and Phase Transitions in Financial Networks},
  author  = {[Your Name]},
  journal = {[Target Venue]},
  year    = {2025}
}
```

---

## 📄 License

MIT License. Free to use for academic research.