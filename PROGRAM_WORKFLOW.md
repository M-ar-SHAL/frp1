# FAPT-GNN Program Workflow & Architecture

This document breaks down the overall workflow of the FAPT-GNN (Fragility-Aware Phase Transition Graph Neural Network) system, explaining what each file does, how data flows through the program, and where training/inference occurs.

## 1. High-Level Data Flow

The system follows a 5-stage pipeline:
1. **Data Ingestion:** Download historical stock data and market indicators.
2. **Preprocessing & Graph Creation:** Clean data, compute features, and build daily financial graphs.
3. **Model Processing:** Encode data, run through GNN and Transformer layers to understand market states.
4. **Training (Learning Phase):** Optimize the model against historical crashes.
5. **Output (Inference Phase):** Predict crash probabilities and system instability.

---

## 2. File & Component Breakdown

### A. The Data Pipeline (`data/` folder)
*These files are responsible for taking raw market inputs and preparing them for the neural network.*

*   **`data/data_pipeline.py`** (The Input Layer)
    *   **What it does:** Downloads NIFTY 50 stock prices (OHLCV) and India VIX data from Yahoo Finance.
*   **`data/feature_engineering.py`**
    *   **What it does:** Takes the raw prices and computes mathematical features for each stock (Volatility, Centrality, Sentiment/VIX, Liquidity).
*   **`data/graph_builder.py`**
    *   **What it does:** Constructs a daily "market graph" where stocks are nodes and their relationships (correlations, sector ties) are edges. 
*   **`data/crash_labeler.py`**
    *   **What it does:** Defines what a "crash" is (e.g., bottom 5% returns or >7% drawdown) to teach the model what to look for.

### B. The Model Architecture (`models/` folder)
*These files define the neural network's architecture.*

*   **`models/fapt_gnn.py`** (The Master Model)
    *   **What it does:** This is the main wrapper that strings all the individual neural network components together.
*   **`models/fragility_encoder.py`** (Step 1)
    *   **What it does:** Converts raw stock features into an initial "fragility score" [0-1] for each individual stock.
*   **`models/gnn_core.py`** (Step 2)
    *   **What it does:** A Graph Attention Network layer.
*   **`models/energy_layer.py`** (Step 3)
    *   **What it does:** Takes the updated graph and computes a single network-wide "Energy" score.
*   **`models/temporal_model.py`** (Step 4)
    *   **What it does:** A Transformer that looks at a rolling 30-day window (sequence) of the Energy scores.
*   **`models/phase_head.py`** (Step 5 - Output)
    *   **What it does:** Analyzes the Transformer's output and spits out the final three predictions: Crash Probability, Time-to-Crash, Instability Index.

### C. Training & Optimization (`training/` folder)
*These files optimize the model.*

*   **`training/trainer.py`** (The Teacher)
    *   **What it does:** Manages the training loop.
*   **`training/losses.py`**
    *   **What it does:** Contains the mathematical formulas (multi-objective loss) to tell the model *how wrong* it is.
*   **`training/evaluate.py`**
    *   **What it does:** Tests the trained model against unseen data.

### D. Entry Points & Execution Scripts (Root folder)

*   **`train_fast.py` / `debug_train2.py`**
    *   **What they do:** Scripts to quickly trigger the training pipeline.
*   **`dashboard.py`** (The GUI)
    *   **What it does:** A Streamlit web application. It acts as the interactive frontend.
*   **`diagnose_model.py` / `final_verification.py`**
    *   **What they do:** Diagnostic scripts used to test data integrity and outputs.

---

## 3. Summary Workflow Execution

If you run `python dashboard.py`, the flow looks like this:
1. `dashboard.py` triggers the data pipeline to fetch today's prices.
2. Prices are converted to an interconnected graph using `feature_engineering.py` and `graph_builder.py`.
3. The pre-trained `fapt_gnn.py` model is loaded from disk.
4. The graph passes through `fragility_encoder.py` -> `gnn_core.py` -> `energy_layer.py`.
5. The sequence passes to `temporal_model.py`.
6. `phase_head.py` predicts final metrics.
7. Streamlit displays these predictions.
