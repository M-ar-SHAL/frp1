# FAPT-GNN Project Glossary

This glossary defines the complex technical, mathematical, and financial terminology used throughout the FAPT-GNN (Fragility-Aware Phase Transition Graph Neural Network) project.

### 1. Fragility (Systemic Fragility)
* **General Definition:** The quality of being easily broken, damaged, or destroyed.
* **In FAPT-GNN:** A mathematical score between 0.0 and 1.0 assigned to every single stock. A stock with a high fragility score (e.g., 0.9) is highly unstable—meaning it is extremely vulnerable to market shocks, has high volatility, and could trigger a massive sell-off. If many interconnected stocks reach high fragility, the entire market becomes "Systemically Fragile" (ready to crash).

### 2. Fragility Encoder
* **General Definition:** A neural network component that compresses raw data into a specific latent representation.
* **In FAPT-GNN:** The specific piece of Python code (`models/fragility_encoder.py`) that reads a stock's raw statistics (like daily trading volume, how volatile its price is, and the VIX sentiment) and translates those raw numbers into the 0.0 to 1.0 Fragility Score mentioned above. 

### 3. Phase Transition (Critical Phase Transition)
* **General Definition:** A concept from physics where a system suddenly changes from one state to a completely different state (like water freezing into solid ice).
* **In FAPT-GNN:** A stock market crash is treated mathematically as a "Phase Transition." The market suddenly snaps from a "stable, normal trading state" into a "chaotic, crashing panic state." The goal of this entire AI is to detect the hidden, mathematical boiling point right before this snap occurs.

### 4. Graph Neural Network (GNN)
* **General Definition:** A type of artificial intelligence designed specifically to analyze networks and webs of data (like social networks or internet routers) instead of flat spreadsheets.
* **In FAPT-GNN:** The system maps the NIFTY 50 stock market as a web. Stocks are **Nodes** (dots), and their relationships (like if they are both IT companies, or if they have correlated price movements) are **Edges** (lines connecting the dots). The GNN (`gnn_core.py`) pushes data across these lines so the AI can understand how a problem in one stock might impact its neighbors.

### 5. Attention Mechanism (Graph Attention / GAT)
* **General Definition:** A deep learning technique that allows an AI to dynamically decide mathematically which pieces of data are "most important" and ignore the noise.
* **In FAPT-GNN:** When the GNN is looking at the web of stocks, it uses attention to figure out who is influencing whom. **Novelty:** This project uses *Fragility-Aware Attention*. If HDFC Bank is incredibly fragile today, the Attention Mechanism forces all connected stocks to pay massive attention to HDFC Bank's status, perfectly mimicking how financial panic spreads.

### 6. System Energy (Energy Layer / E(t))
* **General Definition:** In physics, energy is a measurable property that must be transferred to an object to do work or heat it up.
* **In FAPT-GNN:** The project borrows physics math to measure the "stress" in the stock market. The `EnergyLayer` calculates a single number for the whole network. If the energy is low, the market is relaxed. If the energy is spiking rapidly (high acceleration), it means stress is building intensely beneath the surface, warning that a crash (Phase Transition) is imminent.

### 7. Temporal Transformer
* **General Definition:** A highly advanced neural network architecture (the same tech behind ChatGPT) designed to read sequences of data over time and find long-term patterns.
* **In FAPT-GNN:** A market doesn't crash based on one bad day—it builds up over weeks. The Temporal Transformer (`temporal_model.py`) looks at a rolling "window" of the last 30 days of System Energy. It tracks the momentum and trajectory of the stress to figure out if it will resolve itself or break the market.

### 8. Contagion (Financial Contagion)
* **General Definition:** The spread of an economic crisis from one market or institution to another, much like a biological virus.
* **In FAPT-GNN:** The core reason we use a Graph Neural Network. The model mathematically tracks contagion by passing "stress values" across the edges of the graph. If one node (stock) fails, the model simulates how that failure infects connected nodes to calculate total market risk.

### 9. Phase Transition Head
* **General Definition:** The final layer ("head") of a neural network that translates the AI's complex internal thoughts into human-readable outputs.
* **In FAPT-GNN:** The `phase_head.py` takes the final analysis from the Temporal Transformer and spits out three simple numbers for the user interface: The probability of a crash (0-100%), the estimated time left until the crash (in days), and the current Instability Index.

### 10. Walk-Forward Splitting (Walk-Forward Validation)
* **General Definition:** A secure machine learning backtesting strategy used specifically for time-series data to prevent the AI from cheating.
* **In FAPT-GNN:** You cannot shuffle stock market data randomly (if you train the AI on 2022 data, it can easily "predict" 2020 because it already knows the future). Walk-forward splitting mathematically strictly partitions the data sequentially. The AI learns on 2015-2020, validates on 2021, and is tested on 2022. It ensures the AI's crash predictions are completely blind to the future.

### 11. Multi-Objective Loss
* **General Definition:** A mathematical formula used during training to punish the AI for getting answers wrong, forcing it to optimize for multiple different goals simultaneously.
* **In FAPT-GNN:** The AI isn't just told to guess *if* a crash happens. The `losses.py` file punishes the AI if it misses a crash (Classification Loss), punishes the AI if it gets the timing wrong (Time-to-Crash Loss), and punishes the AI if the physics energy math doesn't align with reality (Energy Regularization). It is forced to learn all three skills at once.

### 12. Shock Simulator
* **General Definition:** A function that applies hypothetical stress to a system to see if it breaks.
* **In FAPT-GNN:** An engine (`ShockSimulator` in `phase_head.py`) that allows you to artificially inject poison into the system. You can mathematically tell the model "Assume Reliance Industries goes completely bankrupt right now" (Node Failure). The simulator measures how violently the System Energy spikes to calculate the market's true resilience.
