"""Debug script to capture full error trace from the training pipeline."""
import sys, traceback, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import yaml
    with open("experiments/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Override to minimal settings
    config["data"]["start_date"] = "2018-01-01"
    config["training"]["epochs"] = 3
    config["model"]["seq_len"] = 20

    from experiments.run_experiment import run_experiment
    run_experiment(config)
    print("\n=== SUCCESS ===")
except Exception:
    print("\n=== FULL ERROR TRACEBACK ===")
    traceback.print_exc()
