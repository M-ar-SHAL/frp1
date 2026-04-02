import sys
sys.path.insert(0, '.')

print("[Verify 1] Loading data pipeline...")
from data.data_pipeline import load_all_data
data = load_all_data(start='2023-01-01')
print(f"[OK] Loaded {len(data['prices'])} trading days")

print("\n[Verify 2] Creating labels...")
from data.crash_labeler import create_labels
labels = create_labels(
    nifty_series=data['nifty'],
    returns=data['prices'].pct_change(),
    percentile=5,
    drawdown_threshold=-5,
    forward_days=5
)
print(f"[OK] Created labels: {labels['crash_label'].sum()} crash days")

print("\n[SUCCESS] All fixes verified - No encoding errors detected")
