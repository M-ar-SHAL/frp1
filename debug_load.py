import traceback
try:
    from data.data_pipeline import load_all_data
    d = load_all_data(start="2010-01-01")
except Exception as e:
    traceback.print_exc()
