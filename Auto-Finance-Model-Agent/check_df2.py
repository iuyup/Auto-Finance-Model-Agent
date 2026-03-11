# Q2/check_df2.py
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.report import make_data_report, print_data_report

if __name__ == "__main__":
    df = pd.read_parquet("data.pq")
    rep = make_data_report(df, label_prefix="Y", x_prefix="X")
    print_data_report(rep)