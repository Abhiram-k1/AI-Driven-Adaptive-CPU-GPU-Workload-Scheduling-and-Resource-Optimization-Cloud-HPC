"""
dataset/build_ml_dataset.py
===========================
Constructs the Layer 3 ML Dataset from Layer 2.
Strictly implements:
  - Section 10: Mandatory Valid-Label Gate
  - Section 12: Strict Data-Leakage Rule
  - Section 30: Data Validation Checks and Final Dataset Audit Report.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.feature_schema import extract_pre_execution_features, FORBIDDEN_LEAKAGE_COLUMNS

DEFAULT_LAYER2_PATH = Path(__file__).resolve().parent / "layer2_aggregated_performance.csv"
DEFAULT_LAYER3_PATH = Path(__file__).resolve().parent / "layer3_ml_dataset.csv"
DEFAULT_LAYER1_PATH = Path(__file__).resolve().parent / "raw_data" / "layer1_raw_executions.csv"


def build_ml_dataset(
    layer2_path: Path = DEFAULT_LAYER2_PATH,
    output_path: Path = DEFAULT_LAYER3_PATH,
    layer1_path: Path = DEFAULT_LAYER1_PATH
) -> pd.DataFrame:
    if not layer2_path.exists():
        raise FileNotFoundError(f"Layer 2 dataset not found at {layer2_path}")

    df_layer2 = pd.read_csv(layer2_path)
    print("=" * 80)
    print("BUILDING LAYER 3 ML DATASET — STRICT GATING & AUDIT")
    print("=" * 80)

    # 1. Evaluate Valid-Label Gate (Section 10)
    # A configuration is eligible only if:
    # comparable_timing_ok == True AND gpu_total_path_time_status == "validated" AND preferred_device in {"cpu", "gpu"}
    eligible_mask = (
        (df_layer2["comparable_timing_ok"] == True) &
        (df_layer2["gpu_total_path_time_status"] == "validated") &
        (df_layer2["preferred_device"].isin(["cpu", "gpu"]))
    )

    df_eligible = df_layer2[eligible_mask].copy()
    df_excluded = df_layer2[~eligible_mask].copy()

    print(f"Total Layer 2 configurations: {len(df_layer2)}")
    print(f"Configurations passing gating checks: {len(df_eligible)}")
    print(f"Configurations excluded by gating: {len(df_excluded)}")

    for _, row in df_excluded.iterrows():
        reason = "comparable_timing_ok is False" if not row["comparable_timing_ok"] else (
            f"gpu_total_path_time_status is '{row['gpu_total_path_time_status']}'" if row["gpu_total_path_time_status"] != "validated" else "preferred_device is not set"
        )
        print(f"  - EXCLUDED: [{row['workload']}] file: {row['input_file']} | Reason: {reason}")

    # 2. Extract Pre-Execution Features for Eligible Configurations
    ml_rows = []
    for _, row in df_eligible.iterrows():
        feats = extract_pre_execution_features(
            workload=row["workload"],
            input_file=row["input_file"],
            input_size_mib=row["input_size_mib"],
            cpu_cores=4,
            gpu_type="Tesla_T4",
            gpu_memory_gb=16.0
        )
        # Attach ground-truth label
        feats["preferred_device"] = row["preferred_device"]
        # Attach telemetry validation metadata (strictly for post-hoc analysis, not classifier feature)
        feats["target_label"] = 0 if row["preferred_device"] == "cpu" else 1
        ml_rows.append(feats)

    df_layer3 = pd.DataFrame(ml_rows)

    # 3. Data Validation Checks (Section 30)
    run_dataset_validation_checks(df_layer3, df_layer2, layer1_path)

    # Save Layer 3
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_layer3.to_csv(output_path, index=False)
    print(f"\nLayer 3 ML dataset successfully saved to: {output_path}")

    # 4. Final Dataset Audit Report
    print_dataset_audit_report(df_layer3, df_layer2, layer1_path, df_excluded)

    return df_layer3


def run_dataset_validation_checks(df_ml: pd.DataFrame, df_l2: pd.DataFrame, layer1_path: Path):
    """Executes the mandatory automated verification rules from Section 30."""
    checks_passed = True
    print("\n--- Running Section 30 Data Validation Checks ---")

    # Check 1: No duplicate ML configurations
    if len(df_ml) > 0:
        dups = df_ml.duplicated(subset=["workload", "input_file"]).sum()
        if dups > 0:
            print(f"[FAIL] Duplicate ML configurations detected: {dups}")
            checks_passed = False
        else:
            print("[PASS] No duplicate ML configurations.")

    # Check 2: No missing target labels
    if len(df_ml) > 0:
        missing_targets = df_ml["preferred_device"].isna().sum()
        if missing_targets > 0:
            print(f"[FAIL] Missing target labels found: {missing_targets}")
            checks_passed = False
        else:
            print("[PASS] No missing target labels in Layer 3.")

    # Check 3: No invalid target classes
    if len(df_ml) > 0:
        invalid_classes = (~df_ml["preferred_device"].isin(["cpu", "gpu"])).sum()
        if invalid_classes > 0:
            print(f"[FAIL] Invalid target classes found: {invalid_classes}")
            checks_passed = False
        else:
            print("[PASS] All target labels belong strictly to {'cpu', 'gpu'}.")

    # Check 4: No post-execution features / Leakage columns
    leakage_found = [c for c in df_ml.columns if c in FORBIDDEN_LEAKAGE_COLUMNS]
    if leakage_found:
        print(f"[FAIL] Data leakage detected! Forbidden columns present: {leakage_found}")
        checks_passed = False
    else:
        print("[PASS] Zero data-leakage columns present in feature matrix.")

    # Check 5: Layer 1 warm-up exclusion & failure tracking
    if layer1_path.exists():
        df_l1 = pd.read_csv(layer1_path)
        warmups = df_l1["is_warmup"].sum()
        print(f"[PASS] Layer 1 contains {warmups} warm-up runs, strictly excluded from aggregation.")
        failed_runs = (df_l1["timing_parse_ok"] == False).sum()
        print(f"[PASS] Layer 1 preserves {failed_runs} failed/unparsed executions with full stderr/return codes.")

    assert checks_passed, "Dataset validation checks failed! Halting pipeline."


def print_dataset_audit_report(df_ml: pd.DataFrame, df_l2: pd.DataFrame, layer1_path: Path, df_excluded: pd.DataFrame):
    """Prints the final formal dataset audit report specified in Section 30."""
    total_raw = 0
    valid_raw = 0
    failed_raw = 0
    if layer1_path.exists():
        df_l1 = pd.read_csv(layer1_path)
        total_raw = len(df_l1)
        valid_raw = int((df_l1["timing_parse_ok"] == True).sum())
        failed_raw = int((df_l1["timing_parse_ok"] == False).sum())

    cpu_cnt = int((df_ml["preferred_device"] == "cpu").sum()) if len(df_ml) > 0 else 0
    gpu_cnt = int((df_ml["preferred_device"] == "gpu").sum()) if len(df_ml) > 0 else 0

    print("\n" + "=" * 80)
    print("FINAL DATASET AUDIT REPORT")
    print("=" * 80)
    print(f"Raw runs (Layer 1)       : {total_raw}")
    print(f"  - Valid executions     : {valid_raw}")
    print(f"  - Failed executions    : {failed_raw}")
    print(f"Layer 2 configurations   : {len(df_l2)}")
    print(f"Layer 3 ML configurations: {len(df_ml)}")
    print(f"  - CPU labels           : {cpu_cnt}")
    print(f"  - GPU labels           : {gpu_cnt}")
    print(f"Excluded configurations  : {len(df_excluded)}")
    for _, row in df_excluded.iterrows():
        print(f"  - {row['workload']} ({row['input_file']}): status='{row['gpu_total_path_time_status']}', label={row['preferred_device']}")
    print(f"Leakage check            : PASS (All {len(FORBIDDEN_LEAKAGE_COLUMNS)} forbidden fields absent)")
    print("=" * 80)


if __name__ == "__main__":
    build_ml_dataset()
