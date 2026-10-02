"""
Verify programmatic zero overlap between the Step 64 Blind Holdout and all existing datasets.
"""
import sys
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

HOLDOUT_PATH = Path("experimental/v4/results/blind_final_holdout_500.npy")

def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=" * 85)
    print("STEP 64: RIGOROUS PROGRAMMATIC ZERO-OVERLAP AUDIT")
    print("=" * 85)

    assert HOLDOUT_PATH.exists(), f"Holdout file {HOLDOUT_PATH} not found!"
    holdout_sha256 = compute_file_hash(HOLDOUT_PATH)
    print(f"Blind Holdout File       : {HOLDOUT_PATH}")
    print(f"Blind Holdout File SHA256: {holdout_sha256}")

    holdout_data = np.load(HOLDOUT_PATH, allow_pickle=True).item()
    print(f"Base Seed                : {holdout_data['metadata']['base_seed']}")
    print(f"Creation Timestamp       : {holdout_data['metadata']['creation_timestamp']}")
    print(f"Number of Conditions     : {len(holdout_data['conditions'])}")

    all_blind_X = np.concatenate([split["X"] for split in holdout_data["conditions"].values()], axis=0)
    all_blind_Y = np.concatenate([split["Y"] for split in holdout_data["conditions"].values()], axis=0)
    total_blind_samples = len(all_blind_X)
    print(f"Total Blind Samples      : {total_blind_samples}")

    existing_datasets = [
        ("v4_train_2000.npy", Path("experimental/v4/datasets/v4_train_2000.npy")),
        ("v4_val_500.npy", Path("experimental/v4/datasets/v4_val_500.npy")),
        ("v4_full_2500.npy", Path("experimental/v4/datasets/v4_full_2500.npy")),
        ("physics_test_200.npy", Path("data/physics_test_200.npy")),
        ("physics_val_200.npy", Path("data/physics_val_200.npy")),
        ("physics_train_800.npy", Path("data/physics_train_800.npy")),
        ("physics_full.npy", Path("data/physics_full.npy")),
        ("train.npy", Path("data/train.npy")),
        ("test.npy", Path("data/test.npy")),
        ("full.npy", Path("data/full.npy")),
    ]

    print(f"\n{'Dataset Name':<25} | {'Samples':<8} | {'SHA256 (first 16)':<18} | {'Overlap Count':<14} | {'Min L-inf Dist':<15}")
    print("-" * 90)

    total_existing_samples = 0
    total_overlap_found = 0
    audit_records = []

    for ds_name, ds_path in existing_datasets:
        if not ds_path.exists():
            continue
        ds_hash = compute_file_hash(ds_path)[:16]

        data = np.load(ds_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        
        if isinstance(data, dict):
            ex_X = data["X"]
        else:
            ex_X = data

        n_samples = len(ex_X)
        total_existing_samples += n_samples

        # Determine common feature dimension (128 currents, or 130 with angle features)
        common_dim = min(ex_X.shape[1], all_blind_X.shape[1])
        b_feat = all_blind_X[:, :common_dim]
        ex_feat = ex_X[:, :common_dim]

        min_dist = float("inf")
        overlap_count = 0

        # Vectorized min distance check per chunk
        chunk_size = 500
        for i in range(0, total_blind_samples, chunk_size):
            b_chunk = b_feat[i:i+chunk_size]
            for b_vec in b_chunk:
                d = np.max(np.abs(ex_feat - b_vec), axis=1)
                md = np.min(d)
                if md < 1e-5:
                    overlap_count += 1
                if md < min_dist:
                    min_dist = float(md)

        total_overlap_found += overlap_count
        audit_records.append({
            "name": ds_name,
            "samples": n_samples,
            "sha256": ds_hash,
            "overlap": overlap_count,
            "min_dist": min_dist,
        })
        print(f"{ds_name:<25} | {n_samples:<8} | {ds_hash:<18} | {overlap_count:<14} | {min_dist:<15.6f}")

    print("-" * 90)
    print(f"Total Existing Samples Audited : {total_existing_samples}")
    print(f"Total Blind Holdout Samples    : {total_blind_samples}")
    print(f"Total Overlap Count Found      : {total_overlap_found}")
    print(f"Minimum Distance Over Entire DB: {min(r['min_dist'] for r in audit_records):.6f}")
    assert total_overlap_found == 0, f"FATAL: Detected {total_overlap_found} overlapping samples!"
    print("VERIFICATION RESULT: STRICT ZERO-OVERLAP CONFIRMED.")

if __name__ == "__main__":
    main()
