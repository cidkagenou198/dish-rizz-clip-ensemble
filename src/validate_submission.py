"""Check a submission file before uploading it (needs only pandas).

    python validate_submission.py submission.csv --data-dir /kaggle/input/<competition>

In a notebook:
    from validate_submission import check
    check("submission.csv", data_dir="/kaggle/input/<competition>")

Format: query_id,image_ids  - image_ids = 20 distinct test-gallery ids separated by spaces, best first.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

COLUMNS = ["query_id", "image_ids"]
TOP_K = 20


def check(submission, data_dir="."):
    """Raise ValueError describing every problem, or return the number of valid rows."""
    data_dir = Path(data_dir)
    queries = set(pd.read_csv(data_dir / "test_queries.csv", dtype=str).query_id)
    gallery = set(pd.read_csv(data_dir / "test_gallery.csv", dtype=str).image_id)
    sub = pd.read_csv(submission, dtype=str, keep_default_na=False)
    if [c.strip() for c in sub.columns] != COLUMNS:
        raise ValueError(f"Header must be exactly: {','.join(COLUMNS)}")
    sub.columns = COLUMNS
    sub = sub.apply(lambda s: s.str.strip())
    lists = sub.image_ids.str.split()
    errors = []
    if sub.query_id.duplicated().any():
        errors.append(f"duplicated query_id: {sub.query_id[sub.query_id.duplicated()].tolist()[:5]}")
    if queries - set(sub.query_id):
        errors.append(f"{len(queries - set(sub.query_id))} test queries are missing")
    if set(sub.query_id) - queries:
        errors.append(f"unknown query_id: {sorted(set(sub.query_id) - queries)[:5]}")
    if (lists.map(len) != TOP_K).any():
        errors.append(f"every row must contain exactly {TOP_K} image ids")
    bad = sorted({v for ids in lists for v in ids if v not in gallery})
    if bad:
        errors.append(f"{len(bad)} image ids not in test_gallery.csv, e.g. {bad[:5]}")
    if lists.map(lambda ids: len(set(ids)) != len(ids)).any():
        errors.append("image ids within a row must be distinct")
    if errors:
        raise ValueError("Invalid submission:\n - " + "\n - ".join(errors))
    return len(sub)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("submission")
    p.add_argument("--data-dir", default=".")
    a = p.parse_args()
    try:
        n = check(a.submission, a.data_dir)
    except ValueError as e:
        print(e)
        sys.exit(1)
    print(f"OK: {n} rows")
