"""
Build models/portfolio_scoring_snapshot.json from Home Credit application_train.csv.

Usage (from project root):
    set CREDIT_RISK_PORTFOLIO_CSV=D:\\path\\to\\application_train.csv
    python scripts/build_portfolio_snapshot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.portfolio_loader import resolve_portfolio_csv  # noqa: E402
from src.ml.portfolio_analytics import (  # noqa: E402
    PORTFOLIO_SNAPSHOT_PATH,
    build_portfolio_snapshot_from_csv,
)


def main() -> None:
    csv_path = resolve_portfolio_csv()
    if csv_path is None:
        print(
            "ERROR: application_train.csv not found.\n"
            "Place the file at data/application_train.csv or set CREDIT_RISK_PORTFOLIO_CSV."
        )
        sys.exit(1)

    print(f"Scoring portfolio from: {csv_path}")
    payload = build_portfolio_snapshot_from_csv(csv_path)
    print(f"Wrote {PORTFOLIO_SNAPSHOT_PATH}")
    print(f"  Scored records: {payload['scored_records']:,}")
    print(f"  Observed default rate: {payload['observed_default_rate_pct']}%")
    print(f"  Approval rate: {payload['approval_rate_pct']}%")
    print(f"  High risk exposure: {payload['high_risk_pct']}% ({payload['high_risk_count']:,})")
    print(f"  Scoring runtime: {payload['scoring_runtime_seconds']}s")
    for row in payload["risk_band_distribution"]:
        print(f"  {row['band']}: {row['pct']}% ({row['count']:,})")


if __name__ == "__main__":
    main()
