import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from valorant_predictor.vct_events import import_vct_season


def main() -> None:
    parser = argparse.ArgumentParser(description="Import curated VCT event history.")
    parser.add_argument("--season-year", type=int, default=2025)
    parser.add_argument("--refresh-existing", action="store_true")
    parser.add_argument("--limit-new-matches", type=int)
    parser.add_argument("--pause-seconds", type=float, default=1.2)
    args = parser.parse_args()

    _, summary = import_vct_season(
        season_year=args.season_year,
        refresh_existing=args.refresh_existing,
        limit_new_matches=args.limit_new_matches,
        pause_seconds=max(0.0, args.pause_seconds),
        progress=lambda step, current, total, message: print(
            f"[{step}] {current}/{total} {message}",
            flush=True,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
