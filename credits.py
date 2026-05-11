import os
import json
from datetime import date, timedelta
from dotenv import load_dotenv
from runwayml import RunwayML

load_dotenv()

RESULTS_FILE = "output/videos/results.json"


def load_local_usage() -> int:
    try:
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        return sum(r.get("credits_used", 0) for r in results if r.get("status") == "success")
    except FileNotFoundError:
        return 0


def main():
    if not os.getenv("RUNWAYML_API_SECRET"):
        raise EnvironmentError("RUNWAYML_API_SECRET not set — check your .env file")

    client = RunwayML()

    # Balance & tier
    org = client.organization.retrieve()
    balance = org.credit_balance
    tier = getattr(org, "usage_tier", "unknown")

    # Usage breakdown for last 30 days
    today = date.today()
    start = today - timedelta(days=30)
    usage = client.organization.retrieve_usage(
        start_date=start.isoformat(),
        before_date=today.isoformat(),
    )

    total_used = sum(getattr(entry, "credits_used", 0) for entry in (usage or []))
    local_used = load_local_usage()

    print("=" * 40)
    print("  Runway Credit Summary")
    print("=" * 40)
    print(f"  Balance remaining : {balance:,} credits")
    print(f"  Usage tier        : {tier}")
    print(f"  Used (last 30d)   : {total_used:,} credits")
    print(f"  Used this session : {local_used:,} credits (from results.json)")
    print("=" * 40)

    if usage:
        print("\n  Breakdown by model (last 30 days):")
        by_model: dict[str, int] = {}
        for entry in usage:
            model = getattr(entry, "model", "unknown")
            credits = getattr(entry, "credits_used", 0)
            by_model[model] = by_model.get(model, 0) + credits
        for model, credits in sorted(by_model.items(), key=lambda x: -x[1]):
            print(f"    {model:<25} {credits:>8,} credits")


if __name__ == "__main__":
    main()
