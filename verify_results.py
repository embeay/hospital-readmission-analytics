
"""Verify model outputs against the tested Python 3.12 environment."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXPECTED = {
    "logistic": {
        "file": ROOT / "src/logistic_v2_results/metrics.json",
        "roc_auc": 0.660770888,
        "pr_auc": 0.228678018,
        "top_10pct_events": 400,
        "top_20pct_events": 666,
    },
    "xgboost": {
        "file": ROOT / "src/xgboost_v2_results/metrics.json",
        "roc_auc": 0.663847572,
        "pr_auc": 0.232728542,
        "top_10pct_events": 424,
        "top_20pct_events": 667,
    },
}

TOLERANCE = 0.0001


def verify_model(name, expected):
    path = expected["file"]

    if not path.exists():
        print(f"FAIL: {name} results file not found: {path}")
        return False

    with path.open(encoding="utf-8") as file:
        results = json.load(file)

    test = results["test"]
    passed = True

    for field in ("n", "events"):
        expected_value = {"n": 14985, "events": 1752}[field]
        actual = test[field]
        ok = actual == expected_value
        print(f"{name}: {field} = {actual} — {'PASS' if ok else 'FAIL'}")
        passed &= ok

    for field in ("roc_auc", "pr_auc",
                  "top_10pct_events", "top_20pct_events"):
        actual = test[field]
        target = expected[field]

        if field.endswith("_events"):
            ok = actual == target
        else:
            ok = abs(actual - target) <= TOLERANCE

        print(
            f"{name}: {field} = {actual:.9f} "
            f"(expected {target:.9f}) — {'PASS' if ok else 'FAIL'}"
        )
        passed &= ok

    if name == "xgboost":
        checks = results.get("shap_checks", {})
        ok = (
            checks.get("matrix_sparse") is True
            and checks.get("n_checked") == 1500
            and checks.get(
                "max_abs_probability_reconstruction_error", 1
            ) < 1e-5
        )
        print(f"{name}: SHAP validation — {'PASS' if ok else 'FAIL'}")
        passed &= ok

    return passed


def main():
    outcomes = [
        verify_model(name, expected)
        for name, expected in EXPECTED.items()
    ]

    if all(outcomes):
        print("\nSUCCESS: Both models match the verified benchmarks.")
    else:
        print("\nFAILURE: One or more verification checks failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
