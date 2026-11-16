"""Platform-level Cox regression and nonparametric survival diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "OnlineTime_YMD", "Bankrupt_WDZJ", "Collapse", "RegCapital", "Joinasso"
]
MODEL_COLUMNS = ["duration", "event", "RegCapital", "Joinasso"]
GROUP_LABELS = {0: "Non-member", 1: "Member"}


def _parse_yyyymmdd(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.to_datetime(
        numeric.astype("Int64").astype("string"),
        format="%Y%m%d",
        errors="coerce",
    )


def load_data(path: Path, sheet_name: str = "Platform Data") -> pd.DataFrame:
    """Load a platform-level workbook and validate its required columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Input workbook not found: {path}")
    data = pd.read_excel(path, sheet_name=sheet_name)
    missing = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return data


def prepare_survival_data(data: pd.DataFrame) -> pd.DataFrame:
    """Construct survival durations and retain model-specific complete cases.

    The latest valid recorded bankruptcy date defines the observation-window
    cutoff for platforms without a recorded collapse. Same-day events are
    retained. A uniform one-day offset is added to every elapsed duration so
    all model durations are strictly positive; this preserves ordering and ties.
    """
    prepared = data.copy()
    launch_date = _parse_yyyymmdd(prepared["OnlineTime_YMD"])
    collapse_date = _parse_yyyymmdd(prepared["Bankrupt_WDZJ"])

    censoring_date = collapse_date.max()
    if pd.isna(censoring_date):
        raise ValueError("No valid bankruptcy dates are available for the censoring cutoff.")

    event = pd.to_numeric(prepared["Collapse"], errors="coerce")
    invalid_events = event.notna() & ~event.isin([0, 1])
    if invalid_events.any():
        raise ValueError("Collapse must be coded as a binary 0/1 variable.")

    end_date = collapse_date.where(event.eq(1), censoring_date)
    elapsed_days = (end_date - launch_date).dt.days

    model_data = pd.DataFrame(
        {
            "elapsed_days": elapsed_days,
            "event": event,
            "RegCapital": pd.to_numeric(prepared["RegCapital"], errors="coerce"),
            "Joinasso": pd.to_numeric(prepared["Joinasso"], errors="coerce"),
        },
        index=prepared.index,
    )
    model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna()

    if model_data.empty:
        raise ValueError("No complete observations remain for the survival model.")
    if (model_data["elapsed_days"] < 0).any():
        raise ValueError("Negative survival durations were found; review the date fields.")
    if not set(model_data["event"].unique()).issubset({0, 1}):
        raise ValueError("The event indicator must contain only 0/1 values.")
    if model_data["event"].nunique() != 2:
        raise ValueError("The survival sample must contain both events and censored cases.")
    if not set(model_data["Joinasso"].unique()).issubset({0, 1}):
        raise ValueError("Joinasso must be coded as a binary 0/1 variable.")

    model_data["duration"] = model_data.pop("elapsed_days") + 1
    return model_data[MODEL_COLUMNS].astype(
        {"duration": float, "event": int, "Joinasso": int}
    )


def fit_cox_model(model_data: pd.DataFrame):
    """Fit the multivariable Cox proportional hazards model."""
    from lifelines import CoxPHFitter

    model = CoxPHFitter()
    model.fit(model_data, duration_col="duration", event_col="event")
    return model


def fit_survival_diagnostics(model, model_data: pd.DataFrame) -> dict:
    """Calculate Kaplan-Meier estimates, a log-rank test, and a PH test.

    The proportional-hazards test uses the rank time transformation. The
    Kaplan-Meier and log-rank results are unadjusted group comparisons.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test, proportional_hazard_test

    km_fits = {}
    summary_rows = []
    for membership, label in GROUP_LABELS.items():
        group = model_data.loc[model_data["Joinasso"].eq(membership)]
        if group.empty:
            raise ValueError(f"No observations found for {label.lower()} platforms.")

        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=group["duration"],
            event_observed=group["event"],
            label=label,
        )
        km_fits[label] = kmf
        summary_rows.append(
            {
                "Group": label,
                "Platforms": len(group),
                "Events": int(group["event"].sum()),
                "Censored": int((group["event"] == 0).sum()),
                "Mean_observed_duration": float(group["duration"].mean()),
                "Median_observed_duration": float(group["duration"].median()),
                "KM_median_survival": float(kmf.median_survival_time_),
            }
        )

    non_member = model_data.loc[model_data["Joinasso"].eq(0)]
    member = model_data.loc[model_data["Joinasso"].eq(1)]
    logrank = logrank_test(
        non_member["duration"],
        member["duration"],
        event_observed_A=non_member["event"],
        event_observed_B=member["event"],
    )
    ph_test = proportional_hazard_test(model, model_data, time_transform="rank")

    return {
        "km_fits": km_fits,
        "group_summary": pd.DataFrame(summary_rows).set_index("Group"),
        "logrank": logrank,
        "ph_test": ph_test,
    }


def run_analysis(model_data: pd.DataFrame) -> dict:
    """Fit the Cox model and all reported survival diagnostics."""
    model = fit_cox_model(model_data)
    diagnostics = fit_survival_diagnostics(model, model_data)
    return {"cox_model": model, **diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate P2P platform survival and run survival diagnostics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/platform_records.xlsx"),
        help="Path to the platform-level workbook.",
    )
    parser.add_argument("--sheet", default="Platform Data")
    args = parser.parse_args()

    raw_data = load_data(args.input, args.sheet)
    model_data = prepare_survival_data(raw_data)
    results = run_analysis(model_data)

    print("PLATFORM SURVIVAL: COX PROPORTIONAL HAZARDS MODEL")
    print(f"Complete observations: {len(model_data):,}")
    print(f"Observed events: {int(model_data['event'].sum()):,}")
    results["cox_model"].print_summary()

    print("\nPROPORTIONAL-HAZARDS ASSUMPTION TEST (RANK TRANSFORM)")
    print(results["ph_test"].summary.to_string())

    print("\nSURVIVAL SUMMARY BY ASSOCIATION MEMBERSHIP")
    print(results["group_summary"].to_string())

    print("\nKAPLAN-MEIER MEDIAN SURVIVAL TIMES (DAYS)")
    for label, kmf in results["km_fits"].items():
        print(f"{label}: {kmf.median_survival_time_}")

    print("\nLOG-RANK TEST")
    print(results["logrank"].summary.to_string())


if __name__ == "__main__":
    main()
