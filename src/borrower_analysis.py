"""Borrower-level credit-default and bidding-activity analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

PREDICTORS = ["CREDIT", "HOUSE", "CAR", "EDUCATION", "WORKTIME"]
LOGIT_REQUIRED = ["DEFAULT", *PREDICTORS]
OLS_REQUIRED = ["BIDS", *PREDICTORS]


def load_data(path: Path, sheet_name: str = "Data Borrower") -> pd.DataFrame:
    """Load a borrower-level workbook and validate its required columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Input workbook not found: {path}")
    data = pd.read_excel(path, sheet_name=sheet_name)
    required = set(LOGIT_REQUIRED + OLS_REQUIRED)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return data


def fit_models(data: pd.DataFrame):
    """Fit Logit and OLS models on specification-specific complete cases."""
    logit_data = data[LOGIT_REQUIRED].apply(pd.to_numeric, errors="coerce").dropna()
    ols_data = data[OLS_REQUIRED].apply(pd.to_numeric, errors="coerce").dropna()

    if logit_data.empty or ols_data.empty:
        raise ValueError("No complete observations remain for one or more models.")
    if not set(logit_data["DEFAULT"].unique()).issubset({0, 1}):
        raise ValueError("DEFAULT must be coded as a binary 0/1 variable.")
    if logit_data["DEFAULT"].nunique() != 2:
        raise ValueError("DEFAULT must contain both outcome classes to fit Logit.")

    y_logit = logit_data["DEFAULT"].astype(int)
    x_logit = sm.add_constant(logit_data[PREDICTORS], has_constant="add")
    logit_result = sm.Logit(y_logit, x_logit).fit(disp=False)

    y_ols = ols_data["BIDS"].astype(float)
    x_ols = sm.add_constant(ols_data[PREDICTORS], has_constant="add")
    ols_result = sm.OLS(y_ols, x_ols).fit()

    return logit_result, ols_result, len(logit_data), len(ols_data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate borrower default and bidding-activity models."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/borrower_records.xlsx"),
        help="Path to the borrower-level workbook.",
    )
    parser.add_argument("--sheet", default="Data Borrower")
    args = parser.parse_args()

    data = load_data(args.input, args.sheet)
    logit, ols, n_logit, n_ols = fit_models(data)

    print("BORROWER DEFAULT: LOGIT MODEL")
    print(f"Complete observations: {n_logit:,}")
    print(logit.summary())
    print("\nBORROWER BIDDING ACTIVITY: OLS MODEL")
    print(f"Complete observations: {n_ols:,}")
    print(ols.summary())


if __name__ == "__main__":
    main()
