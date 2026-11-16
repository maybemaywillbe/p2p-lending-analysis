# P2P Lending: Borrower Credit Risk and Platform Survival

This project examines borrower-level loan outcomes and platform-level survival in two course-provided P2P lending datasets. It combines Logit and OLS regression with Cox regression and nonparametric survival analysis.

## Research questions

1. How are borrower credit grades and selected personal characteristics associated with loan default?
2. How are borrower characteristics associated with the number of investors participating in a loan?
3. How are registered capital and industry-association membership associated with platform survival?

## Data provenance and access

The analysis uses the datasets supplied with the FIN3210 Week 2 assignment materials:

- **Renrendai borrower dataset** (`FIN3210 Week 2 Renrendai loans.xlsx`): 10,000 loan records, with loan posting dates from 12 October 2010 to 17 October 2016.
- **P2P platform dataset** (`FIN3210 Week 2 p2p lending platforms.xlsx`): 1,000 platform records, with launch dates from 9 April 2009 to 24 May 2018. Recorded bankruptcy dates range from 1 June 2012 to 4 September 2019.

The workbooks and record-level data are not included in this repository. The upstream data provider and a separate redistribution licence are not identified in the supplied course materials; no further source attribution or data licence is asserted here. Obtain the workbooks through an authorized source before running the analysis.

Expected worksheets and fields:

| Analysis | Worksheet | Required columns |
|---|---|---|
| Borrower | `Data Borrower` | `DEFAULT`, `BIDS`, `CREDIT`, `HOUSE`, `CAR`, `EDUCATION`, `WORKTIME` |
| Platform | `Platform Data` | `OnlineTime_YMD`, `Bankrupt_WDZJ`, `Collapse`, `RegCapital`, `Joinasso` |

The borrower data dictionary defines `DEFAULT` as 1 for `BAD_DEBT` or `OVER_DUE` and 0 otherwise. `CREDIT` is coded from 1 (HR) to 7 (AA); `HOUSE` and `CAR` are binary indicators; `EDUCATION` and `WORKTIME` are ordinal categories. Platform `RegCapital` is measured in ten-thousand yuan, and `Joinasso` indicates industry-association membership.

Place locally available workbooks in a `data/` directory or supply another path with `--input`. The `data/` directory and common data-file formats are excluded by `.gitignore`.

## Methods

### Borrower-level models

- **Logit regression:** models the binary `DEFAULT` outcome using `CREDIT`, `HOUSE`, `CAR`, `EDUCATION`, and `WORKTIME`.
- **OLS regression:** models `BIDS`, the number of investors participating in a loan, using the same five predictors.
- Each model uses its own complete-case sample. Numeric conversion and binary-outcome validation are performed before estimation.

### Platform-level survival analysis

- **Cox proportional hazards model:** estimates the association of `RegCapital` and `Joinasso` with platform collapse.
- **Kaplan–Meier estimates:** compare unadjusted survival distributions by association membership.
- **Log-rank test:** tests the unadjusted difference between the two Kaplan–Meier groups.
- **Proportional-hazards diagnostic:** uses the rank time transformation to test the Cox model covariates.

`Collapse` is the event indicator: 1 denotes a platform recorded as bankrupt and 0 denotes a non-collapse observation. Launch and bankruptcy dates are parsed from `YYYYMMDD` values. For platforms without a recorded collapse, the latest valid bankruptcy date in the supplied platform dataset is used as the right-censoring cutoff, following the assignment's analysis convention. Elapsed time is measured in calendar days from launch to collapse or censoring. Same-day events are retained; one day is added uniformly to every duration to ensure positive model input. This common offset preserves ordering and ties, but the reported duration scale (including Kaplan–Meier medians) is one day greater than raw elapsed calendar time.

## Reported empirical results

The figures below are from the original analysis of the course-provided workbooks. Because the workbooks are not distributed, these numerical estimates cannot be independently reproduced from a clean clone of this repository alone. The scripts reproduce the specified analyses when compatible source workbooks are supplied.

### Borrower-level results

Both borrower models use 9,990 complete observations.

- **Logit:** McFadden pseudo-$R^2$ = 0.2151. `CREDIT` (coefficient = -1.8806), `CAR` (-0.2748), and `EDUCATION` (-0.4583) are negatively associated with the default indicator; `WORKTIME` (0.0699) is positively associated. `HOUSE` is not statistically significant (p = 0.113).
- **OLS:** $R^2$ = 0.0595. `CREDIT` (2.4536), `CAR` (11.9676), and `WORKTIME` (4.5106) are positively associated with the number of bids; `EDUCATION` (-4.6468) is negatively associated. `HOUSE` is not statistically significant (p = 0.362).

### Platform-level results

The Cox model uses 968 complete observations and includes 774 observed collapse events. Its concordance index is approximately 0.577.

- **Industry-association membership:** estimated hazard ratio = 0.25 (95% CI: 0.16–0.38; p < 0.005).
- **Registered capital:** not statistically significant (p = 0.84).

The rank-based proportional-hazards test flags `Joinasso` (test statistic = 12.0676, p = 0.000513), while it does not flag `RegCapital` (test statistic = 1.0014, p = 0.3170). The estimated membership hazard ratio should therefore not be interpreted as a constant relative hazard over the full observation period.

The unadjusted Kaplan–Meier comparison is:

| Measure | Non-members | Members |
|---|---:|---:|
| Platforms | 915 | 53 |
| Observed collapse events | 752 | 22 |
| Right-censored observations | 163 | 31 |
| Kaplan–Meier median duration (days, including the common +1 offset) | 551 | 1,995 |

The log-rank test statistic is 48.9867 (p = 2.577 × 10⁻¹²). These group comparisons are unadjusted and do not control for registered capital or other platform characteristics. The membership group is comparatively small. All reported estimates describe associations, not causal effects.

## Setup

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python src/borrower_analysis.py --input data/borrower_records.xlsx
python src/platform_survival.py --input data/platform_records.xlsx
```

The default worksheet names are `Data Borrower` and `Platform Data`. Use `--sheet` to specify a different worksheet name. The platform script prints the Cox model, proportional-hazards test, group survival summary, Kaplan–Meier medians, and log-rank test.

## Tests

The test suite uses synthetic observations and does not require the course workbooks.

```bash
python -m pytest -q
```

## Repository contents

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── borrower_analysis.py
│   └── platform_survival.py
└── tests/
    └── test_analysis.py
```

## Rights

Copyright (c) 2026 Siyan Wang. All rights reserved. No licence is granted to copy, modify, redistribute, or commercially use the original code in this repository without the copyright holder's prior written permission. Third-party packages remain subject to their respective licences.
