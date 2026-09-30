"""Descriptive acquisition-bias statistics on canonical image contents."""

import numpy as np
import pandas as pd
from scipy import stats

from .transformers import LABELS, METRICS

SEED = 42


def holm_adjust(pvalues):
    """Holm family-wise adjustment; preserve input order."""
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.maximum.accumulate(
        values[order] * np.arange(len(values), 0, -1)
    ).clip(0, 1)
    result = np.empty_like(values)
    result[order] = adjusted
    return result


def mutual_information(x, y):
    """Discrete plug-in mutual information in nats, with zero cells omitted."""
    table = np.zeros((int(x.max()) + 1, int(y.max()) + 1), dtype=float)
    np.add.at(table, (x, y), 1)
    joint = table / table.sum()
    independent = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / independent[mask])))


def mi_diagnostic(values, target, continuous):
    """Ten pooled quantile bins; report MI and its permutation bias."""
    if continuous:
        bins = pd.qcut(values, q=10, labels=False, duplicates="drop")
        x = np.asarray(bins, dtype=int)
    else:
        x = pd.factorize(values, sort=True)[0]
    y = pd.factorize(target, sort=True)[0]
    observed = mutual_information(x, y)
    rng = np.random.default_rng(SEED)
    null = np.array([
        mutual_information(x, rng.permutation(y)) for _ in range(499)
    ])
    return {
        "mi_nats": observed, "mi_bins": len(set(x)),
        "mi_permutation_null_mean": float(null.mean()),
        "mi_above_null_mean": max(0.0, observed - float(null.mean())),
        "mi_permutation_p": (1 + int((null >= observed).sum())) / 500,
    }


def analyze_metadata(frame, output):
    """Rank comparisons; inferential independence remains unverified."""
    summaries, associations, matrix = [], [], []
    for variable in METRICS:
        arrays = []
        for label in LABELS:
            x = frame.loc[frame.target_label == label, variable].to_numpy()
            arrays.append(x)
            q1, median, q3 = np.quantile(x, [0.25, 0.5, 0.75], method="linear")
            w, pvalue = stats.shapiro(x)
            summaries.append({
                "variable": variable, "target_label": label, "N": len(x),
                "mean": x.mean(), "median": median,
                "standard_deviation": x.std(ddof=1),
                "Q1": q1, "Q3": q3, "IQR": q3 - q1,
                "minimum": x.min(), "maximum": x.max(),
                "skewness": stats.skew(x, bias=False),
                "excess_kurtosis": stats.kurtosis(x, fisher=True, bias=False),
                "shapiro_W": w, "shapiro_p": pvalue,
            })
        h, pvalue = stats.kruskal(*arrays, nan_policy="raise")
        variance_stat, variance_p = stats.levene(*arrays, center="median")
        associations.append({
            "variable": variable, "method": "Kruskal-Wallis",
            "N": len(frame), "statistic": h, "df": len(arrays) - 1,
            "p_value": pvalue,
            "effect_size_name": "eta_squared_H",
            "effect_size": max(0.0, (h - 2) / (len(frame) - 3)),
            "effect_size_formula": "max(0,(H-k+1)/(N-k))",
            "brown_forsythe_statistic": variance_stat,
            "brown_forsythe_p": variance_p,
            "interpretation": "rank/distribution association; not causation",
            **mi_diagnostic(frame[variable], frame.target_label, True),
        })
    for variable in ["image_mode", "channel_count"]:
        table = pd.crosstab(frame[variable], frame.target_label).reindex(
            columns=LABELS, fill_value=0
        )
        table.to_csv(output / f"{variable}_contingency.csv")
        chi2, pvalue, dof, expected = stats.chi2_contingency(
            table, correction=False
        )
        valid = bool((expected >= 5).all())
        expected_frame = pd.DataFrame(
            expected, index=table.index, columns=table.columns
        )
        expected_frame.to_csv(
            output / f"{variable}_expected_counts.csv"
        )
        associations.append({
            "variable": variable, "method": "Pearson Chi-square",
            "N": len(frame), "statistic": chi2, "df": dof,
            "p_value": pvalue if valid else np.nan,
            "effect_size_name": "Cramers_V",
            "effect_size": np.sqrt(
                chi2 / (table.values.sum() * min(np.array(table.shape) - 1))
            ),
            "effect_size_formula": "sqrt(chi2/(N*min(r-1,c-1)))",
            "chi_square_expected_counts_valid": valid,
            "minimum_expected_count": expected.min(),
            "interpretation": "categorical association; not causation",
            **mi_diagnostic(frame[variable], frame.target_label, False),
        })
    associations = pd.DataFrame(associations)
    # For df=2, the chi-square survival function is exactly exp(-stat/2).
    # Its logarithm remains representable when floating-point p underflows.
    associations["log10_p_value"] = [
        -row.statistic / (2 * np.log(10)) if row.df == 2
        else stats.chi2.logsf(row.statistic, row.df) / np.log(10)
        for row in associations.itertuples()
    ]
    associations["p_value_underflow"] = associations.p_value.eq(0)
    valid = associations.p_value.notna()
    associations.loc[valid, "p_holm_8_tests"] = holm_adjust(
        associations.loc[valid, "p_value"]
    )
    for row in associations.itertuples():
        matrix.append({
            "diagnostic_variable": row.variable,
            "data_type": "continuous" if row.variable in METRICS
            else "categorical",
            "target_type": "nominal three-class label",
            "association_method": row.method,
            "effect_size_method": row.effect_size_name,
            "purpose": "acquisition/storage shortcut-risk diagnostic",
            "used_as_CNN_feature": "NO",
        })
    pd.DataFrame(summaries).to_csv(
        output / "class_metadata_summary.csv", index=False
    )
    associations.to_csv(
        output / "metadata_target_associations.csv", index=False
    )
    pd.DataFrame(matrix).to_csv(
        output / "feature_target_test_matrix.csv", index=False
    )
    correlation = frame[list(METRICS)].corr(method="spearman")
    correlation.to_csv(output / "metadata_correlation_matrix.csv")
    pairs = [{"variable_1": a, "variable_2": b,
              "spearman_rho": correlation.loc[a, b]}
             for i, a in enumerate(METRICS) for b in METRICS[i + 1:]
             if abs(correlation.loc[a, b]) >= 0.85]
    pd.DataFrame(pairs, columns=[
        "variable_1", "variable_2", "spearman_rho",
    ]).to_csv(output / "high_metadata_correlations.csv", index=False)
    return associations, correlation


def quality_class_analysis(frame, flags):
    """One content per class denominator; original IQR fences are unchanged."""
    flags = flags.drop_duplicates(["sha256_hash", "metric"])
    counts = flags.groupby("sha256_hash").size()
    rows = []
    for label in LABELS:
        part = frame[frame.target_label == label]
        nflags = part.sha256_hash.map(counts).fillna(0)
        row = {
            "target_class": label, "total_images": len(part),
            "flagged_images": int(nflags.gt(0).sum()),
            "flag_rate": float(nflags.gt(0).mean()),
            "single_flag_count": int(nflags.eq(1).sum()),
            "multiple_flag_count": int(nflags.gt(1).sum()),
        }
        for metric in METRICS:
            ids = flags.loc[flags.metric == metric, "sha256_hash"]
            count = int(part.sha256_hash.isin(ids).sum())
            row[f"{metric}_flag_count"] = count
            row[f"{metric}_flag_rate"] = count / len(part)
        rows.append(row)
    return pd.DataFrame(rows)
