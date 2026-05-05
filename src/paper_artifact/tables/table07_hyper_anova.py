from __future__ import annotations

"""
Stage 3 builder for paper Table 7.

This module is intended to live at:
    src/paper_artifact/tables/table07_hyper_anova.py

It adapts the original hyperparameter-shift ANOVA script with only artifact IO
changes.

Input:
    <ctx.input_root>/beers_summary.xlsx
    <ctx.input_root>/flights_summary.xlsx
    <ctx.input_root>/hospital_summary.xlsx
    <ctx.input_root>/rayyan_summary.xlsx

Output:
    <ctx.output_dir>/table_7/table7_hyper_anova_all.csv

No table10, table11, audit, XLSX, or auxiliary outputs are generated.

Core logic preserved from the original script:
1. Hyperparameter shifts are computed relative to Mode within the same
   dataset x error-rate bin x clustering algorithm group.
2. GroundTruth is excluded from shift tables and ANOVA. Mode is kept only as
   the zero baseline for delta construction, and excluded from ANOVA.
3. Missing or unparseable hyperparameters are treated as NaN and dropped only
   for the affected parameter.
4. The parser supports both legacy strings and JSON/Python-dict-like strings.
5. ANOVA uses numeric error_bin as the error-rate main effect.
"""

import ast
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
except Exception as exc:
    raise RuntimeError(
        "This builder requires statsmodels. Install it with `pip install statsmodels`."
    ) from exc

from src.paper_artifact.io import ArtifactResult, BuildContext


ARTIFACT = {
    "id": "table07_hyper_anova",
    "paper_id": "Table 7",
    "label": "Table 7: Hyperparameter-shift ANOVA",
    "description": "Build Table 7 from the four *_summary.xlsx workbooks under results/.",
    "enabled": True,
}


# ----------------------------------------------------------------------
# Constants.
# ----------------------------------------------------------------------
DATASETS = ["beers", "flights", "hospital", "rayyan"]
ERROR_BINS = [5, 10, 15, 20, 25, 30]

CLEAN_ORDER_FULL = [
    "Mode",
    "Baran",
    "HoloClean",
    "BigDansing",
    "BoostClean",
    "Horizon",
    "SCAReD",
    "Unified",
    "UniClean",
]
BASELINE_CLEANER = "Mode"
ORACLE_CLEANERS = {"GroundTruth", "Oracle", "GT"}

ANOVA_INCLUDE_MODE = False
ANOVA_INCLUDE_ORACLE = False

# Keep old ANOVA logic: numeric error_bin main effect.
ERROR_AS_CATEGORICAL = False


# ----------------------------------------------------------------------
# Column and name normalization.
# ----------------------------------------------------------------------
COLUMN_ALIASES: Dict[str, list[str]] = {
    "dataset_id": ["dataset_id", "dirty_id", "instance_id", "id"],
    "error_rate": ["error_rate", "q_tot", "qtot", "total_error_rate"],
    "cleaning_method": ["cleaning_method", "cleaner", "method", "cleaning"],
    "cluster_method": ["cluster_method", "clusterer", "clustering_method", "algorithm"],
    "parameters": ["parameters", "params", "hyperparameters", "best_params", "best_parameters"],
}

METHOD_ALIASES = {
    "mode": "Mode",
    "modeimpute": "Mode",
    "modeimputation": "Mode",
    "modeimputer": "Mode",
    "none": "Mode",
    "baran": "Baran",
    "holoclean": "HoloClean",
    "holo": "HoloClean",
    "bigdansing": "BigDansing",
    "bigdans": "BigDansing",
    "boostclean": "BoostClean",
    "horizon": "Horizon",
    "scared": "SCAReD",
    "unified": "Unified",
    "uniclean": "UniClean",
    "groundtruth": "GroundTruth",
    "groundtruthclean": "GroundTruth",
    "groundtruthcleaned": "GroundTruth",
    "gt": "GroundTruth",
    "oracle": "GroundTruth",
}


def norm_key(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def normalize_method(x: Any) -> str:
    key = norm_key(x)
    return METHOD_ALIASES.get(key, str(x).strip())


def find_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    norm_to_col = {norm_key(c): c for c in df.columns}
    for alias in COLUMN_ALIASES[canonical]:
        key = norm_key(alias)
        if key in norm_to_col:
            return norm_to_col[key]
    return None


def require_column(df: pd.DataFrame, canonical: str, path: Path) -> str:
    col = find_column(df, canonical)
    if col is None:
        raise ValueError(
            f"Missing required column {canonical!r} in {path}. "
            f"Available columns: {list(df.columns)}"
        )
    return col


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_error_rate_to_percent(s: pd.Series) -> pd.Series:
    values = to_numeric(s)
    max_val = values.max(skipna=True)
    if pd.notna(max_val) and max_val <= 1.0:
        values = values * 100.0
    return values


def cluster_family(method: Any) -> str:
    m = str(method).upper().strip()
    if m.startswith("KMEAN"):
        return "KMEANS"
    if m.startswith("DBSCAN"):
        return "DBSCAN"
    if m.startswith("GMM"):
        return "GMM"
    if m.startswith("HC") or "AGGLO" in m or "HIER" in m:
        return "HC"
    return "OTHER"


# ----------------------------------------------------------------------
# Hyperparameter parsing.
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ParamSpec:
    raw_col: str
    delta_col: str
    kind: str  # "numeric" or "categorical"


FAMILY_PARAM_SPECS: Dict[str, list[ParamSpec]] = {
    "KMEANS": [ParamSpec("k", "Δk", "numeric")],
    "DBSCAN": [
        ParamSpec("eps", "Δeps", "numeric"),
        ParamSpec("min_samples", "Δmin_samples", "numeric"),
    ],
    "GMM": [
        ParamSpec("n_components", "Δn_components", "numeric"),
        ParamSpec("cov_type", "Δcov", "categorical"),
    ],
    "HC": [
        ParamSpec("linkage", "Δlinkage", "categorical"),
    ],
}

PARAM_COLS = ["k", "n_components", "eps", "min_samples", "cov_type", "linkage"]
_NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _to_float(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        if pd.isna(x):
            return None
        return float(x)
    txt = str(x).strip().strip("'\"")
    if txt == "" or txt.lower() in {"none", "nan", "null"}:
        return None
    try:
        return float(txt)
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    f = _to_float(x)
    if f is None:
        return None
    return int(round(f))


def _to_str(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().strip("'\"")
    if s == "" or s.lower() in {"none", "nan", "null", "{}"}:
        return None
    return s


def _is_numeric_string(x: Any) -> bool:
    return _to_float(x) is not None


def _try_parse_mapping(text: str) -> dict[str, Any]:
    t = str(text).strip()
    if t == "" or t in {"{}", "[]"} or t.lower() in {"none", "nan", "null"}:
        return {}

    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        obj = ast.literal_eval(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return {}


def _regex_value(patterns: Iterable[str], text: str) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().strip("'\"")
    return None


def parse_parameters(value: Any, family: str) -> Dict[str, Any]:
    text = "" if pd.isna(value) else str(value).strip()
    mapping_raw = _try_parse_mapping(text)
    mapping = {norm_key(k): v for k, v in mapping_raw.items()}

    k_val = mapping.get("k") or mapping.get("nclusters") or mapping.get("ncluster")
    ncomp_val = mapping.get("ncomponents") or mapping.get("ncomponent")
    eps_val = mapping.get("eps") or mapping.get("epsilon")
    min_samples_val = (
        mapping.get("minsamples")
        or mapping.get("minsample")
        or mapping.get("minpts")
        or mapping.get("minpoints")
    )
    cov_val = (
        mapping.get("covariancetype")
        or mapping.get("covtype")
        or mapping.get("covariance")
        or mapping.get("cov")
    )
    linkage_val = mapping.get("linkage") or mapping.get("metric") or mapping.get("affinity")

    if k_val is None:
        k_val = _regex_value(
            [
                rf"(?:^|[{{,;\s])k\s*[:=]\s*({_NUM_RE})",
                rf"n[_\s-]*clusters?\s*[:=]\s*({_NUM_RE})",
            ],
            text,
        )

    if ncomp_val is None:
        ncomp_val = _regex_value(
            [
                rf"n[_\s-]*components?\s*[:=]\s*({_NUM_RE})",
                rf"n[_\s-]*clusters?\s*[:=]\s*({_NUM_RE})",
            ],
            text,
        )

    if eps_val is None:
        eps_val = _regex_value(
            [
                rf"(?:eps|epsilon)\s*[:=]\s*({_NUM_RE})",
            ],
            text,
        )

    if min_samples_val is None:
        min_samples_val = _regex_value(
            [
                rf"min[_\s-]*(?:samples?|pts?|points?)\s*[:=]\s*({_NUM_RE})",
            ],
            text,
        )

    if cov_val is None:
        cov_val = _regex_value(
            [
                r"covariance[_\s-]*type\s*[:=]\s*([A-Za-z0-9_.\-]+)",
                r"cov[_\s-]*type\s*[:=]\s*([A-Za-z0-9_.\-]+)",
                r"cov\s*[:=]\s*([A-Za-z0-9_.\-]+)",
            ],
            text,
        )

    if linkage_val is None:
        linkage_val = _regex_value(
            [
                r"linkage\s*[:=]\s*([A-Za-z0-9_.\-]+)",
                r"metric\s*[:=]\s*([A-Za-z0-9_.\-]+)",
                r"affinity\s*[:=]\s*([A-Za-z0-9_.\-]+)",
            ],
            text,
        )

    out: Dict[str, Any] = {
        "k": _to_int(k_val),
        "n_components": _to_int(ncomp_val),
        "eps": _to_float(eps_val),
        "min_samples": _to_int(min_samples_val),
        "cov_type": _to_str(cov_val),
        "linkage": _to_str(linkage_val),
    }

    fam = family.upper()

    if fam == "KMEANS":
        if out["k"] is None:
            out["k"] = out["n_components"]

    elif fam == "GMM":
        if out["n_components"] is None:
            out["n_components"] = out["k"]

    elif fam == "DBSCAN":
        if out["eps"] is None and out["cov_type"] is not None and _is_numeric_string(out["cov_type"]):
            out["eps"] = _to_float(out["cov_type"])
            out["cov_type"] = None

        if out["min_samples"] is None and out["n_components"] is not None:
            out["min_samples"] = out["n_components"]

        if out["min_samples"] is None and out["k"] is not None:
            out["min_samples"] = out["k"]

    elif fam == "HC":
        if out["linkage"] is None and out["cov_type"] is not None and not _is_numeric_string(out["cov_type"]):
            out["linkage"] = out["cov_type"]

        if out["linkage"] is None:
            link = mapping.get("linkage")
            metric = mapping.get("metric") or mapping.get("affinity")
            pieces = [str(x).strip() for x in [link, metric] if x is not None and str(x).strip()]
            if pieces:
                out["linkage"] = "-".join(pieces)

    return out


# ----------------------------------------------------------------------
# Input loading.
# ----------------------------------------------------------------------
def _candidate_files(input_dir: Path, dataset: str) -> list[Path]:
    preferred = [
        input_dir / f"{dataset}_summary.xlsx",
        input_dir / f"{dataset}-summary.xlsx",
    ]
    files = [p for p in preferred if p.exists()]
    if files:
        return files
    return sorted(input_dir.glob(f"*{dataset}*.xlsx"))


def load_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    df0 = pd.read_excel(path, engine="openpyxl")

    col_dataset_id = find_column(df0, "dataset_id")
    col_error_rate = require_column(df0, "error_rate", path)
    col_cleaning = require_column(df0, "cleaning_method", path)
    col_cluster = require_column(df0, "cluster_method", path)
    col_params = require_column(df0, "parameters", path)

    df = pd.DataFrame()
    df["dataset"] = dataset_name
    df["dataset_id"] = df0[col_dataset_id] if col_dataset_id is not None else np.arange(len(df0))
    df["error_rate_raw"] = to_numeric(df0[col_error_rate])
    df["error_rate_pct"] = normalize_error_rate_to_percent(df0[col_error_rate])
    df["error_bin"] = ((df["error_rate_pct"] / 5.0).round() * 5).astype("Int64")
    df["cleaning_method_raw"] = df0[col_cleaning].astype(str).str.strip()
    df["cleaning_method"] = df["cleaning_method_raw"].map(normalize_method)
    df["cluster_method_raw"] = df0[col_cluster].astype(str).str.strip()
    df["cluster_method"] = df["cluster_method_raw"].astype(str).str.upper().str.strip()
    df["algo_family"] = df["cluster_method"].map(cluster_family)
    df["parameters_raw"] = df0[col_params]
    df["source_file"] = path.name

    parsed_records = []
    for _, row in df.iterrows():
        parsed_records.append(parse_parameters(row["parameters_raw"], row["algo_family"]))

    parsed = pd.DataFrame(parsed_records, index=df.index)

    for col in PARAM_COLS:
        if col not in parsed.columns:
            parsed[col] = None

    for col in PARAM_COLS:
        df[col] = parsed[col]

    return df


def concat_all(input_dir: Path) -> pd.DataFrame:
    frames = []

    for dataset in DATASETS:
        files = _candidate_files(input_dir, dataset)
        if not files:
            raise FileNotFoundError(f"No summary workbook found for dataset={dataset!r} in {input_dir}")

        path = files[0]
        frames.append(load_dataset(path, dataset))

    df = pd.concat(frames, ignore_index=True)
    df = df[df["algo_family"].isin(FAMILY_PARAM_SPECS)].copy()
    df = df[df["error_bin"].isin(ERROR_BINS)].copy()
    return df


# ----------------------------------------------------------------------
# Delta computation.
# ----------------------------------------------------------------------
def _median_numeric(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").median())


def _mode_categorical(s: pd.Series) -> Any:
    vals = s.dropna().astype(str)
    vals = vals[vals.str.strip() != ""]
    if vals.empty:
        return np.nan
    return vals.mode().iat[0]


def _baseline_value(mode_rows: pd.DataFrame, raw_col: str, kind: str) -> Any:
    if mode_rows.empty:
        return np.nan
    if kind == "numeric":
        return _median_numeric(mode_rows[raw_col])
    return _mode_categorical(mode_rows[raw_col])


def _delta_value(value: Any, base: Any, kind: str) -> float:
    if kind == "numeric":
        v = _to_float(value)
        b = _to_float(base)
        if v is None or b is None:
            return np.nan
        return float(v - b)

    v = _to_str(value)
    b = _to_str(base)
    if v is None or b is None:
        return np.nan
    return 0.0 if v == b else 1.0


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "dataset_id", "error_bin", "cluster_method"]

    for _, grp in df.groupby(group_cols, observed=False, dropna=False):
        fam = grp["algo_family"].dropna().iat[0] if not grp["algo_family"].dropna().empty else "OTHER"
        specs = FAMILY_PARAM_SPECS.get(fam, [])
        mode_rows = grp[grp["cleaning_method"].eq(BASELINE_CLEANER)]

        base_vals = {
            spec.delta_col: _baseline_value(mode_rows, spec.raw_col, spec.kind)
            for spec in specs
        }

        for _, r in grp.iterrows():
            out = r.to_dict()
            for spec in specs:
                base = base_vals.get(spec.delta_col, np.nan)
                out[spec.delta_col] = _delta_value(r.get(spec.raw_col, np.nan), base, spec.kind)
            rows.append(out)

    delta_df = pd.DataFrame(rows)
    delta_df["is_oracle"] = (
        delta_df["cleaning_method"].isin(ORACLE_CLEANERS)
        | delta_df["cleaning_method"].eq("GroundTruth")
    )
    delta_df["is_mode"] = delta_df["cleaning_method"].eq(BASELINE_CLEANER)
    return delta_df


# ----------------------------------------------------------------------
# ANOVA.
# ----------------------------------------------------------------------
def _sig_star(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _empty_anova_row(metric: str, n: int = 0, reason: str = "") -> Dict[str, Any]:
    return {
        "Metric": metric,
        "n": n,
        "n_cleaners": np.nan,
        "n_error_bins": np.nan,
        "Err_R2": np.nan,
        "Err_F": np.nan,
        "Err_p": np.nan,
        "Err_sig": "",
        "Clean_R2": np.nan,
        "Clean_F": np.nan,
        "Clean_p": np.nan,
        "Clean_sig": "",
        "Inter_R2": np.nan,
        "Inter_F": np.nan,
        "Inter_p": np.nan,
        "Inter_sig": "",
        "reason": reason,
    }


def anova_single_metric(delta_df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    use = delta_df.copy()

    if not ANOVA_INCLUDE_ORACLE:
        use = use[~use["is_oracle"]]
    if not ANOVA_INCLUDE_MODE:
        use = use[~use["is_mode"]]

    if metric not in use.columns:
        return _empty_anova_row(metric, reason="metric column missing")

    sub = use[[metric, "error_bin", "cleaning_method"]].copy()
    sub = sub.dropna(subset=[metric, "error_bin", "cleaning_method"])
    sub = sub.rename(columns={metric: "y"})
    sub["error_bin"] = pd.to_numeric(sub["error_bin"], errors="coerce")
    sub = sub.dropna(subset=["error_bin"])

    n = len(sub)
    n_cleaners = sub["cleaning_method"].nunique()
    n_error_bins = sub["error_bin"].nunique()

    if n < 6 or n_cleaners < 2 or n_error_bins < 2 or sub["y"].nunique() < 2:
        row = _empty_anova_row(metric, n=n, reason="insufficient variation or cells")
        row["n_cleaners"] = n_cleaners
        row["n_error_bins"] = n_error_bins
        return row

    sub["cleaning_method"] = sub["cleaning_method"].astype("category")

    if ERROR_AS_CATEGORICAL:
        sub["error_bin_cat"] = sub["error_bin"].astype(int).astype(str).astype("category")
        formula = "y ~ C(error_bin_cat) + C(cleaning_method) + C(error_bin_cat):C(cleaning_method)"
        term_err = "C(error_bin_cat)"
        term_clean = "C(cleaning_method)"
        term_inter = "C(error_bin_cat):C(cleaning_method)"
    else:
        formula = "y ~ error_bin + C(cleaning_method) + error_bin:C(cleaning_method)"
        term_err = "error_bin"
        term_clean = "C(cleaning_method)"
        term_inter = "error_bin:C(cleaning_method)"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ols(formula, data=sub, missing="drop").fit()
            aov = sm.stats.anova_lm(model, typ=2)
    except Exception as exc:
        row = _empty_anova_row(metric, n=n, reason=f"anova failed: {type(exc).__name__}: {exc}")
        row["n_cleaners"] = n_cleaners
        row["n_error_bins"] = n_error_bins
        return row

    total_ss = float(aov["sum_sq"].sum()) if "sum_sq" in aov else np.nan

    def get(term: str, col: str) -> float:
        if term in aov.index and col in aov.columns:
            try:
                return float(aov.loc[term, col])
            except Exception:
                return np.nan
        return np.nan

    def r2(term: str) -> float:
        ss = get(term, "sum_sq")
        if pd.isna(ss) or pd.isna(total_ss) or total_ss <= 0:
            return np.nan
        return ss / total_ss

    err_p = get(term_err, "PR(>F)")
    clean_p = get(term_clean, "PR(>F)")
    inter_p = get(term_inter, "PR(>F)")

    return {
        "Metric": metric,
        "n": n,
        "n_cleaners": n_cleaners,
        "n_error_bins": n_error_bins,
        "Err_R2": r2(term_err),
        "Err_F": get(term_err, "F"),
        "Err_p": err_p,
        "Err_sig": _sig_star(err_p),
        "Clean_R2": r2(term_clean),
        "Clean_F": get(term_clean, "F"),
        "Clean_p": clean_p,
        "Clean_sig": _sig_star(clean_p),
        "Inter_R2": r2(term_inter),
        "Inter_F": get(term_inter, "F"),
        "Inter_p": inter_p,
        "Inter_sig": _sig_star(inter_p),
        "reason": "",
    }


def compute_anova_table(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, specs in FAMILY_PARAM_SPECS.items():
        fam_df = delta_df[delta_df["algo_family"].eq(fam)].copy()
        for spec in specs:
            row = anova_single_metric(fam_df, spec.delta_col)
            row["Family"] = fam
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Paper Table 7 formatting.
# ----------------------------------------------------------------------
DISPLAY_ORDER = [
    ("KMEANS", "Δk", "Δk"),
    ("DBSCAN", "Δeps", "Δε"),
    ("DBSCAN", "Δmin_samples", "ΔminPts"),
    ("GMM", "Δn_components", "Δn_comp"),
    ("GMM", "Δcov", "Δcov (GMM)"),
    ("HC", "Δlinkage", "Δlinkage (HC)"),
]


def _fmt_num(x: Any, ndigits: int = 3) -> str:
    if x is None or pd.isna(x):
        return ""
    try:
        value = float(x)
    except Exception:
        return ""
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:.{ndigits}f}"


def _fmt_p(p: Any, sig: str = "") -> str:
    if p is None or pd.isna(p):
        return ""
    p = float(p)
    sig = sig or _sig_star(p)

    if p < 0.001:
        # CSV-friendly scientific notation using a multiplication sign.
        s = f"{p:.2e}"
        base, exp = s.split("e")
        exp_i = int(exp)
        return f"{base}×10^{exp_i}{sig}"

    return f"{p:.3f}{sig}"


def make_paper_table7_csv(anova_df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for family, metric, label in DISPLAY_ORDER:
        sub = anova_df[
            anova_df["Family"].eq(family)
            & anova_df["Metric"].eq(metric)
        ]

        if sub.empty:
            rec = {
                "Parameter": label,
                "Error-rate effect R2_eff": "",
                "Error-rate effect F": "",
                "Error-rate effect p": "",
                "Cleaning effect R2_eff": "",
                "Cleaning effect F": "",
                "Cleaning effect p": "",
                "Interaction effect R2_eff": "",
                "Interaction effect F": "",
                "Interaction effect p": "",
            }
        else:
            r = sub.iloc[0]
            rec = {
                "Parameter": label,
                "Error-rate effect R2_eff": _fmt_num(r.get("Err_R2")),
                "Error-rate effect F": _fmt_num(r.get("Err_F")),
                "Error-rate effect p": _fmt_p(r.get("Err_p"), r.get("Err_sig", "")),
                "Cleaning effect R2_eff": _fmt_num(r.get("Clean_R2")),
                "Cleaning effect F": _fmt_num(r.get("Clean_F")),
                "Cleaning effect p": _fmt_p(r.get("Clean_p"), r.get("Clean_sig", "")),
                "Interaction effect R2_eff": _fmt_num(r.get("Inter_R2")),
                "Interaction effect F": _fmt_num(r.get("Inter_F")),
                "Interaction effect p": _fmt_p(r.get("Inter_p"), r.get("Inter_sig", "")),
            }

        rows.append(rec)

    out = pd.DataFrame(rows)
    out_path = out_dir / "table7_hyper_anova_all.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


# ----------------------------------------------------------------------
# Stage 3 entry point.
# ----------------------------------------------------------------------
def build(ctx: BuildContext) -> ArtifactResult:
    out_dir = ctx.output_dir / "table_7"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep output directory minimal.
    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()

    raw = concat_all(ctx.input_root)
    delta_df = compute_deltas(raw)
    anova_df = compute_anova_table(delta_df)

    out_path = make_paper_table7_csv(anova_df, out_dir)

    inputs = []
    for dataset in DATASETS:
        files = _candidate_files(ctx.input_root, dataset)
        if files:
            inputs.append(files[0])

    return ArtifactResult(
        artifact_id=ARTIFACT["id"],
        status="success",
        outputs=[out_path],
        inputs=inputs,
        message=f"Built Table 7 CSV under {out_dir}.",
        metadata={
            "output_subdir": "table_7",
            "expected_output_count": 1,
            "actual_output_count": 1,
            "anova_include_mode": ANOVA_INCLUDE_MODE,
            "anova_include_oracle": ANOVA_INCLUDE_ORACLE,
            "error_as_categorical": ERROR_AS_CATEGORICAL,
        },
    )
