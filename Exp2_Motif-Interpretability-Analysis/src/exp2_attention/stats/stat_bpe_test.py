from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection


@dataclass
class TestConfig:
    alternative: str = "two-sided"  # 'two-sided' | 'greater' | 'less'


def choose_normality(diff: np.ndarray) -> Tuple[str, float, bool]:
    """
    对差值向量选择正态性检验：样本量小用Shapiro，大用D'Agostino
    返回 (测试名, p值, 是否认为正态)
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    if diff.size == 0:
        return ("NA", np.nan, False)
    if diff.size <= 5000:
        stat, p = stats.shapiro(diff)
        return ("shapiro", float(p), bool(p >= 0.05))
    stat, p = stats.normaltest(diff)
    return ("dagostino", float(p), bool(p >= 0.05))


def paired_test(motif: np.ndarray, random_: np.ndarray, cfg: TestConfig) -> Dict:
    motif = np.asarray(motif, dtype=float)
    random_ = np.asarray(random_, dtype=float)
    mask = ~np.isnan(motif) & ~np.isnan(random_)
    motif = motif[mask]
    random_ = random_[mask]
    n = motif.size
    if n == 0:
        return {"normality": "NA", "normality_p": np.nan, "test": "NA", "n": 0,
                "stat": np.nan, "pval": np.nan, "effect": np.nan, "effect_name": "NA",
                "mean_delta": np.nan}

    diff = motif - random_
    name, p_norm, is_normal = choose_normality(diff)
    if is_normal:
        t, p = stats.ttest_rel(motif, random_, alternative=cfg.alternative)
        sd = float(np.std(diff, ddof=1))
        d = np.nan if sd == 0.0 else float(np.mean(diff) / sd)
        return {
            "normality": name,
            "normality_p": float(p_norm),
            "test": "ttest_rel",
            "n": int(n),
            "stat": float(t),
            "pval": float(p),
            "effect": d,
            "effect_name": "cohens_d",
            "mean_delta": float(np.mean(diff)),
        }

    w = stats.wilcoxon(motif, random_, alternative=cfg.alternative, zero_method="zsplit")
    # 近似从p值获得Z分数（双侧与单侧）
    if cfg.alternative == "two-sided":
        z = float(stats.norm.isf(w.pvalue / 2.0))
    else:
        z = float(stats.norm.isf(w.pvalue))
    r = z / np.sqrt(n) if n > 0 else np.nan
    return {
        "normality": name,
        "normality_p": float(p_norm),
        "test": "wilcoxon",
        "n": int(n),
        "stat": float(w.statistic),
        "pval": float(w.pvalue),
        "effect": float(r),
        "effect_name": "r",
        "mean_delta": float(np.mean(diff)),
    }


def fdr_bh(df: pd.DataFrame, pcol: str = "pval", alpha: float = 0.05) -> pd.DataFrame:
    p = df[pcol].values
    mask = ~np.isnan(p)
    rej, p_adj = fdrcorrection(p[mask], alpha=alpha)
    p_full = np.full_like(p, fill_value=np.nan, dtype=float)
    p_full[mask] = p_adj
    df["pval_adj"] = p_full
    rej_full = np.zeros_like(mask, dtype=bool)
    rej_full[mask] = rej
    df["rejected_fdr_0.05"] = rej_full
    return df


