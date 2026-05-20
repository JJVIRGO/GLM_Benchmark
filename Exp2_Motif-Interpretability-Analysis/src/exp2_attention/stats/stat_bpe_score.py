from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def parse_attention_vector(attn_str: str) -> np.ndarray:
    """
    将逗号分隔的注意力向量字符串解析为 float ndarray
    """
    if isinstance(attn_str, (list, tuple, np.ndarray)):
        return np.asarray(attn_str, dtype=np.float32)
    if not isinstance(attn_str, str):
        return np.asarray([], dtype=np.float32)
    attn_str = attn_str.strip()
    if not attn_str:
        return np.asarray([], dtype=np.float32)
    return np.array([float(x) for x in attn_str.split(',')], dtype=np.float32)


def compute_statistic(attn: np.ndarray, mapping: np.ndarray) -> float:
    """
    统计量：
    (attn · mapping / motif_token_count) / ((1 - attn[0]) / (N - 1))
    """
    if attn is None or mapping is None:
        return np.nan
    if len(attn) == 0 or len(mapping) == 0:
        return np.nan
    n = len(attn)
    if n != len(mapping):
        return np.nan
    denom = (1.0 - float(attn[0])) / max(n - 1, 1)
    k = int(np.sum(mapping))
    if k <= 0 or denom == 0.0:
        return np.nan
    numer = float(np.dot(attn, mapping)) / k
    return numer / denom


def aggregate_random(scores: np.ndarray, method: str = "mean") -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.nan
    if method == "median":
        return float(np.nanmedian(scores))
    return float(np.nanmean(scores))


@dataclass
class SampleScore:
    sequence_name: str
    motif_id: int
    layer: int
    head: int
    motif_score: float
    random_mean: float
    random_std: float
    delta: float


def pack_sample_scores(
    sequence_name: str,
    motif_id: int,
    layer: int,
    head: int,
    motif_score: float,
    random_scores: np.ndarray,
    aggregate_method: str,
    random_mean: float = None,
    random_std: float = None,
) -> Dict:
    """
    打包逐样本分数。

    支持两种输入形式：
    1) 传入 random_scores（保持兼容原实现，由本函数计算均值/方差）
    2) 直接传入 random_mean/random_std（用于解析解背景，避免生成随机样本）
    """
    r_mean: float
    r_std: float
    if random_mean is not None:
        r_mean = float(random_mean)
        r_std = float(random_std) if random_std is not None else np.nan
    else:
        random_scores = np.asarray(random_scores, dtype=float)
        r_mean = aggregate_random(random_scores, aggregate_method)
        r_std = float(np.nanstd(random_scores)) if random_scores.size else np.nan
    delta = float(motif_score - r_mean) if (not np.isnan(motif_score) and not np.isnan(r_mean)) else np.nan
    return {
        "sequence_name": sequence_name,
        "motif_id": motif_id,
        "layer": layer,
        "head": head,
        "motif_score": float(motif_score) if not np.isnan(motif_score) else np.nan,
        "random_mean": r_mean,
        "random_std": r_std,
        "delta": delta,
    }


def build_attention_index(attn_df: pd.DataFrame) -> Dict[Tuple[str, int, int], np.ndarray]:
    """
    构建基于 (sequence_name, layer, head) 的注意力向量索引
    仅使用 CLS -> 所有 token 的向量（attn vector）
    """
    index: Dict[Tuple[str, int, int], np.ndarray] = {}
    # 使用 itertuples 提升遍历性能
    cols = [c for c in ["sequence_name", "layer", "head", "attention_vector"] if c in attn_df.columns]
    for row in attn_df.loc[:, cols].itertuples(index=False, name=None):
        # row 结构: (sequence_name, layer, head, attention_vector)
        seq, layer, head, attn_vec_str = row
        vec = parse_attention_vector(attn_vec_str)
        index[(str(seq), int(layer), int(head))] = vec
    return index



