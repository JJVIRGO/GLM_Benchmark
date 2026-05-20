from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from ast import literal_eval


@dataclass
class TokenMapping:
    """
    Token级别的映射信息
    """
    motif_tokens: np.ndarray  # motif对应的token位置（1表示属于motif，0表示不属于）
    background_tokens: List[np.ndarray]  # 随机背景的token位置列表


def load_mapping_data(mapping_csv: str) -> pd.DataFrame:
    """
    加载映射数据，解析mapping列
    """
    df = pd.read_csv(mapping_csv)
    # 解析mapping列，将字符串转换为numpy数组
    df['mapping_parsed'] = df['mapping'].apply(lambda x: np.array(literal_eval(x), dtype=int))
    return df


def extract_token_positions(mapping_array: np.ndarray, vector_length: int) -> np.ndarray:
    """
    从映射数组中提取token位置（值为1的位置）
    """
    token_positions = np.where(mapping_array == 1)[0]
    # 对齐到注意力向量长度
    if len(token_positions) > vector_length:
        token_positions = token_positions[:vector_length]
    return token_positions


def sample_background_tokens(
    motif_tokens: np.ndarray,
    vector_length: int,
    num_random: int,
    seed: int
) -> List[np.ndarray]:
    """
    基于motif的token数量，随机选择背景token位置

    Args:
        motif_tokens: motif对应的token位置数组
        vector_length: 注意力向量长度
        num_random: 随机样本数量
        seed: 随机种子

    Returns:
        背景token位置的列表，每个元素是一个numpy数组
    """
    rng = np.random.default_rng(seed)

    # 获取所有可用的token位置（排除特殊token）
    available_positions = np.arange(1, vector_length - 1, dtype=int)  # 排除CLS和可能的SEP

    # 确保有足够的可用位置
    k = int(len(motif_tokens))
    if len(available_positions) < k:
        raise ValueError(f"可用token位置不足: 需要{k}, 只有{len(available_positions)}")

    background_samples: List[np.ndarray] = []
    # 预分配并一次生成，避免为每个样本创建新的RNG
    for _ in range(num_random):
        selected_positions = rng.choice(available_positions, size=k, replace=False)
        background_mapping = np.zeros(vector_length, dtype=int)
        background_mapping[selected_positions] = 1
        background_samples.append(background_mapping)

    return background_samples


def build_token_mappings_optimized(
    mapping_df: pd.DataFrame,
    sequence_name: str,
    motif_idx: int,
    vector_length: int,
    num_random: int,
    seed: int,
    sub_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    基于现有的映射数据构建token级别的映射
    这是优化版本，直接使用已有的mapping信息

    Args:
        mapping_df: 映射数据DataFrame
        sequence_name: 序列名称
        motif_idx: motif索引
        vector_length: 注意力向量长度
        num_random: 随机样本数量
        seed: 随机种子

    Returns:
        (motif_mapping, background_mappings)
    """
    # 在该 sequence 对应的子表中，按出现顺序选择第 motif_idx 行
    if sub_df is None:
        sub_df = mapping_df[mapping_df['sequence_name'] == sequence_name]
    if motif_idx < 0 or motif_idx >= len(sub_df):
        raise ValueError(f"未找到序列 {sequence_name} 的motif {motif_idx}")

    # 获取mapping数组
    motif_mapping = sub_df.iloc[motif_idx]['mapping_parsed']

    # 对齐长度
    if len(motif_mapping) > vector_length:
        motif_mapping = motif_mapping[:vector_length]
    elif len(motif_mapping) < vector_length:
        padding = np.zeros(vector_length - len(motif_mapping), dtype=motif_mapping.dtype)
        motif_mapping = np.concatenate([motif_mapping, padding])

    # 解析解替代背景抽样：不再生成随机背景，返回 None 作为占位
    background_mappings = None  # type: ignore

    return motif_mapping, background_mappings


def compute_statistic_optimized(
    attn: np.ndarray,
    motif_mapping: np.ndarray,
    background_mappings,
    aggregate_method: str = "mean"
) -> Dict:
    """
    优化版本的统计计算，直接使用token映射

    Args:
        attn: 注意力向量
        motif_mapping: motif的token映射
        background_mappings: 背景token映射列表
        aggregate_method: 聚合方法

    Returns:
        包含motif_score和random_scores的字典
    """
    if attn is None or motif_mapping is None:
        return {"motif_score": np.nan, "random_scores": np.array([])}

    attn = np.asarray(attn, dtype=np.float32)
    motif_mapping = np.asarray(motif_mapping, dtype=np.float32)

    if attn.size == 0 or motif_mapping.size == 0:
        return {"motif_score": np.nan, "random_scores": np.array([])}

    n = attn.shape[0]
    if n != motif_mapping.shape[0]:
        return {"motif_score": np.nan, "random_scores": np.array([])}

    # 计算motif得分
    denom = (1.0 - float(attn[0])) / max(n - 1, 1)
    k = float(np.sum(motif_mapping))
    if k <= 0.0 or denom == 0.0:
        motif_score = np.nan
    else:
        numer = float(attn @ motif_mapping) / k
        motif_score = numer / denom

    # 计算随机背景得分
    if background_mappings is None:
        # 解析解：不再采样，直接使用可用token的均值/中位数
        if n <= 2:
            random_scores = np.empty((0,), dtype=np.float32)
            random_mean = np.nan
            random_std = np.nan
        else:
            available = attn[1:n-1]
            if aggregate_method == "median":
                random_mean = float(np.nanmedian(available) / denom)
                random_std = np.nan  # 中位数解析解标准差较复杂，置为 NaN
            else:
                mu = float(np.nanmean(available))
                # 近似标准差：SRSWOR 的样本均值方差 var = (sigma^2 / k) * ((N-k)/(N-1))
                # 其中 k 为 motif token 数
                if k > 0 and (n - 2) > 1:
                    sigma2 = float(np.nanvar(available))
                    N = float(n - 2)
                    var_mean = (sigma2 / k) * ((N - k) / (N - 1.0))
                    random_std = float(np.sqrt(var_mean) / denom)
                else:
                    random_std = np.nan
                random_mean = float(mu / denom)
            random_scores = np.empty((0,), dtype=np.float32)
    elif isinstance(background_mappings, np.ndarray):
        # 期望形状：(num_random, n)
        bg_mat = np.asarray(background_mappings, dtype=np.float32)
        if bg_mat.ndim == 1:
            bg_mat = bg_mat.reshape(1, -1)
    else:
        # list of arrays -> 2D
        try:
            bg_mat = np.asarray(background_mappings, dtype=np.float32)
            if bg_mat.ndim == 1:
                bg_mat = bg_mat.reshape(1, -1)
        except Exception:
            # 回退：逐个堆叠
            bg_mat = np.stack([np.asarray(x, dtype=np.float32) for x in background_mappings], axis=0) if len(background_mappings) > 0 else np.empty((0, n), dtype=np.float32)

    if bg_mat.size == 0:
        random_scores = np.empty((0,), dtype=np.float32)
        random_mean = np.nan
        random_std = np.nan
    else:
        k_vec = np.sum(bg_mat, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            dots = bg_mat @ attn
            random_scores = np.where(k_vec > 0.0, (dots / k_vec) / denom, np.nan).astype(np.float32)
        if aggregate_method == "median":
            random_mean = float(np.nanmedian(random_scores))
        else:
            random_mean = float(np.nanmean(random_scores))
        random_std = float(np.nanstd(random_scores)) if random_scores.size > 0 else np.nan

    return {
        "motif_score": float(motif_score) if not np.isnan(motif_score) else np.nan,
        "random_mean": random_mean if not (isinstance(random_mean, float) and np.isnan(random_mean)) else np.nan,
        "random_std": random_std,
        "random_scores": random_scores,
        "delta": float(motif_score - random_mean) if (not np.isnan(motif_score) and not np.isnan(random_mean)) else np.nan,
    }


def compute_statistic_optimized_batch(
    attn_mat: np.ndarray,
    motif_mapping: np.ndarray,
    aggregate_method: str = "mean",
) -> Dict[str, np.ndarray]:
    """
    批量计算多个 (layer, head) 的 motif 分数与解析解背景统计。

    Args:
        attn_mat: 形状为 (L, N) 的注意力矩阵，L 为 (layer, head) 数量
        motif_mapping: 长度 N 的 0/1 向量
        aggregate_method: "mean" 或 "median"

    Returns:
        {
          "motif_score": (L,),
          "random_mean": (L,),
          "random_std": (L,),
        }
    """
    if attn_mat is None or motif_mapping is None:
        return {"motif_score": np.array([]), "random_mean": np.array([]), "random_std": np.array([])}
    A = np.asarray(attn_mat, dtype=np.float32)
    m = np.asarray(motif_mapping, dtype=np.float32)
    if A.ndim != 2 or m.ndim != 1:
        return {"motif_score": np.array([]), "random_mean": np.array([]), "random_std": np.array([])}
    L, n = A.shape
    if m.shape[0] != n or n == 0:
        return {"motif_score": np.array([]), "random_mean": np.array([]), "random_std": np.array([])}

    denom_vec = (1.0 - A[:, 0].astype(np.float32)) / max(n - 1, 1)
    k = float(np.sum(m))
    # motif 分数
    dots = A @ m
    with np.errstate(divide='ignore', invalid='ignore'):
        numer = np.where(k > 0.0, dots / k, np.nan)
        motif_score = numer / denom_vec

    # 背景解析解
    if n <= 2:
        random_mean = np.full(L, np.nan, dtype=np.float32)
        random_std = np.full(L, np.nan, dtype=np.float32)
    else:
        available = A[:, 1:n-1]
        if aggregate_method == "median":
            mu = np.nanmedian(available, axis=1).astype(np.float32)
            random_mean = mu / denom_vec
            random_std = np.full(L, np.nan, dtype=np.float32)
        else:
            mu = np.nanmean(available, axis=1).astype(np.float32)
            random_mean = mu / denom_vec
            if k > 0.0 and (n - 2) > 1:
                sigma2 = np.nanvar(available, axis=1).astype(np.float32)
                N = float(n - 2)
                var_mean = (sigma2 / k) * ((N - k) / (N - 1.0))
                with np.errstate(invalid='ignore'):
                    random_std = np.sqrt(var_mean) / denom_vec
            else:
                random_std = np.full(L, np.nan, dtype=np.float32)

    return {
        "motif_score": motif_score.astype(np.float32),
        "random_mean": random_mean.astype(np.float32),
        "random_std": random_std.astype(np.float32),
    }
