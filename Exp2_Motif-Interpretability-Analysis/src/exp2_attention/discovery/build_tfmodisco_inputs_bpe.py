import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from exp2_attention.paths import DISCOVERY_ROOT

DEFAULT_DISCOVERY_ROOT = str(DISCOVERY_ROOT)

# ------------------------------
# 配置
# ------------------------------
@dataclass
class BPEConfig:
    tf_name: str
    model_type: str  # DNABERT-2 | GENA_LM_BERT | GROVER
    csv_path: str
    attn_dir: str
    out_dir: str
    tokenizer_path: str
    max_sequences: Optional[int] = None
    dtype: str = "float32"
    eps: float = 1e-8
    center: str = "mean"  # none/mean/median/zscore
    fake_negative: bool = False
    fake_neg_base: str = "A"
    fake_neg_scale: float = 0.01


# ------------------------------
# 工具函数
# ------------------------------
BASE_TO_CH = {"A": 0, "C": 1, "G": 2, "T": 3}


def build_ohe(seqs: List[str], dtype: str = "float32") -> np.ndarray:
    if len(seqs) == 0:
        raise ValueError("空序列列表")
    lengths = [len(s) for s in seqs]
    L_min = min(lengths)
    L_max = max(lengths)
    if L_min != L_max:
        # 为保证 TF-MoDISco 的 [N,4,L] 刚性，这里裁剪为最短长度
        seqs = [s[:L_min] for s in seqs]
    N = len(seqs)
    L = len(seqs[0])
    ohe = np.zeros((N, 4, L), dtype=dtype)
    for n, s in enumerate(seqs):
        for p, ch in enumerate(s):
            idx = BASE_TO_CH.get(ch)
            if idx is not None:
                ohe[n, idx, p] = 1.0
    return ohe


def scatter_to_real_base_channel(ohe_1n4l: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
    _, L = ohe_1n4l.shape
    out = np.zeros_like(ohe_1n4l, dtype=np.float32)
    if L == 0:
        return out
    real_ch = np.argmax(ohe_1n4l, axis=0)
    has_real = (ohe_1n4l.sum(axis=0) > 0)
    idx = np.where(has_real)[0]
    if idx.size > 0:
        out[real_ch[idx], idx] = base_scores[idx]
    return out


def parse_attention_vector(attn_value) -> np.ndarray:
    # 兼容 list/tuple/ndarray 和 字符串 "0.1,0.2,0.3"
    if isinstance(attn_value, (list, tuple, np.ndarray)):
        return np.asarray(attn_value, dtype=np.float32)
    if not isinstance(attn_value, str):
        return np.asarray([], dtype=np.float32)
    s = attn_value.strip()
    if not s:
        return np.asarray([], dtype=np.float32)
    return np.array([float(x) for x in s.split(',')], dtype=np.float32)


def compute_fc_vector_bpe(attn_vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    计算 BPE 模型逐 token 的 FC（去除 CLS 与 EOS/SEP）。
    FC[t] = attn[t] / ((1 - attn[0] - attn[-1]) / (n - 2)), t in [1, n-2]
    返回长度为 n-2 的向量（不含首尾特殊 token）。
    """
    n = int(attn_vec.shape[0])
    if n <= 2:
        return np.zeros((0,), dtype=np.float32)
    denom = (1.0 - float(attn_vec[0]) - float(attn_vec[-1])) / max(n - 2, 1)
    if abs(denom) < eps:
        denom = eps
    return (attn_vec[1:-1].astype(np.float32) / denom).astype(np.float32)


def tokens_to_base_contrib_offsets(
    fc_tokens: np.ndarray,
    valid_offsets: List[Tuple[int, int]],
    L: int,
) -> np.ndarray:
    """
    将逐 token FC（对应有效非特殊 token 的顺序）均分到其覆盖的碱基范围内。
    - fc_tokens: [T]
    - valid_offsets: 长度为 T 的 (start, end) 列表，start/end 为字符级区间，半开 [start,end)
    - 返回: [L]
    """
    T = int(fc_tokens.shape[0])
    base_scores = np.zeros((L,), dtype=np.float32)
    if T <= 0 or L <= 0:
        return base_scores
    for j in range(min(T, len(valid_offsets))):
        start, end = valid_offsets[j]
        if end <= start:
            continue
        start_clip = max(0, int(start))
        end_clip = min(L, int(end))
        span = max(end_clip - start_clip, 0)
        if span <= 0:
            continue
        inc = float(fc_tokens[j]) / float(span)
        base_scores[start_clip:end_clip] += inc
    return base_scores


def resolve_default_paths(model_type: str, tf_name: str, output_model_name: Optional[str] = None) -> Tuple[str, str, str]:
    model_dir_name = output_model_name or model_type
    csv_path = f"{DEFAULT_DISCOVERY_ROOT}/predict_true/{model_dir_name}/{tf_name}_train_true.csv"
    attn_dir = f"{DEFAULT_DISCOVERY_ROOT}/predict_true/{model_dir_name}/attention/{tf_name}_attention_weight"
    out_dir = f"{DEFAULT_DISCOVERY_ROOT}/predict_true/{model_dir_name}/tfmodisco_inputs/{tf_name}"
    os.makedirs(out_dir, exist_ok=True)
    return csv_path, attn_dir, out_dir


def detect_allowed_layers(part_files: List[str], model_type: str) -> Optional[set]:
    """
    根据模型类型返回允许处理的层集合。
    - DNABERT-2/GENA_LM_BERT: None（表示全层）
    - GROVER: 最后三层
    """
    if model_type != "GROVER":
        return None
    layers = set()
    for f in part_files:
        try:
            df = pd.read_parquet(f, columns=["layer"])  # 仅取列，避免大内存
            if not df.empty and "layer" in df.columns:
                vals = [int(x) for x in pd.unique(df["layer"]) if pd.notna(x)]
                layers.update(vals)
        except Exception:
            continue
    if not layers:
        return None
    sorted_layers = sorted(layers)
    last3 = set(sorted_layers[-3:])
    return last3


def build_token_offsets(
    tokenizer: AutoTokenizer, sequences: List[str]
) -> List[List[Tuple[int, int]]]:
    """
    生成每条序列的有效 token（排除特殊 token）对应的 (start,end) 偏移列表。
    要求 tokenizer 为 fast（具备 offsets）。
    """
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("需要 fast tokenizer 以提供 offsets 映射")
    enc = tokenizer(
        sequences,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    all_offsets = enc["offset_mapping"]
    all_special = enc.get("special_tokens_mask", None)

    result: List[List[Tuple[int, int]]] = []
    for i in range(len(sequences)):
        offs_i = all_offsets[i]
        if all_special is not None:
            spec_i = all_special[i]
        else:
            spec_i = [0] * len(offs_i)
        valid_i: List[Tuple[int, int]] = []
        for (st, ed), sp in zip(offs_i, spec_i):
            st_i, ed_i = int(st), int(ed)
            if sp == 1:
                continue
            if ed_i <= st_i:
                continue
            valid_i.append((st_i, ed_i))
        result.append(valid_i)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["DNABERT-2", "GENA_LM_BERT", "GROVER"]) 
    parser.add_argument("--output_model_name", type=str, default=None,
                        help="输入/输出目录使用的模型名；不影响tokenizer/model_type逻辑")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--attn_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="HuggingFace tokenizer 路径/名称；GROVER 需 trust_remote_code=True")
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--center", type=str, default="mean",
                        choices=["none", "mean", "median", "zscore"],
                        help="对每条序列的贡献向量做去中心/标准化以引入负值")
    parser.add_argument("--fake_negative", action="store_true", default=False,
                        help="若最终所有贡献均为非负，则追加一条虚构负样本以满足 TF-MoDISco 双侧阈值计算。建议与 --center none 联用以不改动原FC。")
    parser.add_argument("--fake_neg_base", type=str, default="A", choices=["A","C","G","T"],
                        help="虚构负样本序列所用的碱基（整段相同碱基）")
    parser.add_argument("--fake_neg_scale", type=float, default=0.01,
                        help="虚构负样本的幅度比例；实际幅度 = 该比例 * max|hypscores|（若为0则退化为1e-3）")
    args = parser.parse_args()

    # 默认路径解析
    csv_path = args.csv_path
    attn_dir = args.attn_dir
    out_dir = args.out_dir
    if csv_path is None or attn_dir is None or out_dir is None:
        _csv, _attn, _out = resolve_default_paths(args.model_type, args.tf_name, args.output_model_name)
        csv_path = csv_path or _csv
        attn_dir = attn_dir or _attn
        out_dir = out_dir or _out
    os.makedirs(out_dir, exist_ok=True)

    cfg = BPEConfig(
        tf_name=args.tf_name,
        model_type=args.model_type,
        csv_path=csv_path,
        attn_dir=attn_dir,
        out_dir=out_dir,
        tokenizer_path=args.tokenizer_path,
        max_sequences=args.max_sequences,
        center=args.center,
        fake_negative=args.fake_negative,
        fake_neg_base=args.fake_neg_base,
        fake_neg_scale=args.fake_neg_scale,
    )

    print(f"TF: {cfg.tf_name}")
    print(f"模型: {cfg.model_type}")
    print(f"序列CSV: {cfg.csv_path}")
    print(f"注意力目录: {cfg.attn_dir}")
    print(f"输出目录: {cfg.out_dir}")
    print(f"tokenizer: {cfg.tokenizer_path}")
    print(f"center={cfg.center}")
    print(f"fake_negative={cfg.fake_negative}, fake_neg_base={cfg.fake_neg_base}, fake_neg_scale={cfg.fake_neg_scale}")

    # 读取序列
    df_seq = pd.read_csv(cfg.csv_path)
    if "sequence" not in df_seq.columns:
        raise ValueError("CSV 缺少 sequence 列")
    if cfg.max_sequences is not None:
        df_seq = df_seq.iloc[: int(cfg.max_sequences)].copy()
    sequences = df_seq["sequence"].astype(str).tolist()
    sequence_names = [f"seq_{i}" for i in range(len(sequences))]

    # 构建 OHE（必要时裁剪为最短长度）
    ohe = build_ohe(sequences, dtype="float32")
    N, _, L = ohe.shape
    print(f"样本数 N={N}, 长度 L={L}")

    # 加载 tokenizer（GROVER 需 trust_remote_code）
    print("加载tokenizer...")
    if cfg.model_type == "GROVER":
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)

    # 生成 offsets 映射（非特殊 token）
    print("生成 token offsets 映射...")
    seq_valid_offsets = build_token_offsets(tokenizer, sequences)  # List[List[(start,end)]]
    if len(seq_valid_offsets) != N:
        raise RuntimeError("offsets 数量与序列数量不一致")

    # 预计算每条序列的有效 token 数（用于后续聚合 shape）
    seq_expected_tokens: List[int] = [len(v) for v in seq_valid_offsets]
    fc_token_max: List[Optional[np.ndarray]] = [None] * N  # 用于跨层跨头逐 token 取最大值

    # 列举注意力分块文件
    if not os.path.isdir(cfg.attn_dir):
        raise FileNotFoundError(f"注意力目录不存在: {cfg.attn_dir}")
    part_files = sorted(
        [os.path.join(cfg.attn_dir, f) for f in os.listdir(cfg.attn_dir) if f.endswith('.parquet')]
    )
    if len(part_files) == 0:
        raise FileNotFoundError(f"未在目录中找到 parquet: {cfg.attn_dir}")

    # 层选择：GROVER 取最后三层，其余全层
    allowed_layers = detect_allowed_layers(part_files, cfg.model_type)
    if allowed_layers is None:
        print("层选择: 全层（DNABERT-2/GENA_LM_BERT 或未能检测到层信息）")
    else:
        print(f"层选择: GROVER 最后三层 {sorted(allowed_layers)}")

    # 构建 seq_name → index 的映射（与生成数据一致）
    name_to_idx: Dict[str, int] = {f"seq_{i}": i for i in range(N)}

    # 聚合：按经验层与各头对逐 token FC 取最大值
    total_rows = 0
    used_rows = 0
    print("读取注意力并跨层跨头聚合（逐token最大）...")
    for pf in tqdm(part_files, desc="Reading attention files"):
        df = pd.read_parquet(pf)
        total_rows += len(df)
        if df.empty:
            continue
        # 层过滤
        if "layer" in df.columns and allowed_layers is not None:
            df = df[df["layer"].astype(int).isin(allowed_layers)]
        # 无论是否过滤，若没有必要列则跳过
        needed_cols = [c for c in ["sequence_name", "attention_vector"] if c in df.columns]
        if len(needed_cols) < 2:
            continue
        # 只保留必要列和层列（用于统计）
        keep_cols = needed_cols + (["layer"] if "layer" in df.columns else [])
        df = df.loc[:, keep_cols]
        if df.empty:
            continue
        used_rows += len(df)

        for row in df.itertuples(index=False):
            # 兼容 presence of layer col or not
            if "layer" in df.columns and len(row) == 3:
                seq_name, attn_val, _layer = row  # type: ignore
            else:
                seq_name, attn_val = row  # type: ignore
            idx = name_to_idx.get(str(seq_name))
            if idx is None or idx < 0 or idx >= N:
                continue
            attn_vec = parse_attention_vector(attn_val)
            if attn_vec.size <= 2:
                continue
            fc_tokens = compute_fc_vector_bpe(attn_vec, eps=cfg.eps)  # 长度≈有效 token 数

            T_exp = int(seq_expected_tokens[idx])
            if T_exp <= 0:
                continue
            T_eff = int(min(T_exp, int(fc_tokens.size)))
            if T_eff <= 0:
                continue
            if fc_token_max[idx] is None:
                fc_token_max[idx] = np.full((T_exp,), -np.inf, dtype=np.float32)
            current = fc_token_max[idx]
            np.maximum(current[:T_eff], fc_tokens[:T_eff], out=current[:T_eff])

    print(f"读取总行数: {total_rows}，被使用行数: {used_rows}")

    # 计算碱基层面贡献
    hypscores = np.zeros_like(ohe, dtype=np.float32)  # [N, 4, L]
    print("\n计算每条序列的碱基层面重要性...")
    for i in tqdm(range(N), desc="Generating hypscores"):
        T_exp = seq_expected_tokens[i]
        if T_exp <= 0:
            continue
        m = fc_token_max[i]
        if m is None:
            continue
        fc_best = np.where(np.isfinite(m), m, 0.0).astype(np.float32)

        # token → base（根据 offsets 均分映射）
        base_scores = tokens_to_base_contrib_offsets(fc_best, seq_valid_offsets[i], L=L)

        # 可选去中心/标准化与兜底扰动（制造双侧）
        if cfg.center != "none":
            x = base_scores.astype(np.float32, copy=True)
            if cfg.center == "mean":
                mu = float(np.mean(x)) if x.size > 0 else 0.0
                x = x - mu
            elif cfg.center == "median":
                med = float(np.median(x)) if x.size > 0 else 0.0
                x = x - med
            elif cfg.center == "zscore":
                mu = float(np.mean(x)) if x.size > 0 else 0.0
                sd = float(np.std(x)) if x.size > 0 else 0.0
                if sd < cfg.eps:
                    sd = cfg.eps
                x = (x - mu) / sd

            if not (np.any(x > 0) and np.any(x < 0)):
                scale = float(np.max(np.abs(x))) if x.size > 0 else 1.0
                ramp = np.linspace(-1.0, 1.0, num=L, dtype=np.float32)
                x = x + (1e-3 * (scale if scale > 0 else 1.0)) * ramp

            base_scores = x

        # 写入真实碱基通道
        hypscores[i] = scatter_to_real_base_channel(ohe[i], base_scores)

    # 若需要且当前没有任何负值，则追加一条虚构负样本
    if cfg.fake_negative:
        has_neg = bool(np.any(hypscores < 0))
        if not has_neg:
            base_idx = BASE_TO_CH.get(cfg.fake_neg_base, 0)
            fake_ohe = np.zeros((4, L), dtype=np.float32)
            if L > 0:
                fake_ohe[base_idx, :] = 1.0
            max_abs = float(np.max(np.abs(hypscores))) if hypscores.size > 0 else 0.0
            amp = cfg.fake_neg_scale * max_abs
            if amp <= 0:
                amp = 1e-3
            fake_base_scores = -np.ones((L,), dtype=np.float32) * float(amp)
            fake_hyps = scatter_to_real_base_channel(fake_ohe, fake_base_scores)
            ohe = np.concatenate([ohe, fake_ohe[np.newaxis, ...]], axis=0)
            hypscores = np.concatenate([hypscores, fake_hyps[np.newaxis, ...]], axis=0)
            print("已追加一条虚构负样本以启用双侧阈值计算，不改动原序列FC")

    # 保存 TF-MoDISco 输入
    ohe_path = os.path.join(cfg.out_dir, "ohe1.npz")
    hyps_path = os.path.join(cfg.out_dir, "hypscores1.npz")
    np.savez_compressed(ohe_path, ohe)
    np.savez_compressed(hyps_path, hypscores)
    print(f"已保存: {ohe_path}")
    print(f"已保存: {hyps_path}")


if __name__ == "__main__":
    main()
