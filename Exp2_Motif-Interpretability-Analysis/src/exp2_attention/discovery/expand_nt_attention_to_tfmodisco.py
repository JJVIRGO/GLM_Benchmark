import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from exp2_attention.paths import DISCOVERY_ROOT


# ------------------------------
# 配置
# ------------------------------
@dataclass
class ExpandConfig:
    tf_name: str
    csv_path: str
    attn_dir: str
    out_dir: str
    layer: str = "last"  # "last" 或 指定整数层号（如 "28"）
    kmer_size: int = 6        # NT 为 6
    max_sequences: Optional[int] = None  # 可选：仅处理前N条
    dtype: str = "float32"
    eps: float = 1e-8
    center: str = "mean"  # none/mean/median/zscore，用于产生一定的负值以适配 TF-MoDISco
    fake_negative: bool = False  # 若为True且最终无负值，则追加一条虚构负样本
    fake_neg_base: str = "A"     # 虚构样本所用碱基
    fake_neg_scale: float = 0.01 # 虚构负样本幅度 = scale * max|hypscores|


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
        # 如需保留全长，可按需修改为padding
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


def compute_fc_vector(attn_vec: np.ndarray, is_nt: bool = True, eps: float = 1e-8) -> np.ndarray:
    """
    将一条注意力向量转换为逐token的 Fold-Change。
    约定：attn_vec[0] 为 CLS，对 NT 不存在 SEP。
    FC[t] = attn[t] / ((1 - attn[0]) / (n - 1))，t>=1
    返回形状为 T = n-1 的向量（不含CLS）。
    """
    n = int(attn_vec.shape[0])
    if n <= 1:
        return np.zeros((0,), dtype=np.float32)
    # NT: exclude_sep=False → 分母 (1 - CLS)/(n-1)
    denom = (1.0 - float(attn_vec[0])) / max(n - 1, 1)
    if abs(denom) < eps:
        denom = eps
    # 去掉 CLS 位
    return (attn_vec[1:].astype(np.float32) / denom).astype(np.float32)


def tokens_to_base_contrib_mean(fc_tokens: np.ndarray, L: int, kmer_size: int = 6) -> np.ndarray:
    """
    将逐token FC 向量（来自不重叠kmer）均分到其覆盖的碱基上。
    - fc_tokens: 形状 [T]，T 约等于 L / kmer_size
    - 返回: 形状 [L] 的碱基层面贡献
    """
    T = fc_tokens.shape[0]
    if T <= 0 or L <= 0 or kmer_size <= 0:
        return np.zeros((L,), dtype=np.float32)

    # 每个token的均分值
    vals_per_base = (fc_tokens / float(kmer_size)).astype(np.float32)

    # 将每个token的分数重复kmer_size次
    repeated_vals = np.repeat(vals_per_base, kmer_size)

    # 创建一个空的碱基分数数组
    base_scores = np.zeros((L,), dtype=np.float32)

    # 将重复后的分数填充到base_scores中，注意不要越界
    num_to_fill = min(len(repeated_vals), L)
    base_scores[:num_to_fill] = repeated_vals[:num_to_fill]

    return base_scores


def scatter_to_real_base_channel(ohe_1n4l: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
    """
    将每个位点的标量贡献写入对应真实碱基通道，其余通道置0。
    ohe_1n4l: [4, L]
    base_scores: [L]
    返回: [4, L]
    """
    _, L = ohe_1n4l.shape
    out = np.zeros_like(ohe_1n4l, dtype=np.float32)
    if L == 0:
        return out
    real_ch = np.argmax(ohe_1n4l, axis=0)  # [L]
    has_real = (ohe_1n4l.sum(axis=0) > 0)
    idx = np.where(has_real)[0]
    if idx.size > 0:
        out[real_ch[idx], idx] = base_scores[idx]
    return out


def resolve_default_paths(tf_name: str) -> Tuple[str, str, str]:
    base_dir = DISCOVERY_ROOT / "predict_true" / "NT"
    csv_path = base_dir / f"{tf_name}_train_true.csv"
    attn_dir = base_dir / "attention" / f"{tf_name}_attention_weight"
    out_dir = base_dir / "tfmodisco_inputs" / tf_name
    os.makedirs(out_dir, exist_ok=True)
    return str(csv_path), str(attn_dir), str(out_dir)


def detect_last_layer(attn_part_files: List[str]) -> int:
    last_layer = None
    for f in attn_part_files:
        try:
            df = pd.read_parquet(f, columns=["layer"])
            mx = int(df["layer"].max()) if not df.empty else None
            if mx is not None:
                last_layer = mx if last_layer is None else max(last_layer, mx)
        except Exception:
            continue
    if last_layer is None:
        raise RuntimeError("无法从注意力分块中检测到层号")
    return last_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_name", type=str, required=True)
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--attn_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--layer", type=str, default="last")
    parser.add_argument("--kmer_size", type=int, default=6)
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

    # 解析默认路径
    csv_path = args.csv_path
    attn_dir = args.attn_dir
    out_dir = args.out_dir
    if csv_path is None or attn_dir is None or out_dir is None:
        _csv, _attn, _out = resolve_default_paths(args.tf_name)
        csv_path = csv_path or _csv
        attn_dir = attn_dir or _attn
        out_dir = out_dir or _out
    os.makedirs(out_dir, exist_ok=True)

    cfg = ExpandConfig(
        tf_name=args.tf_name,
        csv_path=csv_path,
        attn_dir=attn_dir,
        out_dir=out_dir,
        layer=args.layer,
        kmer_size=args.kmer_size,
        max_sequences=args.max_sequences,
        center=args.center,
        fake_negative=args.fake_negative,
        fake_neg_base=args.fake_neg_base,
        fake_neg_scale=args.fake_neg_scale,
    )

    print(f"TF: {cfg.tf_name}")
    print(f"序列CSV: {cfg.csv_path}")
    print(f"注意力目录: {cfg.attn_dir}")
    print(f"输出目录: {cfg.out_dir}")
    print(f"kmer={cfg.kmer_size}")
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

    # 构建 OHE（统一长度，必要时裁剪为最短）
    ohe = build_ohe(sequences, dtype="float32")
    N, _, L = ohe.shape
    print(f"样本数 N={N}, 长度 L={L}")

    # 为每个序列预留逐token FC 的累加器（延迟创建以节省内存）
    # 记录每个序列的期望 token 数（L // k）
    seq_expected_tokens = [L // cfg.kmer_size for _ in range(N)]
    # 改为在层14-28与各头间逐token取最大值
    fc_token_max: List[Optional[np.ndarray]] = [None] * N

    # 列举注意力分块文件
    if not os.path.isdir(cfg.attn_dir):
        raise FileNotFoundError(f"注意力目录不存在: {cfg.attn_dir}")
    part_files = sorted(
        [os.path.join(cfg.attn_dir, f) for f in os.listdir(cfg.attn_dir) if f.endswith('.parquet')]
    )
    if len(part_files) == 0:
        raise FileNotFoundError(f"未在目录中找到 parquet: {cfg.attn_dir}")

    # 固定选用层范围: 14-28（对应第15-29层）
    layer_min, layer_max = 14, 28
    print(f"选用层范围: {layer_min}-{layer_max}，跨层跨头逐token取最大值")

    # 构建 seq_name → index 的映射（与 get_score.py 的命名保持一致）
    name_to_idx = {f"seq_{i}": i for i in range(N)}

    # 逐分块读取并累计最后一层各头的逐token FC
    total_rows = 0
    used_rows = 0
    print("Reading attention files and accumulating scores (max over layers 14-28 and heads)...")
    for pf in tqdm(part_files, desc="Reading attention files"):
        df = pd.read_parquet(pf)
        total_rows += len(df)
        # 仅保留层范围内的记录
        if "layer" in df.columns:
            df = df[df["layer"].astype(int).between(layer_min, layer_max)]
        else:
            # 若缺少层信息，无法参与本次聚合
            continue
        if df.empty:
            continue
        used_rows += len(df)

        # 只保留必要列，后续按列名访问
        needed_cols = [c for c in ["sequence_name", "attention_vector"] if c in df.columns]
        if len(needed_cols) < 2:
            continue
        df = df.loc[:, needed_cols]

        # 逐行：同一序列的对应token取最大值
        for row in df.itertuples(index=False):
            seq_name = getattr(row, "sequence_name", None)
            attn_list = getattr(row, "attention_vector", None)
            if seq_name is None or attn_list is None:
                continue
            idx = name_to_idx.get(str(seq_name))
            if idx is None or idx < 0 or idx >= N:
                continue
            attn_vec = np.asarray(attn_list, dtype=np.float32)
            fc_tokens = compute_fc_vector(attn_vec, is_nt=True, eps=cfg.eps)

            T_exp = int(seq_expected_tokens[idx])
            if T_exp <= 0 or fc_tokens.size == 0:
                continue
            T_eff = int(min(T_exp, int(fc_tokens.size)))
            if T_eff <= 0:
                continue
            if fc_token_max[idx] is None:
                # 用 -inf 初始化以便取最大值
                fc_token_max[idx] = np.full((T_exp,), -np.inf, dtype=np.float32)
            # 取最大
            current = fc_token_max[idx]
            np.maximum(current[:T_eff], fc_tokens[:T_eff], out=current[:T_eff])

    print(f"读取总行数: {total_rows}，层14-28行数: {used_rows}")

    # 计算头间平均 FC，并映射到碱基层面
    hypscores = np.zeros_like(ohe, dtype=np.float32)  # [N, 4, L]
    print("\nCalculating final importance scores per sequence...")
    for i in tqdm(range(N), desc="Generating hypscores"):
        T_exp = seq_expected_tokens[i]
        if T_exp <= 0:
            continue
        m = fc_token_max[i]
        if m is None:
            continue
        # 将未赋值位置（-inf）置为0
        fc_best = np.where(np.isfinite(m), m, 0.0).astype(np.float32)

        # token → base（聚合）
        base_scores = tokens_to_base_contrib_mean(
            fc_best, L=L, kmer_size=cfg.kmer_size
        )

        # 可选：对每条序列做去中心/标准化，使得部分位置为负，便于 TF-MoDISco 双侧建模
        if cfg.center != "none":
            x = base_scores.astype(np.float32, copy=True)
            if cfg.center == "mean":
                m = float(np.mean(x)) if x.size > 0 else 0.0
                x = x - m
            elif cfg.center == "median":
                med = float(np.median(x)) if x.size > 0 else 0.0
                x = x - med
            elif cfg.center == "zscore":
                m = float(np.mean(x)) if x.size > 0 else 0.0
                sd = float(np.std(x)) if x.size > 0 else 0.0
                if sd < cfg.eps:
                    sd = cfg.eps
                x = (x - m) / sd

            # 兜底：若仍未出现双侧（例如常数序列），叠加极小幅度的线性扰动以制造正负样本
            if not (np.any(x > 0) and np.any(x < 0)):
                # 以当前量级为尺度，加入 1e-3 幅度的 [-1,1] 线性坡度
                scale = float(np.max(np.abs(x))) if np.max(np.abs(x)) > 0 else 1.0
                ramp = np.linspace(-1.0, 1.0, num=L, dtype=np.float32)
                x = x + (1e-3 * scale) * ramp

            base_scores = x

        # 写入真实碱基通道
        hypscores[i] = scatter_to_real_base_channel(ohe[i], base_scores)

    # 保存为 TF-MoDISco 输入
    # 若需要且当前没有任何负值，则追加一条虚构负样本
    if cfg.fake_negative:
        has_neg = bool(np.any(hypscores < 0))
        if not has_neg:
            base_idx = BASE_TO_CH.get(cfg.fake_neg_base, 0)
            fake_ohe = np.zeros((4, L), dtype=np.float32)
            if L > 0:
                fake_ohe[base_idx, :] = 1.0
            # 计算幅度
            max_abs = float(np.max(np.abs(hypscores))) if hypscores.size > 0 else 0.0
            amp = cfg.fake_neg_scale * max_abs
            if amp <= 0:
                amp = 1e-3
            fake_base_scores = -np.ones((L,), dtype=np.float32) * float(amp)
            fake_hyps = scatter_to_real_base_channel(fake_ohe, fake_base_scores)

            # 追加到数组
            ohe = np.concatenate([ohe, fake_ohe[np.newaxis, ...]], axis=0)
            hypscores = np.concatenate([hypscores, fake_hyps[np.newaxis, ...]], axis=0)
            print("已追加一条虚构负样本以启用双侧阈值计算，不改动原序列FC")

    ohe_path = os.path.join(cfg.out_dir, "ohe1.npz")
    hyps_path = os.path.join(cfg.out_dir, "hypscores1.npz")
    np.savez_compressed(ohe_path, ohe)
    np.savez_compressed(hyps_path, hypscores)
    print(f"已保存: {ohe_path}")
    print(f"已保存: {hyps_path}")


if __name__ == "__main__":
    main()
