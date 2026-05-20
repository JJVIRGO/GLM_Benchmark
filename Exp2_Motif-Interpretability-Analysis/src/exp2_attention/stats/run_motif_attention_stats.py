import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import sys

import numpy as np
import pandas as pd
try:
    import polars as pl  # type: ignore
    _HAS_POLARS = True
except Exception:
    pl = None  # type: ignore
    _HAS_POLARS = False
import time

# 支持作为包内模块或脚本直接运行
try:
    from .stat_bpe_mapping import ModelSpec, get_model_spec, load_tokenizer, build_mappings, compute_offsets
    from .stat_bpe_score import (
        build_attention_index,
        compute_statistic,
        pack_sample_scores,
        parse_attention_vector,
    )
    from .stat_bpe_test import TestConfig, fdr_bh, paired_test
    from .stat_bpe_optimized import load_mapping_data, build_token_mappings_optimized, compute_statistic_optimized, compute_statistic_optimized_batch
except Exception:
    import sys as _sys
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if _CURRENT_DIR not in _sys.path:
        _sys.path.append(_CURRENT_DIR)
    from stat_bpe_mapping import ModelSpec, get_model_spec, load_tokenizer, build_mappings, compute_offsets
    from stat_bpe_score import (
        build_attention_index,
        compute_statistic,
        pack_sample_scores,
        parse_attention_vector,
    )
    from stat_bpe_test import TestConfig, fdr_bh, paired_test
    from stat_bpe_optimized import load_mapping_data, build_token_mappings_optimized, compute_statistic_optimized, compute_statistic_optimized_batch

# 进度条（若可用）
try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore
    _HAS_TQDM = False


from exp2_attention.paths import REPO_ROOT, STAT_ROOT

DEFAULT_MOTIF_ROOT = str(REPO_ROOT)


def resolve_paths(tf_name: str, model_type: str, output_model_name: str, motif_root: str,
                  attention_path: str = None, mapping_csv_arg: str = None, out_dir_arg: str = None):
    data_dir = f"{motif_root}/Data/processed_data/{tf_name}/{output_model_name}"
    attn_csv = os.path.join(data_dir, f"attention_scores_{output_model_name}_original.csv")
    attn_parquet_dir = os.path.join(data_dir, f"attention_scores_{output_model_name}_original")
    attn_default = attention_path or (attn_csv if os.path.exists(attn_csv) else attn_parquet_dir)
    # 映射文件优先 threshold，否则 ratio
    map_threshold = os.path.join(data_dir, f"motif_mapping_{output_model_name}_threshold.csv")
    map_ratio = os.path.join(data_dir, f"motif_mapping_{output_model_name}_ratio.csv")
    mapping_csv = mapping_csv_arg or (map_threshold if os.path.exists(map_threshold) else map_ratio)
    if not os.path.exists(attn_default):
        raise FileNotFoundError(f"未找到注意力文件: {attn_default}")
    if not os.path.exists(mapping_csv):
        raise FileNotFoundError(f"未找到映射文件: {map_threshold} 或 {map_ratio}")
    out_dir = out_dir_arg or os.path.join(str(STAT_ROOT), tf_name, output_model_name)
    os.makedirs(out_dir, exist_ok=True)
    return attn_default, mapping_csv, out_dir


def build_attention_index_from_path(attn_path: str, needed_sequences: set) -> Dict[Tuple[str, int, int], np.ndarray]:
    """Build attention index from either legacy CSV or Parquet part directory."""
    attn_index: Dict[Tuple[str, int, int], np.ndarray] = {}
    cols = ["sequence_name", "layer", "head", "attention_vector"]
    if os.path.isdir(attn_path):
        part_files = sorted(glob.glob(os.path.join(attn_path, "*.parquet")))
        if not part_files:
            raise FileNotFoundError(f"注意力目录中没有 parquet part 文件: {attn_path}")
        for i, part_file in enumerate(part_files, start=1):
            chunk = pd.read_parquet(part_file, columns=cols)
            if "sequence_name" in chunk.columns:
                chunk = chunk[chunk["sequence_name"].astype(str).isin(needed_sequences)]
            attn_index.update(build_attention_index(chunk))
            print(f"  Parquet part {i}/{len(part_files)}: 行数={len(chunk)} 累计键={len(attn_index)}")
        return attn_index

    if attn_path.endswith(".parquet"):
        attn_df = pd.read_parquet(attn_path, columns=cols)
        attn_df = attn_df[attn_df["sequence_name"].astype(str).isin(needed_sequences)]
        return build_attention_index(attn_df)

    if _HAS_POLARS:
        try:
            scan = pl.scan_csv(attn_path)
            scan = scan.select([c for c in cols if c in scan.columns])
            if "sequence_name" in scan.columns:
                scan = scan.filter(pl.col("sequence_name").cast(pl.Utf8).is_in(list(needed_sequences)))
            df_pl = scan.collect(streaming=True)
            print(f"Polars 收集完成，行数={df_pl.height}")
            return build_attention_index(df_pl.to_pandas())
        except Exception as e:
            print(f"Polars 读取失败，回退到 pandas 分块: {e}")

    _chunk_id = 0
    chunks = pd.read_csv(attn_path, chunksize=200000)
    for chunk in chunks:
        if "sequence_name" in chunk.columns:
            chunk = chunk[chunk["sequence_name"].astype(str).isin(needed_sequences)]
        attn_index.update(build_attention_index(chunk))
        _chunk_id += 1
        print(f"  CSV chunk#{_chunk_id} 行数={len(chunk)} 累计键={len(attn_index)}")
    return attn_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["GROVER", "GENA_LM_BERT", "DNABERT-2", "NT"])
    parser.add_argument("--output_model_name", type=str, default=None,
                        help="输入/输出文件使用的模型名；不影响模型逻辑")
    parser.add_argument("--motif_root", type=str, default=DEFAULT_MOTIF_ROOT)
    parser.add_argument("--attention_path", type=str, default=None)
    parser.add_argument("--mapping_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--num_random", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aggregate", type=str, default="mean", choices=["mean", "median"])
    parser.add_argument("--alternative", type=str, default="two-sided", choices=["two-sided", "greater", "less"])
    parser.add_argument("--mode", type=str, default="original", choices=["original", "optimized"],
                        help="运行模式：original使用原始方法，optimized使用token级优化方法")
    parser.add_argument("--progress", type=str, default="auto", choices=["auto", "bar", "log", "none"],
                        help="进度展示方式：auto优先用进度条，无则降级为日志")
    parser.add_argument("--log_every", type=int, default=500, help="log模式下每处理多少个motif打印一次")
    args = parser.parse_args()

    # 自动检测TTY，优化progress='auto'行为
    # 如果是auto模式，则根据是否为TTY以及tqdm是否安装，决定实际的进度展示方式
    effective_progress = args.progress
    if effective_progress == "auto":
        # in a tty and tqdm is available -> bar
        # not in a tty or no tqdm -> log
        if sys.stdout.isatty() and _HAS_TQDM:
            effective_progress = "bar"
        else:
            effective_progress = "log"

    t0 = time.time()
    print("开始计算BPE模型motif得分...")
    # 分阶段计时
    stage_times: Dict[str, float] = {
        "read_mapping_csv": 0.0,
        "build_attention_index": 0.0,
        "lh_map_build": 0.0,
        "tokenizer_load": 0.0,
        "offsets_compute": 0.0,
        "mapping_build": 0.0,
        "scoring": 0.0,
        "row_pack": 0.0,
        "per_sample_df_create": 0.0,
        "per_sample_df_write": 0.0,
        "motif_mean_compute": 0.0,
        "motif_mean_write": 0.0,
        "paired_tests_compute": 0.0,
        "paired_tests_write": 0.0,
    }
    output_model_name = args.output_model_name or args.model_type
    attn_path, mapping_csv, out_dir = resolve_paths(
        args.tf_name,
        args.model_type,
        output_model_name,
        args.motif_root,
        args.attention_path,
        args.mapping_csv,
        args.out_dir,
    )
    print(f"注意力数据路径: {attn_path}")
    print(f"映射数据路径: {mapping_csv}")
    print(f"输出路径: {out_dir}")
    # 读入 motif 映射的原始表（含 sequence、start、end、sequence_name）
    t_read_map = time.time()
    if args.mode == "optimized":
        # 优化模式：预加载并解析所有映射数据
        map_df = load_mapping_data(mapping_csv)
        print("优化模式：使用预解析的映射数据")
    else:
        # 原始模式：直接读取CSV
        map_df = pd.read_csv(mapping_csv)
    stage_times["read_mapping_csv"] = time.time() - t_read_map
    print(f"读入映射数据完毕")
    print(f"映射表: {len(map_df)} 行, 唯一序列: {map_df['sequence_name'].nunique() if 'sequence_name' in map_df.columns else 'N/A'}, 耗时 {stage_times['read_mapping_csv']:.2f}s")
    # 一些数据集字段名兼容
    if "sequence_name" not in map_df.columns and "sequence_id" in map_df.columns:
        map_df = map_df.rename(columns={"sequence_id": "sequence_name"})

    # 优先：根据映射表中涉及的序列集合，分块读取注意力数据并构建索引，降低内存
    needed_sequences = set(map_df["sequence_name"].astype(str).unique().tolist())
    attn_index: Dict[Tuple[str, int, int], np.ndarray] = {}
    try:
        print("开始读取注意力数据并构建索引...")
        t_attn_total = time.time()
        attn_index = build_attention_index_from_path(attn_path, needed_sequences)
        stage_times["build_attention_index"] = time.time() - t_attn_total
        print(f"读取注意力数据并构建索引完毕：{len(attn_index)} 键，总耗时 {stage_times['build_attention_index']:.2f}s")
    except Exception:
        t_attn = time.time()
        attn_df = pd.read_csv(attn_path)
        attn_df = attn_df[attn_df["sequence_name"].astype(str).isin(needed_sequences)]
        attn_index = build_attention_index(attn_df)
        stage_times["build_attention_index"] = time.time() - t_attn
        print(f"一次性读取注意力数据并构建索引完毕：{len(attn_index)} 键，耗时 {stage_times['build_attention_index']:.2f}s")
    # 模型与tokenizer
    if args.mode == "original":
        spec = get_model_spec(args.model_type)
        t_tok = time.time()
        tokenizer = load_tokenizer(spec)
        stage_times["tokenizer_load"] = time.time() - t_tok
        print(f"tokenizer: {tokenizer}")
    else:
        # 优化模式不需要tokenizer
        tokenizer = None
        print("优化模式：跳过tokenizer加载")

    # 从注意力数据推断每个 sequence 的 vector 长度（任选一个 layer/head）
    seq_to_vec_len: Dict[str, int] = {}
    for (seq, layer, head), vec in attn_index.items():
        seq_to_vec_len[seq] = len(vec)
    if not seq_to_vec_len:
        raise RuntimeError("注意力索引为空，无法推断向量长度")

    # 优化：预构建 sequence -> [(layer, head)] 映射
    t_lh_map = time.time()
    seq_to_lh_map = defaultdict(list)
    for seq, layer, head in attn_index.keys():
        seq_to_lh_map[seq].append((layer, head))
    stage_times["lh_map_build"] = time.time() - t_lh_map
    print(f"构建 sequence-to-(layer,head) 映射完毕, {len(seq_to_lh_map)}个序列, 耗时 {stage_times['lh_map_build']:.2f}s")

    # 输出逐样本评分数据
    per_sample_rows: List[Dict] = []

    # 进度设置
    total_motifs = int(len(map_df))
    use_bar = (effective_progress == "bar") and _HAS_TQDM
    use_log = (effective_progress == "log")
    processed = 0
    pbar = None
    if use_bar and tqdm is not None:
        pbar = tqdm(total=total_motifs, desc="Motifs", ncols=100)

    # 为加速：按 sequence_name 分组，避免重复计算 offsets
    grouped = map_df.groupby("sequence_name", sort=False)
    offsets_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    num_sequences = 0
    for seq_name, sub in grouped:
        if seq_name not in seq_to_vec_len:
            # 没有该序列的注意力向量，跳过
            continue

        if args.mode == "original":
            sequence = sub["sequence"].iloc[0]
        else:
            # 优化模式不需要sequence文本
            sequence = ""

        vector_len = seq_to_vec_len[seq_name]

        # 该 sequence 下有哪些 layer/head 可用
        lh_pairs = seq_to_lh_map.get(seq_name, [])
        if not lh_pairs:
            continue

        # 遍历该 sequence 的所有 motif 行
        num_sequences += 1
        sub_reset = sub.reset_index(drop=True)
        # 预先堆叠该 sequence 的全部 (layer, head) 注意力为矩阵，便于批量计算
        if args.mode != "original":
            # 优化模式：构造 (L, N) 矩阵
            attn_mat = np.vstack([attn_index[(seq_name, int(layer), int(head))] for (layer, head) in lh_pairs]).astype(np.float32)
        for motif_idx, row in sub_reset.iterrows():
            motif_start = int(row["start"])  # 闭区间起点
            motif_end_inclusive = int(row["end"])  # 闭区间终点
            motif_end = motif_end_inclusive  # 原项目 end 为包含端
            motif_end = motif_end_inclusive + 1  # 转为半开区间 [start, end)

            # 根据模式选择不同的映射构建方法
            if args.mode == "original":
                # 先计算并缓存映射（与 layer/head 无关），并复用 offsets
                if seq_name not in offsets_cache:
                    t_offs = time.time()
                    offsets_cache[seq_name] = (None, None)  # 占位
                    offs, _ = compute_offsets(sequence, tokenizer)
                    offsets_cache[seq_name] = (offs, None)
                    stage_times["offsets_compute"] += time.time() - t_offs
                offs = offsets_cache[seq_name][0]

                t_map = time.time()
                motif_map, random_maps = build_mappings(
                    sequence=sequence,
                    motif_start=motif_start,
                    motif_end=motif_end,
                    vector_length=vector_len,
                    tokenizer=tokenizer,
                    num_random=args.num_random,
                    seed=args.seed + motif_idx,
                    offsets=offs,
                )
                stage_times["mapping_build"] += time.time() - t_map
            else:
                # 优化模式：直接使用预解析的映射数据
                t_map = time.time()
                motif_map, _ = build_token_mappings_optimized(
                    mapping_df=map_df,
                    sequence_name=seq_name,
                    motif_idx=motif_idx,
                    vector_length=vector_len,
                    num_random=args.num_random,
                    seed=args.seed + motif_idx,
                    sub_df=sub_reset,
                )
                stage_times["mapping_build"] += time.time() - t_map

            if args.mode == "original":
                # 逐 (layer, head) 计算（保持原逻辑）
                random_maps_np = np.empty((0, vector_len))
                for (layer, head) in lh_pairs:
                    attn_vec = attn_index[(seq_name, layer, head)]
                    t_score = time.time()
                    motif_score = compute_statistic(attn_vec, motif_map)
                    # 原始模式仍可用随机映射（若需要），此处沿用旧实现
                    rand_scores = random_maps_np  # 空置以走 pack 的均值=NaN 分支
                    stage_times["scoring"] += time.time() - t_score
                    t_pack = time.time()
                    row_dict = pack_sample_scores(
                        sequence_name=seq_name,
                        motif_id=int(motif_idx),
                        layer=int(layer),
                        head=int(head),
                        motif_score=motif_score,
                        random_scores=rand_scores,
                        aggregate_method=args.aggregate,
                    )
                    per_sample_rows.append(row_dict)
                    stage_times["row_pack"] += time.time() - t_pack
            else:
                # 批量 (layer, head) 计算
                t_score = time.time()
                batch_res = compute_statistic_optimized_batch(
                    attn_mat=attn_mat,
                    motif_mapping=motif_map,
                    aggregate_method=args.aggregate,
                )
                stage_times["scoring"] += time.time() - t_score
                # 打包结果
                t_pack = time.time()
                ms = batch_res["motif_score"]
                rm = batch_res["random_mean"]
                rs = batch_res["random_std"]
                for i, (layer, head) in enumerate(lh_pairs):
                    row_dict = pack_sample_scores(
                        sequence_name=seq_name,
                        motif_id=int(motif_idx),
                        layer=int(layer),
                        head=int(head),
                        motif_score=float(ms[i]) if i < len(ms) else np.nan,
                        random_scores=np.empty((0,), dtype=float),
                        aggregate_method=args.aggregate,
                        random_mean=float(rm[i]) if i < len(rm) else np.nan,
                        random_std=float(rs[i]) if i < len(rs) else np.nan,
                    )
                    per_sample_rows.append(row_dict)
                stage_times["row_pack"] += time.time() - t_pack

            # 进度更新（按 motif 粒度）
            processed += 1
            if pbar is not None:
                pbar.update(1)
            elif use_log and args.log_every > 0 and (processed % args.log_every == 0):
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 1e-6)
                remain = (total_motifs - processed) / max(rate, 1e-6)
                total_profile = sum(stage_times.values())
                # 选出前3耗时阶段
                top = sorted(stage_times.items(), key=lambda x: x[1], reverse=True)[:3]
                top_str = ", ".join([f"{k}:{v:.1f}s" for k, v in top])
                print(
                    f"Processed {processed}/{total_motifs} motifs | {rate:.1f} motifs/s | ETA {remain/60:.1f} min | "
                    f"profile(sum={total_profile:.1f}s): {top_str}"
                )

    if pbar is not None:
        pbar.close()
    print(f"完成motif得分计算，总计 {processed}/{total_motifs}，耗时 {(time.time()-t0)/60:.1f} 分钟")

    t_df = time.time()
    per_sample_df = pd.DataFrame(per_sample_rows)
    stage_times["per_sample_df_create"] = time.time() - t_df
    mode_suffix = f"_{args.mode}" if args.mode != "original" else ""
    per_sample_path = os.path.join(out_dir, f"per_sample_scores_{output_model_name}_n{args.num_random}{mode_suffix}.parquet")
    t_write = time.time()
    per_sample_df.to_parquet(per_sample_path, index=False)
    stage_times["per_sample_df_write"] = time.time() - t_write

    # 新增：保存 motif 的真实统计值在每层每头的平均值（列: layer, head, score）
    t_motif_mean = time.time()
    motif_mean_df = (
        per_sample_df.groupby(["layer", "head"], sort=True)["motif_score"].mean().reset_index()
        .rename(columns={"motif_score": "score"})
    )
    motif_mean_path = os.path.join(out_dir, f"motif_scores_{output_model_name}{mode_suffix}.csv")
    stage_times["motif_mean_compute"] = time.time() - t_motif_mean
    t_mm_write = time.time()
    motif_mean_df.to_csv(motif_mean_path, index=False)
    stage_times["motif_mean_write"] = time.time() - t_mm_write

    # 统计检验：按 (layer, head) 聚合
    results = []
    cfg = TestConfig(alternative=args.alternative)
    lh_uniq = per_sample_df[["layer", "head"]].drop_duplicates()
    total_lh = int(len(lh_uniq))
    pbar2 = None
    if use_bar and tqdm is not None:
        pbar2 = tqdm(total=total_lh, desc="Paired tests", ncols=100)
    processed_lh = 0
    t_paired = time.time()
    for (layer, head), g in per_sample_df.groupby(["layer", "head"], sort=True):
        motif = g["motif_score"].astype(float).values
        rnd = g["random_mean"].astype(float).values
        res = paired_test(motif, rnd, cfg)
        res.update({"layer": int(layer), "head": int(head)})
        results.append(res)
        processed_lh += 1
        if pbar2 is not None:
            pbar2.update(1)
        elif use_log and args.log_every > 0 and (processed_lh % args.log_every == 0):
            print(f"Paired tests {processed_lh}/{total_lh}")
    test_df = pd.DataFrame(results)
    test_df = fdr_bh(test_df, pcol="pval", alpha=0.05)
    stage_times["paired_tests_compute"] = time.time() - t_paired
    if pbar2 is not None:
        pbar2.close()

    paired_path = os.path.join(out_dir, f"paired_tests_{output_model_name}_n{args.num_random}{mode_suffix}.csv")
    t_pt_write = time.time()
    test_df.to_csv(paired_path, index=False)
    stage_times["paired_tests_write"] = time.time() - t_pt_write

    print(f"逐样本评分已保存: {per_sample_path}")
    print(f"motif平均得分已保存: {motif_mean_path}")
    print(f"成对检验结果已保存: {paired_path}")

    # 阶段耗时汇总
    print("阶段耗时统计（秒）：")
    total_profile = sum(stage_times.values())
    for k, v in sorted(stage_times.items(), key=lambda x: x[1], reverse=True):
        pct = (v / max(total_profile, 1e-9)) * 100.0
        print(f"  {k:24s} {v:8.2f}s  ({pct:5.1f}%)")
    print(f"总运行时间: {(time.time()-t0):.2f}s, 计入profile的时间: {total_profile:.2f}s, 序列数: {num_sequences}, motif数: {processed}")


if __name__ == "__main__":
    main()


    """
    # 原始模式
    python run_stat_bpe.py --tf_name CTCF --model_type DNABERT-2 --num_random 1000 --seed 42 --aggregate mean --alternative greater --mode original

    # 优化模式（推荐）
    python run_stat_bpe.py --tf_name MYC --model_type NT --num_random 10 --seed 42 --aggregate mean --alternative greater --mode optimized
    python run_stat_bpe.py --tf_name CTCF --model_type GROVER --num_random 1000 --seed 42 --aggregate mean --alternative greater --mode optimized
    python run_stat_bpe.py --tf_name MYC --model_type GENA_LM_BERT --num_random 1000 --seed 42 --aggregate mean --alternative greater --mode optimized
    python run_stat_bpe.py --tf_name MYC --model_type DNABERT-2 --num_random 10 --seed 42 --aggregate mean --alternative greater --mode optimized
    """
