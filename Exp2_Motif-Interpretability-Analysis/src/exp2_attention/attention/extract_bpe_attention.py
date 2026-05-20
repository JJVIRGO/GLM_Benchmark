import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import ast  # 用于解析字符串表示的列表
import argparse
import os
import re
import shutil
import sys
from safetensors.torch import load_file
import multiprocessing
import time
from pathlib import Path

from exp2_attention.paths import DISCOVERY_ROOT, MODEL_ROOT, OUTPUT_ROOT, PROCESSED_DATA_DIR, REPO_ROOT

DEFAULT_PROJECT_ROOT = str(REPO_ROOT)
DEFAULT_MOTIF_ROOT = str(REPO_ROOT)
DEFAULT_GFM_ROOT = str(MODEL_ROOT)


def disable_dnabert2_triton_attention(model: torch.nn.Module):
    """Force DNABERT2 remote-code attention to use the PyTorch fallback."""
    patched_modules = set()
    for module in model.modules():
        if module.__class__.__name__ != "BertUnpadSelfAttention":
            continue
        module_obj = sys.modules.get(module.__class__.__module__)
        if module_obj is not None and hasattr(module_obj, "flash_attn_qkvpacked_func"):
            module_obj.flash_attn_qkvpacked_func = None
            patched_modules.add(module.__class__.__module__)
    if patched_modules:
        print("Disabled DNABERT2 Triton attention in: " + ", ".join(sorted(patched_modules)))


def find_latest_checkpoint(base_path):
    """
    在给定的基础路径中找到最新的检查点

    参数:
    base_path: motif_{TF} 目录的路径

    返回:
    最新的检查点完整路径
    """
    # 查找所有checkpoint开头的目录
    checkpoint_dirs = [d for d in os.listdir(base_path) if d.startswith('checkpoint-')]

    if not checkpoint_dirs:
        raise ValueError(f"在 {base_path} 中没有找到检查点目录")

    # 提取checkpoint编号并找到最大的
    checkpoint_nums = []
    for dir_name in checkpoint_dirs:
        match = re.search(r'checkpoint-(\d+)', dir_name)
        if match:
            checkpoint_nums.append((int(match.group(1)), dir_name))

    if not checkpoint_nums:
        raise ValueError(f"在 {base_path} 中没有找到有效的检查点目录")

    # 选择编号最大的检查点
    latest_num, latest_dir = max(checkpoint_nums, key=lambda x: x[0])
    latest_path = os.path.join(base_path, latest_dir)

    print(f"选择最新的检查点: {latest_path} (step {latest_num})")
    return latest_path

parser = argparse.ArgumentParser()
parser.add_argument("--tf_name", type=str, default=None, help="转录因子名称")
parser.add_argument("--model_type", type=str, default="GROVER",
                   choices=["GROVER", "GENA_LM_BERT", "DNABERT-2"],
                   help="BPE模型类型")
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--input_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--output_model_name", type=str, default=None,
                    help="输出目录/文件使用的模型名；不影响模型加载逻辑")
parser.add_argument("--project_root", type=str, default=DEFAULT_PROJECT_ROOT)
parser.add_argument("--motif_root", type=str, default=DEFAULT_MOTIF_ROOT)
parser.add_argument("--gfm_root", type=str, default=DEFAULT_GFM_ROOT)
parser.add_argument("--batch_size", type=int, default=128, help="批处理大小，统一为128")
parser.add_argument("--write_chunk_size", type=int, default=512,
                    help="每处理多少条序列写出一个Parquet part，避免全量attention常驻内存")
args = parser.parse_args()
output_model_name = args.output_model_name or args.model_type

# 根据模型类型设置路径和参数
def get_model_config(model_type):
    """根据模型类型获取配置信息"""
    gfm_root = args.gfm_root

    # 设置原始模型路径和训练时的max_length
    if model_type == "GROVER":
        original_model_path = f"{gfm_root}/GROVER"
        trust_remote_code = True
        model_max_length = 310  # 与训练时保持一致
    elif model_type == "GENA_LM_BERT":
        original_model_path = f"{gfm_root}/GENA_LM_BERT"
        trust_remote_code = True
        model_max_length = 310  # 与训练时保持一致
    elif model_type == "DNABERT-2":
        original_model_path = f"{gfm_root}/DNABERT-2-117M"
        trust_remote_code = True
        model_max_length = 512  # 与 train_BPE_copy.py / DNABERT2 checkpoint 保持一致

    return original_model_path, trust_remote_code, model_max_length

# 如果提供了tf_name，则自动构造路径
if args.tf_name:
    motif_root = args.motif_root

    # 根据模型类型设置checkpoint目录
    model_type_short = args.model_type.replace("_LM_BERT", "").replace("-", "")
    motif_dir = str(OUTPUT_ROOT / "finetune" / model_type_short / f"motif_{args.tf_name}")

    # 自动选择最新的检查点
    if args.model_path is None:
        args.model_path = find_latest_checkpoint(motif_dir)

    # 自动设置输入路径 - 优先使用 predict_true 数据集；若不存在则回退到映射文件
    if args.input_path is None:
        predict_true_path = str(
            DISCOVERY_ROOT / "predict_true" / output_model_name / f"{args.tf_name}_train_true.csv"
        )
        if os.path.exists(predict_true_path):
            args.input_path = predict_true_path
        else:
            # 回退：优先使用threshold类型的映射文件，如果不存在则使用ratio类型
            threshold_path = str(
                PROCESSED_DATA_DIR
                / args.tf_name
                / output_model_name
                / f"motif_mapping_{output_model_name}_threshold.csv"
            )
            ratio_path = str(
                PROCESSED_DATA_DIR
                / args.tf_name
                / output_model_name
                / f"motif_mapping_{output_model_name}_ratio.csv"
            )

            if os.path.exists(threshold_path):
                args.input_path = threshold_path
            elif os.path.exists(ratio_path):
                args.input_path = ratio_path
            else:
                raise FileNotFoundError(
                    f"在 predict_true 与 {args.tf_name}/{args.model_type} 目录下均找不到输入文件"
                )

    # 自动设置输出路径 - 保存到 predict_true 目录，命名为 *_attention_weight.parquet
    if args.output_path is None:
        args.output_path = str(
            DISCOVERY_ROOT
            / "predict_true"
            / output_model_name
            / "attention"
            / f"{args.tf_name}_attention_weight.parquet"
        )

model_path = args.model_path
input_path = args.input_path
output_path = args.output_path

print(f"模型类型: {args.model_type}")
print(f"输出模型名: {output_model_name}")
print(f"模型路径: {model_path}")
print(f"输入文件: {input_path}")
print(f"输出文件: {output_path}")

# 获取模型配置
original_model_path, trust_remote_code, model_max_length = get_model_config(args.model_type)

# 加载tokenizer (使用训练时的max_length)
# 注意：DNABERT2的tokenizer将在模型加载部分从checkpoint路径加载
if args.model_type != "DNABERT-2":
    print("加载tokenizer...")
    if args.model_type == "GROVER":
        tokenizer = AutoTokenizer.from_pretrained(
            original_model_path,
            trust_remote_code=True,
            model_max_length=model_max_length
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            original_model_path,
            model_max_length=model_max_length
        )

# DNABERT2需要特殊处理：checkpoint目录需包含remote-code源码文件
if args.model_type == "DNABERT-2":
    print("检查DNABERT-2 checkpoint源码文件...")

    # 获取原始模型路径
    original_model_path = f"{args.gfm_root}/DNABERT-2-117M"

    # 需要复制的文件列表（所有DNABERT-2模型需要的Python模块文件）
    files_to_copy = [
        "bert_layers.py",
        "flash_attn_triton.py",
        "bert_padding.py",
        "configuration_bert.py"
    ]

    # 新一轮复跑不改动原checkpoint；若checkpoint缺少源码文件，直接报错。
    for file_name in files_to_copy:
        src_file = os.path.join(original_model_path, file_name)
        dst_file = os.path.join(model_path, file_name)

        if not os.path.exists(dst_file):
            raise FileNotFoundError(
                f"DNABERT-2 checkpoint缺少 {file_name}: {dst_file}; "
                f"可用源文件应位于 {src_file}"
            )

    print("DNABERT-2源码文件检查完成，准备加载模型...")

    # 从checkpoint路径加载tokenizer
    print("从checkpoint路径加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )

    # 从checkpoint路径加载模型
    print("加载DNABERT-2模型和checkpoint...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,  # 从checkpoint路径加载
        num_labels=2,  # 二分类任务
        output_attentions=True,  # 确保输出注意力分数
        trust_remote_code=True,
        local_files_only=True,
    )
    if os.environ.get("DISABLE_DNABERT2_TRITON") == "1":
        disable_dnabert2_triton_attention(model)
    model = model.to('cuda')

else:
    # 直接从checkpoint路径加载模型（这样能正确处理扩展的位置嵌入）
    print("加载模型和checkpoint...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,  # 直接从checkpoint路径加载
        num_labels=2,  # 二分类任务
        output_attentions=True,  # 确保输出注意力分数
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    ).to('cuda')

print("模型加载完成")

def parse_mapping(mapping_str):
    """
    将字符串形式的mapping解析为数值列表
    """
    if isinstance(mapping_str, str):
        try:
            # 尝试解析字符串为列表
            mapping_list = ast.literal_eval(mapping_str)
            return np.array(mapping_list, dtype=float)
        except:
            # 如果解析失败，尝试按逗号分割
            try:
                # 直接按逗号分割并转换为float
                return np.array([float(x.strip()) for x in mapping_str.strip('[]').split(',')], dtype=float)
            except ValueError:
                # 处理numpy.float64格式的情况
                import re
                # 使用正则表达式提取数值部分
                cleaned_str = re.sub(r'np\.float64\(([^)]+)\)', r'\1', mapping_str.strip('[]'))
                try:
                    values = [float(x.strip()) for x in cleaned_str.split(',')]
                    return np.array(values, dtype=float)
                except ValueError:
                    # 如果还是失败，使用eval作为最后手段
                    try:
                        values = eval(mapping_str.strip('[]'))
                        return np.array(values, dtype=float)
                    except:
                        print(f"Error: Unable to parse mapping string: {mapping_str}")
                        return np.array([])
    elif isinstance(mapping_str, list):
        # 如果已经是列表，直接转换为numpy数组
        return np.array(mapping_str, dtype=float)
    else:
        # 其他情况，返回空数组
        print(f"Warning: Unknown mapping type {type(mapping_str)}")
        return np.array([])

def extract_all_attentions_batch(model, tokenizer, sequences, batch_size=32):
    """
    批量提取多个序列的所有层和头的CLS注意力权重向量

    参数:
    model: 预训练模型
    tokenizer: 分词器
    sequences: DNA序列列表
    batch_size: 批处理大小，默认64

    返回:
    all_attention_weights: 包含所有序列、所有层和头的CLS注意力权重向量列表
    [sequence_idx][(layer_idx, head_idx)] = attention_vector (numpy array)
    """
    all_attention_weights = []

    # 确定打印进度的频率 (每处理约1000条序列打印一次)
    print_every_n_batches = max(1, 1000 // batch_size)

    # 分批处理序列
    for i, batch_start in enumerate(range(0, len(sequences), batch_size)):
        # 进度显示
        if i > 0 and i % print_every_n_batches == 0:
            processed_count = batch_start
            print(f"  ... 已处理 {processed_count} / {len(sequences)} 条序列")

        batch_end = min(batch_start + batch_size, len(sequences))
        batch_sequences = sequences[batch_start:batch_end]

        # 对序列批进行编码
        inputs = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 获取模型输出的注意力分数
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # 处理batch中的每个序列
        for seq_idx_in_batch in range(len(batch_sequences)):
            attention_weights = {}

            # 遍历所有层
            for layer_idx, layer_attention in enumerate(outputs.attentions):
                # 遍历所有头
                for head_idx in range(layer_attention.shape[1]):
                    # 提取CLS token对所有token的注意力权重向量
                    # BPE模型的CLS token也在位置0
                    cls_attention = layer_attention[seq_idx_in_batch, head_idx, 0, :].cpu().numpy()

                    # 直接保存完整的注意力权重向量（包括对CLS自身和其他所有token）
                    attention_weights[(layer_idx, head_idx)] = cls_attention

            all_attention_weights.append(attention_weights)

    return all_attention_weights

def analyze_sequences_all_layers_heads(model, tokenizer, sequences_df, batch_size=32):
    """
    批量分析多个序列在所有层和头的CLS注意力权重向量

    参数:
    model: 预训练模型
    tokenizer: 分词器
    sequences_df: 包含序列信息的DataFrame
    batch_size: 批处理大小，默认64

    返回:
    all_attention_weights: 包含所有序列的注意力权重向量
    格式: [sequence_idx][(layer_idx, head_idx)] = attention_vector
    """
    # 提取所有序列和sequence_name
    sequences = sequences_df['sequence'].tolist()
    if 'sequence_name' in sequences_df.columns:
        sequence_names = sequences_df['sequence_name'].tolist()
    else:
        # 若无 sequence_name，则以顺序编号生成占位名
        sequence_names = [f"seq_{i}" for i in range(len(sequences))]

    print(f"开始批量处理 {len(sequences)} 个序列，batch_size={batch_size}")

    # 使用batch处理提取所有注意力权重
    all_attention_weights = extract_all_attentions_batch(model, tokenizer, sequences, batch_size)

    # 打印进度
    print(f"已完成处理 {len(all_attention_weights)} 个序列")

    return all_attention_weights, sequence_names

# 主程序
result_df = pd.read_csv(input_path)


def save_attention_part(chunk_idx, chunk_weights, chunk_names, output_dir):
    """将一个序列块的attention保存为单个Parquet文件。"""
    results_data = []
    # 遍历数据块中的每个序列
    for sequence_weights, sequence_name in zip(chunk_weights, chunk_names):
        # 遍历每个层和头
        for (layer_idx, head_idx), attention_vector in sequence_weights.items():
            results_data.append({
                'sequence_name': sequence_name,
                'layer': layer_idx,
                'head': head_idx,
                # 直接保存为列表，Parquet格式原生支持
                'attention_vector': attention_vector.tolist(),
                'vector_length': len(attention_vector)
            })

    if not results_data:
        return 0

    # 将处理好的数据转换为DataFrame并保存
    df = pd.DataFrame(results_data)
    chunk_output_file = output_dir / f'part_{chunk_idx:04d}.parquet'
    df.to_parquet(chunk_output_file, engine='pyarrow')
    
    # 返回处理的记录数，用于统计
    return len(results_data)


start_time = time.time()
output_path_obj = Path(output_path)
output_dir = output_path_obj.parent / output_path_obj.stem
if output_dir.exists():
    print(f"清理已存在的输出目录: {output_dir}")
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

total_sequences = len(result_df)
total_records = 0
part_idx = 0
num_layers = 0
num_heads = 0
write_chunk_size = max(1, int(args.write_chunk_size))
print(f"开始流式处理 {total_sequences} 个序列，输出目录: {output_dir}")

for start_idx in range(0, total_sequences, write_chunk_size):
    end_idx = min(start_idx + write_chunk_size, total_sequences)
    chunk_df = result_df.iloc[start_idx:end_idx].reset_index(drop=True)
    print(f"处理序列块 {start_idx}:{end_idx} / {total_sequences}")
    chunk_weights, chunk_names = analyze_sequences_all_layers_heads(
        model, tokenizer, chunk_df, batch_size=args.batch_size
    )
    if chunk_weights and num_layers == 0:
        sample_weights = chunk_weights[0]
        num_layers = max(key[0] for key in sample_weights.keys()) + 1
        num_heads = max(key[1] for key in sample_weights.keys()) + 1
        print(f"模型层数: {num_layers}, 注意力头数: {num_heads}")
    total_records += save_attention_part(part_idx, chunk_weights, chunk_names, output_dir)
    part_idx += 1
    del chunk_weights
    torch.cuda.empty_cache()

print(f"模型类型: {args.model_type}")
print(f"总共处理了 {total_sequences} 个序列")
print("计算完成，释放模型和GPU缓存...")
del model
torch.cuda.empty_cache()
end_time = time.time()
print(f"结果已全部保存至目录: {output_dir}")
print(f"总共保存了 {total_records} 条注意力向量记录")
print(f"流式写入耗时: {end_time - start_time:.2f} 秒")
