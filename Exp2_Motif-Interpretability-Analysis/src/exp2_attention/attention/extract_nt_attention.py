import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
# 设置中文字体，以防绘图时出现乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import seaborn as sns
import numpy as np
import pandas as pd
import ast  # 用于解析字符串表示的列表
import argparse
import os
import re
from safetensors.torch import load_file
import multiprocessing
import time
import shutil
from pathlib import Path

from exp2_attention.paths import DISCOVERY_ROOT, MODEL_ROOT, OUTPUT_ROOT, PROCESSED_DATA_DIR, model_path as resolve_model_path


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
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--input_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=128, help="批处理大小，用于加速attention提取")
args = parser.parse_args()

# 如果提供了tf_name，则自动构造路径
if args.tf_name:
    motif_dir = str(OUTPUT_ROOT / "finetune" / "NT" / f"motif_{args.tf_name}")

    # 自动选择最新的检查点
    if args.model_path is None:
        args.model_path = find_latest_checkpoint(motif_dir)

    # 自动设置输入路径 - 优先使用 predict_true 数据集；若不存在则回退到映射文件
    if args.input_path is None:
        predict_true_path = str(DISCOVERY_ROOT / "predict_true" / "NT" / f"{args.tf_name}_train_true.csv")
        if os.path.exists(predict_true_path):
            args.input_path = predict_true_path
        else:
            # 回退：优先使用threshold类型的映射文件，如果不存在则使用ratio类型
            threshold_path = str(PROCESSED_DATA_DIR / args.tf_name / "NT" / "motif_mapping_NT_threshold.csv")
            ratio_path = str(PROCESSED_DATA_DIR / args.tf_name / "NT" / "motif_mapping_NT_ratio.csv")

            if os.path.exists(threshold_path):
                args.input_path = threshold_path
            elif os.path.exists(ratio_path):
                args.input_path = ratio_path
            else:
                raise FileNotFoundError(
                    f"在 predict_true 与 {args.tf_name}/NT 目录下均找不到输入文件"
                )

    # 自动设置输出路径 - 保存到 predict_true 目录，命名为 *_attention_weight.parquet
    if args.output_path is None:
        args.output_path = str(
            DISCOVERY_ROOT
            / "predict_true"
            / "NT"
            / "attention"
            / f"{args.tf_name}_attention_weight.parquet"
        )

model_path = args.model_path
input_path = args.input_path
output_path = args.output_path

print(f"模型路径: {model_path}")
print(f"输入文件: {input_path}")
print(f"输出文件: {output_path}")

# 先加载原始模型，然后加载检查点权重
original_model_path = resolve_model_path("NT", MODEL_ROOT)
print("加载原始模型...")
model = AutoModelForSequenceClassification.from_pretrained(
    original_model_path,
    num_labels=2,  # 二分类任务
    output_attentions=True,  # 确保输出注意力分数
    trust_remote_code=True,
    local_files_only=True
).to('cuda')

# 加载检查点权重
print("加载检查点权重...")
checkpoint = load_file(os.path.join(model_path, 'model.safetensors'))
model.load_state_dict(checkpoint, strict=False)
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
    batch_size: 批处理大小

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
    batch_size: 批处理大小，默认32

    返回:
    all_attention_weights: 包含所有序列的注意力权重向量
    格式: [sequence_idx][(layer_idx, head_idx)] = attention_vector
    """
    # 提取所有序列和sequence_name
    sequences = sequences_df['sequence'].tolist()
    if 'sequence_name' in sequences_df.columns:
        sequence_names = sequences_df['sequence_name'].tolist()
    else:
        sequence_names = [f"seq_{i}" for i in range(len(sequences))]

    print(f"开始批量处理 {len(sequences)} 个序列，batch_size={batch_size}")

    # 使用batch处理提取所有注意力权重
    all_attention_weights = extract_all_attentions_batch(model, tokenizer, sequences, batch_size)

    # 打印进度
    print(f"已完成处理 {len(all_attention_weights)} 个序列")

    return all_attention_weights, sequence_names

# 主程序
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
result_df = pd.read_csv(input_path)

# 分析所有序列在所有层和头的CLS注意力权重向量
all_attention_weights, sequence_names = analyze_sequences_all_layers_heads(model, tokenizer, result_df, batch_size=args.batch_size)

# 获取模型结构信息
if all_attention_weights:
    sample_weights = all_attention_weights[0]
    num_layers = max(key[0] for key in sample_weights.keys()) + 1
    num_heads = max(key[1] for key in sample_weights.keys()) + 1
    print(f"模型层数: {num_layers}, 注意力头数: {num_heads}")
else:
    num_layers = 0
    num_heads = 0

print(f"模型类型: NT")
print(f"总共处理了 {len(result_df)} 个序列")


# --- 全新的并行写入逻辑 ---

def process_and_save_chunk(chunk_data):
    """
    工作函数：处理数据块并将其保存为单个Parquet文件。
    在独立的进程中运行。
    """
    chunk_idx, chunk_weights, chunk_names, output_dir = chunk_data
    
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


# 1. 在开始CPU密集型任务前，主动释放GPU资源
print("计算完成，释放模型和GPU缓存...")
del model
torch.cuda.empty_cache()

# 2. 使用多进程并行处理和写入
if all_attention_weights:
    start_time = time.time()
    
    # 将输出路径从单个文件转换为一个目录，用于存放分块的Parquet文件
    output_path_obj = Path(output_path)
    output_dir = output_path_obj.parent / output_path_obj.stem # e.g., ".../CTCF_attention_weight"
    
    # 如果目录已存在，先清理
    if output_dir.exists():
        print(f"清理已存在的输出目录: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    print(f"开始使用多进程并行写入Parquet文件至目录: {output_dir}")
    
    # 根据CPU核心数确定进程数，设置上限避免资源过度消耗
    num_processes = min(max(1, multiprocessing.cpu_count() // 2), 16)
    print(f"使用 {num_processes} 个并行进程...")

    # 将全部数据分割成小块，每个进程处理一块
    num_sequences = len(all_attention_weights)
    chunk_size = (num_sequences + num_processes - 1) // num_processes
    
    tasks = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_sequences)
        if start_idx >= end_idx:
            continue
        
        chunk_weights = all_attention_weights[start_idx:end_idx]
        chunk_names = sequence_names[start_idx:end_idx]
        tasks.append((i, chunk_weights, chunk_names, output_dir))
    
    # 启动进程池并执行任务
    total_records = 0
    with multiprocessing.Pool(processes=num_processes) as pool:
        print(f"开始处理 {len(tasks)} 个数据块...")
        # imap_unordered可以让我们在任务完成时立即获得结果，方便展示进度
        results = pool.imap_unordered(process_and_save_chunk, tasks)
        
        for i, num_records in enumerate(results):
            total_records += num_records
            print(f"  ... 已完成 {i + 1}/{len(tasks)} 个数据块的处理")

    end_time = time.time()
    
    print(f"结果已全部保存至目录: {output_dir}")
    print(f"总共保存了 {total_records} 条注意力向量记录")
    print(f"并行写入耗时: {end_time - start_time:.2f} 秒")
else:
    print("警告: 没有生成任何注意力权重数据")