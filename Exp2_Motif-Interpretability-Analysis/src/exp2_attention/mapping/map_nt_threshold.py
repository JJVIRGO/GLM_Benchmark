import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
import pandas as pd
import argparse

from exp2_attention.paths import PROCESSED_DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--fasta_file", type=str, default=None)
parser.add_argument("--fimo_file", type=str, default=None)
parser.add_argument("--result_df_path", type=str, default=None)
parser.add_argument("--tf_name", type=str, default=None, help="转录因子名称")
parser.add_argument("--threshold_mode", action="store_true", help="使用50%阈值二值化模式")
parser.add_argument("--processed_data_dir", type=str, default=str(PROCESSED_DATA_DIR))
parser.add_argument("--tokenizer_path", type=str, default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
args = parser.parse_args()

# 如果提供了tf_name，则自动构造路径
if args.tf_name:
    tf_dir = os.path.join(args.processed_data_dir, args.tf_name)
    if args.fasta_file is None:
        args.fasta_file = os.path.join(tf_dir, f"{args.tf_name}_positive_sequences.fasta")
    if args.fimo_file is None:
        args.fimo_file = os.path.join(tf_dir, "fimo_results", "fimo_filtered.tsv")
    if args.result_df_path is None:
        suffix = "threshold" if args.threshold_mode else "ratio"
        args.result_df_path = os.path.join(tf_dir, "NT", f"motif_mapping_NT_{suffix}.csv")

fasta_file = args.fasta_file
fimo_file = args.fimo_file
result_df_path = args.result_df_path


tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

# 解析fasta文件，创建sequence_name到序列的映射
def parse_fasta(fasta_file):
    sequences = {}
    with open(fasta_file, 'r') as f:
        current_seq_name = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq_name is not None:
                    sequences[current_seq_name] = ''.join(current_seq)
                current_seq_name = line[1:]  # 移除'>'
                current_seq = []
            else:
                current_seq.append(line)
        if current_seq_name is not None:
            sequences[current_seq_name] = ''.join(current_seq)
    return sequences

fasta_sequences = parse_fasta(fasta_file)
fimo_data = pd.read_csv(fimo_file, sep='\t')

complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}

def convert_sequence(sequence):
    sequence = ''.join([complement.get(base, 'N') for base in sequence[::-1]])
    return sequence

# 创建结果DataFrame
results = []

# 处理fimo数据中的每一行
for idx, row in fimo_data.iterrows():
    sequence_name = row['sequence_name']
    strand = row['strand']
    start = row['start']
    end = row['stop']

    # 检查sequence_name是否存在于fasta文件中
    if sequence_name in fasta_sequences:
        sequence = fasta_sequences[sequence_name]

        if strand == "+":
            motif_length = end - start + 1
            results.append({
                'sequence_name': sequence_name,
                'sequence': sequence,
                'start': start,
                'end': end,
                'strand': strand,
                'motif_length': motif_length
            })
        elif strand == "-":
            # 对负链进行反向互补
            sequence = convert_sequence(sequence)
            # 调整坐标：负链的start和end需要反转
            total_length = len(sequence)
            temp = start
            start = total_length - end
            end = total_length - temp
            motif_length = end - start + 1
            results.append({
                'sequence_name': sequence_name,
                'sequence': sequence,
                'start': start,
                'end': end,
                'strand': strand,
                'motif_length': motif_length
            })
    else:
        print(f"警告：序列名 {sequence_name} 在fasta文件中不存在")


# 将结果保存到CSV
result_df = pd.DataFrame(results)

def create_motif_token_mapping(sequence, motif_start, motif_end, tokenizer):
    """
    创建motif到token的映射关系（基于50%阈值二值化）

    参数:
    sequence: 完整的DNA序列 (1000bp)
    motif_start: motif在序列中的起始位置 (0-based)
    motif_end: motif在序列中的结束位置 (0-based)
    tokenizer: 用于序列分词的tokenizer

    返回:
    mapping: 长度为171的列表，1表示该token与motif重叠比例>50%，0表示<=50%或无重叠
    """
    # 对序列进行编码
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        padding=False,
        truncation=False
    )

    # 获取token IDs和tokens
    token_ids = inputs['input_ids'][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # 创建一个171长度的映射列表，初始值为0
    mapping = [0] * 171

    # token映射关系：
    # - token[0] = <cls>
    # - token[1:167] (索引1-166) = 六聚体
    # - token[167:171] (索引167-170) = 单碱基

    # 处理六聚体token (索引1-166)
    for i in range(1, 167):
        # 确定当前六聚体的位置范围
        # 每个六聚体token对应原序列中的6个连续碱基
        hexamer_start = (i - 1) * 6
        hexamer_end = hexamer_start + 6 - 1  # -1是因为位置是0-based

        # 计算与motif的重叠碱基数
        overlap_start = max(hexamer_start, motif_start)
        overlap_end = min(hexamer_end, motif_end)
        overlap = max(0, overlap_end - overlap_start + 1)

        # 计算重叠比例并应用50%阈值
        token_length = 6
        overlap_ratio = overlap / token_length
        if overlap_ratio > 0.5:
            mapping[i] = 1  # 重叠比例大于50%
        else:
            mapping[i] = 0  # 重叠比例小于等于50%或无重叠

    # 处理单碱基token (索引167-170)
    for i in range(167, 171):
        # 计算单碱基在原序列中的位置
        # 单碱基token从原序列第996位开始(996-999)
        base_position = 996 + (i - 167)

        # 计算重叠比例并应用50%阈值（单碱基长度为1）
        token_length = 1
        if motif_start <= base_position <= motif_end:
            overlap_ratio = 1.0  # 完全重叠
        else:
            overlap_ratio = 0.0  # 无重叠

        if overlap_ratio > 0.5:
            mapping[i] = 1  # 重叠比例大于50%
        else:
            mapping[i] = 0  # 重叠比例小于等于50%或无重叠

    # <cls>保持为0

    return mapping, tokens

result_df["mapping"] = None
mappings = []

for i in range(len(result_df)):
    sequence = result_df["sequence"][i]
    motif_start = result_df["start"][i]
    motif_end = result_df["end"][i]
    mapping, tokens = create_motif_token_mapping(sequence, motif_start, motif_end, tokenizer)
    mappings.append(mapping)

result_df["mapping"] = mappings

# 确保输出目录存在
os.makedirs(os.path.dirname(result_df_path), exist_ok=True)
result_df.to_csv(result_df_path, index=False)

print(f"处理完成，结果已保存到: {result_df_path}")
print(f"总共处理了 {len(result_df)} 个motif")
