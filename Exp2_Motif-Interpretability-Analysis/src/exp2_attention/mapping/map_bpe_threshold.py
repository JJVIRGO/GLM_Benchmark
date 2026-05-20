import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
import pandas as pd
import argparse

from exp2_attention.paths import MODEL_ROOT, model_path

parser = argparse.ArgumentParser()
parser.add_argument("--fasta_file", type=str, default=None)
parser.add_argument("--fimo_file", type=str, default=None)
parser.add_argument("--result_df_path", type=str, default=None)
parser.add_argument("--model_type", type=str, default="GROVER",
                   choices=["GROVER", "GENA_LM_BERT", "DNABERT-2"],
                   help="Model type for token mapping")
parser.add_argument("--threshold_mode", action="store_true",
                   help="使用50%阈值二值化模式")
parser.add_argument("--model_root", type=str, default=str(MODEL_ROOT))
parser.add_argument("--tokenizer_path", type=str, default=None)
args = parser.parse_args()

fasta_file = args.fasta_file
fimo_file = args.fimo_file
result_df_path = args.result_df_path
model_type = args.model_type

# 根据模型类型设置tokenizer路径
tokenizer_path = args.tokenizer_path or model_path(model_type, args.model_root)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

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

def create_motif_token_mapping_BPE(sequence, motif_start, motif_end, tokenizer):
    """
    BPE模型的motif到token映射（可变长度token，基于50%阈值二值化）
    """
    # 对序列进行编码，获取offset_mapping
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        padding=False,
        truncation=False,
        return_offsets_mapping=True
    )

    # 获取token IDs、tokens和offset mapping
    token_ids = inputs['input_ids'][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    offset_mapping = inputs['offset_mapping'][0].tolist()

    # 创建与token数量相同的映射列表，初始值为0
    mapping = [0] * len(tokens)

    # 处理每个token（跳过特殊token [CLS] 和 [SEP]）
    for i in range(1, len(tokens) - 1):  # 跳过[CLS]和[SEP]
        token_start, token_end = offset_mapping[i]

        # 计算与motif的重叠碱基数
        overlap_start = max(token_start, motif_start)
        overlap_end = min(token_end, motif_end)
        overlap = max(0, overlap_end - overlap_start)

        # 获取token长度
        token_length = token_end - token_start

        # 计算重叠比例并应用50%阈值
        if token_length > 0:
            overlap_ratio = overlap / token_length
            if overlap_ratio > 0.5:
                mapping[i] = 1  # 重叠比例大于50%
            else:
                mapping[i] = 0  # 重叠比例小于等于50%或无重叠

    # [CLS]和[SEP]保持为0
    return mapping, tokens

# 使用BPE映射函数
mapping_function = create_motif_token_mapping_BPE

result_df["mapping"] = None
mappings = []

for i in range(len(result_df)):
    sequence = result_df["sequence"][i]
    motif_start = result_df["start"][i]
    motif_end = result_df["end"][i]
    mapping, tokens = mapping_function(sequence, motif_start, motif_end, tokenizer)
    mappings.append(mapping)

result_df["mapping"] = mappings

# 根据模型类型和阈值模式修改输出文件名
base_path = os.path.splitext(result_df_path)[0]
if args.threshold_mode:
    result_df_path = f"{base_path}_{model_type}_threshold.csv"
else:
    result_df_path = f"{base_path}_{model_type}_ratio.csv"

# 确保输出目录存在
os.makedirs(os.path.dirname(result_df_path), exist_ok=True)
result_df.to_csv(result_df_path, index=False)

print(f"模型类型: {model_type}")
print(f"处理完成，结果已保存到: {result_df_path}")
print(f"总共处理了 {len(result_df)} 个motif")
print(f"Token数量: {len(tokens)}")
print(f"映射示例 (前10个): {mapping[:10]}")
