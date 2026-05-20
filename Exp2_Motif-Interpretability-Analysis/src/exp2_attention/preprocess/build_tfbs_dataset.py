import os
import subprocess
import argparse
import pandas as pd
import numpy as np
import glob
from multiprocessing import Pool
import dask.dataframe as dd
import warnings

from exp2_attention.paths import REFERENCE_DIR, TF_LIST

warnings.filterwarnings("ignore")

# 设置配置和路径
def setup_config(tf_name, peak_file, output_base_dir, reference_dir=REFERENCE_DIR, total_samples=1000000, num_processes=8, random_seed=42):
    config = {}
    
    # Base directories and paths
    reference_dir = os.fspath(reference_dir)
    
    # Settings
    config['TF'] = tf_name
    config['RAW_BED'] = peak_file
    config['TF_DIR'] = os.path.join(output_base_dir, tf_name)
    
    # 设置参考文件路径
    config['BLACKLIST_FILE'] = os.path.join(reference_dir, "hg38-blacklist.v2.bed")
    config['HG38_FA'] = os.path.join(reference_dir, "hg38.fa")
    config['CHROM_SIZE'] = os.path.join(reference_dir, "hg38.chrom.sizes")
    config['FILTERED_CHROM_SIZE'] = os.path.join(config['TF_DIR'], "hg38.chrom.sizes.filtered")
    
    # 设置参数
    config['WINDOW_SIZE'] = 1001
    config['MAX_NUM'] = 9999999  # Effectively no limit
    config['TOTAL_SAMPLES'] = total_samples  # 添加采样数目
    config['NUM_PROCESSES'] = num_processes  # 添加核心数目
    config['RANDOM_SEED'] = random_seed  # 固定随机种子
    
    # 设置输出文件路径
    config['NO_BLACKLIST_BED'] = os.path.join(config['TF_DIR'], f"{tf_name}_no_blacklist.bed")
    config['CORE_REGIONS_FILE'] = os.path.join(config['TF_DIR'], f"{tf_name}_core_regions.bed")
    config['POSITIVE_SEQUENCE_FILE'] = os.path.join(config['TF_DIR'], f"{tf_name}_positive_sequences.csv")
    config['NEGATIVE_REGIONS'] = os.path.join(config['TF_DIR'], f"{tf_name}_negative_regions.bed")
    config['NEGATIVE_SEQUENCE_FILE'] = os.path.join(config['TF_DIR'], f"{tf_name}_negative_sequences.csv")
    config['GC_STATISTICS'] = os.path.join(config['TF_DIR'], f"{tf_name}_negative_gc.txt")
    config['BALANCED_DATASET_FILE'] = os.path.join(config['TF_DIR'], f"{tf_name}_balanced_dataset.csv")
    
    # 设置临时文件路径
    config['FINAL_NEGATIVE_REGIONS'] = os.path.join(config['TF_DIR'], f"{tf_name}_final_negative_regions.bed")
    config['RANDOM_REGIONS'] = os.path.join(config['TF_DIR'], f"{tf_name}_random_regions.bed")
    config['REGIONS_STATS'] = os.path.join(config['TF_DIR'], f"{tf_name}_regions_stats.txt")
    
    return config

# 过滤黑名单区域
def filter_blacklist(config):
    print(f"过滤黑名单区域...")
    cmd = f"bedtools intersect -a {config['RAW_BED']} -b {config['BLACKLIST_FILE']} -v > {config['NO_BLACKLIST_BED']}"
    os.system(cmd)
    return

# 处理正样本
def process_positive_samples(config):
    print(f"处理正样本...")
    # 读取峰值床文件
    peaks_df = pd.read_csv(config['NO_BLACKLIST_BED'], sep='\t', header=None,
                          names=['chrom', 'start', 'end', 'peak_id', 'score'])
    print(f"从 {config['NO_BLACKLIST_BED']} 中加载了 {len(peaks_df)} 个peak。")

    # 从过滤后的chrom.sizes文件加载允许的染色体
    with open(config['FILTERED_CHROM_SIZE'], 'r') as f:
        allowed_chroms = {line.split('\t')[0].strip() for line in f}
    
    original_count = len(peaks_df)
    peaks_df = peaks_df[peaks_df['chrom'].isin(allowed_chroms)].copy()
    print(f"过滤非主要染色体后，剩余 {len(peaks_df)} 个peak (从 {original_count} 个中)。")

    # 排序并获取顶部峰值
    peaks_df = (peaks_df
                .sort_values(by="score", ascending=False)
                .assign(center=lambda x: ((x['start'] + x['end'])//2)))

    # 在峰值中心创建窗口
    start_coords = peaks_df['center'] - config['WINDOW_SIZE']//2
    end_coords = peaks_df['center'] + (config['WINDOW_SIZE']//2 + 1)
    
    # 修正错误：过滤掉起始坐标为负数的区域
    valid_indices = start_coords >= 0
    
    filtered_df = pd.DataFrame({
        'chrom': peaks_df.loc[valid_indices, 'chrom'],
        'start': start_coords[valid_indices],
        'end': end_coords[valid_indices]
    })
    print(f"过滤掉窗口起始为负数的区域后，剩余 {len(filtered_df)} 个。")
    
    # 记录窗口化后的区域数量
    print(f"创建了 {len(filtered_df)} 个窗口区域，准备用bedtools提取序列。")

    # 保存到床文件
    filtered_df.to_csv(config['CORE_REGIONS_FILE'], sep="\t", header=False, index=False)

    # 用bedtools提取序列
    fasta_command = f'bedtools getfasta -fi {config["HG38_FA"]} -bed {config["CORE_REGIONS_FILE"]} -fo temp_positive.fa'
    os.system(fasta_command)

    # 读取fasta并提取序列及区域信息
    sequences = []
    extracted_regions = []
    try:
        with open('temp_positive.fa', 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2): # Header at 0, 2, 4... Seq at 1, 3, 5...
                header = lines[i].strip()
                if header.startswith('>'):
                    # bedtools getfasta的header格式为 >chr:start-end
                    extracted_regions.append(header[1:])
                    if (i + 1) < len(lines):
                        sequences.append(lines[i+1].strip().upper())
    except FileNotFoundError:
        print("警告: temp_positive.fa 未找到，bedtools可能未能提取任何序列。")

    print(f"bedtools成功提取了 {len(sequences)} 条序列。")
    
    # 比较原始区域和成功提取的区域，找出差异
    original_regions_set = set(filtered_df.apply(lambda row: f"{row['chrom']}:{row['start']}-{row['end']}", axis=1))
    extracted_regions_set = set(extracted_regions)
    
    dropped_regions = original_regions_set - extracted_regions_set
    print(f"有 {len(dropped_regions)} 个区域在提取序列时被丢弃。")
    if dropped_regions:
        print("被丢弃区域的一些例子:")
        for i, region in enumerate(list(dropped_regions)[:5]):
            print(f"  - {region}")
        print("丢弃原因可能是：1. 区域坐标为负数（太靠近染色体边缘）；2. 所在染色体不存在于参考基因组中。")

    os.system('rm temp_positive.fa')

    # 将序列和区域信息放入DataFrame
    seq_df = pd.DataFrame({'sequence': sequences})

    # 检查序列中是否含有'N'
    initial_count = len(seq_df)
    if initial_count > 0:
        seq_df['has_N'] = seq_df['sequence'].str.contains('N')
        
        num_with_N = seq_df['has_N'].sum()
        print(f"在成功提取的 {initial_count} 条序列中，有 {num_with_N} 条含有'N'碱基。")

        # 过滤掉含有'N'的序列
        if num_with_N > 0:
            seq_df = seq_df[~seq_df['has_N']].copy()
            print(f"过滤掉含有'N'的序列后，剩余 {len(seq_df)} 条。")
        
        # 删除辅助列
        seq_df.drop(columns=['has_N'], inplace=True)
    
    seq_df["label"] = 1

    # 如果有序列，则计算GC含量
    if not seq_df.empty:
        seq_df['gc_content'] = seq_df['sequence'].apply(lambda x: (x.count('G') + x.count('C')) / len(x))
        seq_df.sort_values(by='gc_content', ascending=False, inplace=True)
    else:
        # 确保空dataframe也有gc_content列
        seq_df['gc_content'] = pd.Series(dtype='float64')

    seq_df.to_csv(config['POSITIVE_SEQUENCE_FILE'], index=False)

    print(f"最终正样本数量: {len(seq_df)}")
    if not seq_df.empty:
        print(f"正样本平均GC含量: {seq_df['gc_content'].mean():.3f}")
        print(f"正样本GC含量标准差: {seq_df['gc_content'].std():.3f}")
    
    return seq_df

# 生成随机区域
def generate_random_regions(config):
    print(f"生成{config['TOTAL_SAMPLES']}个随机区域并过滤...")
    random_command = (
        f'bedtools random -l {config["WINDOW_SIZE"]} -n {config["TOTAL_SAMPLES"]} -g {config["FILTERED_CHROM_SIZE"]} -seed {config["RANDOM_SEED"]} | '
        f'bedtools subtract -a stdin -b {config["RAW_BED"]} | '
        f'bedtools subtract -a stdin -b {config["BLACKLIST_FILE"]} > {config["RANDOM_REGIONS"]}'
    )
    os.system(random_command)
    return

# 处理单个区块的函数
def process_chunk(args):
    chunk_id, temp_dir, hg38_fa = args
    chunk_file = f"{temp_dir}/chunk_{chunk_id}"
    if not os.path.exists(chunk_file):
        return None
        
    chunk_stats = f"{temp_dir}/chunk_{chunk_id}.stats"
    
    # 运行bedtools nuc
    print(f"进程{chunk_id}开始处理...")
    nuc_cmd = f'bedtools nuc -fi {hg38_fa} -bed {chunk_file} > {chunk_stats}'
    subprocess.run(nuc_cmd, shell=True)
    print(f"进程{chunk_id}处理完成")
    
    return chunk_stats

# 并行计算核苷酸统计信息
def parallel_nuc_calculation(config):
    print("并行计算核苷酸统计信息...")
    # 获取文件行数
    num_lines = int(subprocess.check_output(f"wc -l {config['RANDOM_REGIONS']}", shell=True).decode().split()[0])
    total_regions = num_lines
    
    num_processes = config['NUM_PROCESSES']
    chunk_size = total_regions // num_processes + 1
    
    # 创建临时目录
    temp_dir = os.path.join(config['TF_DIR'], "temp_regions")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 分割文件
    print(f"将{config['RANDOM_REGIONS']}分割为{num_processes}个块...")
    split_command = f'split -l {chunk_size} -a 1 --numeric-suffixes=1 {config["RANDOM_REGIONS"]} {temp_dir}/chunk_'
    os.system(split_command)
    
    # 准备参数列表，每个进程处理一个块
    args_list = [(i, temp_dir, config['HG38_FA']) for i in range(1, num_processes+1)]
    
    # 并行处理所有块
    print(f"启动{num_processes}个进程并行处理...")
    with Pool(processes=num_processes) as pool:
        stat_files = pool.map(process_chunk, args_list)
    
    # 合并结果
    stat_files = [f for f in stat_files if f is not None]
    if stat_files:
        # 写入头部
        os.system(f'head -n 1 {stat_files[0]} > {config["REGIONS_STATS"]}')
        
        # 写入数据
        for stat_file in stat_files:
            os.system(f'tail -n +2 {stat_file} >> {config["REGIONS_STATS"]}')
    
    # 清理临时文件
    os.system(f'rm -rf {temp_dir}')
    print("并行处理完成")
    return

# 过滤区域
def filter_regions(config):
    print("过滤区域...")
    try:
        # 使用Dask分块读取大文件
        df = dd.read_csv(config['REGIONS_STATS'], sep='\t', assume_missing=True, blocksize="100MB")
        
        # 基本质控过滤：去除含N和长度不符合要求的区域
        filtered_df = df[(df['13_num_N'] == 0) & (df['15_seq_len'] == config['WINDOW_SIZE'])]
        
        # 一次性计算并加载到内存
        all_regions = filtered_df[['#1_usercol', '2_usercol', '3_usercol', '4_usercol']].compute()
        all_regions.columns = ['chrom', 'start', 'end', 'id']
        
        # 删除重复区域
        all_regions.drop_duplicates(subset=['chrom', 'start', 'end'], inplace=True)
        print(f"过滤后的唯一区域总数: {len(all_regions)}")
        
    except Exception as e:
        print(f"处理区域时出错: {e}")
        all_regions = pd.DataFrame(columns=['chrom', 'start', 'end', 'id'])

    # 检查是否有区域
    if len(all_regions) == 0:
        print("警告: 过滤后没有可用区域")
        os.system(f'rm {config["RANDOM_REGIONS"]} {config["REGIONS_STATS"]}')
        return None

    # 确保所有数值列为整数
    all_regions['start'] = all_regions['start'].astype(int)
    all_regions['end'] = all_regions['end'].astype(int)
    all_regions['id'] = all_regions['id'].astype(int)

    # 保存过滤后的区域
    all_regions.to_csv(config['FINAL_NEGATIVE_REGIONS'], sep='\t', header=False, index=False)
    print(f"保存过滤后的负区域: {len(all_regions)}")

    # 清理临时文件
    os.system(f'rm {config["RANDOM_REGIONS"]} {config["REGIONS_STATS"]}')
    
    return all_regions

# 提取序列
def extract_sequences(config):
    print("提取DNA序列...")
    fasta_command = f'bedtools getfasta -fi {config["HG38_FA"]} -bed {config["FINAL_NEGATIVE_REGIONS"]} -fo stdout'
    neg_sequences = []
    try:
        process = subprocess.Popen(fasta_command, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
        current_seq = ""
        for line in process.stdout:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    neg_sequences.append(current_seq.upper())
                    current_seq = ""
            else:
                current_seq += line
        if current_seq:  # 添加最后一个序列
            neg_sequences.append(current_seq.upper())
    except Exception as e:
        print(f"提取序列时出错: {e}")
        neg_sequences = []

    # 清理
    if os.path.exists(config['FINAL_NEGATIVE_REGIONS']):
        os.system(f"rm {config['FINAL_NEGATIVE_REGIONS']}")
        
    return neg_sequences

# 计算GC含量
def calculate_gc_content(neg_sequences, batch_size=10000):
    print("计算GC含量...")
    if not neg_sequences:
        return None
        
    # 创建空的结果DataFrame
    neg_seq_df = pd.DataFrame(columns=['sequence', 'label', 'gc_content'])
    
    # 分批处理序列
    for i in range(0, len(neg_sequences), batch_size):
        batch = neg_sequences[i:i+batch_size]
        batch_df = pd.DataFrame({
            'sequence': batch,
            'label': 0
        })
        
        # 计算GC含量
        batch_df['gc_content'] = batch_df['sequence'].apply(
            lambda x: (x.count('G') + x.count('C')) / len(x) if len(x) > 0 else 0
        )
        
        # 合并结果
        neg_seq_df = pd.concat([neg_seq_df, batch_df])
        
        # 清除批处理变量以释放内存
        del batch, batch_df
    
    print(f"负样本平均GC含量: {neg_seq_df['gc_content'].mean():.3f}")
    print(f"可用负样本总数: {len(neg_seq_df)}")
    
    return neg_seq_df

# 构建平衡数据集
def build_balanced_dataset(positive_df, neg_seq_df, config):
    print("构建平衡数据集...")
    num_positives = len(positive_df)

    print(f"正样本数量: {num_positives}")
    print(f"可用负样本数量: {len(neg_seq_df)}")

    # 为GC含量创建bins
    bins = 10
    pos_hist, pos_bin_edges = np.histogram(positive_df['gc_content'], bins=bins)
    bin_ranges = list(zip(pos_bin_edges[:-1], pos_bin_edges[1:]))

    # 创建空DataFrame存储匹配的负样本
    matched_neg_df = pd.DataFrame(columns=neg_seq_df.columns)
    
    # 用于统计的变量
    total_needed = 0
    total_available = 0
    
    # 设置随机种子以确保可重复性
    np.random.seed(config['RANDOM_SEED'])

    # 对于每个GC含量bin，采样相同数量的负序列
    for i, (bin_start, bin_end) in enumerate(bin_ranges):
        pos_count = pos_hist[i]
        
        # 跳过空bins
        if pos_count == 0:
            continue
        
        # 找到这个GC范围内的负序列
        bin_neg_df = neg_seq_df[(neg_seq_df['gc_content'] >= bin_start) & 
                                (neg_seq_df['gc_content'] < bin_end)]
        
        # 记录总需求和可用数量（用于最终统计）
        total_needed += pos_count
        total_available += len(bin_neg_df)
        
        # 基于可用性决定采样，不用有放回采样
        if len(bin_neg_df) >= pos_count:
            # 使用固定随机种子确保可重复性
            sampled_neg = bin_neg_df.sample(n=pos_count, random_state=config['RANDOM_SEED'])
        else:
            print(f"警告: bin {i} ({bin_start:.2f}-{bin_end:.2f})中负样本不足。"
                  f"可用: {len(bin_neg_df)}, 需要: {pos_count}")
            # 不使用有放回采样，只使用全部可用样本
            sampled_neg = bin_neg_df.copy()
        
        matched_neg_df = pd.concat([matched_neg_df, sampled_neg])

    # 验证数量
    print(f"匹配的负序列数量: {len(matched_neg_df)}")
    print(f"需要的负样本总数: {total_needed}, 实际可用的负样本总数: {total_available}")

    # 如果总体样本不足，打印汇总警告
    if total_available < total_needed:
        print(f"警告: 整体负样本数量不足。总共需要: {total_needed}, 实际可用: {total_available}, 缺口: {total_needed - total_available}")

    print(f"正样本平均GC含量: {positive_df['gc_content'].mean():.3f}")
    print(f"匹配后负样本平均GC含量: {matched_neg_df['gc_content'].mean():.3f}")

    # 创建组合数据集
    balanced_df = pd.concat([positive_df, matched_neg_df])
    balanced_df = balanced_df.drop('gc_content', axis=1)
    
    # 随机打乱数据集，但使用固定随机种子确保可重复性
    balanced_df = balanced_df.sample(frac=1, random_state=config['RANDOM_SEED']).reset_index(drop=True)
    
    balanced_df.to_csv(config['BALANCED_DATASET_FILE'], index=False)

    print(f"创建数据集，包含 {len(balanced_df)} 个序列")
    print(f"正样本: {len(positive_df)}, 负样本: {len(matched_neg_df)}")
    
    return balanced_df

# 新增函数：创建过滤后的chrom.sizes文件
def create_filtered_chrom_sizes(config):
    """
    Filters the chromosome sizes file to keep only main chromosomes.
    """
    filtered_chrom_path = config['FILTERED_CHROM_SIZE']
    if os.path.exists(filtered_chrom_path):
        print("Filtered chromosome sizes file already exists.")
        return

    print("Creating filtered chromosome sizes file...")
    main_chroms = {f'chr{i}' for i in range(1, 23)} | {'chrX', 'chrY', 'chrM'}
    
    with open(config['CHROM_SIZE'], 'r') as infile, open(filtered_chrom_path, 'w') as outfile:
        for line in infile:
            chrom = line.split('\t')[0]
            if chrom in main_chroms:
                outfile.write(line)
    print(f"Filtered chromosome sizes file created at: {filtered_chrom_path}")

# 主函数
def main(tf_name, peak_file, output_base_dir, total_samples=1500000, num_processes=8):
    print(f"开始处理转录因子: {tf_name}")
    
    # 设置配置和常量
    config = setup_config(tf_name, peak_file, output_base_dir, reference_dir=reference_dir, total_samples=total_samples, num_processes=num_processes, random_seed=random_seed)
    
    # 确保输出目录存在
    os.makedirs(config['TF_DIR'], exist_ok=True)
    
    # 创建一个只包含主要染色体的chrom.sizes文件
    create_filtered_chrom_sizes(config)
    
    # 1. 过滤黑名单区域
    filter_blacklist(config)
    
    # 2. 处理正样本
    positive_df = process_positive_samples(config)
    
    # 3. 生成随机区域作为负样本
    generate_random_regions(config)
    
    # 4. 并行计算核苷酸统计信息
    parallel_nuc_calculation(config)
    
    # 5. 过滤区域
    all_regions = filter_regions(config)
    if all_regions is None:
        print("警告: 过滤后没有可用区域，处理终止")
        return
    
    # 6. 提取负样本序列
    neg_sequences = extract_sequences(config)
    if not neg_sequences:
        print("警告: 未能提取负样本序列，处理终止")
        return
    
    # 7. 计算GC含量
    neg_seq_df = calculate_gc_content(neg_sequences)
    if neg_seq_df is None:
        print("警告: 计算GC含量失败，处理终止")
        return
    
    # 8. 构建平衡数据集
    balanced_df = build_balanced_dataset(positive_df, neg_seq_df, config)
    
    print(f"转录因子 {tf_name} 处理完成!")
    return balanced_df

def cli():
    parser = argparse.ArgumentParser(description="Build TFBS binary-classification datasets from ChIP-seq peaks.")
    parser.add_argument("--peaks-dir", required=True, help="Directory containing <TF>_peaks.bed files.")
    parser.add_argument("--output-dir", required=True, help="Directory for processed TF datasets.")
    parser.add_argument("--reference-dir", default=os.fspath(REFERENCE_DIR), help="Directory with hg38.fa, hg38.chrom.sizes, and hg38 blacklist BED.")
    parser.add_argument("--tf-list", nargs="+", default=TF_LIST, help="Transcription factors to process.")
    parser.add_argument("--total-samples", type=int, default=1500000, help="Number of random negative candidate regions.")
    parser.add_argument("--num-processes", type=int, default=8, help="Parallel workers for bedtools nuc.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for negative region sampling.")
    args = parser.parse_args()

    for tf_name in args.tf_list:
        peak_file = os.path.join(args.peaks_dir, f"{tf_name}_peaks.bed")
        if os.path.exists(peak_file):
            print(f"开始处理转录因子: {tf_name}")
            main(tf_name, peak_file, args.output_dir, reference_dir=args.reference_dir,
                 total_samples=args.total_samples, num_processes=args.num_processes, random_seed=args.seed)
        else:
            print(f"警告: 转录因子 {tf_name} 的峰值文件不存在: {peak_file}")


if __name__ == "__main__":
    cli()
