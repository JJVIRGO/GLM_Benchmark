import os
import re
import shutil
import argparse
from typing import Tuple, List

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safetensors.torch import load_file
from tqdm.auto import tqdm

from exp2_attention.paths import REPO_ROOT, MODEL_ROOT, PROCESSED_DATA_DIR, OUTPUT_ROOT, DISCOVERY_ROOT

DEFAULT_PROJECT_ROOT = str(REPO_ROOT)
DEFAULT_MOTIF_ROOT = str(REPO_ROOT)
DEFAULT_GFM_ROOT = str(MODEL_ROOT)

# 允许 tokenizer 多线程并行，以提升分词速度
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
# 允许 TF32（Ampere 及以上 GPU），可提升 matmul 性能
try:
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass


def find_latest_checkpoint(base_path: str) -> str:
    """
    在给定的基础路径中找到最新的检查点目录（名称形如 checkpoint-XXXX）。

    返回该目录的完整路径。
    """
    checkpoint_dirs = [d for d in os.listdir(base_path) if d.startswith('checkpoint-')]
    if not checkpoint_dirs:
        raise ValueError(f"在 {base_path} 中没有找到检查点目录")

    checkpoint_nums: List[Tuple[int, str]] = []
    for dir_name in checkpoint_dirs:
        match = re.search(r'checkpoint-(\d+)', dir_name)
        if match:
            checkpoint_nums.append((int(match.group(1)), dir_name))

    if not checkpoint_nums:
        raise ValueError(f"在 {base_path} 中没有找到有效的检查点目录")

    latest_num, latest_dir = max(checkpoint_nums, key=lambda x: x[0])
    latest_path = os.path.join(base_path, latest_dir)
    print(f"选择最新的检查点: {latest_path} (step {latest_num})")
    return latest_path


def get_model_config(model_type: str):
    """根据模型类型获取原始模型路径、trust_remote_code 与 tokenizer 的最大长度。"""
    gfm_root = os.environ.get("DLM_GFM_ROOT", DEFAULT_GFM_ROOT)

    if model_type == "GROVER":
        original_model_path = f"{gfm_root}/GROVER"
        trust_remote_code = True
        model_max_length = 310
    elif model_type == "GENA_LM_BERT":
        original_model_path = f"{gfm_root}/GENA_LM_BERT"
        trust_remote_code = True
        model_max_length = 310
    elif model_type == "DNABERT-2":
        original_model_path = f"{gfm_root}/DNABERT-2-117M"
        trust_remote_code = True
        model_max_length = 310
    elif model_type == "NT":
        original_model_path = f"{gfm_root}/NT/NT_500M_model"
        trust_remote_code = True
        model_max_length = 310
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return original_model_path, trust_remote_code, model_max_length


def auto_resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    """基于 tf_name 与 model_type 自动推断 model_path/input_path/output_path。"""
    if args.tf_name:
        output_model_name = args.output_model_name or args.model_type
        model_type_short = args.model_type.replace("_LM_BERT", "").replace("-", "")
        motif_dir = os.path.join(str(OUTPUT_ROOT), "finetune", model_type_short, f"motif_{args.tf_name}")

        if args.model_path is None:
            args.model_path = find_latest_checkpoint(motif_dir)

        if args.input_path is None:
            train_path = os.path.join(str(PROCESSED_DATA_DIR), args.tf_name, "train.csv")
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"找不到训练集文件: {train_path}")
            args.input_path = train_path

        if args.output_path is None:
            args.output_path = os.path.join(str(DISCOVERY_ROOT), "predict_true", output_model_name, f"{args.tf_name}_train_true.csv")

    return args


def prepare_model_and_tokenizer(args: argparse.Namespace):
    """按模型类型加载 tokenizer 与模型（从 checkpoint），并移动到 GPU。"""
    gfm_root = args.gfm_root
    os.environ["DLM_GFM_ROOT"] = gfm_root
    original_model_path, trust_remote_code, model_max_length = get_model_config(args.model_type)

    # tokenizer
    if args.model_type == "NT":
        # NT 的 tokenizer 与模型分离，使用公开的 tokenizer
        print("加载NT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
            trust_remote_code=True,
            use_fast=True,
            model_max_length=model_max_length,
        )
    elif args.model_type != "DNABERT-2":
        print("加载tokenizer...")
        if args.model_type == "GROVER":
            tokenizer = AutoTokenizer.from_pretrained(
                original_model_path,
                trust_remote_code=True,
                use_fast=True,
                model_max_length=model_max_length,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                original_model_path,
                use_fast=True,
                model_max_length=model_max_length,
            )
    else:
        print("检查DNABERT-2 checkpoint源码文件...")
        original_model_path = f"{gfm_root}/DNABERT-2-117M"
        files_to_copy = [
            "bert_layers.py",
            "flash_attn_triton.py",
            "bert_padding.py",
            "configuration_bert.py",
        ]
        for file_name in files_to_copy:
            src_file = os.path.join(original_model_path, file_name)
            dst_file = os.path.join(args.model_path, file_name)
            if not os.path.exists(dst_file):
                raise FileNotFoundError(
                    f"DNABERT-2 checkpoint缺少 {file_name}: {dst_file}; "
                    f"可用源文件应位于 {src_file}"
                )

        print("从checkpoint路径加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=True,
            model_max_length=model_max_length,
        )

    # model
    print("加载模型和checkpoint...")
    if args.model_type == "DNABERT-2":
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=2,
            trust_remote_code=True,
            local_files_only=True,
        ).to('cuda')
    elif args.model_type == "NT":
        # 先加载原始NT模型，再加载checkpoint权重（safetensors）
        model = AutoModelForSequenceClassification.from_pretrained(
            original_model_path,
            num_labels=2,
            trust_remote_code=True,
            local_files_only=True,
        ).to('cuda')

        ckpt_file = os.path.join(args.model_path, 'model.safetensors')
        if os.path.exists(ckpt_file):
            checkpoint = load_file(ckpt_file)
            model.load_state_dict(checkpoint, strict=False)
        else:
            # 兜底：尝试直接从checkpoint目录加载
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    args.model_path,
                    num_labels=2,
                    trust_remote_code=True,
                    local_files_only=True,
                ).to('cuda')
            except Exception as e:
                raise FileNotFoundError(
                    f"未找到NT的 safetensors 权重文件，且无法直接从 {args.model_path} 加载: {e}"
                )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=2,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        ).to('cuda')

    model.eval()
    return model, tokenizer


def predict_labels(model, tokenizer, sequences: List[str], batch_size: int = 64, progress_desc: str = "predict"):
    """对给定序列批量预测，返回预测标签与正类概率。"""
    all_pred_labels = []
    all_pred_probs = []

    pbar = tqdm(total=len(sequences), desc=progress_desc, unit="seq", dynamic_ncols=True, miniters=1)
    for start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[start:start + batch_size]
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # 非阻塞传输到 GPU（若未 pin memory，则等效为阻塞）
        inputs = {k: v.to(model.device, non_blocking=True) for k, v in inputs.items()}

        # DNABERT-2 的 remote-code attention path 在当前环境更稳定；
        # 与 get_score_BPE.py 保持一致，避免 inference_mode 触发底层崩溃。
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_labels = torch.argmax(probs, dim=-1)

        all_pred_labels.append(pred_labels.detach().cpu().numpy())
        all_pred_probs.append(probs[:, 1].detach().cpu().numpy())
        pbar.update(len(batch_seqs))
    pbar.close()

    pred_labels = np.concatenate(all_pred_labels, axis=0) if all_pred_labels else np.array([])
    pred_probs = np.concatenate(all_pred_probs, axis=0) if all_pred_probs else np.array([])
    return pred_labels, pred_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_name", type=str, default=None, help="转录因子名称")
    parser.add_argument(
        "--model_type",
        type=str,
        default="GROVER",
        choices=["GROVER", "GENA_LM_BERT", "DNABERT-2", "NT"],
        help="模型类型",
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=None, help="训练集CSV路径，包含sequence,label两列")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--output_model_name", type=str, default=None,
                        help="输出目录使用的模型名；不影响模型加载逻辑")
    parser.add_argument("--project_root", type=str, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--motif_root", type=str, default=DEFAULT_MOTIF_ROOT)
    parser.add_argument("--gfm_root", type=str, default=DEFAULT_GFM_ROOT)
    parser.add_argument("--batch_size", type=int, default=512, help="批处理大小，默认512")
    args = parser.parse_args()

    # 自动补全路径
    args = auto_resolve_paths(args)

    print(f"模型类型: {args.model_type}")
    print(f"输出模型名: {args.output_model_name or args.model_type}")
    print(f"模型路径: {args.model_path}")
    print(f"输入文件: {args.input_path}")
    print(f"输出文件: {args.output_path}")

    # 读取数据
    df = pd.read_csv(args.input_path)
    required_cols = {"sequence", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError("输入CSV必须包含 'sequence' 与 'label' 两列")

    sequences = df["sequence"].tolist()

    # 准备模型与tokenizer
    model, tokenizer = prepare_model_and_tokenizer(args)

    # 预测
    desc = f"Predict {args.model_type}/{args.tf_name if args.tf_name else 'N/A'}"
    pred_labels, pred_probs = predict_labels(
        model,
        tokenizer,
        sequences,
        batch_size=args.batch_size,
        progress_desc=desc,
    )
    if len(pred_labels) != len(df):
        raise RuntimeError("预测数量与输入样本数量不一致")

    # 过滤预测为 True (1) 的样本
    true_mask = pred_labels == 1
    true_count = int(true_mask.sum())
    print(f"总样本数: {len(df)} | 预测为 True(1) 的样本数: {true_count}")

    df_true = df.loc[true_mask, ["sequence", "label"]].copy()

    # 保存
    out_dir = os.path.dirname(args.output_path)
    os.makedirs(out_dir, exist_ok=True)
    df_true.to_csv(args.output_path, index=False)
    print(f"已保存预测为 True 的样本至: {args.output_path}")


if __name__ == "__main__":
    main()


