from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer

from exp2_attention.paths import MODEL_ROOT


@dataclass
class ModelSpec:
    """
    模型与tokenizer配置
    """
    model_type: str
    model_path: str
    model_max_length: int
    trust_remote_code: bool = True


def get_model_spec(model_type: str) -> ModelSpec:
    """
    按项目约定返回模型与tokenizer的配置
    """
    model_dirs = {
        "GROVER": "GROVER",
        "GENA_LM_BERT": "GENA_LM_BERT",
        "DNABERT-2": "DNABERT-2-117M",
        "DNABERT2_5.6": "DNABERT-2-117M",
    }
    if model_type not in model_dirs:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return ModelSpec(
        model_type=model_type,
        model_path=str(MODEL_ROOT / model_dirs[model_type]),
        model_max_length=310,
        trust_remote_code=True,
    )

def load_tokenizer(spec: ModelSpec):
    """
    加载与训练时一致配置的tokenizer
    """
    if spec.model_type == "GROVER":
        return AutoTokenizer.from_pretrained(
            spec.model_path, trust_remote_code=True, model_max_length=spec.model_max_length
        )
    return AutoTokenizer.from_pretrained(spec.model_path, model_max_length=spec.model_max_length)


def compute_offsets(sequence: str, tokenizer) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算token offsets（包含特殊token），并返回 input_ids
    使用与 map_BPE_threshold.py 一致的无padding/无截断设置，以保留原始offset
    """
    enc = tokenizer(
        sequence,
        return_tensors="pt",
        padding=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    offsets = np.array(enc["offset_mapping"][0])
    input_ids = np.array(enc["input_ids"][0])
    return offsets, input_ids


def map_interval_to_tokens(offsets: np.ndarray, start: int, end: int) -> np.ndarray:
    """
    将区间 [start, end) 映射为与token数量相同的0/1向量
    规则：若与token的重叠比例 > 0.5，则该token置为1。
    """
    mapping = np.zeros(len(offsets), dtype=int)
    if len(offsets) == 0:
        return mapping
    for i in range(1, len(offsets) - 1):
        token_start, token_end = int(offsets[i][0]), int(offsets[i][1])
        overlap_start = max(token_start, start)
        overlap_end = min(token_end, end)
        overlap = max(0, overlap_end - overlap_start)
        token_len = max(1, token_end - token_start)
        mapping[i] = 1 if (overlap / token_len) > 0.5 else 0
    return mapping


def align_length(array: np.ndarray, target_len: int) -> np.ndarray:
    """
    将映射向量长度与注意力向量长度对齐
    超长截断；不足右侧零填充
    """
    if len(array) > target_len:
        return array[:target_len]
    if len(array) < target_len:
        padding = np.zeros(target_len - len(array), dtype=array.dtype)
        return np.concatenate([array, padding], axis=0)
    return array


def sample_random_intervals(
    sequence_length: int,
    motif_length: int,
    num_samples: int,
    exclude: Optional[Tuple[int, int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[int, int]]:
    """
    在 [0, sequence_length) 上采样与 motif 等长的随机区间，避免与 exclude 重叠
    返回半开区间 [start, end)
    """
    if rng is None:
        rng = np.random.default_rng()
    if motif_length <= 0:
        return []
    max_start = max(0, sequence_length - motif_length)
    intervals: List[Tuple[int, int]] = []
    for _ in range(num_samples):
        while True:
            start = int(rng.integers(0, max_start + 1))
            end = start + motif_length
            if exclude is None:
                intervals.append((start, end))
                break
            ex_s, ex_e = exclude
            if end <= ex_s or start >= ex_e:
                intervals.append((start, end))
                break
    return intervals


def build_mappings(
    sequence: str,
    motif_start: int,
    motif_end: int,
    vector_length: int,
    tokenizer,
    num_random: int = 1000,
    seed: int = 42,
    offsets: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    基于给定序列与motif区间构建：
      - motif的token映射
      - 若干随机区间的token映射
    并将所有映射对齐到注意力向量长度
    """
    if offsets is None:
        offsets, _ = compute_offsets(sequence, tokenizer)
    motif_map = map_interval_to_tokens(offsets, motif_start, motif_end)
    motif_map = align_length(motif_map, vector_length)

    seq_len = len(sequence)
    motif_len = max(0, motif_end - motif_start)
    rng = np.random.default_rng(seed)
    random_maps: List[np.ndarray] = []
    for rs, re in sample_random_intervals(seq_len, motif_len, num_random, (motif_start, motif_end), rng):
        rand_map = map_interval_to_tokens(offsets, rs, re)
        random_maps.append(align_length(rand_map, vector_length))

    return motif_map, random_maps


