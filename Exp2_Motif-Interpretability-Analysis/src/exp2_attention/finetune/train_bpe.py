"""
BPE-based DNA sequence classification training script.

This script supports training with different BPE-encoded DNA language models:
- DNABERT-2 (default): Uses DNABERT-2-117M model
- GENA: Uses GENA_LM_BigBird model
- GROVER: Uses GROVER model with optimized vocabulary for human genome

Usage examples:
    # Train with DNABERT-2 (default)
    python train_BPE_copy.py --data_path /path/to/data --output_dir ./output

    # Train with GENA model
    python train_BPE_copy.py --model_type gena --data_path /path/to/data --output_dir ./output

    # Train with GROVER model
    python train_BPE_copy.py --model_type grover --data_path /path/to/data --output_dir ./output

    # Use custom model path
    python train_BPE_copy.py --model_type dnabert2 --model_name_or_path /custom/path --data_path /path/to/data --output_dir ./output
"""

import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
import transformers
if os.environ.get("DISABLE_PEFT_FOR_TRAINER") == "1":
    import transformers.utils.import_utils as _transformers_import_utils
    _transformers_import_utils._peft_available = False
if os.environ.get("PATCH_ACCELERATE_FOR_TRAINER") == "1":
    import inspect
    import accelerate

    _accelerator_init = accelerate.Accelerator.__init__
    _accelerator_params = set(inspect.signature(_accelerator_init).parameters)

    def _compatible_accelerator_init(self, *args, **kwargs):
        kwargs = {key: value for key, value in kwargs.items() if key in _accelerator_params}
        return _accelerator_init(self, *args, **kwargs)

    accelerate.Accelerator.__init__ = _compatible_accelerator_init
import sklearn
import numpy as np
from torch.utils.data import Dataset

import sys
from exp2_attention.paths import MODEL_ROOT, model_path

gfm_model_use = MODEL_ROOT.parent / "GFM_model_use"
if gfm_model_use.exists():
    sys.path.append(str(gfm_model_use))


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
        logging.warning("Disabled DNABERT2 Triton attention in: %s", ", ".join(sorted(patched_modules)))

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_type: str = field(default="dnabert2", metadata={"help": "Type of model to use: 'dnabert2', 'gena', or 'grover'"})
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=2)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    report_to: str = field(default="tensorboard", metadata={"help": "The list of integrations to report the results and logs to."})
    logging_dir: Optional[str] = field(default=None, metadata={"help": "TensorBoard log directory. Will default to runs/**CURRENT_DATETIME_HOSTNAME** in the output directory."})
    

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa




class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], attention_mask=self.attention_mask[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.LongTensor(labels)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall, auprc, auroc with sklearn.
"""
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray, probabilities: np.ndarray = None):
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    metrics = {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }

    # Add AUPRC and AUROC if probabilities are provided
    if probabilities is not None:
        valid_probabilities = probabilities[valid_mask]

        # For binary classification, use the positive class probability
        if valid_probabilities.ndim == 1:
            pos_probabilities = valid_probabilities
        else:
            pos_probabilities = valid_probabilities[:, 1]  # Use probability of positive class

        # Calculate AUPRC (Area Under Precision-Recall Curve)
        precision_curve, recall_curve, _ = sklearn.metrics.precision_recall_curve(valid_labels, pos_probabilities)
        metrics["auprc"] = sklearn.metrics.auc(recall_curve, precision_curve)

        # Calculate AUROC (Area Under ROC Curve)
        metrics["auroc"] = sklearn.metrics.roc_auc_score(valid_labels, pos_probabilities)

    return metrics

# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return logits


"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Get predictions from logits
    predictions = np.argmax(logits, axis=-1)

    # Get probabilities using softmax for AUPRC and AUROC calculation
    # Use scipy.special.softmax for numerical stability
    from scipy.special import softmax
    probabilities = softmax(logits, axis=-1)

    return calculate_metric_with_sklearn(predictions, labels, probabilities)



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set default model path based on model type.
    if model_args.model_name_or_path is None:
        model_args.model_name_or_path = model_path(model_args.model_type)

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                      data_path=os.path.join(data_args.data_path, "train.csv"))
    val_dataset = SupervisedDataset(tokenizer=tokenizer,
                                     data_path=os.path.join(data_args.data_path, "val.csv"))
    test_dataset = SupervisedDataset(tokenizer=tokenizer,
                                     data_path=os.path.join(data_args.data_path, "test.csv"))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


    # load model
    if model_args.model_type == "gena":
        from GENA_LM.src.gena_lm.modeling_bert import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=train_dataset.num_labels,
        )
    elif model_args.model_type == "grover":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    else:  # dnabert2
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )

    if model_args.model_type == "dnabert2" and os.environ.get("DISABLE_DNABERT2_TRITON") == "1":
        disable_dnabert2_triton_attention(model)

    # Set up TensorBoard logging
    if training_args.logging_dir is None:
        training_args.logging_dir = os.path.join(training_args.output_dir, "runs")

    # Add TensorBoard callback
    callbacks = []
    if training_args.report_to == "tensorboard":
        from transformers.integrations import TensorBoardCallback
        callbacks.append(TensorBoardCallback())

    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator,
                                   callbacks=callbacks)
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)




if __name__ == "__main__":
    train()
