#!/usr/bin/env bash
set -euo pipefail

# Make `mamba` available when invoked from a non-interactive shell.
if ! command -v mamba >/dev/null 2>&1; then
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniforge3}"
  for init in "${HOME}/miniforge3/etc/profile.d/conda.sh" "${HOME}/miniforge3/etc/profile.d/mamba.sh"; do
    if [[ -f "${init}" ]]; then
      # shellcheck disable=SC1090
      source "${init}"
    fi
  done
fi

usage() {
  cat <<'USAGE'
Usage:
  run_matrix.sh [options] [extra train.py args...]

Options:
  --smoke-test                  Pass smoke-test sample caps to train.py.
  --dataset NT|GUE|all          Dataset filter (default all).
  --model MODEL|all             Model filter (default all).
  --task TASK|all               Task filter (NT task name or GUE subdir/task).
  --cuda-visible-devices IDS    GPU IDs exposed to each run (default: CUDA_VISIBLE_DEVICES or 0).
  --output-root PATH            Training output root (default: finetune/outputs).
  --dry-run                     Print planned commands without training.
  --skip-completed              Skip tasks whose test_metrics.json already exists.
  -h, --help                    Show this help.

Runs the full Experiment 1 model/task matrix on one visible GPU by default.
Failed model-task pairs are logged and the matrix continues.
USAGE
}

smoke=0
dataset_filter="all"
model_filter="all"
task_filter="all"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
output_root=""
dry_run=0
skip_completed=0
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-test)
      smoke=1
      shift
      ;;
    --dataset)
      dataset_filter="$2"
      shift 2
      ;;
    --model)
      model_filter="$2"
      shift 2
      ;;
    --task)
      task_filter="$2"
      shift 2
      ;;
    --cuda-visible-devices)
      cuda_visible_devices="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --skip-completed)
      skip_completed=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exp_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${exp_root}/../.." && pwd)"
export PYTHONPATH="${exp_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
conda_env="${CONDA_ENV:-glm_hf}"
cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${cuda_visible_devices}"
if [[ -z "${output_root}" ]]; then
  output_root="${exp_root}/outputs"
fi

models=(dnabert2_117m ntv2_500m_multi hyenadna_large_1m gena_bigbird_t2t grover)
nt_tasks=(
  promoter_all promoter_tata promoter_no_tata enhancers enhancers_types
  splice_sites_all splice_sites_acceptors splice_sites_donors H2AFZ H3K27ac
  H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K9ac H3K9me3 H4K20me1
)
gue_tasks=(
  EMP/H3 EMP/H3K14ac EMP/H3K36me3 EMP/H3K4me1 EMP/H3K4me2 EMP/H3K4me3
  EMP/H3K79me3 EMP/H3K9ac EMP/H4 EMP/H4ac prom/prom_300_all
  prom/prom_300_notata prom/prom_300_tata prom/prom_core_all prom/prom_core_notata
  prom/prom_core_tata splice/reconstructed tf/0 tf/1 tf/2 tf/3 tf/4
)

run_task() {
  local model="$1"
  local dataset="$2"
  local task="$3"
  local log_dir="${exp_root}/logs/${model}/${dataset}/${task}"
  local task_out="${output_root}/${model}/${dataset}/${task}"
  mkdir -p "${log_dir}"

  if [[ "${skip_completed}" == "1" && -f "${task_out}/test_metrics.json" ]]; then
    echo "SKIP ${model} ${dataset}/${task} (test_metrics.json exists)"
    return 0
  fi

  local args=(--model "${model}" --dataset "${dataset}" --task "${task}" --output-root "${output_root}")
  if [[ "${smoke}" == "1" ]]; then
    args+=(--smoke-test --max-train-samples 4 --max-eval-samples 4 --token-length-sample-size 64)
  fi
  args+=("${extra_args[@]}")

  echo "BEGIN ${model} ${dataset}/${task} cuda=${CUDA_VISIBLE_DEVICES} output_root=${output_root}"
  if [[ "${dry_run}" == "1" ]]; then
    echo "DRYRUN python -m glm_finetune.train ${args[*]}"
    return 0
  fi
  if mamba run -n "${conda_env}" python -m glm_finetune.train "${args[@]}" > "${log_dir}/train.log" 2>&1; then
    echo "OK ${model} ${dataset}/${task}"
  else
    echo "FAILED ${model} ${dataset}/${task}; see ${log_dir}/train.log"
    return 1
  fi
}

status=0
for model in "${models[@]}"; do
  if [[ "${model_filter}" != "all" && "${model_filter}" != "${model}" ]]; then
    continue
  fi

  if [[ "${dataset_filter}" == "all" || "${dataset_filter}" == "NT" ]]; then
    for task in "${nt_tasks[@]}"; do
      if [[ "${task_filter}" != "all" && "${task_filter}" != "${task}" ]]; then
        continue
      fi
      run_task "${model}" NT "${task}" || status=1
    done
  fi

  if [[ "${dataset_filter}" == "all" || "${dataset_filter}" == "GUE" ]]; then
    for task in "${gue_tasks[@]}"; do
      if [[ "${task_filter}" != "all" && "${task_filter}" != "${task}" ]]; then
        continue
      fi
      run_task "${model}" GUE "${task}" || status=1
    done
  fi
done

exit "${status}"
