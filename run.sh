#!/bin/bash
#SBATCH --job-name=fairness-safety
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --time=12:00:00

set -u

if [[ $# -ne 1 ]]; then
    echo "Usage: bash run.sh <config.yaml|config-directory>" >&2
    exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TARGET="$(realpath "$1")"
STATE_DIR="$ROOT/.run_state"
mkdir -p "$STATE_DIR" || exit 1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${CONDA_ENV:-}" ]]; then
    source /apps/local/conda_init.sh
    conda activate "$CONDA_ENV"
    PYTHON=(python)
else
    PYTHON=(python)
fi

if [[ -d "$TARGET" ]]; then
    mapfile -t CONFIGS < <(find "$TARGET" -type f \( -name '*.yaml' -o -name '*.yml' \) | sort)
elif [[ -f "$TARGET" && "$TARGET" =~ \.ya?ml$ ]]; then
    CONFIGS=("$TARGET")
else
    echo "Not a YAML file or directory: $TARGET" >&2
    exit 2
fi

# Split a config directory across a contiguous/stepped SLURM array. For
# example, --array=0-7 assigns one eighth of the sorted configs to each GPU.
if [[ -d "$TARGET" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    task_min="${SLURM_ARRAY_TASK_MIN:-0}"
    task_step="${SLURM_ARRAY_TASK_STEP:-1}"
    task_count="${SLURM_ARRAY_TASK_COUNT:-1}"
    delta=$((SLURM_ARRAY_TASK_ID - task_min))
    if ((delta < 0 || delta % task_step != 0)); then
        echo "Invalid SLURM array task: $SLURM_ARRAY_TASK_ID" >&2
        exit 2
    fi
    slot=$((delta / task_step))
    if ((slot >= ${#CONFIGS[@]})); then
        echo "No config assigned to array task $SLURM_ARRAY_TASK_ID"
        exit 0
    fi
    SELECTED=()
    for ((index = slot; index < ${#CONFIGS[@]}; index += task_count)); do
        SELECTED+=("${CONFIGS[index]}")
    done
    CONFIGS=("${SELECTED[@]}")
    echo "ARRAY task $SLURM_ARRAY_TASK_ID: ${#CONFIGS[@]} config(s)"
fi

CURRENT_LOCK=""
trap '[[ -n "$CURRENT_LOCK" ]] && rmdir "$CURRENT_LOCK" 2>/dev/null || true' EXIT INT TERM
failed=0

for config in "${CONFIGS[@]}"; do
    path_hash="$(printf '%s' "$config" | sha256sum | cut -c1-12)"
    key="$(basename "${config%.*}")-$path_hash"
    done_file="$STATE_DIR/$key.done"
    lock_dir="$STATE_DIR/$key.running"
    if [[ -f "$done_file" ]]; then
        echo "SKIP completed: $config"
        continue
    fi
    if ! mkdir "$lock_dir" 2>/dev/null; then
        echo "SKIP running: $config"
        continue
    fi
    CURRENT_LOCK="$lock_dir"
    echo "RUN: $config"
    if "${PYTHON[@]}" -m fairness_safety.runner "$config"; then
        touch "$done_file"
    else
        echo "FAILED: $config" >&2
        failed=1
    fi
    rmdir "$lock_dir"
    CURRENT_LOCK=""
done

exit "$failed"
