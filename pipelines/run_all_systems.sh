#!/usr/bin/env bash
# =============================================================================
# Cluster driver — runs every reimplemented CLPsych 2025 system (Task B) plus
# the EAIonFlux rerun (Task B) with the ORIGINAL paper models, via vLLM with
# tensor parallelism across the cluster's 4 GPUs.
#
# TASK B ONLY for now. Every system also implements Task C (--task c, or
# --task bc to run both in one model load) — see the commented Task C blocks
# at the bottom of this script. uOttawa's Task C needs its own Task B
# summaries, so for Task C use its --task bc.
#
# Systems run (in order):
#   1. BULUSI     Task B   Qwen2.5-72B-Instruct   (5-shot contriever + ms-marco rerank)
#   2. Aquarius   Task B   Qwen2.5-32B-Instruct   (one-shot, BGE-large retrieval)
#   3. ISM        Task B   Meta-Llama-3-8B-Instruct (10 candidates + KDE selection)
#   4. uOttawa    Task B   Llama-3.3-70B-Instruct (variant S2, one-shot; see
#                        the commented S1/S3 blocks for the other submissions)
#   5. EAIonFlux  Task B  Llama-3.3-70B-Instruct (k=5, paper original model)
#
# Resumable: each system appends per-post/per-timeline results to a JSONL
# checkpoint and skips completed items on restart, so a crashed or preempted
# run can be restarted by simply rerunning this script (reruns cost only the
# missing generations). The 70B/72B runs take hours.
#
# Everything is local: generation loads local model weights via vLLM; set the
# *_PATH variables below to your downloaded copies (huggingface-cli download
# ...). HF_HUB_OFFLINE=1 guarantees no network calls at runtime. Pre-download
# the (small) embedding/reranker models listed in the PREFETCH note first —
# they load from the HF cache.
#
# Outputs land in $OUT_DIR/<system>/ as official submission-format JSONs
# (same structure as dev.json / the gold data, gold evidence + gold well-being
# filled in), so they plug straight into eval/task31_eval.py and the official
# run.py — copy $OUT_DIR back to the workstation and score there.
#
# Typical use (bare cluster node):
#   chmod +x pipelines/run_all_systems.sh
#   ./pipelines/run_all_systems.sh
# or wrapped in your scheduler, e.g.:
#   sbatch -N1 --gres=gpu:4 --wrap='bash pipelines/run_all_systems.sh'
# =============================================================================
set -euo pipefail

# ── Configuration (override via environment variables) ──────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

# Source data: directory of per-timeline shared-task JSON files
# (the same directory you would pass to pipeline.py / prepare_data.py).
DATA_DIR="${DATA_DIR:-/path/to/train-clpsych2025-v2}"

# Output root (checkpoints + submissions + settings logs per system).
OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/cluster_run}"

# Local model weights — set these to your huggingface-cli download targets:
#   huggingface-cli download Qwen/Qwen2.5-72B-Instruct
#   huggingface-cli download Qwen/Qwen2.5-32B-Instruct
#   huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct
#   huggingface-cli download meta-llama/Llama-3.3-70B-Instruct
QWEN25_72B_PATH="${QWEN25_72B_PATH:-/path/to/Qwen2.5-72B-Instruct}"
QWEN25_32B_PATH="${QWEN25_32B_PATH:-/path/to/Qwen2.5-32B-Instruct}"
LLAMA3_8B_PATH="${LLAMA3_8B_PATH:-/path/to/Meta-Llama-3-8B-Instruct}"
LLAMA33_70B_PATH="${LLAMA33_70B_PATH:-/path/to/Llama-3.3-70B-Instruct}"

# If the GPUs cannot hold a bf16 70B/72B per card, quantize, e.g.:
#   QUANTIZATION="fp8"   (H100)  or  QUANTIZATION="awq"  (with an AWQ checkpoint)
QUANTIZATION="${QUANTIZATION:-}"

TP="${TP:-4}"            # vLLM tensor parallelism (4 GPUs)
GPU_UTIL="${GPU_UTIL:-0.9}"
BATCH="${BATCH:-16}"      # posts per generation batch (checkpoint saved after each)
ISM_BATCH="${ISM_BATCH:-8}"   # ISM: n_rounds(10) generations per post

# Keep everything local at runtime. NOTE: the first run needs the small
# embedding / reranker models in the HF cache; either pre-download them
# (huggingface-cli download facebook/contriever
#  cross-encoder/ms-marco-MiniLM-L-6-v2  BAAI/bge-large-en-v1.5
#  sentence-transformers/all-MiniLM-L6-v2
#  sentence-transformers/all-mpnet-base-v2  Linq-AI-Research/Linq-Embed-Mistral)
# or set HF_HUB_OFFLINE=0 for that first run only.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONIOENCODING=utf-8

QUANT_FLAG=()
[ -n "$QUANTIZATION" ] && QUANT_FLAG=(--quantization "$QUANTIZATION")

step() {
    echo ""
    echo "==================================================================="
    echo ">>> $*"
    echo "==================================================================="
    date
}

# ── 1. BULUSI (Task B, Qwen2.5-72B) ─────────────────────────────────────────
step "BULUSI Task B — Qwen2.5-72B (5-shot: contriever top-50 -> ms-marco rerank -> top-5, LOTO)"
"$PYTHON" "$REPO_ROOT/pipelines/bulusi_model.py" \
    --data_dir "$DATA_DIR" --output_dir "$OUT_DIR/bulusi" \
    --task b --backend vllm --model "$QWEN25_72B_PATH" \
    --tensor_parallel_size "$TP" --gpu_memory_utilization "$GPU_UTIL" \
    --batch_size "$BATCH" "${QUANT_FLAG[@]+"${QUANT_FLAG[@]}"}"

# ── 2. Aquarius (Task B, Qwen2.5-32B) ────────────────────────────────────────
step "Aquarius Task B — Qwen2.5-32B (one-shot, BGE-large-en-v1.5 retrieval, LOTO)"
"$PYTHON" "$REPO_ROOT/pipelines/aquarius_model.py" \
    --data_dir "$DATA_DIR" --output_dir "$OUT_DIR/aquarius" \
    --task b --backend vllm --model "$QWEN25_32B_PATH" \
    --tensor_parallel_size "$TP" --gpu_memory_utilization "$GPU_UTIL" \
    --batch_size "$BATCH"

# ── 3. ISM (Task B, Meta-Llama-3-8B) ────────────────────────────────────────
# 10 candidates per post (temp 0.1) + Gaussian KDE (Scott) selection over
# all-MiniLM-L6-v2 summary embeddings. TP=1 lets the 8B fit on one GPU — but
# if you run the systems strictly sequentially, keep TP so this driver uses
# the same 4-GPU layout.
step "ISM Task B — Meta-Llama-3-8B (10 candidates + KDE selection, LOTO)"
"$PYTHON" "$REPO_ROOT/pipelines/ism_model.py" \
    --data_dir "$DATA_DIR" --output_dir "$OUT_DIR/ism" \
    --task b --backend vllm --model "$LLAMA3_8B_PATH" \
    --tensor_parallel_size "$TP" --gpu_memory_utilization "$GPU_UTIL" \
    --batch_size "$ISM_BATCH"

# ── 4. uOttawa (Task B, Llama-3.3-70B) ───────────────────────────────────────
step "uOttawa Task B (S2 one-shot, their best submission) — Llama-3.3-70B"
"$PYTHON" "$REPO_ROOT/pipelines/uottawa_model.py" \
    --data_dir "$DATA_DIR" --output_dir "$OUT_DIR/uottawa" \
    --task b --backend vllm --model "$LLAMA33_70B_PATH" --variant s2 \
    --tensor_parallel_size "$TP" --gpu_memory_utilization "$GPU_UTIL" \
    --batch_size "$BATCH" "${QUANT_FLAG[@]+"${QUANT_FLAG[@]}"}"

# The paper's other two official submissions (each ~1 extra Task B pass):
# step "uOttawa Task B (S1 zero-shot)"
# "$PYTHON" "$REPO_ROOT/pipelines/uottawa_model.py" \
#     --data_dir "$DATA_DIR" --output_dir "$OUT_DIR/uottawa" \
#     --backend vllm --model "$LLAMA33_70B_PATH" --variant s1 \
#     --tensor_parallel_size "$TP" --gpu_memory_utilization "$GPU_UTIL" \
#     --batch_size "$BATCH" "${QUANT_FLAG[@]+"${QUANT_FLAG[@]}"}"
# step "uOttawa Task B (S3 few-shot, 7 examples)"
# "$PYTHON" "$REPO_ROOT/pipelines/uottawa_model.py" \
#     --data_dir "$DATA_DIR" --output_dir "$OUT_DIR/uottawa" \
#     --backend vllm --model "$LLAMA33_70B_PATH" --variant s3 \
#     --tensor_parallel_size "$TP" --gpu_memory_utilization "$GPU_UTIL" \
#     --batch_size "$BATCH" "${QUANT_FLAG[@]+"${QUANT_FLAG[@]}"}"

# ── 5. EAIonFlux rerun (Task B, Llama-3.3-70B — the paper's model) ──────────
step "EAIonFlux Task B — Llama-3.3-70B (k=5, ICL)"
"$PYTHON" "$REPO_ROOT/pipelines/eaionflux_model.py" \
    --data_dir "$DATA_DIR" --output_dir "$OUT_DIR/eaionflux" \
    --backend vllm --model "$LLAMA33_70B_PATH" --task B --k 5 \
    --skip_zero_shot \
    --tensor_parallel_size "$TP" --gpu_memory_utilization "$GPU_UTIL" \
    "${QUANT_FLAG[@]+"${QUANT_FLAG[@]}"}"

# ── Task C (for later use — not run for now) ────────────────────────────────
# Every system implements Task C too. To add Task C runs, uncomment the blocks
# you want and rerun this script (checkpoints make it resumable, so only the
# Task C generations are added). Score Task C submissions with
# clpsych2025/evaluation/run.py --tasks C.
#
# step "BULUSI Task C"
# "$PYTHON" "$REPO_ROOT/pipelines/bulusi_model.py" \
#     --data_dir "$DATA_DIR" --output_dir "$OUT_DIR/bulusi" \
#     --task c --backend vllm --model "$QWEN25_72B_PATH" \
#     --tensor_parallel_size "$TP" --gpu_memory_utilization "$GPU_UTIL" \
#     --batch_size "$BATCH" "${QUANT_FLAG[@]+"${QUANT_FLAG[@]}"}"
#
# step "Aquarius Task C"
# "$PYTHON" "$REPO_ROOT/pipelines/aquarius_model.py" \
#     --data_dir "$DATA_DIR" --output_dir "$OUT_DIR/aquarius" \
#     --task c --backend vllm --model "$QWEN25_32B_PATH" \
#     --tensor_parallel_size "$TP" --gpu_memory_utilization "$GPU_UTIL" \
#     --batch_size "$BATCH"
#
# step "ISM Task C"
# "$PYTHON" "$REPO_ROOT/pipelines/ism_model.py" \
#     --data_dir "$DATA_DIR" --output_dir "$OUT_DIR/ism" \
#     --task c --backend vllm --model "$LLAMA3_8B_PATH" \
#     --tensor_parallel_size "$TP" --gpu_memory_utilization "$GPU_UTIL" \
#     --batch_size "$ISM_BATCH"
#
# uOttawa Task C cascades its own Task B summaries -> use --task bc (Task B is
# resumed from its checkpoint, so only missing posts + Task C are generated).
# step "uOttawa Task C (S2, cascades Task B)"
# "$PYTHON" "$REPO_ROOT/pipelines/uottawa_model.py" \
#     --data_dir "$DATA_DIR" --output_dir "$OUT_DIR/uottawa" \
#     --task bc --backend vllm --model "$LLAMA33_70B_PATH" --variant s2 \
#     --tensor_parallel_size "$TP" --gpu_memory_utilization "$GPU_UTIL" \
#     --batch_size "$BATCH" "${QUANT_FLAG[@]+"${QUANT_FLAG[@]}"}"
#
# step "EAIonFlux Task C (cascades Task B)"
# "$PYTHON" "$REPO_ROOT/pipelines/eaionflux_model.py" \
#     --data_dir "$DATA_DIR" --output_dir "$OUT_DIR/eaionflux" \
#     --backend vllm --model "$LLAMA33_70B_PATH" --task C --k 5 \
#     --tensor_parallel_size "$TP" --gpu_memory_utilization "$GPU_UTIL" \
#     "${QUANT_FLAG[@]+"${QUANT_FLAG[@]}"}"

# ── Summary ──────────────────────────────────────────────────────────────────
step "All systems complete. Task B submission files (copy back and score locally):"
find "$OUT_DIR" -name "*taskb_submission.json" -print || true
echo ""
echo "Score the Task B systems locally with, e.g.:"
echo "  python eval/task31_eval.py --submissions \\"
echo "    $OUT_DIR/bulusi/bulusi_taskb_submission.json \\"
echo "    $OUT_DIR/aquarius/aquarius_taskb_submission.json \\"
echo "    $OUT_DIR/ism/ism_taskb_submission.json \\"
echo "    $OUT_DIR/uottawa/uottawa_s2_taskb_submission.json \\"
echo "    $OUT_DIR/eaionflux/eaionflux_k5_taskb_submission.json"
echo "(Task C submissions — if you enable the blocks above — are scored with"
echo " run.py --tasks C; each system's dir also holds a *_settings.md decoding"
echo " log — the source for the paper's reimplementation section)"
