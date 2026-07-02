#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -uo pipefail

# Helper function to print styled logs
log_info() {
    echo -e "\e[32m[INFO]\e[0m $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "\e[31m[ERROR]\e[0m $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_warn() {
    echo -e "\e[33m[WARN]\e[0m $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Helper function to check if a specific command line option is present
has_option() {
    local opt="$1"
    shift
    for arg in "$@"; do
        if [[ "$arg" == "$opt" ]]; then
            return 0
        fi
    done
    return 1
}

show_usage() {
    echo "Usage: $0 <model_path> --backend <vllm|sglang> [additional_args...]"
    echo "   or: $0 --model <model_path> --backend <vllm|sglang> [additional_args...]"
    echo ""
    echo "Options:"
    echo "  -m, --model <path>      Path to the local model directory or Hugging Face repo ID"
    echo "  -b, --backend <name>    Backend to use: 'vllm' or 'sglang'"
    echo "  -h, --help              Show this help message"
}

# Parse command line arguments
MODEL_PATH=""
BACKEND=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;
        -m|--model)
            if [[ -z "${2:-}" ]]; then
                log_error "Missing value for --model option"
                exit 1
            fi
            MODEL_PATH="$2"
            shift 2
            ;;
        -b|--backend)
            if [[ -z "${2:-}" ]]; then
                log_error "Missing value for --backend option"
                exit 1
            fi
            BACKEND="$2"
            shift 2
            ;;
        -*)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            if [[ -z "${MODEL_PATH}" ]]; then
                MODEL_PATH="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

# Validate inputs
if [[ -z "${MODEL_PATH}" ]]; then
    log_error "Model path/ID is required."
    show_usage
    exit 1
fi

if [[ -z "${BACKEND}" ]]; then
    log_error "Backend is required (use --backend vllm or --backend sglang)."
    show_usage
    exit 1
fi

# Normalize backend input
BACKEND=$(echo "${BACKEND}" | tr '[:upper:]' '[:lower:]')

if [[ "${BACKEND}" != "vllm" && "${BACKEND}" != "sglang" ]]; then
    log_error "Invalid backend: '${BACKEND}'. Supported backends: 'vllm', 'sglang'."
    exit 1
fi

# Phase 1 baselines must not silently use unvalidated custom kernels.
export USE_CUSTOM_RMSNORM="${USE_CUSTOM_RMSNORM:-0}"
export USE_CUSTOM_SWIGLU="${USE_CUSTOM_SWIGLU:-0}"
export USE_TRITON_KERNELS="${USE_TRITON_KERNELS:-0}"

# Define path configurations
VENV_PATH=""
PYTHON_BIN=""
SCRIPT_BIN=""
ARGS=()

if [[ "${BACKEND}" == "vllm" ]]; then
    VENV_PATH="/data/.venv-vllm"
    PYTHON_BIN="${VENV_PATH}/bin/python"
    SCRIPT_BIN="/data/inference/kernel-swing/serve_vllm.py"
    if [[ ! -f "${SCRIPT_BIN}" ]]; then
        SCRIPT_BIN="${VENV_PATH}/bin/vllm"
    fi
    
    # Configure defaults optimized for RTX 4050 6GB VRAM
    ARGS=("serve" "${MODEL_PATH}")
    
    # 1. Enforce a small max sequence length to fit in VRAM
    if ! has_option "--max-model-len" "${EXTRA_ARGS[@]}"; then
        ARGS+=("--max-model-len" "2048")
    fi
    # 2. Restrict GPU memory allocation to prevent Out-Of-Memory errors
    if ! has_option "--gpu-memory-utilization" "${EXTRA_ARGS[@]}"; then
        ARGS+=("--gpu-memory-utilization" "0.80")
    fi
    # 3. Enforce eager mode to disable CUDA graph memory overhead (saves ~1-2 GB of VRAM)
    if ! has_option "--enforce-eager" "${EXTRA_ARGS[@]}"; then
        ARGS+=("--enforce-eager")
    fi
    # 4. Disable flashinfer sampler compilation (can cause compiler hangs / memory issues on low VRAM GPUs)
    export VLLM_USE_FLASHINFER_SAMPLER=0

    # Append any user overrides/extra args
    ARGS+=("${EXTRA_ARGS[@]}")

elif [[ "${BACKEND}" == "sglang" ]]; then
    VENV_PATH="/data/.venv-sglang"
    PYTHON_BIN="${VENV_PATH}/bin/python"
    SCRIPT_BIN="${VENV_PATH}/bin/sglang"
    
    # Configure defaults optimized for RTX 4050 6GB VRAM
    ARGS=("launch_server" "--model-path" "${MODEL_PATH}")
    
    # 1. Enforce a small max sequence length to fit in VRAM
    if ! has_option "--max-model-len" "${EXTRA_ARGS[@]}"; then
        ARGS+=("--max-model-len" "2048")
    fi
    # 2. Limit the memory fraction allocated to the static KV cache
    if ! has_option "--mem-fraction-static" "${EXTRA_ARGS[@]}"; then
        ARGS+=("--mem-fraction-static" "0.80")
    fi
    
    # Append any user overrides/extra args
    ARGS+=("${EXTRA_ARGS[@]}")
fi

# Verify the virtual environment directory exists
if [[ ! -d "${VENV_PATH}" ]]; then
    log_error "Virtual environment directory not found: ${VENV_PATH}"
    exit 1
fi

# Verify the python binary exists and is executable
if [[ ! -x "${PYTHON_BIN}" ]]; then
    log_error "Python executable not found or not executable: ${PYTHON_BIN}"
    exit 1
fi

# Verify the script exists
if [[ ! -f "${SCRIPT_BIN}" ]]; then
    log_error "Serving script not found: ${SCRIPT_BIN}"
    exit 1
fi

# Print startup information
log_info "Initializing model serving..."
log_info "  - Backend:      ${BACKEND}"
log_info "  - Model Path:   ${MODEL_PATH}"
log_info "  - Environment:  ${VENV_PATH}"
log_info "  - Extra Args:   ${EXTRA_ARGS[*]:-(none)}"
log_info "  - RMSNorm:      USE_CUSTOM_RMSNORM=${USE_CUSTOM_RMSNORM}"
log_info "  - SwiGLU:       USE_CUSTOM_SWIGLU=${USE_CUSTOM_SWIGLU}"
log_info "  - Command:      ${PYTHON_BIN} ${SCRIPT_BIN} ${ARGS[*]}"

# Run the command and catch exit status
set +e
"${PYTHON_BIN}" "${SCRIPT_BIN}" "${ARGS[@]}"
EXIT_CODE=$?
set -e

# Handle termination logging
if [[ ${EXIT_CODE} -ne 0 ]]; then
    log_error "Server process failed or terminated with error (Exit Code: ${EXIT_CODE})."
    exit ${EXIT_CODE}
else
    log_info "Server process stopped clean (Exit Code: 0)."
fi
