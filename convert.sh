#!/usr/bin/env bash
#
# convert.sh - One-shot YOLOv8 (.pt) -> Hailo HEF converter
#
# Target hardware : Raspberry Pi 5 + Hailo AI Hat (Hailo-8L)
# Pipeline        : .pt -> ONNX -> Hailo HAR (parse) -> optimized HAR (quantize) -> HEF (compile)
#
# Just run:
#     ./convert.sh                      # converts the default exp-2.pt
#     ./convert.sh my_model.pt          # converts my_model.pt
#     ./convert.sh my_model.pt --force  # rebuild every stage from scratch
#     ./convert.sh --num-calib 1024     # more calibration images = better accuracy
#
# Edit the CONFIG section below to change defaults. No need to run any command
# one by one - this script does everything: venv check, dependency check, export,
# parse, calibration, optimization, and HEF compilation.

set -euo pipefail

# ---------------------------- CONFIG ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/hailo_venv"
DFC_WHL="${SCRIPT_DIR}/hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"
CONVERT_PY="${SCRIPT_DIR}/convert_hef.py"

# Defaults passed to convert_hef.py (override via CLI flags after the model name)
DEFAULT_WEIGHTS="exp-2.pt"
DEFAULT_HW_ARCH="hailo8l"          # Pi 5 + Hailo AI Hat = Hailo-8L
DEFAULT_CALIB_DIR="dataset_yolo/images/val"
DEFAULT_NUM_CALIB="300"
# ----------------------------------------------------------------------------

# Colors
G="\033[1;32m"; C="\033[1;36m"; Y="\033[1;33m"; R="\033[1;31m"; B="\033[1m"; X="\033[0m"
say()  { printf "${C}[*]${X} %s\n" "$*"; }
ok()   { printf "${G}[OK]${X} %s\n"  "$*"; }
warn() { printf "${Y}[!]${X} %s\n"   "$*"; }
err()  { printf "${R}[ERR]${X} %s\n" "$*" >&2; }
die()  { err "$*"; exit 1; }

usage() {
    cat <<EOF
${B}Usage:${X}
  $0 [WEIGHTS.pt] [OPTIONS]

${B}Options${X} (forwarded to the python converter):
  --calib-dir DIR       Calibration image folder (default: ${DEFAULT_CALIB_DIR})
  --imgsz N             Input image size        (default: read from checkpoint)
  --num-calib N         # calibration images    (default: ${DEFAULT_NUM_CALIB})
  --batch-size N        Batch size for calib/bias-correction/QFT (default: auto)
  --qft-epochs N        QFT training epochs     (default: 4)
  --hw-arch ARCH        hailo8l | hailo8        (default: ${DEFAULT_HW_ARCH})
  --out-dir DIR         Output directory        (default: this folder)
  --force               Rebuild every stage
  --setup-only          Only create/verify the venv + deps, then exit
  -h, --help            Show this help

${B}Examples${X}
  $0                                   # convert ./exp-2.pt with defaults
  $0 my_model.pt                       # convert a different model
  $0 --num-calib 1024 --force          # best-accuracy rebuild
EOF
    exit 0
}

# ---------------------------- arg parsing -----------------------------------
# Separate the model path (first non-flag arg) from the rest (passed to python).
WEIGHTS=""
PY_ARGS=()
SETUP_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        --setup-only) SETUP_ONLY=1; shift ;;
        --*)  PY_ARGS+=("$1"); shift ;;
        *)   if [[ -z "$WEIGHTS" ]]; then WEIGHTS="$1"; else PY_ARGS+=("$1"); fi; shift ;;
    esac
done
[[ -z "$WEIGHTS" ]] && WEIGHTS="${DEFAULT_WEIGHTS}"

# ---------------------------- banner ----------------------------------------
printf "${G}================================================================${X}\n"
printf "${G}  YOLOv8 -> Hailo HEF converter  (Hailo-8L / Raspberry Pi 5)${X}\n"
printf "${G}================================================================${X}\n"
say "model    : ${WEIGHTS}"
say "venv     : ${VENV_DIR}"
say "hw arch  : ${DEFAULT_HW_ARCH}"

# ---------------------------- 0. preflight checks ---------------------------
[[ -f "${CONVERT_PY}" ]] || die "convert_hef.py missing in ${SCRIPT_DIR}"
[[ -f "${DFC_WHL}"     ]] || die "Hailo DFC wheel missing: ${DFC_WHL}"
# Resolve weights to an absolute path.
[[ "${WEIGHTS}" = /* ]] || WEIGHTS="${SCRIPT_DIR}/${WEIGHTS}"
[[ -f "${WEIGHTS}" ]] || die "Weights file not found: ${WEIGHTS}"

# ---------------------------- 1. venv ---------------------------------------
setup_venv() {
    say "checking Python 3.10 venv..."
    if [[ ! -x "${PYTHON}" ]]; then
        say "creating venv at ${VENV_DIR} ..."
        if ! command -v python3.10 >/dev/null 2>&1; then
            die "python3.10 not found. Install it first (sudo apt install python3.10 python3.10-venv)."
        fi
        python3.10 -m venv "${VENV_DIR}"
        "${PIP}" install --upgrade pip wheel >/dev/null
        ok "venv created"
    else
        ok "venv exists"
    fi
}

# ---------------------------- 2. dependencies -------------------------------
# DFC pins: protobuf==3.20.3, numpy==1.26.4, scipy==1.12.0, onnx==1.16.0,
# onnxruntime==1.18.0, onnxsim==0.4.36, typing-extensions==4.12.2.
# ultralytics pulls newer versions, so we re-pin after installing everything.
PIN_DEPS=(
    "protobuf==3.20.3"
    "numpy==1.26.4"
    "scipy==1.12.0"
    "onnx==1.16.0"
    "onnxruntime==1.18.0"
    "onnxsim==0.4.36"
    "typing-extensions==4.12.2"
    "opencv-python==4.10.0.84"
)

install_deps() {
    say "checking dependencies..."
    if "${PYTHON}" -c "import hailo_sdk_client, ultralytics, onnx, onnxruntime" 2>/dev/null; then
        patch_gpu_threshold
        ok "all key dependencies present"
        return
    fi
    warn "some dependencies missing - installing (this happens once)..."

    # pygraphviz needs relaxed flags + graphviz headers on this toolchain.
    if ! "${PYTHON}" -c "import pygraphviz" 2>/dev/null; then
        if ! command -v gcc >/dev/null 2>&1; then
            die "gcc not found. Install build-essential + libgraphviz-dev first."
        fi
        say "building pygraphviz (needs libgraphviz-dev)..."
        CFLAGS="-Wno-error=incompatible-pointer-types -Wno-incompatible-pointer-types" \
            "${PIP}" install --no-binary :all: pygraphviz \
            --global-option=build_ext \
            --global-option="-I/usr/include/graphviz" \
            --global-option="-L/usr/lib/x86_64-linux-gnu" || \
            die "pygraphviz build failed. Try: sudo apt install libgraphviz-dev graphviz"
    fi

    say "installing Hailo DFC + ultralytics (may take several minutes)..."
    "${PIP}" install "${DFC_WHL}" || die "DFC install failed"
    "${PIP}" install "ultralytics==8.3.96" onnxscript || die "ultralytics install failed"
    say "pinning compatible versions..."
    "${PIP}" install "${PIN_DEPS[@]}" || die "version pinning failed"
    "${PYTHON}" -c "import hailo_sdk_client, ultralytics, onnx, onnxruntime" \
        || die "dependency verification failed even after install"
    patch_gpu_threshold
    ok "dependencies installed"
}

# Hailo DFC only picks a GPU if its memory usage <= 5% (nvidia_smi_gpu_selector.py).
# A laptop GPU driving a display sits well above 5%, so Hailo wrongly falls back to CPU.
# Bump the threshold to 80% so the GPU is used even when the screen is on. Idempotent.
patch_gpu_threshold() {
    local f="${VENV_DIR}/lib/python3.10/site-packages/hailo_model_optimization/acceleras/utils/nvidia_smi_gpu_selector.py"
    [[ -f "$f" ]] || return
    if grep -q "def select_least_used_gpu(max_memory_utilization=0.05)" "$f"; then
        sed -i 's/def select_least_used_gpu(max_memory_utilization=0.05)/def select_least_used_gpu(max_memory_utilization=0.8)/' "$f"
        ok "patched Hailo GPU threshold (5%% -> 80%%, so the display GPU is used)"
    fi
}

setup_venv
install_deps

if [[ ${SETUP_ONLY} -eq 1 ]]; then
    ok "setup complete (--setup-only). Re-run without that flag to convert."
    exit 0
fi

# ---------------------------- 3. XLA / CUDA env -------------------------------
# TensorFlow XLA JIT compiler needs libdevice.10.bc for GPU kernels.
# Auto-detect from common CUDA install locations.
if [[ -z "${XLA_FLAGS:-}" ]]; then
    _libdevice=""
    for _d in /usr/local/cuda-13.1/nvvm/libdevice /usr/local/cuda/nvvm/libdevice /usr/lib/nvidia-cuda-toolkit/libdevice; do
        [[ -f "$_d/libdevice.10.bc" ]] && { _libdevice="$_d/libdevice.10.bc"; break; }
    done
    if [[ -n "$_libdevice" ]]; then
        export XLA_FLAGS="--xla_gpu_cuda_data_dir=$(dirname "$(dirname "$_libdevice")")"
    fi
fi

# ---------------------------- 4. run the pipeline ---------------------------
say "launching conversion pipeline..."
# Use exec so the python process owns the terminal (Ctrl-C works cleanly).
exec "${PYTHON}" "${CONVERT_PY}" \
    --weights  "${WEIGHTS}" \
    --calib-dir "${DEFAULT_CALIB_DIR}" \
    --num-calib "${DEFAULT_NUM_CALIB}" \
    --hw-arch   "${DEFAULT_HW_ARCH}" \
    "${PY_ARGS[@]}"
