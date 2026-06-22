#!/usr/bin/env python3
"""
convert_hef.py - YOLOv8 (.pt) -> ONNX -> Hailo HAR -> optimized HAR -> HEF (Hailo-8L)

Validated end-to-end conversion pipeline for the Raspberry Pi 5 + Hailo AI Hat (Hailo-8L).

Usage (driven by convert.sh; works standalone too):

    python3 convert_hef.py                       # full pipeline, default weights
    python3 convert_hef.py my_model.pt --force   # rebuild every stage
    python3 convert_hef.py --num-calib 1024      # more calibration = better accuracy

DESIGN NOTE
-----------
The Hailo DFC optimizer/allocator is built on TensorFlow eager mode. Running the ONNX
export, HAR parse, model optimization, and HEF compile all inside ONE Python process
accumulates TF state and can deadlock (verified empirically) when re-run with --force.
To stay rock-solid, each heavy stage runs in its OWN isolated subprocess. The main
process is a lightweight orchestrator that spawns `python convert_hef.py --stage <n>`.
This way every run is as reliable as a cold start.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image


# ---------- pretty printing --------------------------------------------------
COLOR_GREEN = "\033[1;32m"
COLOR_CYAN = "\033[1;36m"
COLOR_YELLOW = "\033[1;33m"
COLOR_RED = "\033[1;31m"
COLOR_RESET = "\033[0m"


def log(msg, color=COLOR_CYAN):
    print(f"{color}[*]{COLOR_RESET} {msg}", flush=True)


def section(title):
    bar = "=" * 64
    print(f"\n{COLOR_GREEN}{bar}\n  {title}\n{bar}{COLOR_RESET}", flush=True)


def fail(msg):
    print(f"{COLOR_RED}[ERROR]{COLOR_RESET} {msg}", flush=True)
    sys.exit(1)


def need(cond, msg):
    if not cond:
        fail(msg)


# ---------- discovery --------------------------------------------------------
def probe_yolo(ckpt_path):
    """Inspect a YOLOv8 checkpoint to recover task/imgsz/class count."""
    import torch
    from ultralytics import YOLO

    model = YOLO(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("train_args", {})
    imgsz = train_args.get("imgsz") or 640
    nc = int(model.model.yaml.get("nc", len(model.names)))
    return {
        "task": model.task,
        "imgsz": int(imgsz),
        "nc": nc,
        "names": model.names,
        "arch": "yolov8",
    }


# ---------- stage 1: ONNX export --------------------------------------------
def patch_onnx_kernel_shapes(onnx_path):
    """Inject missing kernel_shape attrs into Conv/ConvTranspose nodes.

    Newer torch dynamo exporter omits kernel_shape on Conv nodes when the
    weight shape is unambiguous, but the Hailo parser requires it explicit.
    """
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(onnx_path)
    weights = {init.name: init for init in model.graph.initializer}
    fixed = 0
    for node in model.graph.node:
        if node.op_type not in ("Conv", "ConvTranspose"):
            continue
        if any(a.name == "kernel_shape" for a in node.attribute):
            continue
        if len(node.input) < 2 or node.input[1] not in weights:
            continue
        w = numpy_helper.to_array(weights[node.input[1]])
        node.attribute.append(helper.make_attribute("kernel_shape", list(w.shape[2:])))
        fixed += 1
    if fixed:
        onnx.checker.check_model(model)
        onnx.save(model, onnx_path)
        log(f"Patched {fixed} Conv node(s) with missing kernel_shape")


def run_export_onnx(weights, imgsz, onnx_path, force):
    if os.path.exists(onnx_path) and not force:
        log(f"ONNX already exists: {onnx_path}  (use --force to rebuild)")
        return
    section(f"1/4  Export {os.path.basename(weights)} -> ONNX (imgsz={imgsz})")
    from ultralytics import YOLO

    model = YOLO(weights)
    # opset=11 + no simplify keeps Conv kernel_shape attributes that Hailo needs.
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=11,
        simplify=False,
        dynamic=False,
        half=False,
    )
    produced = os.path.splitext(weights)[0] + ".onnx"
    if produced != onnx_path and os.path.exists(produced):
        shutil.move(produced, onnx_path)
    need(os.path.exists(onnx_path), f"ONNX export failed: {onnx_path} not found")
    patch_onnx_kernel_shapes(onnx_path)
    log(f"ONNX ready: {onnx_path}")


# ---------- stage 2: parse ONNX -> HAR --------------------------------------
# Hailo's parser detects the YOLOv8 NMS structure and prints a recommendation of the
# correct pre-DFL output heads. We capture that recommendation automatically so the
# pipeline works for any YOLO variant (v8n/s/m/l/x, v11, different imgsz) — no hardcoding.
# This fallback is only used if auto-detection fails (shouldn't happen with valid ONNX).
YOLOV8_END_NODES_FALLBACK = [
    "node_conv2d_47", "node_conv2d_50", "node_conv2d_53",
    "node_conv2d_56", "node_conv2d_59", "node_conv2d_62",
]


def detect_yolo_end_nodes(runner, onnx_path):
    """Probe-parse to capture Hailo's recommended pre-DFL end nodes for THIS model.

    Hailo emits a line like:
        In order to use HailoRT post-processing capabilities, these end node names
        should be used: node_conv2d_47 node_conv2d_50 ...
    The SDK prints via a logger that bypasses sys.stdout redirection, so we run the
    probe in a SUBPROCESS and capture its real stdout. Works for any YOLO variant
    (v8n/s/m/l/x, v11, different imgsz). Returns: list[str] or None.
    """
    import re
    import subprocess
    import sys

    # Pick a safe intermediate end node to force Hailo's NMS-detection logic.
    # node_cat_16 is the DFL input concat on a standard yolov8 export; if absent
    # (different arch), fall back to the final graph output.
    import onnx
    m = onnx.load(onnx_path)
    node_names = {n.name for n in m.graph.node}
    out_names = [o.name for o in m.graph.output]
    probe_end = "node_cat_16" if "node_cat_16" in node_names else (out_names[0] if out_names else "output0")

    hw_arch = getattr(runner, "_hw_arch", None) or "hailo8l"
    snippet = (
        "import sys\n"
        "from hailo_sdk_client import ClientRunner\n"
        f"r = ClientRunner(hw_arch={hw_arch!r})\n"
        "try:\n"
        f"    r.translate_onnx_model({onnx_path!r}, net_name='probe',\n"
        f"        start_node_names=['images'], end_node_names=[{probe_end!r}])\n"
        "except Exception:\n"
        "    pass\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True, text=True,
    )
    joined = proc.stdout + "\n" + proc.stderr
    m = re.search(
        r"(?:should be used|end node names should be used)[:]\s*([^\n.]+)",
        joined,
    )
    if m:
        names = m.group(1).split()
        names = [n.strip("',\"") for n in names if re.match(r"^[a-zA-Z][a-zA-Z0-9_]+$", n)]
        if names:
            return names
    return None


def run_parse_har(onnx_path, har_path, hw_arch, force):
    if os.path.exists(har_path) and not force:
        log(f"HAR already exists: {har_path}  (use --force to rebuild)")
        return
    section("2/4  Parse ONNX -> Hailo HAR")
    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(hw_arch=hw_arch)

    # Auto-detect the correct pre-DFL end nodes for THIS model (works for any
    # YOLO variant/size/imgsz). Hailo's parser inspects the graph and prints a
    # recommendation; we capture and use it.
    log("Auto-detecting YOLO pre-DFL output heads for this model...")
    end_nodes = detect_yolo_end_nodes(runner, onnx_path)
    if end_nodes:
        log(f"Detected {len(end_nodes)} end nodes: {end_nodes}")
    else:
        end_nodes = YOLOV8_END_NODES_FALLBACK
        log(f"Auto-detect failed; using YOLOv8n fallback: {end_nodes}")

    runner = ClientRunner(hw_arch=hw_arch)
    runner.translate_onnx_model(
        onnx_path,
        net_name="yolov8n_custom",
        start_node_names=["images"],
        end_node_names=end_nodes,
    )
    runner.save_har(har_path)
    log(f"HAR ready: {har_path}")


# ---------- GPU helpers ------------------------------------------------------
def get_gpu_info():
    """Return (total_vram_mb, free_vram_mb) or (None, None) if no NVIDIA GPU."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
        line = out.strip().split("\n")[0].split(", ")
        return int(line[0]), int(line[1])
    except Exception:
        return None, None


def pick_batch_size(requested_bs=None):
    """Choose a safe QFT batch_size based on available GPU memory.

    The Hailo QFT algorithm builds two TF graphs (forward + backward), each
    holding a full copy of the model plus intermediate activations.  On a
    4 GB RTX 2050 with ~3.3 GB free, batch_size=4 is about the limit for
    yolov8n.  Larger models need even smaller batches.

    The user can override this via --batch-size.
    """
    total, free = get_gpu_info()
    if total is None:
        # No GPU detected — DFC will run on CPU, batch_size doesn't matter
        # much for VRAM but keep it small to limit RAM usage.
        return requested_bs or 4

    if requested_bs is not None:
        log(f"Using user-specified batch_size={requested_bs} (GPU: {total} MB total, {free} MB free)")
        return requested_bs

    # Auto-pick based on free VRAM.
    # Empirical: yolov8n QFT needs ~600-700 MB per batch element on GPU.
    if free >= 8000:
        bs = 8
    elif free >= 4000:
        bs = 4
    elif free >= 2000:
        bs = 2
    else:
        bs = 1
    log(f"Auto-selected batch_size={bs} (GPU: {total} MB total, {free} MB free)")
    return bs


# ---------- ALLS model script builder -----------------------------------------
def build_alls_script(batch_size, num_calib, epochs=4):
    """Build an ALLS model-optimization script string.

    The script explicitly enables BOTH bias correction AND QFT with tuned
    parameters.  This avoids the SDK's auto-detection which:
      - At level 1 (dataset < 1024): enables bias correction but SKIPS QFT
      - At level 2 (dataset >= 1024): enables QFT but SKIPS bias correction

    We want both, with batch sizes that fit in the available GPU memory.

    Parameters
    ----------
    batch_size : int
        Batch size for calibration, bias correction, and QFT.
    num_calib : int
        Number of calibration / training images.
    epochs : int
        QFT training epochs (default 4).
    """
    # Clamp dataset_size for QFT to what we actually have.
    dataset_size = min(num_calib, 1024)
    # val_batch_size is independent of training batch_size and also consumes
    # GPU memory.  Keep it proportional but capped to avoid OOM.
    val_batch_size = min(batch_size * 16, 64)
    val_images = min(num_calib * 4, 4096)

    script = (
        # Base calibration config with explicit batch size
        f"model_optimization_config(calibration, batch_size={batch_size}, calibset_size={num_calib})\n"
        # Set optimization level to 2 so the flow runs QFT, but override
        # individual algorithm settings below.
        f"model_optimization_flavor(optimization_level=2, batch_size={batch_size})\n"
        # Bias correction: enabled explicitly (level 2 disables it by default).
        # NOTE: ALLS enum values must NOT be quoted (e.g. policy=enabled).
        f"post_quantization_optimization(bias_correction, "
        f"policy=enabled, batch_size={batch_size}, calibset_size={num_calib})\n"
        # QFT: enabled with reduced batch_size to fit in GPU memory.
        # dataset_size controls how many images QFT uses for training.
        f"post_quantization_optimization(finetune, "
        f"policy=enabled, batch_size={batch_size}, dataset_size={dataset_size}, "
        f"epochs={epochs}, val_batch_size={val_batch_size}, val_images={val_images})\n"
    )
    return script


# ---------- stage 3: calibrate + optimize -----------------------------------
def collect_calib_images(calib_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(calib_dir, ext)))
    return sorted(set(files))


def load_calib(images, imgsz, limit):
    images = images[:limit]
    need(len(images) > 0, "No calibration images found.")
    size = (imgsz, imgsz)
    arrs = []
    for p in images:
        img = Image.open(p).convert("RGB").resize(size, Image.BILINEAR)
        arrs.append(np.asarray(img, dtype=np.float32) / 255.0)
    return np.stack(arrs)


def run_optimize(har_path, opt_har_path, calib_dir, imgsz, num_calib,
                 force, batch_size=None, qft_epochs=4):
    if os.path.exists(opt_har_path) and not force:
        log(f"Optimized HAR already exists: {opt_har_path}  (use --force to rebuild)")
        return
    section(f"3/4  Quantize + optimize  (calib: {calib_dir}, n={num_calib})")
    from hailo_sdk_client import ClientRunner

    images = collect_calib_images(calib_dir)
    calib_data = load_calib(images, imgsz, num_calib)
    log(f"Calibration tensor shape: {calib_data.shape}")

    # Auto-detect safe batch_size for the GPU, or use user override.
    bs = pick_batch_size(batch_size)

    # Build an explicit ALLS script to enable both bias correction AND QFT
    # with batch sizes that fit in the available GPU memory.
    alls_script = build_alls_script(
        batch_size=bs,
        num_calib=num_calib,
        epochs=qft_epochs,
    )
    log("ALLS optimization script:")
    for line in alls_script.strip().split("\n"):
        log(f"  {line}")

    runner = ClientRunner(har=har_path)
    runner.load_model_script(alls_script)
    runner.optimize(calib_data)
    runner.save_har(opt_har_path)
    log(f"Optimized HAR ready: {opt_har_path}")


# ---------- stage 4: compile HAR -> HEF -------------------------------------
def run_compile(opt_har_path, hef_path, force):
    if os.path.exists(hef_path) and not force:
        log(f"HEF already exists: {hef_path}  (use --force to rebuild)")
        return
    section("4/4  Compile HAR -> HEF  (this can take several minutes)")
    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(har=opt_har_path)
    hef_bytes = runner.compile()
    with open(hef_path, "wb") as f:
        f.write(hef_bytes)
    log(f"HEF ready: {hef_path}  ({len(hef_bytes)/1e6:.2f} MB)")


# ---------- subprocess stage dispatcher -------------------------------------
def run_stage_in_subprocess(stage, args):
    """Spawn a fresh python process for one heavy stage (avoids TF-eager state
    accumulation that can deadlock multi-stage runs in a single process)."""
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--stage", stage,
        "--weights", args.weights,
        "--calib-dir", args.calib_dir,
        "--imgsz", str(args.imgsz),
        "--num-calib", str(args.num_calib),
        "--hw-arch", args.hw_arch,
        "--out-dir", args.out_dir,
    ]
    # Forward optional params if set.
    if args.batch_size:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.qft_epochs:
        cmd += ["--qft-epochs", str(args.qft_epochs)]

    log(f"running stage '{stage}' in isolated subprocess...")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        fail(f"Stage '{stage}' failed with exit code {proc.returncode}")


# ---------- single-stage entry (called by the orchestrator) -----------------
def stage_dispatch(args):
    here = os.path.dirname(os.path.abspath(__file__))
    stem = os.path.splitext(os.path.basename(args.weights))[0]
    out_dir = args.out_dir or here
    onnx_path = os.path.join(out_dir, f"{stem}.onnx")
    har_path = os.path.join(out_dir, f"{stem}.har")
    opt_har_path = os.path.join(out_dir, f"{stem}_optim.har")
    hef_path = os.path.join(out_dir, f"{stem}.hef")

    if args.stage == "onnx":
        run_export_onnx(args.weights, args.imgsz, onnx_path, True)
    elif args.stage == "parse":
        run_parse_har(onnx_path, har_path, args.hw_arch, True)
    elif args.stage == "optimize":
        run_optimize(har_path, opt_har_path, args.calib_dir, args.imgsz,
                     args.num_calib, True, args.batch_size, args.qft_epochs)
    elif args.stage == "compile":
        run_compile(opt_har_path, hef_path, True)
    else:
        fail(f"Unknown stage: {args.stage}")


# ---------- full pipeline orchestrator --------------------------------------
def run_pipeline(args):
    here = os.path.dirname(os.path.abspath(__file__))
    stem = os.path.splitext(os.path.basename(args.weights))[0]
    out_dir = args.out_dir or here
    onnx_path = os.path.join(out_dir, f"{stem}.onnx")
    har_path = os.path.join(out_dir, f"{stem}.har")
    opt_har_path = os.path.join(out_dir, f"{stem}_optim.har")
    hef_path = os.path.join(out_dir, f"{stem}.hef")

    section("YOLO -> Hailo HEF converter")
    info = probe_yolo(args.weights)
    imgsz = args.imgsz if args.imgsz else info["imgsz"]
    args.imgsz = imgsz
    print(f"  task    : {info['task']}")
    print(f"  arch    : {info['arch']}")
    print(f"  imgsz   : {imgsz}")
    print(f"  classes : {info['nc']}  {info['names']}")
    print(f"  hw_arch : {args.hw_arch}")
    print(f"  weights : {args.weights}")
    print(f"  calib   : {args.calib_dir}  (using up to {args.num_calib} images)")

    # Each heavy stage in its own process -> no TF-eager state accumulation,
    # so --force rebuilds are as reliable as a cold start.
    if not (os.path.exists(onnx_path) and not args.force):
        run_stage_in_subprocess("onnx", args)
    else:
        log(f"ONNX already exists: {onnx_path}  (use --force to rebuild)")

    if not (os.path.exists(har_path) and not args.force):
        run_stage_in_subprocess("parse", args)
    else:
        log(f"HAR already exists: {har_path}  (use --force to rebuild)")

    if not (os.path.exists(opt_har_path) and not args.force):
        run_stage_in_subprocess("optimize", args)
    else:
        log(f"Optimized HAR already exists: {opt_har_path}  (use --force to rebuild)")

    if not (os.path.exists(hef_path) and not args.force):
        run_stage_in_subprocess("compile", args)
    else:
        log(f"HEF already exists: {hef_path}  (use --force to rebuild)")

    section("DONE")
    print(f"  {COLOR_GREEN}HEF:{COLOR_RESET} {hef_path}")
    print(f"  size: {os.path.getsize(hef_path)/1e6:.2f} MB")
    print("  Copy the .hef to your Raspberry Pi and load it with HailoRT.\n")


# ---------- main -------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Convert a YOLOv8 .pt model to a Hailo HEF (Hailo-8L)."
    )
    parser.add_argument("--weights", dest="weights",
                        default=os.path.join(here, "exp-2.pt"),
                        help="Path to the YOLOv8 .pt checkpoint (default: ./exp-2.pt)")
    parser.add_argument("weights_pos", nargs="?", default=None,
                        help="Positional alias for --weights")
    parser.add_argument("--stage", default=None,
                        choices=["onnx", "parse", "optimize", "compile"],
                        help="INTERNAL: run only one stage (used by subprocess dispatch).")
    parser.add_argument("--calib-dir", default=os.path.join(here, "dataset_yolo", "images", "val"),
                        help="Folder of images used for calibration (default: ./dataset_yolo/images/val)")
    parser.add_argument("--imgsz", type=int, default=0,
                        help="Input size (default: read from checkpoint, else 640)")
    parser.add_argument("--num-calib", type=int, default=300,
                        help="Number of calibration images (default: 300; 1024+ for best accuracy)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for calibration/bias-correction/QFT (default: auto-detect from GPU VRAM)")
    parser.add_argument("--qft-epochs", type=int, default=4,
                        help="QFT training epochs (default: 4; more epochs = better accuracy, slower)")
    parser.add_argument("--hw-arch", default="hailo8l", choices=["hailo8l", "hailo8"],
                        help="Target hardware arch (default: hailo8l for the Pi 5 AI Hat)")
    parser.add_argument("--out-dir", default=here, help="Where to write artifacts (default: this folder)")
    parser.add_argument("--force", action="store_true", help="Rebuild every stage")
    args = parser.parse_args()

    # Positional weights overrides --weights flag.
    if args.weights_pos:
        args.weights = args.weights_pos
    # Resolve/normalize args early.
    if not os.path.isabs(args.weights):
        args.weights = os.path.abspath(args.weights)
    if not os.path.isabs(args.calib_dir):
        args.calib_dir = os.path.abspath(args.calib_dir)
    args.out_dir = os.path.abspath(args.out_dir)

    # If imgsz not specified, fill it from the checkpoint (needed by subprocesses).
    if not args.imgsz and not args.stage:
        try:
            args.imgsz = probe_yolo(args.weights)["imgsz"]
        except Exception:
            args.imgsz = 640
    elif not args.imgsz:
        args.imgsz = 640  # stage subprocess path: orchestrator already resolved it

    if args.stage:
        # Single-stage worker process. Minimal validation, run the stage.
        need(os.path.exists(args.weights), f"Weights not found: {args.weights}")
        stage_dispatch(args)
        return

    need(os.path.exists(args.weights), f"Weights not found: {args.weights}")
    need(os.path.isdir(args.calib_dir), f"Calib dir not found: {args.calib_dir}")
    run_pipeline(args)


if __name__ == "__main__":
    main()
