# YOLOv8 → Hailo HEF Converter

Convert an Ultralytics YOLOv8 model (`.pt`) into a Hailo HEF file ready to run on:

* Raspberry Pi 5
* Hailo AI Hat (Hailo-8L)
* HailoRT 5.3+

The converter performs the entire pipeline automatically:

```text
.pt → ONNX → HAR → Optimized HAR → HEF
```

---

# Requirements

## Host Machine

Supported conversion host:

* Ubuntu 22.04 / 24.04 x86_64
* Python 3.10
* 8 GB RAM minimum
* NVIDIA GPU (optional but recommended)

## Target Device

* Raspberry Pi 5
* Hailo AI Hat (Hailo-8L)
* HailoRT 5.3+

---

# Required Downloads

Due to Hailo licensing restrictions, required Hailo packages are NOT included in this repository.

Download the following files manually from the Hailo Developer Zone:

## 1. Hailo Dataflow Compiler (Required)

Download:

```text
hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
```

Place it in the project root:

```text
project/
├── convert.sh
├── convert_hef.py
├── hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
└── ...
```

## 2. HailoRT Driver (Deployment)

Download:

```text
hailort-pcie-driver_5.3.0_all.deb
```

Required only on Raspberry Pi deployment targets.

## 3. Hailo Model Zoo (Optional)

Download:

```text
hailo_gen_ai_model_zoo_5.3.0_amd64.deb
```

Optional.

Not required for HEF compilation.

---

# System Dependencies

Install required Ubuntu packages:

```bash
sudo apt update

sudo apt install -y \
    python3.10 \
    python3.10-venv \
    build-essential \
    graphviz \
    libgraphviz-dev \
    gcc
```

---

# Quick Start

Clone repository:

```bash
git clone https://github.com/erqyan/Easy-Convert-Pytorch-To-HEF.git
cd Easy-Convert-Pytorch-To-HEF
```

Put:

```text
hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
```

into the project directory.

Run:

```bash
chmod +x convert.sh

./convert.sh
```

or:

```bash
./convert.sh my_model.pt
```

The script automatically:

1. Creates a Python 3.10 virtual environment.
2. Installs Hailo DFC.
3. Installs Ultralytics YOLO.
4. Applies compatible dependency pinning.
5. Exports ONNX.
6. Parses ONNX into HAR.
7. Calibrates and quantizes.
8. Compiles HEF.

Output:

```text
my_model.hef
```

---

# Calibration Dataset

Default calibration folder:

```text
dataset_yolo/images/val
```

Use your own dataset:

```bash
./convert.sh \
    --calib-dir /path/to/images \
    --num-calib 1024 \
    --force
```

Recommended:

* 300 images = fast
* 1024+ images = best accuracy

---

# GPU Acceleration (Optional)

Hailo DFC uses TensorFlow internally.

With NVIDIA GPU:

* Bias Correction enabled
* Better quantization accuracy
* Faster optimization

Install CUDA runtime, cuDNN, and nvcc:

```bash
sudo apt install -y \
    libcublas-13-1 \
    nvidia-cudnn \
    libcufft-13-1 \
    libcusparse-13-1 \
    libcusolver-13-1 \
    libcurand-13-1 \
    libnvjitlink-13-1 \
    cuda-nvcc-13-1 \
    cuda-libraries-13-1
```

The script automatically:

* Patches Hailo GPU threshold (5% → 80%)
* Configures XLA_FLAGS
* Enables TensorFlow GPU execution

No manual configuration required.

---

# Deploy to Raspberry Pi 5

Copy:

```text
my_model.hef
```

to Raspberry Pi.

Install HailoRT:

```bash
sudo dpkg -i hailort-pcie-driver_5.3.0_all.deb
```

Run inference using:

* HailoRT
* Hailo TAPPAS
* hailo-rpi5-examples

---

# Troubleshooting

### pygraphviz build failed

```bash
sudo apt install graphviz libgraphviz-dev build-essential
```

### python3.10 not found

```bash
sudo apt install python3.10 python3.10-venv
```

### Hailo DFC wheel missing

Make sure:

```text
hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
```

exists in the project root.

### No GPU chosen

Not an error.

The conversion will continue on CPU.

### HEF not generated

Run:

```bash
./convert.sh --force
```

and inspect:

```text
*.har
*_optim.har
```

for failures.
