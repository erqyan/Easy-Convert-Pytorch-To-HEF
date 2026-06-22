# YOLOv8 → Hailo HEF Converter

Konversi model **Ultralytics YOLOv8** (`.pt`) menjadi **Hailo HEF** siap pakai di
**Raspberry Pi 5 + Hailo AI Hat (Hailo-8L)** — dengan satu perintah.

---

## 📁 Struktur Folder

| File / Folder | Deskripsi |
|---------------|-----------|
| `convert.sh` | Entry point utama. Cek venv, install/verifikasi dependencies, lalu jalankan pipeline. |
| `convert_hef.py` | Pipeline Python: `.pt → ONNX → HAR → optimized HAR → HEF`. Setiap tahap berat berjalan di subprocess terpisah agar anti-hang. |
| `hailo_venv/` | Virtualenv Python 3.10 (dibuat otomatis). Semua dependency ter-pin di sini. |
| `dataset_yolo/` | Dataset kamu. `images/val` dipakai untuk kalibrasi. |
| `hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl` | Hailo DFC wheel (disediakan). |
| `hailort-pcie-driver_5.3.0_all.deb` | Driver HailoRT (untuk Pi 5). |
| `hailo_gen_ai_model_zoo_5.3.0_amd64.deb` | Hailo Model Zoo (opsional). |
| `exp-2.pt` | Model YOLOv8 contoh (ganti dengan modelmu). |

---

## 🚀 Quick Start

```bash
./convert.sh                 # convert ./exp-2.pt dengan pengaturan default
./convert.sh my_model.pt     # convert model lain
```

Selesai. Script akan otomatis melakukan:

1. **Membuat/memeriksa** virtualenv Python 3.10.
2. **Menginstal** Hailo DFC + ultralytics + pin versi yang kompatibel.
3. **Mengekspor** `.pt → ONNX` dan mem-patch Conv nodes (fix bug torch 2.12).
4. **Mem-parse** `ONNX → HAR` menggunakan 6 pre-DFL output heads untuk HailoRT NMS.
5. **Mengkalibrasi** dengan gambar dari `dataset_yolo/images/val`, lalu optimize (quantize).
6. **Mengompilasi** `HAR → HEF` untuk Hailo-8L (multi-context allocation).

**Output:** `exp-2.hef` → copy ke Raspberry Pi 5, load dengan HailoRT.

---

## ⚙️ Opsi Command Line

```bash
./convert.sh --help
```

| Flag | Default | Keterangan |
|------|---------|------------|
| `WEIGHTS.pt` (positional) | `exp-2.pt` | Model yang ingin dikonversi |
| `--calib-dir DIR` | `dataset_yolo/images/val` | Folder gambar kalibrasi |
| `--num-calib N` | `300` | Jumlah gambar kalibrasi (1024+ disarankan untuk akurasi terbaik) |
| `--imgsz N` | dari checkpoint | Ukuran input gambar |
| `--hw-arch ARCH` | `hailo8l` | `hailo8l` (Pi 5 AI Hat) atau `hailo8` |
| `--out-dir DIR` | folder ini | Lokasi output artifact |
| `--force` | off | Rebuild semua tahap dari awal |
| `--setup-only` | off | Hanya buat/verifikasi venv + deps, lalu exit |

### Contoh

```bash
# Convert model lain dengan kalibrasi maksimal
./convert.sh my_model.pt --num-calib 1024 --force

# Hanya setup environment tanpa convert
./convert.sh --setup-only

# Pakai folder kalibrasi custom
./convert.sh --calib-dir /path/to/my/images
```

---

## 🔁 Menjalankan Ulang

Setiap artifact intermediate (`.onnx`, `.har`, `_optim.har`, `.hef`) **di-reuse** saat
run berikutnya, jadi re-run selesai instan. Gunakan `--force` untuk rebuild
salah satu/semua tahap.

---

## 🧠 Pipeline Detail

```
exp-2.pt ──[stage 1]──> exp-2.onnx ──[stage 2]──> exp-2.har
                                                        │
                                                        ▼
         exp-2.hef <──[stage 4]── exp-2_optim.har <──[stage 3]── (kalibrasi)
```

| Tahap | Proses terpisah? | Fungsi |
|-------|------------------|--------|
| 1. ONNX export | ✓ subprocess | YOLOv8 `.pt` → ONNX (opset 11, no simplify) + patch kernel_shape |
| 2. Parse HAR | ✓ subprocess | ONNX → Hailo HAR dengan pre-DFL output heads |
| 3. Optimize | ✓ subprocess | Kalibrasi + quantization (8-bit) → optimized HAR |
| 4. Compile | ✓ subprocess | HAR → HEF (multi-context allocation, kernel compile) |

> **Kenapa setiap tahap pakai subprocess terpisah?**
> Hailo DFC optimizer/allocator dibangun di atas TensorFlow eager mode. Menjalankan
> ONNX export + parse + optimize + compile dalam **satu proses Python** bisa
> menumpuk state TF dan **deadlock** (terverifikasi empiris) saat di-rerun dengan
> `--force`. Dengan subprocess terisolasi, setiap run seandal cold start.

---

## 🎯 Tips Akurasi Terbaik

- Gunakan **1024+ gambar kalibrasi** yang representatif dengan data deployment:
  ```bash
  ./convert.sh --num-calib 1024 --force
  ```
- Set kalibrasi harus mencakup pencahayaan/sudut/scene yang akan ditemui model di produksi.
- Di mesin **tanpa GPU**, optimizer jalan di level 0 (warning normal). Dengan GPU,
  optimizer jalan di level 1+ dengan **Bias Correction** — algoritma akurasi yang
  memperbaiki bias setelah quantization. Ini signifikan meningkatkan akurasi deteksi.

---

## 🖥️ Enable GPU (Opsional — Optimasi Lebih Baik)

Hailo DFC menggunakan TensorFlow untuk optimasi. GPU memberikan **Bias Correction**
(algoritma perbaikan akurasi setelah quantization) yang tidak jalan di CPU.

### Langkah Install

```bash
# 1. Install CUDA runtime + cuDNN + nvcc (Ubuntu)
sudo apt install -y libcublas-13-1 nvidia-cudnn libcufft-13-1 \
    libcusparse-13-1 libcusolver-13-1 libcurand-13-1 libnvjitlink-13-1 \
    cuda-nvcc-13-1 cuda-libraries-13-1
```

> `convert.sh` otomatis:
> - Patch threshold GPU Hailo (5% → 80%) agar GPU laptop dengan display tetap terdeteksi
> - Set `XLA_FLAGS` agar TF XLA menemukan `libdevice.10.bc` dari CUDA toolkit
>
> Jadi setelah install CUDA packages di atas, langsung jalankan `./convert.sh` — GPU akan otomatis dipakai.

### Verifikasi

```
[info] No GPU chosen, Selected GPU 0        ← GPU terdeteksi
[info] Starting Bias Correction            ← hanya jalan dengan GPU
```

Jika masih muncul `No GPU chosen and no suitable GPU found`, berarti memory GPU kamu
sudah kepenuhan (>80%). Tutup aplikasi lain yang pakai GPU, lalu coba lagi.

---

## 📤 Deploy HEF ke Raspberry Pi 5

1. Copy `exp-2.hef` ke Raspberry Pi 5.
2. Install HailoRT di Pi:
   ```bash
   sudo dpkg -i hailort-pcie-driver_5.3.0_all.deb
   ```
3. Load HEF dengan HailoRT — gunakan pipeline inference `hailo-rpi5-examples`
   atau Hailo TAPPAS.

---

## 🐛 Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `pygraphviz build failed` | `sudo apt install build-essential libgraphviz-dev graphviz` |
| `python3.10 not found` | `sudo apt install python3.10 python3.10-venv` |
| `No GPU chosen` warning | Lihat section **Enable GPU** di atas (opsional, bukan error) |
| Parsing error / output salah | Re-run dengan `--force`, cek log untuk rekomendasi end-node YOLOv8 |
| Optimizer hang saat `--force` | Pastikan pakai versi `convert_hef.py` terbaru (subprocess arch) |
| HEF tidak terbentuk | Cek `exp-2.har` dan `exp-2_optim.har` ada; re-run `--force` |

---

## 📋 Requirements

- **Host (convert machine):** Linux x86_64, Python 3.10, ~8GB RAM, optional NVIDIA GPU
- **Target (deployment):** Raspberry Pi 5 + Hailo AI Hat (Hailo-8L), HailoRT 5.3.0+
- **Model:** Ultralytics YOLOv8 detection (`.pt`)

---

## 📦 Dependencies (auto-installed)

Hailo DFC 3.33.1 + ultralytics 8.3.96 dengan pin versi:
- `protobuf==3.20.3`, `numpy==1.26.4`, `scipy==1.12.0`
- `onnx==1.16.0`, `onnxruntime==1.18.0`, `onnxsim==0.4.36`
- `typing-extensions==4.12.2`, `opencv-python==4.10.0.84`

Semua diinstal otomatis di `hailo_venv/` saat pertama kali run.
