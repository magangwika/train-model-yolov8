
# Training  Dataset YOLOv8

Untuk mulai training dataset pada YOLOv8 di butuhkan beberapa persiapan seperti berikut:




## Tech Stack

**OS:** Windows / Linux (Ubuntu)

**Programming Language:** Python (Latest) tested on 3.10.7, Virtual Environment (venv, anaconda)

**Device:** Device has GPU with CUDA version (11.8, 12.1, 12.4)



## Installation

**Install virtual environment dan packages yang dibutuhkan** (Windows Version)

```bash
  python -m venv .venv
  ./venv/Scripts/activate
```
Linux Version
```bash
  python -m venv .venv
  bash ./venv/bin/activate
```

**Install packages pytorch**. Untuk packages torch, sesuaikan dengan CUDA version yang device anda miliki. selengkapnya https://pytorch.org/get-started/locally/ untuk melihat versi packages lain dari CUDA yang anda miliki

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Install packages ultralytics**. ultralytics adalah packages all in one (AIO) yang berfungsi untuk proses training dataset dan juga proses convert model YOLO / AI (.pt) menjadi beberapa version. seperti onnx dan sebagainya.

```bash
pip install ultralytics
```




    