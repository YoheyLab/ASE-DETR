# ASE-DETR

An Enhanced Detection Transformer for Infrared Vehicle-Pedestrian Detection

## 🚀 Quick Start 

### Requirements 
- Python 3.10.14
- PyTorch 2.2.2
- CUDA 12.1 (for GPU support)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YoheyLab/ASE-DETR.git
cd ASE-DETR
```

2. **Create conda environment**
```bash
conda create -n ase_detr python=3.10.14
conda activate ase_detr
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📂 Dataset

We use [FLIR ADAS Thermal Dataset](https://www.flir.com/oem/adas/adas-dataset-form/) for training and evaluation.

### Dataset Structure
```
dataset/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── labels/
│       ├── img_001.txt
│       └── ...
└── val/
    ├── images/
    └── labels/
```

## 🎯 Training

```bash
cd ultralytics
python train.py
```

## 📚 Citation 

If you find our code helpful in your research, please consider citing:

```bibtex
@misc{asedetr2025,
  author = {Wang, Yuhang and Zhang, Wenqiang and Yu, Junwei and Zhang, Mengya and Mu, Yashuang},
  title = {ASE-DETR},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/YoheyLab/ASE-DETR}},
}
```

Paper under review. This will be updated upon acceptance.
