# SKMamba

Official implementation for **Structure-aware Knowledge-guided Heterogeneous Mamba for Zygomaticomaxillary Suture Assessment**, MICCAI 2026.

SKMamba targets automated zygomaticomaxillary suture maturation staging from CBCT images.

## Status

Code and dataset are being organized and will be released soon.

## Project Structure

```text
SKMamba/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── configs/
│   └── skmamba.yaml
├── scripts/
│   └── train.py
├── models/
│   ├── __init__.py
│   └── skmamba.py
├── datasets/
│   ├── __init__.py
│   └── zms_dataset.py
├── utils/
│   ├── __init__.py
│   ├── seed.py
│   ├── metrics.py
│   └── visualization.py
├── splits/
│   └── README.md
├── features/
│   └── README.md
├── checkpoints/
│   └── iee_mambaout.pth
├── data/
│   └── README.md
└── outputs/
    └── .gitkeep
```

## Installation

```bash
git clone <repository-url>
cd SKMamba
```

Create and activate a conda environment:

```bash
conda create -n skmamba python=3.10 -y
conda activate skmamba
```

Install PyTorch according to your local CUDA environment. For example, on a CUDA 12.9 system:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

If your CUDA version is different, please choose the matching PyTorch installation command from the [official PyTorch installation page](https://pytorch.org/get-started/locally/).

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Check the environment:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Run training with the default configuration:

```bash
python scripts/train.py --config configs/skmamba.yaml
```

## Data Preparation

Place the ZMS classification dataset under:

```text
data/ZMS_classification/
├── A/
├── B/
├── C/
├── D/
└── E/
```

Prepare split files under `splits/`. Each line should contain a relative image path such as:

```text
A/0012 LL.bmp
A/0012 LU.bmp
A/0012 RL.bmp
A/0012 RU.bmp
```

Dataset release note: the dataset will be released via Google Drive and the access link will be provided in this repository.

## Text Feature Preparation

Place the text feature files under `features/`:

```text
features/zms_text_features.npy
features/descriptions.csv
```

The text features and descriptions file are being organized and will be released soon.

## Training

Update paths in `configs/skmamba.yaml` if needed, then run:

```bash
python scripts/train.py --config configs/skmamba.yaml
```

The default configuration saves checkpoints, logs, and figures under `outputs/`.

## Citation

```bibtex
@inproceedings{skmamba2026,
  title={Structure-aware Knowledge-guided Heterogeneous Mamba for Zygomaticomaxillary Suture Assessment},
  author={Author names},
  booktitle={MICCAI},
  year={2026}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
