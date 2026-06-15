import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MultimodalZMSDataset(Dataset):
    def __init__(
        self,
        file_list,
        text_feat_dict,
        transform=None,
        img_size=224,
        text_dim=768,
    ):
        self.file_list = file_list
        self.text_feat_dict = text_feat_dict
        self.transform = transform
        self.img_size = img_size
        self.text_dim = text_dim
        self.label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label_name = self.file_list[idx]
        file_name = os.path.basename(img_path)
        unique_key = f"{label_name}_{file_name}"

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except OSError as exc:
            print(f"Failed to load image {img_path}: {exc}")
            image = torch.zeros((3, self.img_size, self.img_size))

        text_feat = self.text_feat_dict.get(
            unique_key,
            np.zeros(self.text_dim, dtype=np.float32),
        )
        text_feat = torch.from_numpy(text_feat.astype(np.float32)).float()
        return image, text_feat, self.label_map[label_name]


def load_data_and_text(txt_path, feat_path, csv_path, data_root, text_dim=768):
    data_list = _load_split(txt_path, data_root)
    text_dict = _load_text_features(feat_path, csv_path, text_dim)

    if data_list and text_dict:
        matched_count = sum(
            1
            for path, label in data_list
            if f"{label}_{os.path.basename(path)}" in text_dict
        )
        print(
            "Text feature match rate: "
            f"{matched_count}/{len(data_list)} ({matched_count / len(data_list):.2%})"
        )

    return data_list, text_dict


def _load_split(txt_path, data_root):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Missing split file: {txt_path}")

    data_list = []
    with open(txt_path, "r", encoding="utf-8") as file:
        for line in file:
            item = line.strip()
            if not item:
                continue
            normalized = item.replace("\\", "/")
            parts = normalized.split("/")
            if len(parts) < 2:
                raise ValueError(
                    "Each split entry must include a class directory, "
                    f"for example A/example.png. Got: {item}"
                )
            category = parts[-2]
            if category not in {"A", "B", "C", "D", "E"}:
                raise ValueError(f"Unsupported class label in split entry: {item}")
            image_path = normalized if os.path.isabs(normalized) else os.path.join(data_root, normalized)
            data_list.append((image_path, category))
    return data_list


def _load_text_features(feat_path, csv_path, text_dim):
    if not os.path.exists(feat_path) or not os.path.exists(csv_path):
        print("Text features were not found. Zero vectors will be used.")
        return {}

    feats = np.load(feat_path).astype(np.float32)
    if feats.ndim != 2 or feats.shape[1] != text_dim:
        raise ValueError(
            f"Expected text features with shape [N, {text_dim}], got {feats.shape}."
        )

    keys = []
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                keys.append(f"{row[1].strip()}_{row[0].strip()}")

    min_len = min(len(keys), len(feats))
    text_dict = {key: value for key, value in zip(keys[:min_len], feats[:min_len])}
    print(f"Loaded {len(text_dict)} text feature entries.")
    return text_dict
