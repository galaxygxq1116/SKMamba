import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import MultimodalZMSDataset, load_data_and_text
from models import ParallelExpertNet
from utils import plot_confusion_matrix, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train SKMamba.")
    parser.add_argument(
        "--config",
        default="configs/skmamba.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(config_path):
    config_path = Path(config_path).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    with open(config_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    return cfg, config_path.parent.parent


def resolve_path(path_value, project_root):
    path = Path(path_value).expanduser()
    return str(path if path.is_absolute() else project_root / path)


def prepare_config(cfg, project_root):
    for section, keys in {
        "data": ["data_root", "train_txt", "val_txt", "text_feat_path", "csv_path"],
        "checkpoint": ["iee_pretrained_path", "save_path"],
        "output": ["log_path", "plot_cm_path"],
    }.items():
        for key in keys:
            cfg[section][key] = resolve_path(cfg[section][key], project_root)
    return cfg


def center_crop_tensor(x, crop_ratio):
    _, _, height, width = x.shape
    crop_height = int(height * crop_ratio)
    crop_width = int(width * crop_ratio)
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    cropped = x[:, :, top : top + crop_height, left : left + crop_width]
    return F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)


def build_transforms(img_size):
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform


def build_optimizer(model, cfg):
    train_cfg = cfg["train"]
    return optim.AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr": float(train_cfg["backbone_lr"]),
                "name": "backbone",
            },
            {
                "params": model.edge_extractor.parameters(),
                "lr": float(train_cfg["fusion_lr"]),
                "name": "edge_extractor",
            },
            {
                "params": model.fusion_conv.parameters(),
                "lr": float(train_cfg["fusion_lr"]),
                "name": "fusion_conv",
            },
            {
                "params": model.text_expert_s16.parameters(),
                "lr": float(train_cfg["fusion_lr"]),
                "name": "text_expert_s16",
            },
            {
                "params": model.se_block.parameters(),
                "lr": float(train_cfg["head_lr"]),
                "name": "se_block",
            },
            {
                "params": model.head.parameters(),
                "lr": float(train_cfg["head_lr"]),
                "name": "head",
            },
        ],
        weight_decay=float(train_cfg["weight_decay"]),
    )


def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, text_feats, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            text_feats = text_feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits_original = model(imgs, text_feats)
            logits_crop = model(center_crop_tensor(imgs, 0.95), text_feats)
            logits = (logits_original + logits_crop) / 2.0

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100.0 * correct / total if total else 0.0
    return val_acc, all_labels, all_preds


def train(cfg):
    set_seed(int(cfg["train"]["seed"]))

    device_name = cfg["train"]["device"]
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"Training device: {device}")

    os.makedirs(os.path.dirname(cfg["checkpoint"]["save_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(cfg["output"]["log_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(cfg["output"]["plot_cm_path"]), exist_ok=True)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    train_files, text_dict = load_data_and_text(
        data_cfg["train_txt"],
        data_cfg["text_feat_path"],
        data_cfg["csv_path"],
        data_cfg["data_root"],
        text_dim=int(model_cfg["text_dim"]),
    )
    val_files, _ = load_data_and_text(
        data_cfg["val_txt"],
        data_cfg["text_feat_path"],
        data_cfg["csv_path"],
        data_cfg["data_root"],
        text_dim=int(model_cfg["text_dim"]),
    )

    train_transform, val_transform = build_transforms(int(model_cfg["img_size"]))
    train_loader = DataLoader(
        MultimodalZMSDataset(
            train_files,
            text_dict,
            train_transform,
            img_size=int(model_cfg["img_size"]),
            text_dim=int(model_cfg["text_dim"]),
        ),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        MultimodalZMSDataset(
            val_files,
            text_dict,
            val_transform,
            img_size=int(model_cfg["img_size"]),
            text_dim=int(model_cfg["text_dim"]),
        ),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = ParallelExpertNet(
        backbone_name=model_cfg["backbone_name"],
        num_classes=int(model_cfg["num_classes"]),
        text_dim=int(model_cfg["text_dim"]),
        img_size=int(model_cfg["img_size"]),
        drop_path_rate=float(model_cfg["drop_path_rate"]),
        iee_checkpoint=cfg["checkpoint"]["iee_pretrained_path"],
        freeze_iee=True,
    ).to(device)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")

    optimizer = build_optimizer(model, cfg)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg["epochs"]),
        eta_min=float(train_cfg["eta_min"]),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg["label_smoothing"]))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_acc = 0.0
    history = {"train_loss": [], "val_acc": []}
    class_names = ["A", "B", "C", "D", "E"]

    for epoch in range(int(train_cfg["epochs"])):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for imgs, text_feats, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            text_feats = text_feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(imgs, text_feats)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()
        val_acc, all_labels, all_preds = validate(model, val_loader, device)
        avg_loss = running_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{train_cfg['epochs']} | "
            f"Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {elapsed:.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg["checkpoint"]["save_path"])
            plot_confusion_matrix(
                all_labels,
                all_preds,
                class_names,
                cfg["output"]["plot_cm_path"],
            )
            print(f"Best model saved with validation accuracy {best_acc:.2f}%.")

    pd.DataFrame(history).to_csv(cfg["output"]["log_path"], index=False)


def main():
    args = parse_args()
    cfg, project_root = load_config(args.config)
    cfg = prepare_config(cfg, project_root)
    train(cfg)


if __name__ == "__main__":
    main()
