import argparse
import csv
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def distributed_setup():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    return world, local_rank, torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def source_group(path: Path) -> str:
    stem = re.sub(r",\d+\)$", ")", path.stem)
    return f"{path.parent.as_posix()}::{stem}"


def scan_dataset(root: Path):
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class folders were found under: {root}")
    classes = [p.name for p in class_dirs]
    records = []
    for label, class_dir in enumerate(class_dirs):
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append((str(path), label, source_group(path)))
    if not records:
        raise RuntimeError("No supported image files were found.")
    return records, classes


class CCIDDataset(Dataset):
    def __init__(self, records, transform): self.records, self.transform = records, transform
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        path, label, group = self.records[index]
        try:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Unreadable image: {path}") from exc
        return image, int(label), path, group

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=8, spline_order=3):
        super().__init__()
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size))
        self.grid_size, self.spline_order = grid_size, spline_order
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        nn.init.normal_(self.spline_weight, std=0.01)
    def forward(self, x):
        base = F.linear(F.silu(x), self.base_weight, self.base_bias)
        x = x.clamp(-3, 3)
        centers = torch.linspace(-3, 3, self.grid_size, device=x.device, dtype=x.dtype)
        width = 6 / max(self.grid_size - 1, 1)
        basis = (1 - (x.unsqueeze(-1) - centers).abs() / width).clamp_min(0).pow(self.spline_order)
        return base + torch.einsum("big,oig->bo", basis, self.spline_weight)


class HybridBackboneKAN(nn.Module):
    def __init__(self, architecture: str, num_classes: int, dropout: float):
        super().__init__()
        if architecture == "convnext_large":
            self.backbone = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
            dim = self.backbone.classifier[-1].in_features; self.backbone.classifier[-1] = nn.Identity()
        elif architecture == "efficientnet_v2_l":
            self.backbone = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
            dim = self.backbone.classifier[-1].in_features; self.backbone.classifier[-1] = nn.Identity()
        elif architecture == "swin_v2_b":
            self.backbone = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT)
            dim = self.backbone.head.in_features; self.backbone.head = nn.Identity()
        elif architecture == "densenet201":
            self.backbone = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
            dim = self.backbone.classifier.in_features; self.backbone.classifier = nn.Identity()
        elif architecture == "regnet_y_16gf":
            self.backbone = models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.DEFAULT)
            dim = self.backbone.fc.in_features; self.backbone.fc = nn.Identity()
        else: raise ValueError(architecture)
        self.projection = nn.Sequential(nn.Linear(dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout))
        self.kan = nn.Sequential(KANLinear(512, 128), nn.GELU(), nn.Dropout(dropout), KANLinear(128, num_classes))
    def forward(self, x): return self.kan(self.projection(self.backbone(x)))


def backbone_last_conv(model):
    convs = [m for m in model.backbone.modules() if isinstance(m, nn.Conv2d)]
    if not convs: raise RuntimeError("No convolutional layer found for Grad-CAM.")
    return convs[-1]


class GradCAM:
    def __init__(self, model):
        self.model, self.activations, self.gradients = model, None, None
        layer = backbone_last_conv(model)
        layer.register_forward_hook(lambda m, i, o: setattr(self, "activations", o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "gradients", go[0].detach()))
    def __call__(self, image, target=None):
        self.model.eval(); self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        target = logits.argmax(1) if target is None else target
        logits.gather(1, target[:, None]).sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), logits.detach()


def make_loaders(train_records, val_records, args, world, rank):
    train_tf = transforms.Compose([transforms.RandomResizedCrop(args.image_size, scale=(0.72, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
        transforms.ColorJitter(0.15, 0.15, 0.10, 0.03), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.12))])
    val_tf = transforms.Compose([transforms.Resize(int(args.image_size * 1.14)), transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    train_ds, val_ds = CCIDDataset(train_records, train_tf), CCIDDataset(val_records, val_tf)
    train_sampler = DistributedSampler(train_ds, world, rank, shuffle=True, seed=args.seed) if world > 1 else None
    val_sampler = DistributedSampler(val_ds, world, rank, shuffle=False, drop_last=False) if world > 1 else None
    common = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    return (DataLoader(train_ds, shuffle=train_sampler is None, sampler=train_sampler, drop_last=True, **common),
            DataLoader(val_ds, shuffle=False, sampler=val_sampler, **common), train_sampler)


def gather_predictions(local):
    if not dist.is_initialized(): return local
    gathered = [None] * dist.get_world_size(); dist.all_gather_object(gathered, local)
    return [item for shard in gathered for item in shard]


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); output = []
    for images, labels, paths, groups in loader:
        probs = F.softmax(model(images.to(device, non_blocking=True)), dim=1).cpu().numpy()
        output.extend(zip(paths, groups, labels.numpy().tolist(), probs.tolist()))
    # DistributedSampler pads to equal length. A path-keyed dict removes repeats.
    return list({row[0]: row for row in gather_predictions(output)}.values())


def train_fold(architecture, fold, train_records, val_records, classes, args, world, rank, device, run_dir):
    train_loader, val_loader, train_sampler = make_loaders(train_records, val_records, args, world, rank)
    model = HybridBackboneKAN(architecture, len(classes), args.dropout).to(device)
    if world > 1: model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    counts = np.bincount([x[1] for x in train_records], minlength=len(classes))
    weights = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(5, args.epochs // 3), T_mult=1)
    scaler = GradScaler(enabled=device.type == "cuda")
    best_auc, bad_epochs, checkpoint = -1.0, 0, run_dir / "checkpoints" / f"{architecture}_fold{fold}.pt"
    for epoch in range(args.epochs):
        model.train()
        if train_sampler: train_sampler.set_epoch(epoch)
        for images, labels, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                loss = criterion(model(images.to(device, non_blocking=True)), labels.to(device, non_blocking=True))
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        scheduler.step(epoch + 1)
        rows = evaluate(model, val_loader, device)
        if is_main():
            y, p = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
            auc = roc_auc_score(y, p, multi_class="ovr", average="macro")
            print(f"{architecture} | fold {fold} | epoch {epoch+1:03d} | val macro-AUC {auc:.4f}")
            if auc > best_auc:
                best_auc, bad_epochs = auc, 0
                torch.save({"architecture": architecture, "fold": fold, "classes": classes, "state_dict": unwrap(model).state_dict(), "auc": auc}, checkpoint)
            else: bad_epochs += 1
        stop = torch.tensor([bad_epochs >= args.patience], device=device, dtype=torch.int32)
        if dist.is_initialized(): dist.broadcast(stop, 0)
        if stop.item(): break
    if dist.is_initialized(): dist.barrier()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    unwrap(model).load_state_dict(state["state_dict"])
    return model, evaluate(model, val_loader, device), checkpoint


def metrics_table(rows, classes):
    y = np.array([r[2] for r in rows]); p = np.array([r[3] for r in rows]); pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=np.arange(len(classes)))
    result = []
    for i, name in enumerate(classes):
        tp, fn, fp = cm[i, i], cm[i].sum() - cm[i, i], cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - tp - fn - fp
        sen = tp / (tp + fn) if tp + fn else 0.; spe = tn / (tn + fp) if tn + fp else 0.
        pre = tp / (tp + fp) if tp + fp else 0.; f1 = 2 * pre * sen / (pre + sen) if pre + sen else 0.
        result.append({"Class": name, "ACC": (tp+tn)/cm.sum(), "SEN": sen, "SPE": spe, "PRE": pre, "RECALL": sen, "F1 Score": f1,
                       "AUC": roc_auc_score((y == i).astype(int), p[:, i]) if len(np.unique(y == i)) == 2 else np.nan})
    macro = pd.DataFrame(result).select_dtypes("number").mean().to_dict()
    macro.update({"Class": "Macro Average", "ACC": (pred == y).mean()})
    return pd.DataFrame(result + [macro]), cm, y, p, pred


def save_analysis(name, rows, classes, output_dir):
    table, cm, y, p, pred = metrics_table(rows, classes)
    table.to_csv(output_dir / f"{name}_metrics.csv", index=False, float_format="%.6f")
    pd.DataFrame([{ "path": r[0], "group_id": r[1], "true_label": classes[r[2]], "pred_label": classes[int(np.argmax(r[3]))],
                   **{f"prob_{classes[i]}": r[3][i] for i in range(len(classes))}} for r in rows]).to_csv(output_dir / f"{name}_oof_predictions.csv", index=False)
    plt.figure(figsize=(max(8, len(classes)), max(6, len(classes)*.85)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"{name}: OOF Confusion Matrix"); plt.tight_layout(); plt.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=300); plt.close()
    plt.figure(figsize=(8, 7)); aucs = {}
    for i, label in enumerate(classes):
        fpr, tpr, _ = roc_curve((y == i).astype(int), p[:, i]); auc = roc_auc_score((y == i).astype(int), p[:, i]); aucs[label] = float(auc)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1], "k--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"{name}: OOF ROC Curves"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(output_dir / f"{name}_roc_curve.png", dpi=300); plt.close()
    return table.iloc[-1].to_dict(), aucs


def save_gradcam(model, records, classes, args, device, out_dir):
    if not is_main(): return
    tf = transforms.Compose([transforms.Resize(int(args.image_size * 1.14)), transforms.CenterCrop(args.image_size), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    cam = GradCAM(unwrap(model)); sample = CCIDDataset(records[:args.gradcam_samples], tf)
    vis_dir = out_dir / "gradcam"; vis_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(sample)):
        image, label, path, _ = sample[i]; heatmap, logits = cam(image.unsqueeze(0).to(device))
        raw = np.asarray(Image.open(path).convert("RGB").resize((args.image_size, args.image_size))) / 255.0
        fig, ax = plt.subplots(1, 3, figsize=(12, 4)); ax[0].imshow(raw); ax[0].set_title(f"True: {classes[label]}")
        ax[1].imshow(heatmap, cmap="jet"); ax[1].set_title("Grad-CAM")
        ax[2].imshow(raw); ax[2].imshow(heatmap, cmap="jet", alpha=.45); ax[2].set_title(f"Pred: {classes[int(logits.argmax(1))]}")
        [a.axis("off") for a in ax]; fig.tight_layout(); fig.savefig(vis_dir / f"cam_{i:03d}.png", dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-dir", default="data/train", help="Class-folder root (normally data/train)")
    ap.add_argument("--output-dir", default="outputs_ccid")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--epochs", type=int, default=35); ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=12, help="Per-GPU batch size; lower if CUDA OOM occurs")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--weight-decay", type=float, default=2e-4); ap.add_argument("--dropout", type=float, default=.35); ap.add_argument("--label-smoothing", type=float, default=.08)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--gradcam-samples", type=int, default=20); ap.add_argument("--fast", action="store_true")
    args = ap.parse_args(); world, rank, device = distributed_setup(); seed_everything(args.seed + rank)
    if args.fast: torch.backends.cudnn.deterministic = False; torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    records, classes = scan_dataset(Path(args.data_dir))
    labels, groups = np.array([x[1] for x in records]), np.array([x[2] for x in records])
    class_group_counts = [len(set(groups[labels == label])) for label in range(len(classes))]
    if min(class_group_counts) < args.folds:
        raise ValueError("Each class must have at least --folds distinct source groups for group-safe CV.")
    run_dir = Path(args.output_dir); (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if is_main():
        json.dump(vars(args) | {"classes": classes, "n_images": len(records), "n_groups": len(set(groups)), "models": ["convnext_large", "efficientnet_v2_l", "swin_v2_b", "densenet201", "regnet_y_16gf"]}, open(run_dir / "run_config.json", "w"), indent=2)
        pd.DataFrame(records, columns=["path", "label", "group_id"]).to_csv(run_dir / "dataset_manifest.csv", index=False)
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    architectures = ["convnext_large", "efficientnet_v2_l", "swin_v2_b", "densenet201", "regnet_y_16gf"]
    all_summary = []
    for architecture in architectures:
        model_rows = []
        for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups), 1):
            model, rows, checkpoint = train_fold(architecture, fold, [records[i] for i in tr], [records[i] for i in va], classes, args, world, rank, device, run_dir)
            if is_main():
                model_rows.extend(rows)
                save_gradcam(model, [records[i] for i in va], classes, args, device, run_dir / f"{architecture}_fold{fold}")
            del model; torch.cuda.empty_cache()
            if dist.is_initialized(): dist.barrier()
        if is_main():
            summary, aucs = save_analysis(architecture, model_rows, classes, run_dir)
            summary.update({"Model": architecture, "Macro AUC": np.mean(list(aucs.values()))}); all_summary.append(summary)
    if is_main():
        result = pd.DataFrame(all_summary).set_index("Model").sort_values("Macro AUC", ascending=False)
        result.to_csv(run_dir / "model_comparison_table.csv", float_format="%.6f")
        print("\nFinal OOF model comparison:\n", result.round(4).to_string())
        print(f"\nSaved all results to: {run_dir.resolve()}")
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__": main()
