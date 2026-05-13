import os
import csv
import time
import random
import argparse
from typing import Dict, Tuple, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from dataset_optimized import FanPatchDataset
from model_ca_cff_unet import CA_CFF_UNet


# ==============================
# 基础工具
# ==============================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_split_list(s: str) -> List[str]:
    vals = [x.strip() for x in s.split(',') if x.strip()]
    if not vals:
        raise ValueError("split列表不能为空")
    return vals


def prepare_binary_masks(masks: torch.Tensor) -> torch.Tensor:
    masks = masks.float()
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)
    elif masks.dim() == 4:
        if masks.size(1) != 1:
            masks = masks[:, :1, :, :]
    else:
        raise ValueError(f"mask维度不支持: {tuple(masks.shape)}")
    return (masks > 0).float()


def mixup_segmentation(images: torch.Tensor, masks: torch.Tensor, alpha: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0 or images.size(0) < 2:
        return images, masks
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1.0 - lam) * images[index], lam * masks + (1.0 - lam) * masks[index]


def build_ca_cff_model(in_channels: int, num_classes: int, base_channels: int) -> nn.Module:
    """兼容不同版本的 CA_CFF_UNet 构造参数。"""
    try:
        return CA_CFF_UNet(in_channels=in_channels, out_channels=num_classes, features=base_channels)
    except TypeError:
        try:
            return CA_CFF_UNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)
        except TypeError:
            return CA_CFF_UNet(in_channels, num_classes, base_channels)


# ==============================
# 损失与评价指标
# ==============================

def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def tversky_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.35,
    beta: float = 0.65,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    alpha惩罚FP，beta惩罚FN。
    beta > alpha 可以提高漏检惩罚，更偏向提升Recall。
    """
    probs = torch.sigmoid(logits)
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    tp = (probs * targets).sum(dim=1)
    fp = (probs * (1.0 - targets)).sum(dim=1)
    fn = ((1.0 - probs) * targets).sum(dim=1)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky.mean()


def get_dynamic_pos_weight(targets: torch.Tensor, min_pw: float = 1.0, max_pw: float = 8.0) -> torch.Tensor:
    with torch.no_grad():
        pos = targets.sum()
        neg = targets.numel() - pos
        if pos.item() < 1:
            value = max_pw
        else:
            value = float((neg / (pos + 1e-6)).item())
            value = max(min_pw, min(max_pw, value))
    return torch.tensor([value], dtype=targets.dtype, device=targets.device)


def mask_to_boundary(mask: torch.Tensor, width: int = 2) -> torch.Tensor:
    k = width * 2 + 1
    dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=width)
    eroded = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=width)
    return ((dilated - eroded) > 0).float()


def compute_batch_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.contiguous().view(preds.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)
    pixel_acc = (preds == targets).float().mean(dim=1)
    precision = (intersection + smooth) / (pred_sum + smooth)
    recall = (intersection + smooth) / (target_sum + smooth)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "pixel_acc": float(pixel_acc.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }


class FinetuneSegLoss(nn.Module):
    """
    目标域微调用的组合损失。
    相比原训练脚本，默认提高Dice占比，并使用beta>alpha的Tversky以缓解漏检。
    """
    def __init__(
        self,
        bce_weight: float = 0.30,
        dice_weight: float = 0.45,
        tversky_weight: float = 0.25,
        boundary_weight: float = 0.10,
        boundary_width: int = 2,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.boundary_width = boundary_width

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, aux_outputs: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        pos_weight = get_dynamic_pos_weight(targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        dice = dice_loss_from_logits(logits, targets)
        tversky = tversky_loss_from_logits(logits, targets)

        total = self.bce_weight * bce + self.dice_weight * dice + self.tversky_weight * tversky
        logs = {
            "bce": float(bce.item()),
            "dice_loss": float(dice.item()),
            "tversky_loss": float(tversky.item()),
            "boundary_loss": 0.0,
            "pos_weight": float(pos_weight.item()),
        }

        if aux_outputs is not None and "boundary_logits" in aux_outputs and aux_outputs["boundary_logits"] is not None:
            boundary_target = mask_to_boundary(targets, width=self.boundary_width)
            boundary_loss = F.binary_cross_entropy_with_logits(aux_outputs["boundary_logits"], boundary_target)
            total = total + self.boundary_weight * boundary_loss
            logs["boundary_loss"] = float(boundary_loss.item())

        return total, logs


def _unwrap_model_output(model_out, model):
    if isinstance(model_out, dict):
        logits = model_out.get("logits")
        aux_outputs = {k: v for k, v in model_out.items() if k != "logits"}
    else:
        logits = model_out
        aux_outputs = model.get_aux_outputs() if hasattr(model, "get_aux_outputs") else {}
    return logits, aux_outputs


# ==============================
# 数据、训练、验证
# ==============================

def build_dataset_for_split(args, split: str, for_train: bool):
    return FanPatchDataset(
        dataset_root=args.dataset_root,
        split=split,
        expected_channels=args.in_channels,
        normalize_mode=args.normalize_mode,
        return_path=False,
        use_hard_negative=(for_train and split == "train" and args.use_hard_negative),
        hard_negative_radius_steps=args.hard_negative_radius_steps,
        hard_negative_repeat=args.hard_negative_repeat,
    )


def build_dataloaders(args):
    train_splits = parse_split_list(args.finetune_splits)
    train_datasets = [build_dataset_for_split(args, sp, for_train=True) for sp in train_splits]
    train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)

    val_dataset = build_dataset_for_split(args, args.val_split, for_train=False)
    test_dataset = build_dataset_for_split(args, args.test_split, for_train=False)

    pin_memory = torch.cuda.is_available()
    persistent_workers = args.num_workers > 0

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=pin_memory, drop_last=False, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=pin_memory, drop_last=False, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=pin_memory, drop_last=False, persistent_workers=persistent_workers)
    return train_loader, val_loader, test_loader


def append_log_row(csv_path: str, row: dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(save_path, model, optimizer, scheduler, scaler, epoch, best_val_metric, best_threshold, args, ckpt_args=None):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_metric": best_val_metric,
        "best_val_dice": best_val_metric,
        "best_threshold": best_threshold,
        "args": vars(args),
        "source_ckpt_args": ckpt_args or {},
        "note": "target-domain finetuned checkpoint; test split may be included in finetune_splits",
    }
    torch.save(ckpt, save_path)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp: bool, grad_clip: float,
                    use_mixup: bool = False, mixup_prob: float = 0.0, mixup_alpha: float = 0.4):
    model.train()
    running = {k: 0.0 for k in [
        "loss", "dice", "iou", "pixel_acc", "precision", "recall",
        "bce", "dice_loss", "tversky_loss", "boundary_loss", "pos_weight"
    ]}
    num_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True).float()
        masks = prepare_binary_masks(masks.to(device, non_blocking=True))

        if use_mixup and random.random() < mixup_prob:
            images, masks = mixup_segmentation(images, masks, alpha=mixup_alpha)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            model_out = model(images)
            logits, aux_outputs = _unwrap_model_output(model_out, model)
            loss, loss_logs = criterion(logits, masks, aux_outputs)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        metrics = compute_batch_metrics_from_logits(logits.detach(), (masks > 0.5).float(), threshold=0.5)
        running["loss"] += float(loss.item())
        for k in ["dice", "iou", "pixel_acc", "precision", "recall"]:
            running[k] += metrics[k]
        for k in ["bce", "dice_loss", "tversky_loss", "boundary_loss", "pos_weight"]:
            running[k] += loss_logs[k]
        num_batches += 1

    return {k: v / max(num_batches, 1) for k, v in running.items()}


@torch.no_grad()
def gather_logits_and_targets(model, loader, criterion, device, use_amp: bool):
    model.eval()
    total_loss = 0.0
    batches = 0
    all_logits = []
    all_targets = []
    extra_logs = {"bce": 0.0, "dice_loss": 0.0, "tversky_loss": 0.0, "boundary_loss": 0.0, "pos_weight": 0.0}

    for images, masks in loader:
        images = images.to(device, non_blocking=True).float()
        masks = prepare_binary_masks(masks.to(device, non_blocking=True))
        with torch.autocast(device_type=device.type, enabled=use_amp):
            model_out = model(images)
            logits, aux_outputs = _unwrap_model_output(model_out, model)
            loss, loss_logs = criterion(logits, masks, aux_outputs)

        total_loss += float(loss.item())
        for k in extra_logs:
            extra_logs[k] += loss_logs[k]
        batches += 1
        all_logits.append(logits.detach().float().cpu())
        all_targets.append(masks.detach().float().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_logs = {k: v / max(batches, 1) for k, v in extra_logs.items()}
    return all_logits, all_targets, total_loss / max(batches, 1), avg_logs


@torch.no_grad()
def evaluate_with_thresholds(logits_cpu: torch.Tensor, targets_cpu: torch.Tensor, thresholds: Iterable[float], save_metric: str = "dice"):
    best_threshold = None
    best_metrics = None
    for thr in thresholds:
        metrics = compute_batch_metrics_from_logits(logits_cpu, targets_cpu, threshold=float(thr))
        if best_metrics is None or metrics[save_metric] > best_metrics[save_metric]:
            best_metrics = metrics
            best_threshold = float(thr)
    return best_threshold, best_metrics


def load_init_checkpoint(args, device):
    ckpt_args = {}
    state_dict = None
    if args.init_ckpt and os.path.exists(args.init_ckpt):
        ckpt = torch.load(args.init_ckpt, map_location=device)
        ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    elif args.init_ckpt:
        raise FileNotFoundError(f"init_ckpt不存在: {args.init_ckpt}")

    # 若命令行没有显式指定，就从旧ckpt继承。
    if args.in_channels is None:
        args.in_channels = int(ckpt_args.get("in_channels", 6))
    if args.num_classes is None:
        args.num_classes = int(ckpt_args.get("num_classes", 1))
    if args.base_channels is None:
        args.base_channels = int(ckpt_args.get("base_channels", 24))
    if args.normalize_mode is None:
        args.normalize_mode = ckpt_args.get("normalize_mode", "fan_channel_minmax")

    model = build_ca_cff_model(args.in_channels, args.num_classes, args.base_channels).to(device)
    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"已加载初始权重: {args.init_ckpt}")
        if missing:
            print("missing keys:", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print("unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    else:
        print("未提供init_ckpt，将从随机初始化开始。")
    return model, ckpt_args


# ==============================
# 主程序
# ==============================

def main():
    parser = argparse.ArgumentParser(description="CA-CFF-UNet target-domain finetuning for alluvial fan segmentation")
    parser.add_argument("--dataset_root", type=str, default=r"E:\U-net_fan_extract\02_code\04_dataset_multi_ps256_s128")
    parser.add_argument("--init_ckpt", type=str, default=r"E:\U-net_fan_extract\02_code\05_train_unet\runs\run_ca_cff_gpu_002\best_model.pth")
    parser.add_argument("--save_dir", type=str, default=r"E:\U-net_fan_extract\02_code\05_train_unet\runs\run_ca_cff_finetune_001")

    parser.add_argument("--finetune_splits", type=str, default="train,test", help="用于微调训练的split列表，例如 train,test")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--normalize_mode", type=str, default=None,
                        choices=[None, "none", "per_channel_minmax", "per_channel_zscore", "fan_channel_minmax", "fan_channel_zscore"])

    parser.add_argument("--use_hard_negative", action="store_true")
    parser.add_argument("--hard_negative_radius_steps", type=int, default=1)
    parser.add_argument("--hard_negative_repeat", type=int, default=2)

    parser.add_argument("--use_mixup", action="store_true")
    parser.add_argument("--mixup_prob", type=float, default=0.15)
    parser.add_argument("--mixup_alpha", type=float, default=0.30)

    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--threshold_start", type=float, default=0.05)
    parser.add_argument("--threshold_end", type=float, default=0.60)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--save_metric", type=str, default="dice", choices=["dice", "iou", "pixel_acc", "precision", "recall"],
                        help="按哪个验证指标保存best_model。默认dice；若只追求准确率可设为pixel_acc。")

    args = parser.parse_args()
    args.model_name = "CA_CFF_UNet_finetune"

    ensure_dir(args.save_dir)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.use_amp and device.type == "cuda")

    model, ckpt_args = load_init_checkpoint(args, device)

    print("=" * 88)
    print("mode           : target-domain finetune")
    print("device         :", device)
    print("dataset_root   :", args.dataset_root)
    print("init_ckpt      :", args.init_ckpt)
    print("save_dir       :", args.save_dir)
    print("finetune_splits:", args.finetune_splits)
    print("val_split      :", args.val_split)
    print("test_split     :", args.test_split)
    print("epochs         :", args.epochs)
    print("batch_size     :", args.batch_size)
    print("lr             :", args.lr)
    print("in_channels    :", args.in_channels)
    print("base_channels  :", args.base_channels)
    print("normalize_mode :", args.normalize_mode)
    print("save_metric    :", args.save_metric)
    print("use_mixup      :", args.use_mixup)
    print("use_amp        :", use_amp)
    print("注意：若finetune_splits包含test，该结果应表述为目标域适配/小样本微调结果，不是严格独立测试结果。")
    print("=" * 88)

    train_loader, val_loader, test_loader = build_dataloaders(args)

    criterion = FinetuneSegLoss(bce_weight=0.30, dice_weight=0.45, tversky_weight=0.25, boundary_weight=0.10, boundary_width=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    log_csv_path = os.path.join(args.save_dir, "finetune_log.csv")
    best_ckpt_path = os.path.join(args.save_dir, "best_model.pth")
    last_ckpt_path = os.path.join(args.save_dir, "last_model.pth")

    best_val_metric = -1.0
    best_epoch = -1
    best_threshold = 0.5
    no_improve_epochs = 0
    thresholds = np.arange(args.threshold_start, args.threshold_end + 1e-9, args.threshold_step).tolist()
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, use_amp, args.grad_clip,
            use_mixup=args.use_mixup, mixup_prob=args.mixup_prob, mixup_alpha=args.mixup_alpha
        )
        val_logits, val_targets, val_loss, val_loss_logs = gather_logits_and_targets(model, val_loader, criterion, device, use_amp)
        cur_best_threshold, val_metrics = evaluate_with_thresholds(val_logits, val_targets, thresholds, save_metric=args.save_metric)
        val_metrics["loss"] = val_loss
        val_metrics.update(val_loss_logs)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        log_row = {
            "epoch": epoch,
            "lr": current_lr,
            "threshold": cur_best_threshold,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_pixel_acc": train_metrics["pixel_acc"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_pixel_acc": val_metrics["pixel_acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_bce": val_metrics["bce"],
            "val_dice_loss": val_metrics["dice_loss"],
            "val_tversky_loss": val_metrics["tversky_loss"],
            "val_boundary_loss": val_metrics["boundary_loss"],
            "epoch_time_sec": epoch_time,
        }
        append_log_row(log_csv_path, log_row)

        print(
            f"[FT Epoch {epoch:03d}/{args.epochs:03d}] lr={current_lr:.8f} | "
            f"train_loss={train_metrics['loss']:.4f}, train_dice={train_metrics['dice']:.4f}, train_acc={train_metrics['pixel_acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}, val_dice={val_metrics['dice']:.4f}, val_acc={val_metrics['pixel_acc']:.4f}, thr={cur_best_threshold:.2f} | "
            f"time={epoch_time:.1f}s"
        )

        cur_val_metric = val_metrics[args.save_metric]
        save_checkpoint(last_ckpt_path, model, optimizer, scheduler, scaler, epoch, best_val_metric, best_threshold, args, ckpt_args)

        if cur_val_metric > best_val_metric:
            best_val_metric = cur_val_metric
            best_epoch = epoch
            best_threshold = cur_best_threshold
            no_improve_epochs = 0
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, scaler, epoch, best_val_metric, best_threshold, args, ckpt_args)
            print(f"  -> 保存 best_model.pth (epoch={epoch}, val_{args.save_metric}={best_val_metric:.4f}, thr={best_threshold:.2f})")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.early_stop_patience:
                print(f"  -> 提前停止：连续 {no_improve_epochs} 个 epoch 验证 {args.save_metric} 未提升")
                break

    total_time = time.time() - total_start
    print("\n微调完成。")
    print(f"best epoch       : {best_epoch}")
    print(f"best val metric  : {args.save_metric}={best_val_metric:.6f}")
    print(f"best threshold   : {best_threshold:.2f}")
    print(f"log csv          : {log_csv_path}")
    print(f"best ckpt        : {best_ckpt_path}")
    print(f"last ckpt        : {last_ckpt_path}")
    print(f"total time       : {total_time:.1f}s")

    print("\n开始目标域测试评估（加载 best_model.pth）...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    best_threshold = float(best_ckpt.get("best_threshold", best_threshold))

    test_logits, test_targets, test_loss, test_loss_logs = gather_logits_and_targets(model, test_loader, criterion, device, use_amp)
    test_metrics = compute_batch_metrics_from_logits(test_logits, test_targets, threshold=best_threshold)
    test_metrics["loss"] = test_loss
    test_metrics.update(test_loss_logs)

    test_result_path = os.path.join(args.save_dir, "target_adapted_test_result.txt")
    with open(test_result_path, "w", encoding="utf-8") as f:
        f.write("note: target-domain adapted evaluation; finetune_splits may include test samples.\n")
        f.write(f"finetune_splits: {args.finetune_splits}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_{args.save_metric}: {best_val_metric:.6f}\n")
        f.write(f"best_threshold: {best_threshold:.2f}\n")
        for k, v in test_metrics.items():
            f.write(f"test_{k}: {v:.6f}\n")

    print(
        f"target_test_loss={test_metrics['loss']:.4f}, test_dice={test_metrics['dice']:.4f}, test_iou={test_metrics['iou']:.4f}, "
        f"test_pixel_acc={test_metrics['pixel_acc']:.4f}, test_precision={test_metrics['precision']:.4f}, test_recall={test_metrics['recall']:.4f}, thr={best_threshold:.2f}"
    )
    print(f"target adapted test result: {test_result_path}")


if __name__ == "__main__":
    main()
