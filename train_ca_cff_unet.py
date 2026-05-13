import os
import csv
import time
import random
import argparse
from typing import Dict, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_optimized import FanPatchDataset
from model_ca_cff_unet import CA_CFF_UNet


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def prepare_binary_masks(masks: torch.Tensor) -> torch.Tensor:
    """
    将数据集返回的mask统一成[B,1,H,W]的0/1浮点格式。
    兼容 [B,H,W]、[B,1,H,W]，以及像素值为0/255的标签。
    """
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
    """
    语义分割Mixup。影像和mask同步混合，mask会变成软标签。
    """
    if alpha <= 0 or images.size(0) < 2:
        return images, masks

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)

    mixed_images = lam * images + (1.0 - lam) * images[index]
    mixed_masks = lam * masks + (1.0 - lam) * masks[index]

    return mixed_images, mixed_masks


def build_ca_cff_model(in_channels: int, num_classes: int, base_channels: int) -> nn.Module:
    """
    兼容不同版本的CA_CFF_UNet构造参数。
    推荐版本为：
        CA_CFF_UNet(in_channels=6, out_channels=1, features=32/64)
    """
    try:
        return CA_CFF_UNet(in_channels=in_channels, out_channels=num_classes, features=base_channels)
    except TypeError:
        try:
            return CA_CFF_UNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)
        except TypeError:
            return CA_CFF_UNet(in_channels, num_classes, base_channels)


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
    alpha: float = 0.65,
    beta: float = 0.35,
    smooth: float = 1e-6,
) -> torch.Tensor:
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
    boundary = (dilated - eroded) > 0
    return boundary.float()


def compute_batch_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Dict[str, float]:
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


class HybridSegLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.45,
        dice_weight: float = 0.25,
        tversky_weight: float = 0.30,
        boundary_weight: float = 0.20,
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


def build_dataloaders(args):
    train_dataset = FanPatchDataset(
        dataset_root=args.dataset_root,
        split="train",
        expected_channels=args.in_channels,
        normalize_mode=args.normalize_mode,
        return_path=False,
        use_hard_negative=args.use_hard_negative,
        hard_negative_radius_steps=args.hard_negative_radius_steps,
        hard_negative_repeat=args.hard_negative_repeat,
    )
    val_dataset = FanPatchDataset(
        dataset_root=args.dataset_root,
        split="val",
        expected_channels=args.in_channels,
        normalize_mode=args.normalize_mode,
        return_path=False,
        use_hard_negative=False,
    )
    test_dataset = FanPatchDataset(
        dataset_root=args.dataset_root,
        split="test",
        expected_channels=args.in_channels,
        normalize_mode=args.normalize_mode,
        return_path=False,
        use_hard_negative=False,
    )

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


def save_checkpoint(save_path, model, optimizer, scheduler, scaler, epoch, best_val_dice, best_threshold, args):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_dice": best_val_dice,
        "best_threshold": best_threshold,
        "args": vars(args),
    }
    torch.save(ckpt, save_path)


def _unwrap_model_output(model_out, model):
    if isinstance(model_out, dict):
        logits = model_out.get("logits")
        aux_outputs = {k: v for k, v in model_out.items() if k != "logits"}
    else:
        logits = model_out
        aux_outputs = model.get_aux_outputs() if hasattr(model, "get_aux_outputs") else {}
    return logits, aux_outputs


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    use_amp: bool,
    grad_clip: float,
    use_mixup: bool = False,
    mixup_prob: float = 0.0,
    mixup_alpha: float = 0.4,
):
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
def evaluate_with_thresholds(logits_cpu: torch.Tensor, targets_cpu: torch.Tensor, thresholds: Iterable[float]):
    best_threshold = None
    best_metrics = None
    for thr in thresholds:
        metrics = compute_batch_metrics_from_logits(logits_cpu, targets_cpu, threshold=float(thr))
        if best_metrics is None or metrics["dice"] > best_metrics["dice"]:
            best_metrics = metrics
            best_threshold = float(thr)
    return best_threshold, best_metrics


def main():
    parser = argparse.ArgumentParser(description="CA-CFF-UNet training for alluvial fan segmentation")
    parser.add_argument("--dataset_root", type=str, default=r"E:\U-net_fan_extract\02_code\04_dataset_multi_ps256_s128")
    parser.add_argument("--save_dir", type=str, default=r"E:\U-net_fan_extract\02_code\05_train_unet\runs\run_ca_cff_001")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--in_channels", type=int, default=6)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--bilinear", action="store_true")
    parser.add_argument("--normalize_mode", type=str, default="fan_channel_minmax", choices=["none", "per_channel_minmax", "per_channel_zscore", "fan_channel_minmax", "fan_channel_zscore"] )
    parser.add_argument("--use_hard_negative", action="store_true", help="训练集启用 hard negative 过采样")
    parser.add_argument("--hard_negative_radius_steps", type=int, default=1, help="hard negative 距离正样本的网格步数半径")
    parser.add_argument("--hard_negative_repeat", type=int, default=2, help="hard negative 重复采样次数")

    parser.add_argument("--use_mixup", action="store_true", help="启用Mixup分割增强")
    parser.add_argument("--mixup_prob", type=float, default=0.30, help="每个batch启用Mixup的概率")
    parser.add_argument("--mixup_alpha", type=float, default=0.40, help="Mixup的Beta分布alpha参数")

    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=12)
    parser.add_argument("--threshold_start", type=float, default=0.30)
    parser.add_argument("--threshold_end", type=float, default=0.70)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    args = parser.parse_args()
    args.model_name = "CA_CFF_UNet"

    ensure_dir(args.save_dir)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.use_amp and device.type == "cuda")

    print("=" * 88)
    print("device         :", device)
    print("dataset_root   :", args.dataset_root)
    print("save_dir       :", args.save_dir)
    print("epochs         :", args.epochs)
    print("batch_size     :", args.batch_size)
    print("lr             :", args.lr)
    print("min_lr         :", args.min_lr)
    print("in_channels    :", args.in_channels)
    print("model          :", args.model_name)
    print("base_channels  :", args.base_channels)
    print("normalize_mode :", args.normalize_mode)
    print("use_mixup      :", args.use_mixup)
    print("mixup_prob     :", args.mixup_prob)
    print("mixup_alpha    :", args.mixup_alpha)
    print("use_amp        :", use_amp)
    print("=" * 88)

    train_loader, val_loader, test_loader = build_dataloaders(args)
    model = build_ca_cff_model(args.in_channels, args.num_classes, args.base_channels).to(device)
    criterion = HybridSegLoss(bce_weight=0.45, dice_weight=0.25, tversky_weight=0.30, boundary_weight=0.20, boundary_width=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    log_csv_path = os.path.join(args.save_dir, "train_log.csv")
    best_ckpt_path = os.path.join(args.save_dir, "best_model.pth")
    last_ckpt_path = os.path.join(args.save_dir, "last_model.pth")

    best_val_dice = -1.0
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
        cur_best_threshold, val_metrics = evaluate_with_thresholds(val_logits, val_targets, thresholds)
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
            "train_bce": train_metrics["bce"],
            "train_dice_loss": train_metrics["dice_loss"],
            "train_tversky_loss": train_metrics["tversky_loss"],
            "train_boundary_loss": train_metrics["boundary_loss"],
            "train_pos_weight": train_metrics["pos_weight"],
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
            "val_pos_weight": val_metrics["pos_weight"],
            "epoch_time_sec": epoch_time,
        }
        append_log_row(log_csv_path, log_row)

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] lr={current_lr:.6f} | "
            f"train_loss={train_metrics['loss']:.4f}, train_dice={train_metrics['dice']:.4f}, train_iou={train_metrics['iou']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}, val_dice={val_metrics['dice']:.4f}, val_iou={val_metrics['iou']:.4f}, thr={cur_best_threshold:.2f} | "
            f"time={epoch_time:.1f}s"
        )

        save_checkpoint(last_ckpt_path, model, optimizer, scheduler, scaler, epoch, best_val_dice, best_threshold, args)

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            best_threshold = cur_best_threshold
            no_improve_epochs = 0
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, scaler, epoch, best_val_dice, best_threshold, args)
            print(f"  -> 保存 best_model.pth (epoch={epoch}, val_dice={best_val_dice:.4f}, thr={best_threshold:.2f})")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.early_stop_patience:
                print(f"  -> 提前停止：连续 {no_improve_epochs} 个 epoch 验证 Dice 未提升")
                break

    total_time = time.time() - total_start
    print("\n训练完成。")
    print(f"best epoch     : {best_epoch}")
    print(f"best val dice  : {best_val_dice:.6f}")
    print(f"best threshold : {best_threshold:.2f}")
    print(f"log csv        : {log_csv_path}")
    print(f"best ckpt      : {best_ckpt_path}")
    print(f"last ckpt      : {last_ckpt_path}")
    print(f"total time     : {total_time:.1f}s")

    print("\n开始测试集评估（加载 best_model.pth）...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    best_threshold = float(best_ckpt.get("best_threshold", best_threshold))

    test_logits, test_targets, test_loss, test_loss_logs = gather_logits_and_targets(model, test_loader, criterion, device, use_amp)
    test_metrics = compute_batch_metrics_from_logits(test_logits, test_targets, threshold=best_threshold)
    test_metrics["loss"] = test_loss
    test_metrics.update(test_loss_logs)

    test_result_path = os.path.join(args.save_dir, "test_result.txt")
    with open(test_result_path, "w", encoding="utf-8") as f:
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_dice: {best_val_dice:.6f}\n")
        f.write(f"best_threshold: {best_threshold:.2f}\n")
        for k, v in test_metrics.items():
            f.write(f"test_{k}: {v:.6f}\n")

    print(
        f"test_loss={test_metrics['loss']:.4f}, test_dice={test_metrics['dice']:.4f}, test_iou={test_metrics['iou']:.4f}, "
        f"test_precision={test_metrics['precision']:.4f}, test_recall={test_metrics['recall']:.4f}, thr={best_threshold:.2f}"
    )
    print(f"test result    : {test_result_path}")


if __name__ == "__main__":
    main()
