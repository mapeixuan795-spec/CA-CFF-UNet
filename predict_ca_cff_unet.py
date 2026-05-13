import os
import sys
import csv
import argparse
from typing import Dict, List, Tuple, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from dataset_optimized import FanPatchDataset
from model_ca_cff_unet import CA_CFF_UNet


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_thresholds(thresholds_str: str) -> List[float]:
    vals: List[float] = []
    for x in thresholds_str.split(','):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        raise ValueError("--thresholds 不能为空")
    for v in vals:
        if not (0.0 < v < 1.0):
            raise ValueError(f"threshold 必须在 (0,1) 内，当前: {v}")
    return vals


def build_ca_cff_model(in_channels: int, num_classes: int, base_channels: int) -> torch.nn.Module:
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


def unwrap_logits(model_out):
    if isinstance(model_out, dict):
        if "logits" not in model_out:
            raise KeyError("模型返回dict时必须包含 logits 字段")
        return model_out["logits"]
    return model_out


def tensor_chw_to_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.detach().cpu().numpy().astype(np.float32)
    if img.ndim != 3 or img.shape[0] < 3:
        raise ValueError(f"image tensor shape should be [C,H,W] with C>=3, got {img.shape}")
    rgb = np.transpose(img[:3], (1, 2, 0))
    for c in range(3):
        ch = rgb[:, :, c]
        ch_min = ch.min()
        ch_max = ch.max()
        if ch_max > ch_min:
            rgb[:, :, c] = (ch - ch_min) / (ch_max - ch_min + 1e-6)
        else:
            rgb[:, :, c] = 0.0
    return np.clip(rgb, 0.0, 1.0)


def mask_tensor_to_2d(mask_tensor: torch.Tensor) -> np.ndarray:
    mask = mask_tensor.detach().cpu().numpy().astype(np.float32)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim == 2:
        pass
    else:
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError(f"mask shape not supported: {mask.shape}")
    return mask


def make_overlay(rgb: np.ndarray, mask01: np.ndarray, color: Tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    overlay = rgb.copy()
    mask_bool = mask01.astype(bool)
    overlay[mask_bool, 0] = color[0]
    overlay[mask_bool, 1] = overlay[mask_bool, 1] * 0.5 + color[1] * 0.5
    overlay[mask_bool, 2] = overlay[mask_bool, 2] * 0.5 + color[2] * 0.5
    return np.clip(overlay, 0.0, 1.0)


def compute_binary_metrics(pred01: np.ndarray, gt01: np.ndarray, smooth: float = 1e-6) -> Dict[str, float]:
    pred = pred01.astype(np.uint8).reshape(-1)
    gt = gt01.astype(np.uint8).reshape(-1)

    tp = np.sum((pred == 1) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    pixel_acc = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "dice": float(dice),
        "iou": float(iou),
        "pixel_acc": float(pixel_acc),
        "precision": float(precision),
        "recall": float(recall),
    }


def build_model_and_args_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    in_channels = int(ckpt_args.get("in_channels", 6))
    num_classes = int(ckpt_args.get("num_classes", 1))
    base_channels = int(ckpt_args.get("base_channels", 24))

    model = build_ca_cff_model(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
    ).to(device)

    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt_args


def disk_structure(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    return (x * x + y * y) <= radius * radius


def remove_small_components(mask01: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask01.astype(np.uint8)
    labeled, num = ndi.label(mask01 > 0)
    if num == 0:
        return mask01.astype(np.uint8)
    out = np.zeros_like(mask01, dtype=np.uint8)
    areas = ndi.sum(mask01 > 0, labeled, index=np.arange(1, num + 1))
    for idx, area in enumerate(areas, start=1):
        if int(area) >= min_area:
            out[labeled == idx] = 1
    return out


def keep_largest_component(mask01: np.ndarray) -> np.ndarray:
    labeled, num = ndi.label(mask01 > 0)
    if num == 0:
        return mask01.astype(np.uint8)
    areas = ndi.sum(mask01 > 0, labeled, index=np.arange(1, num + 1))
    largest_idx = int(np.argmax(areas)) + 1
    out = np.zeros_like(mask01, dtype=np.uint8)
    out[labeled == largest_idx] = 1
    return out


def postprocess_mask(
    mask01: np.ndarray,
    min_area: int = 0,
    fill_holes: bool = False,
    closing_radius: int = 0,
    opening_radius: int = 0,
    keep_largest: bool = False,
) -> np.ndarray:
    out = (mask01 > 0).astype(np.uint8)

    if fill_holes:
        out = ndi.binary_fill_holes(out > 0).astype(np.uint8)

    if closing_radius > 0:
        out = ndi.binary_closing(out > 0, structure=disk_structure(closing_radius)).astype(np.uint8)

    if opening_radius > 0:
        out = ndi.binary_opening(out > 0, structure=disk_structure(opening_radius)).astype(np.uint8)

    if min_area > 1:
        out = remove_small_components(out, min_area=min_area)

    if keep_largest:
        out = keep_largest_component(out)

    return out.astype(np.uint8)


def save_visualization(
    save_path: str,
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    prob_map: np.ndarray,
    pred_raw_mask: np.ndarray,
    pred_post_mask: np.ndarray,
    metrics_raw: Dict[str, float],
    metrics_post: Dict[str, float],
    sample_name: str,
    threshold: float,
):
    overlay_raw = make_overlay(rgb, pred_raw_mask, color=(1.0, 0.0, 0.0))
    overlay_post = make_overlay(rgb, pred_post_mask, color=(1.0, 0.5, 0.0))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.reshape(2, 4)

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_mask, cmap="gray")
    axes[0, 1].set_title("GT")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(prob_map, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("Prob")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(make_overlay(rgb, gt_mask, color=(0.0, 1.0, 0.0)))
    axes[0, 3].set_title("GT Overlay")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(pred_raw_mask, cmap="gray")
    axes[1, 0].set_title(f"Pred Raw @ {threshold:.2f}")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay_raw)
    axes[1, 1].set_title(
        f"Raw Overlay\nD={metrics_raw['dice']:.3f} P={metrics_raw['precision']:.3f} R={metrics_raw['recall']:.3f}"
    )
    axes[1, 1].axis("off")

    axes[1, 2].imshow(pred_post_mask, cmap="gray")
    axes[1, 2].set_title("Pred Post")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(overlay_post)
    axes[1, 3].set_title(
        f"Post Overlay\nD={metrics_post['dice']:.3f} P={metrics_post['precision']:.3f} R={metrics_post['recall']:.3f}"
    )
    axes[1, 3].axis("off")

    fig.suptitle(sample_name, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_from_totals(
    total_tp: Optional[int] = None,
    total_tn: Optional[int] = None,
    total_fp: Optional[int] = None,
    total_fn: Optional[int] = None,
    *,
    tp: Optional[int] = None,
    tn: Optional[int] = None,
    fp: Optional[int] = None,
    fn: Optional[int] = None,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    兼容两种调用方式：
    1) summarize_from_totals(total_tp=..., total_tn=..., total_fp=..., total_fn=...)
    2) summarize_from_totals(tp=..., tn=..., fp=..., fn=...)
    """
    if total_tp is None:
        total_tp = 0 if tp is None else int(tp)
    if total_tn is None:
        total_tn = 0 if tn is None else int(tn)
    if total_fp is None:
        total_fp = 0 if fp is None else int(fp)
    if total_fn is None:
        total_fn = 0 if fn is None else int(fn)

    overall_dice = (2.0 * total_tp + smooth) / (2.0 * total_tp + total_fp + total_fn + smooth)
    overall_iou = (total_tp + smooth) / (total_tp + total_fp + total_fn + smooth)
    overall_acc = (total_tp + total_tn + smooth) / (total_tp + total_tn + total_fp + total_fn + smooth)
    overall_precision = (total_tp + smooth) / (total_tp + total_fp + smooth)
    overall_recall = (total_tp + smooth) / (total_tp + total_fn + smooth)
    return {
        "dice": float(overall_dice),
        "iou": float(overall_iou),
        "pixel_acc": float(overall_acc),
        "precision": float(overall_precision),
        "recall": float(overall_recall),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict CA-CFF-UNet segmentation results with threshold sweep and post-processing")
    parser.add_argument("--dataset_root", type=str, default=r"E:\U-net_fan_extract\02_code\04_dataset_multi_ps256_s128")
    parser.add_argument("--ckpt_path", type=str, default=r"E:\U-net_fan_extract\02_code\05_train_unet\runs\run_ca_cff_001\best_model.pth")
    parser.add_argument("--output_dir", type=str, default=r"E:\U-net_fan_extract\02_code\05_train_unet\runs\run_ca_cff_001\predict_test_ca_cff")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--thresholds", type=str, default="0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--normalize_mode", type=str, default="auto",
                        choices=["auto", "none", "per_channel_minmax", "per_channel_zscore", "fan_channel_minmax", "fan_channel_zscore"])
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--save_prob_npy", action="store_true")
    parser.add_argument("--save_pred_npy", action="store_true")
    parser.add_argument("--fill_holes", action="store_true")
    parser.add_argument("--closing_radius", type=int, default=0)
    parser.add_argument("--opening_radius", type=int, default=0)
    parser.add_argument("--min_area", type=int, default=0)
    parser.add_argument("--keep_largest", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    if args.save_vis:
        ensure_dir(os.path.join(args.output_dir, "vis"))
    if args.save_prob_npy:
        ensure_dir(os.path.join(args.output_dir, "prob_npy"))
    if args.save_pred_npy:
        ensure_dir(os.path.join(args.output_dir, "pred_npy"))

    thresholds = parse_thresholds(args.thresholds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt_args = build_model_and_args_from_checkpoint(args.ckpt_path, device)
    if args.normalize_mode == "auto":
        dataset_normalize_mode = ckpt_args.get("normalize_mode", "per_channel_minmax")
    else:
        dataset_normalize_mode = args.normalize_mode

    print("=" * 80)
    print("device          :", device)
    print("dataset_root    :", args.dataset_root)
    print("ckpt_path       :", args.ckpt_path)
    print("model           :", ckpt_args.get("model_name", "CA_CFF_UNet"))
    print("base_channels   :", ckpt_args.get("base_channels", 24))
    print("output_dir      :", args.output_dir)
    print("split           :", args.split)
    print("thresholds      :", thresholds)
    print("normalize_mode  :", dataset_normalize_mode)
    print("fill_holes      :", args.fill_holes)
    print("closing_radius  :", args.closing_radius)
    print("opening_radius  :", args.opening_radius)
    print("min_area        :", args.min_area)
    print("keep_largest    :", args.keep_largest)
    print("max_samples     :", args.max_samples)
    print("save_vis        :", args.save_vis)
    print("=" * 80)

    dataset = FanPatchDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        expected_channels=ckpt_args.get("in_channels", 6),
        normalize_mode=dataset_normalize_mode,
        return_path=True,
        use_hard_negative=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    per_sample_rows: List[Dict] = []
    totals_raw = {thr: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for thr in thresholds}
    totals_post = {thr: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for thr in thresholds}
    processed_count = 0

    with torch.no_grad():
        for batch in loader:
            images, masks, img_paths, mask_paths = batch
            images = images.to(device).float()
            logits = unwrap_logits(model(images))
            probs = torch.sigmoid(logits).cpu()
            masks = masks.cpu()

            batch_size = images.size(0)
            for i in range(batch_size):
                if args.max_samples > 0 and processed_count >= args.max_samples:
                    break

                image_i = images[i].cpu()
                mask_i = masks[i]
                prob_i = probs[i]
                img_path = img_paths[i]
                mask_path = mask_paths[i]
                sample_name = os.path.splitext(os.path.basename(img_path))[0]

                rgb = tensor_chw_to_rgb(image_i)
                gt_mask = (mask_tensor_to_2d(mask_i) > 0).astype(np.uint8)
                prob_map = mask_tensor_to_2d(prob_i).astype(np.float32)

                best_post_dice = -1.0
                best_record: Optional[Dict] = None
                best_pred_raw = None
                best_pred_post = None

                for thr in thresholds:
                    pred_raw = (prob_map > thr).astype(np.uint8)
                    pred_post = postprocess_mask(
                        pred_raw,
                        min_area=args.min_area,
                        fill_holes=args.fill_holes,
                        closing_radius=args.closing_radius,
                        opening_radius=args.opening_radius,
                        keep_largest=args.keep_largest,
                    )

                    metrics_raw = compute_binary_metrics(pred_raw, gt_mask)
                    metrics_post = compute_binary_metrics(pred_post, gt_mask)

                    for key in ["tp", "tn", "fp", "fn"]:
                        totals_raw[thr][key] += metrics_raw[key]
                        totals_post[thr][key] += metrics_post[key]

                    if metrics_post["dice"] > best_post_dice:
                        best_post_dice = metrics_post["dice"]
                        best_record = {
                            "sample_name": sample_name,
                            "img_path": img_path,
                            "mask_path": mask_path,
                            "best_threshold": thr,
                            "raw_dice": metrics_raw["dice"],
                            "raw_iou": metrics_raw["iou"],
                            "raw_pixel_acc": metrics_raw["pixel_acc"],
                            "raw_precision": metrics_raw["precision"],
                            "raw_recall": metrics_raw["recall"],
                            "post_dice": metrics_post["dice"],
                            "post_iou": metrics_post["iou"],
                            "post_pixel_acc": metrics_post["pixel_acc"],
                            "post_precision": metrics_post["precision"],
                            "post_recall": metrics_post["recall"],
                        }
                        best_pred_raw = pred_raw
                        best_pred_post = pred_post
                        best_metrics_raw = metrics_raw
                        best_metrics_post = metrics_post

                assert best_record is not None
                per_sample_rows.append(best_record)
                processed_count += 1

                if args.save_vis:
                    vis_path = os.path.join(args.output_dir, "vis", f"{sample_name}.png")
                    save_visualization(
                        save_path=vis_path,
                        rgb=rgb,
                        gt_mask=gt_mask,
                        prob_map=prob_map,
                        pred_raw_mask=best_pred_raw,
                        pred_post_mask=best_pred_post,
                        metrics_raw=best_metrics_raw,
                        metrics_post=best_metrics_post,
                        sample_name=sample_name,
                        threshold=best_record["best_threshold"],
                    )

                if args.save_prob_npy:
                    np.save(os.path.join(args.output_dir, "prob_npy", f"{sample_name}.npy"), prob_map.astype(np.float32))

                if args.save_pred_npy:
                    np.save(os.path.join(args.output_dir, "pred_npy", f"{sample_name}.npy"), best_pred_post.astype(np.uint8))

                print(
                    f"[{processed_count}] {sample_name} | best_thr={best_record['best_threshold']:.2f} | "
                    f"post_dice={best_record['post_dice']:.4f}, post_iou={best_record['post_iou']:.4f}, "
                    f"P={best_record['post_precision']:.4f}, R={best_record['post_recall']:.4f}"
                )

            if args.max_samples > 0 and processed_count >= args.max_samples:
                break

    sample_csv_path = os.path.join(args.output_dir, f"{args.split}_per_sample_best.csv")
    with open(sample_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(per_sample_rows[0].keys()) if per_sample_rows else [
            "sample_name", "img_path", "mask_path", "best_threshold",
            "raw_dice", "raw_iou", "raw_pixel_acc", "raw_precision", "raw_recall",
            "post_dice", "post_iou", "post_pixel_acc", "post_precision", "post_recall",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if per_sample_rows:
            writer.writerows(per_sample_rows)

    threshold_rows: List[Dict] = []
    best_global_thr = None
    best_global_post_dice = -1.0
    for thr in thresholds:
        raw_summary = summarize_from_totals(**totals_raw[thr])
        post_summary = summarize_from_totals(**totals_post[thr])
        row = {
            "threshold": thr,
            "raw_dice": raw_summary["dice"],
            "raw_iou": raw_summary["iou"],
            "raw_pixel_acc": raw_summary["pixel_acc"],
            "raw_precision": raw_summary["precision"],
            "raw_recall": raw_summary["recall"],
            "post_dice": post_summary["dice"],
            "post_iou": post_summary["iou"],
            "post_pixel_acc": post_summary["pixel_acc"],
            "post_precision": post_summary["precision"],
            "post_recall": post_summary["recall"],
        }
        threshold_rows.append(row)
        if row["post_dice"] > best_global_post_dice:
            best_global_post_dice = row["post_dice"]
            best_global_thr = thr

    threshold_csv_path = os.path.join(args.output_dir, f"{args.split}_threshold_sweep.csv")
    with open(threshold_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys()) if threshold_rows else [
            "threshold", "raw_dice", "raw_iou", "raw_pixel_acc", "raw_precision", "raw_recall",
            "post_dice", "post_iou", "post_pixel_acc", "post_precision", "post_recall",
        ])
        writer.writeheader()
        if threshold_rows:
            writer.writerows(threshold_rows)

    summary_path = os.path.join(args.output_dir, f"{args.split}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"split: {args.split}\n")
        f.write(f"processed_count: {processed_count}\n")
        f.write(f"normalize_mode: {dataset_normalize_mode}\n")
        f.write(f"thresholds: {thresholds}\n")
        f.write(f"best_global_threshold_by_post_dice: {best_global_thr}\n")
        f.write(f"fill_holes: {args.fill_holes}\n")
        f.write(f"closing_radius: {args.closing_radius}\n")
        f.write(f"opening_radius: {args.opening_radius}\n")
        f.write(f"min_area: {args.min_area}\n")
        f.write(f"keep_largest: {args.keep_largest}\n\n")
        for row in threshold_rows:
            f.write(
                f"thr={row['threshold']:.2f} | raw_dice={row['raw_dice']:.6f}, raw_iou={row['raw_iou']:.6f}, "
                f"raw_precision={row['raw_precision']:.6f}, raw_recall={row['raw_recall']:.6f} | "
                f"post_dice={row['post_dice']:.6f}, post_iou={row['post_iou']:.6f}, "
                f"post_precision={row['post_precision']:.6f}, post_recall={row['post_recall']:.6f}\n"
            )

    print("\n阈值扫描完成。")
    print(f"processed_count         : {processed_count}")
    print(f"best_global_threshold   : {best_global_thr}")
    print(f"per-sample best csv     : {sample_csv_path}")
    print(f"threshold sweep csv     : {threshold_csv_path}")
    print(f"summary txt             : {summary_path}")
    if args.save_vis:
        print(f"vis dir                 : {os.path.join(args.output_dir, 'vis')}")


if __name__ == "__main__":
    main()
