import os
import csv
import glob
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_patch_name(filename: str) -> Tuple[str, int, int]:
    """
    fan_013_0000_0128.npy -> tile_id='fan_013', top=0, left=128
    最后两个字段视为 top / left，前面部分视为 tile_id。
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"文件名格式不符合预期: {filename}")
    top = int(parts[-2])
    left = int(parts[-1])
    tile_id = "_".join(parts[:-2])
    return tile_id, top, left


def load_2d(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2:
        out = arr
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            out = arr[0]
        elif arr.shape[-1] == 1:
            out = arr[:, :, 0]
        else:
            out = np.squeeze(arr)
            if out.ndim != 2:
                raise ValueError(f"无法转换为2维数组: {path}, shape={arr.shape}")
    else:
        raise ValueError(f"不支持的数组维度: {path}, shape={arr.shape}")
    return out.astype(np.float32)


def load_image_hwc(path: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"图像不是3维: {path}, shape={arr.shape}")
    # 支持 CHW / HWC
    if arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.shape[-1] <= 16 and arr.shape[0] > 16 and arr.shape[1] > 16:
        pass
    else:
        raise ValueError(f"无法识别图像通道布局: {path}, shape={arr.shape}")
    return arr


def binarize(arr: np.ndarray) -> np.ndarray:
    return (arr > 0).astype(np.uint8)


def normalize_rgb_for_display(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32).copy()
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError(f"RGB 显示输入不合法: shape={rgb.shape}")
    rgb = rgb[:, :, :3]
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        ch = rgb[:, :, c]
        valid = np.isfinite(ch)
        if not np.any(valid):
            continue
        lo = np.percentile(ch[valid], 1)
        hi = np.percentile(ch[valid], 99)
        if hi > lo:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        else:
            mn, mx = float(ch[valid].min()), float(ch[valid].max())
            if mx > mn:
                out[:, :, c] = np.clip((ch - mn) / (mx - mn + 1e-6), 0.0, 1.0)
            else:
                out[:, :, c] = 0.0
    return out


def disk_structure(radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    return (x * x + y * y) <= radius * radius


def postprocess_mask(
    mask01: np.ndarray,
    fill_holes: bool = False,
    closing_radius: int = 0,
    opening_radius: int = 0,
    min_area: int = 0,
    keep_largest: bool = False,
) -> np.ndarray:
    m = mask01.astype(bool)

    if opening_radius > 0:
        m = ndi.binary_opening(m, structure=disk_structure(opening_radius))
    if closing_radius > 0:
        m = ndi.binary_closing(m, structure=disk_structure(closing_radius))
    if fill_holes:
        m = ndi.binary_fill_holes(m)

    if min_area > 0:
        lbl, num = ndi.label(m)
        if num > 0:
            counts = np.bincount(lbl.ravel())
            keep = np.zeros_like(m, dtype=bool)
            for idx in range(1, len(counts)):
                if counts[idx] >= int(min_area):
                    keep |= (lbl == idx)
            m = keep

    if keep_largest:
        lbl, num = ndi.label(m)
        if num > 0:
            counts = np.bincount(lbl.ravel())
            if len(counts) > 1:
                largest = int(np.argmax(counts[1:]) + 1)
                m = (lbl == largest)
            else:
                m = np.zeros_like(m, dtype=bool)

    return m.astype(np.uint8)


def compute_metrics(pred01: np.ndarray, gt01: np.ndarray, smooth: float = 1e-6) -> Dict[str, float]:
    pred = pred01.astype(np.uint8).reshape(-1)
    gt = gt01.astype(np.uint8).reshape(-1)

    tp = int(np.sum((pred == 1) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    pixel_acc = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "dice": float(dice),
        "iou": float(iou),
        "pixel_acc": float(pixel_acc),
        "precision": float(precision),
        "recall": float(recall),
    }


def make_overlay(rgb01: np.ndarray, mask01: np.ndarray, color=(1.0, 0.3, 0.0), alpha: float = 0.45) -> np.ndarray:
    out = rgb01.copy()
    m = mask01.astype(bool)
    if not np.any(m):
        return out
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out[m] = (1.0 - alpha) * out[m] + alpha * color_arr[0, 0]
    return np.clip(out, 0.0, 1.0)


def build_error_rgb(pred01: np.ndarray, gt01: np.ndarray) -> np.ndarray:
    # 黑=TN, 红=FP, 蓝=FN, 绿=TP
    pred = pred01.astype(np.uint8)
    gt = gt01.astype(np.uint8)
    err = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.float32)
    err[(pred == 1) & (gt == 0)] = [1.0, 0.0, 0.0]
    err[(pred == 0) & (gt == 1)] = [0.0, 0.0, 1.0]
    err[(pred == 1) & (gt == 1)] = [0.0, 1.0, 0.0]
    return err


def save_compare_figure(
    save_path: str,
    rgb01: np.ndarray,
    gt_map: np.ndarray,
    vote_map: np.ndarray,
    pred_map: np.ndarray,
    metrics: Dict[str, float],
    title: str,
):
    pred_overlay = make_overlay(rgb01, pred_map, color=(1.0, 0.45, 0.0), alpha=0.45)
    gt_overlay = make_overlay(rgb01, gt_map, color=(0.2, 1.0, 0.2), alpha=0.40)
    error_rgb = build_error_rgb(pred_map, gt_map)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    axes[0].imshow(rgb01)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    axes[1].imshow(gt_map, cmap="gray")
    axes[1].set_title("GT")
    axes[1].axis("off")

    im = axes[2].imshow(vote_map, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title("Vote / overlap score")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(pred_map, cmap="gray")
    axes[3].set_title("Prediction")
    axes[3].axis("off")

    axes[4].imshow(pred_overlay)
    axes[4].set_title("Pred Overlay")
    axes[4].axis("off")

    axes[5].imshow(error_rgb)
    axes[5].set_title("Error map (FP=red, FN=blue, TP=green)")
    axes[5].axis("off")

    fig.suptitle(
        f"{title} | Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, "
        f"Acc={metrics['pixel_acc']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="将 patch 预测拼接回整扇面，并输出更直观的整图对比结果")

    parser.add_argument("--dataset_root", type=str,
                        default=r"E:\U-net_fan_extract\02_code\04_dataset_multi_ps256_s128",
                        help="数据集根目录")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="分区")
    parser.add_argument("--pred_dir", type=str,
                        default=r"E:\U-net_fan_extract\02_code\05_train_unet\runs\run_opt_003\predict_thr045_raw\pred_npy",
                        help="predict.py 输出的 pred_npy 目录")
    parser.add_argument("--output_dir", type=str,
                        default=r"E:\U-net_fan_extract\02_code\05_train_unet\runs\run_opt_003\stitched_thr045_raw",
                        help="输出目录")
    parser.add_argument("--patch_size", type=int, default=256, help="patch 尺寸")
    parser.add_argument("--vote_threshold", type=float, default=0.5, help="重叠区域平均后二值化阈值")
    parser.add_argument("--save_fig", action="store_true", help="保存整图可视化")

    # 整图级后处理，可选
    parser.add_argument("--whole_fill_holes", action="store_true", help="整图级填洞")
    parser.add_argument("--whole_closing_radius", type=int, default=0, help="整图级闭运算半径")
    parser.add_argument("--whole_opening_radius", type=int, default=0, help="整图级开运算半径")
    parser.add_argument("--whole_min_area", type=int, default=0, help="整图级最小连通域面积")
    parser.add_argument("--whole_keep_largest", action="store_true", help="整图级只保留最大连通域")

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    npy_out_dir = os.path.join(args.output_dir, "stitched_npy")
    png_out_dir = os.path.join(args.output_dir, "stitched_png")
    ensure_dir(npy_out_dir)
    if args.save_fig:
        ensure_dir(png_out_dir)

    image_dir = os.path.join(args.dataset_root, args.split, "images")
    gt_mask_dir = os.path.join(args.dataset_root, args.split, "masks")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"找不到图像目录: {image_dir}")
    if not os.path.isdir(gt_mask_dir):
        raise FileNotFoundError(f"找不到真值目录: {gt_mask_dir}")
    if not os.path.isdir(args.pred_dir):
        raise FileNotFoundError(f"找不到预测目录: {args.pred_dir}")

    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.npy")))
    if not pred_files:
        raise RuntimeError(f"预测目录下没有 .npy 文件: {args.pred_dir}")

    print("=" * 80)
    print("dataset_root         :", args.dataset_root)
    print("split                :", args.split)
    print("pred_dir             :", args.pred_dir)
    print("output_dir           :", args.output_dir)
    print("patch_size           :", args.patch_size)
    print("vote_threshold       :", args.vote_threshold)
    print("whole_fill_holes     :", args.whole_fill_holes)
    print("whole_closing_radius :", args.whole_closing_radius)
    print("whole_opening_radius :", args.whole_opening_radius)
    print("whole_min_area       :", args.whole_min_area)
    print("whole_keep_largest   :", args.whole_keep_largest)
    print("save_fig             :", args.save_fig)
    print("pred file num        :", len(pred_files))
    print("=" * 80)

    groups = defaultdict(list)
    for pred_path in pred_files:
        tile_id, top, left = parse_patch_name(os.path.basename(pred_path))
        groups[tile_id].append((top, left, pred_path))

    print("发现 tile:", sorted(groups.keys()))

    per_tile_rows: List[Dict] = []
    global_tp = global_tn = global_fp = global_fn = 0

    for tile_id in sorted(groups.keys()):
        patch_items = sorted(groups[tile_id], key=lambda x: (x[0], x[1]))

        max_bottom = 0
        max_right = 0
        for top, left, _ in patch_items:
            max_bottom = max(max_bottom, top + args.patch_size)
            max_right = max(max_right, left + args.patch_size)

        pred_sum = np.zeros((max_bottom, max_right), dtype=np.float32)
        pred_cnt = np.zeros((max_bottom, max_right), dtype=np.float32)
        gt_sum = np.zeros((max_bottom, max_right), dtype=np.float32)
        gt_cnt = np.zeros((max_bottom, max_right), dtype=np.float32)
        rgb_sum = np.zeros((max_bottom, max_right, 3), dtype=np.float32)
        rgb_cnt = np.zeros((max_bottom, max_right, 1), dtype=np.float32)

        for top, left, pred_path in patch_items:
            patch_name = os.path.basename(pred_path)
            gt_path = os.path.join(gt_mask_dir, patch_name)
            img_path = os.path.join(image_dir, patch_name)
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"缺少对应真值 patch: {gt_path}")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"缺少对应图像 patch: {img_path}")

            pred_patch = load_2d(pred_path)
            gt_patch = binarize(load_2d(gt_path)).astype(np.float32)
            img_patch = load_image_hwc(img_path)
            rgb_patch = img_patch[:, :, :3]

            h, w = pred_patch.shape
            pred_sum[top:top + h, left:left + w] += pred_patch.astype(np.float32)
            pred_cnt[top:top + h, left:left + w] += 1.0

            gt_sum[top:top + h, left:left + w] += gt_patch[:h, :w]
            gt_cnt[top:top + h, left:left + w] += 1.0

            rgb_sum[top:top + h, left:left + w, :] += rgb_patch[:h, :w, :]
            rgb_cnt[top:top + h, left:left + w, :] += 1.0

        vote_map = pred_sum / np.maximum(pred_cnt, 1e-6)
        pred_full = (vote_map >= args.vote_threshold).astype(np.uint8)
        gt_full = (gt_sum / np.maximum(gt_cnt, 1e-6) > 0.5).astype(np.uint8)
        rgb_full = rgb_sum / np.maximum(rgb_cnt, 1e-6)
        rgb_disp = normalize_rgb_for_display(rgb_full)

        pred_full = postprocess_mask(
            pred_full,
            fill_holes=args.whole_fill_holes,
            closing_radius=args.whole_closing_radius,
            opening_radius=args.whole_opening_radius,
            min_area=args.whole_min_area,
            keep_largest=args.whole_keep_largest,
        )

        metrics = compute_metrics(pred_full, gt_full)
        global_tp += metrics["tp"]
        global_tn += metrics["tn"]
        global_fp += metrics["fp"]
        global_fn += metrics["fn"]

        np.save(os.path.join(npy_out_dir, f"{tile_id}_pred.npy"), pred_full.astype(np.uint8))
        np.save(os.path.join(npy_out_dir, f"{tile_id}_gt.npy"), gt_full.astype(np.uint8))
        np.save(os.path.join(npy_out_dir, f"{tile_id}_vote.npy"), vote_map.astype(np.float32))
        np.save(os.path.join(npy_out_dir, f"{tile_id}_rgb.npy"), rgb_disp.astype(np.float32))
        np.save(os.path.join(npy_out_dir, f"{tile_id}_count.npy"), pred_cnt.astype(np.float32))

        if args.save_fig:
            fig_path = os.path.join(png_out_dir, f"{tile_id}_compare.png")
            save_compare_figure(
                save_path=fig_path,
                rgb01=rgb_disp,
                gt_map=gt_full,
                vote_map=vote_map,
                pred_map=pred_full,
                metrics=metrics,
                title=tile_id,
            )

        row = {
            "tile_id": tile_id,
            "height": int(pred_full.shape[0]),
            "width": int(pred_full.shape[1]),
            "patch_num": len(patch_items),
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "pixel_acc": metrics["pixel_acc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }
        per_tile_rows.append(row)

        print(
            f"[{tile_id}] size=({pred_full.shape[0]},{pred_full.shape[1]}) "
            f"patches={len(patch_items)} | Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, "
            f"Acc={metrics['pixel_acc']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}"
        )

    smooth = 1e-6
    overall_dice = (2.0 * global_tp + smooth) / (2.0 * global_tp + global_fp + global_fn + smooth)
    overall_iou = (global_tp + smooth) / (global_tp + global_fp + global_fn + smooth)
    overall_acc = (global_tp + global_tn + smooth) / (global_tp + global_tn + global_fp + global_fn + smooth)
    overall_precision = (global_tp + smooth) / (global_tp + global_fp + smooth)
    overall_recall = (global_tp + smooth) / (global_tp + global_fn + smooth)

    csv_path = os.path.join(args.output_dir, f"{args.split}_stitched_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "tile_id", "height", "width", "patch_num",
            "tp", "tn", "fp", "fn",
            "dice", "iou", "pixel_acc", "precision", "recall",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_tile_rows)

    summary_path = os.path.join(args.output_dir, f"{args.split}_stitched_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"split: {args.split}\n")
        f.write(f"tile_num: {len(per_tile_rows)}\n")
        f.write(f"pred_dir: {args.pred_dir}\n")
        f.write(f"patch_size: {args.patch_size}\n")
        f.write(f"vote_threshold: {args.vote_threshold}\n")
        f.write(f"whole_fill_holes: {args.whole_fill_holes}\n")
        f.write(f"whole_closing_radius: {args.whole_closing_radius}\n")
        f.write(f"whole_opening_radius: {args.whole_opening_radius}\n")
        f.write(f"whole_min_area: {args.whole_min_area}\n")
        f.write(f"whole_keep_largest: {args.whole_keep_largest}\n")
        f.write(f"global_tp: {global_tp}\n")
        f.write(f"global_tn: {global_tn}\n")
        f.write(f"global_fp: {global_fp}\n")
        f.write(f"global_fn: {global_fn}\n")
        f.write(f"overall_dice: {overall_dice:.6f}\n")
        f.write(f"overall_iou: {overall_iou:.6f}\n")
        f.write(f"overall_pixel_acc: {overall_acc:.6f}\n")
        f.write(f"overall_precision: {overall_precision:.6f}\n")
        f.write(f"overall_recall: {overall_recall:.6f}\n")

    print("\n拼接完成。")
    print(f"tile_num           : {len(per_tile_rows)}")
    print(f"overall_dice       : {overall_dice:.6f}")
    print(f"overall_iou        : {overall_iou:.6f}")
    print(f"overall_pixel_acc  : {overall_acc:.6f}")
    print(f"overall_precision  : {overall_precision:.6f}")
    print(f"overall_recall     : {overall_recall:.6f}")
    print(f"metrics csv        : {csv_path}")
    print(f"summary txt        : {summary_path}")
    print(f"stitched npy dir   : {npy_out_dir}")
    if args.save_fig:
        print(f"stitched png dir   : {png_out_dir}")


if __name__ == "__main__":
    main()
