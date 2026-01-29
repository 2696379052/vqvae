"""
Train EfficientNetV2 on heatmap 3-class labels (stop-loss / take-profit / timeout).
Default label: label_3c_1.0_30t.cls (1.0 RR, 30-tick SL).
"""

import argparse
import glob
import io
import json
import os
import random
import time
from collections import deque
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torch.nn.functional as F
import webdataset as wds
import timm
from tqdm import tqdm
from sklearn.metrics import classification_report

try:
    from timm.data.mixup import Mixup
    from timm.utils import ModelEmaV2
except Exception:
    Mixup = None
    ModelEmaV2 = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def expand_patterns(patterns: str):
    shards = []
    for part in patterns.split(","):
        part = part.strip()
        if not part:
            continue
        shards.extend(glob.glob(part))
    return sorted(set(shards))


def default_shards(data_dir: str, split: str):
    pattern = os.path.join(data_dir, f"{split}-*.tar*")
    return expand_patterns(pattern)


def _parse_label(x):
    if isinstance(x, (tuple, list)) and len(x) == 1:
        x = x[0]
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8")
    if isinstance(x, np.ndarray):
        x = x.item()
    return int(x)


def _to_tensor(x):
    if isinstance(x, np.ndarray):
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        return torch.from_numpy(x)
    return torch.tensor(x, dtype=torch.float32)


@lru_cache(maxsize=16)
def _gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.astype(np.float32)


def gaussian_blur_heatmaps(heatmaps: np.ndarray, sigma: float) -> np.ndarray:
    """对热图进行高斯模糊（通道独立，使用可分离1D卷积）"""
    if sigma <= 0:
        return heatmaps
    kernel = _gaussian_kernel1d(sigma)
    pad = len(kernel) // 2
    x = heatmaps.astype(np.float32, copy=False)
    squeeze = False
    if x.ndim == 3:
        x = x[None, ...]
        squeeze = True
    # H 方向卷积
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (0, 0)), mode="edge")
    win = sliding_window_view(x_pad, len(kernel), axis=2)
    x = np.tensordot(win, kernel, axes=([-1], [0]))
    # W 方向卷积
    x_pad = np.pad(x, ((0, 0), (0, 0), (0, 0), (pad, pad)), mode="edge")
    win = sliding_window_view(x_pad, len(kernel), axis=3)
    x = np.tensordot(win, kernel, axes=([-1], [0]))
    return x[0] if squeeze else x


def make_load_npy(blur_sigma: float):
    def _load_npy(x):
        if isinstance(x, np.ndarray):
            arr = x
        elif isinstance(x, (bytes, bytearray, memoryview)):
            with io.BytesIO(x) as f:
                arr = np.load(f)
        else:
            arr = np.asarray(x)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if blur_sigma and blur_sigma > 0:
            arr = gaussian_blur_heatmaps(arr, blur_sigma)
        return torch.from_numpy(arr)
    return _load_npy


def _to_label_tensor(x):
    return torch.tensor(_parse_label(x), dtype=torch.long)


def infer_label_key(shards):
    ds = wds.WebDataset(shards)
    sample = next(iter(ds))
    for key in ("label_3c_1.0_30t.cls", "label_3c_1.0.cls", "label_1.0_30t.cls", "label_1.0.cls"):
        if key in sample:
            return key
    cls_keys = sorted([k for k in sample.keys() if k.endswith(".cls")])
    if cls_keys:
        return cls_keys[0]
    raise KeyError("No .cls label key found in dataset sample.")


def ensure_label_key(shards, label_key):
    ds = wds.WebDataset(shards)
    sample = next(iter(ds))
    if label_key in sample:
        return
    available = sorted([k for k in sample.keys() if k.endswith(".cls")])
    raise KeyError(f"Label key '{label_key}' not found. Available: {available}")


def make_loader(
    shards,
    label_key,
    batch_size,
    shuffle=False,
    num_workers=4,
    repeat=False,
    drop_last=False,
    shuffle_buf=1000,
    epoch_samples=None,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=2,
    timeout=0,
    blur_sigma=0.0,
):
    if not shards:
        return None
    # shardshuffle is ignored for resampled datasets; keep it False/0 to avoid warnings
    if repeat:
        shardshuffle_val = 0
    else:
        shardshuffle_val = 100 if shuffle else 0
    dataset = wds.WebDataset(shards, resampled=repeat, shardshuffle=shardshuffle_val)
    if shuffle:
        dataset = dataset.shuffle(shuffle_buf)
    dataset = dataset.to_tuple("input.npy", label_key)
    load_npy = make_load_npy(blur_sigma)
    dataset = dataset.map_tuple(load_npy, _to_label_tensor)
    if repeat and epoch_samples:
        dataset = dataset.with_epoch(epoch_samples)
        dataset = dataset.with_length(epoch_samples)
    dataset = dataset.batched(batch_size, partial=not drop_last)
    loader_kwargs = {
        "num_workers": num_workers,
        "batch_size": None,
        "pin_memory": pin_memory,
        "timeout": timeout,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    loader = wds.WebLoader(dataset, **loader_kwargs)
    return loader


def count_labels(shards, label_key, num_classes, max_samples=None):
    if not shards:
        return None, 0
    ds = wds.WebDataset(shards).to_tuple(label_key).map(_parse_label)
    loader = wds.WebLoader(ds, num_workers=0, batch_size=None)
    counts = np.zeros(num_classes, dtype=np.int64)
    n = 0
    for y in loader:
        counts[y] += 1
        n += 1
        if max_samples and n >= max_samples:
            break
    return counts, n


def class_weights_from_counts(counts):
    inv = 1.0 / (counts + 1e-6)
    weights = inv / inv.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


def soft_target_cross_entropy(logits, targets, class_weights=None):
    log_probs = F.log_softmax(logits, dim=-1)
    if class_weights is not None:
        weights = class_weights.view(1, -1)
        loss = -(targets * log_probs * weights).sum(dim=-1)
    else:
        loss = -(targets * log_probs).sum(dim=-1)
    return loss.mean()


def load_batch_counts(data_dir):
    path = os.path.join(data_dir, "sample_count_cache.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_cached_steps(cache, split, batch_size, num_classes):
    key = f"{split}_bs{batch_size}_nc{num_classes}"
    val = cache.get(key)
    if isinstance(val, (int, float)):
        return int(val)
    # Fallback: infer total samples from any cached batch size for this split
    samples_est = None
    for k, v in cache.items():
        if not isinstance(v, (int, float)):
            continue
        if not k.startswith(f"{split}_bs"):
            continue
        if f"_nc{num_classes}" not in k:
            continue
        try:
            bs_part = k.split("_bs", 1)[1].split("_", 1)[0]
            bs_cached = int(bs_part)
        except Exception:
            continue
        est = int(v) * bs_cached
        samples_est = est if samples_est is None else max(samples_est, est)
    if samples_est:
        return int((samples_est + batch_size - 1) // batch_size)
    return None


def update_confusion(cm, y_true, y_pred, num_classes):
    idx = (y_true * num_classes + y_pred).to("cpu")
    binc = torch.bincount(idx, minlength=num_classes * num_classes)
    cm += binc.reshape(num_classes, num_classes).cpu().numpy()


def compute_metrics(cm):
    total = cm.sum()
    acc = float(np.trace(cm)) / total if total > 0 else 0.0
    per_class_recall = np.zeros(cm.shape[0], dtype=np.float32)
    per_class_f1 = np.zeros(cm.shape[0], dtype=np.float32)
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class_recall[i] = recall
        per_class_f1[i] = f1
    macro_f1 = float(per_class_f1.mean()) if per_class_f1.size > 0 else 0.0
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall.tolist(),
        "confusion_matrix": cm.tolist(),
    }


@torch.inference_mode()
def collect_tp_scores(model, loader, device, steps=None):
    p_list = []
    y_list = []
    for step, (x, y) in enumerate(loader):
        if steps and step >= steps:
            break
        x = x.to(device, non_blocking=True)
        logits = model(x)
        # Only compare stop-loss vs take-profit (ignore timeout class)
        probs = torch.softmax(logits[:, :2], dim=1)
        p_tp = probs[:, 1].detach().cpu().numpy()
        p_list.append(p_tp)
        y_list.append(y.cpu().numpy())
    if not p_list:
        return None, None, None
    p_all = np.concatenate(p_list)
    y_all = np.concatenate(y_list)
    # Exclude timeout samples for threshold analysis
    mask = y_all != 2
    p_all = p_all[mask]
    y_all = y_all[mask]
    if y_all.size == 0:
        return None, None, None
    counts = {"tp": int((y_all == 1).sum()), "sl": int((y_all == 0).sum())}
    return p_all, y_all, counts


def binary_metrics_from_threshold(p_tp, y_true, threshold):
    if p_tp is None:
        return None
    pred_tp = p_tp >= threshold
    true_tp = y_true == 1
    tp = int((pred_tp & true_tp).sum())
    fp = int((pred_tp & (~true_tp)).sum())
    fn = int((~pred_tp & true_tp).sum())
    tn = int((~pred_tp & (~true_tp)).sum())
    total = tp + fp + fn + tn
    precision_tp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_tp = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision_sl = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_sl = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = (tp + tn) / total if total > 0 else 0.0
    pred_tp_rate = (tp + fp) / total if total > 0 else 0.0
    pred_sl_rate = (tn + fn) / total if total > 0 else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "acc": acc,
        "precision_tp": precision_tp,
        "recall_tp": recall_tp,
        "precision_sl": precision_sl,
        "recall_sl": recall_sl,
        "pred_tp_rate": pred_tp_rate,
        "pred_sl_rate": pred_sl_rate,
    }


def find_threshold_for_recall_range(p_tp, y_true, recall_min, recall_max, base_counts=None):
    if p_tp is None:
        return None
    pos = (y_true == 1).astype(np.int64)
    total_pos = int(pos.sum())
    if total_pos == 0:
        return {"note": "no positive samples in validation set"}
    order = np.argsort(-p_tp)
    pos_sorted = pos[order]
    tp_cum = np.cumsum(pos_sorted)
    pred_cum = np.arange(1, len(pos_sorted) + 1)
    recall = tp_cum / total_pos
    precision = np.divide(
        tp_cum, pred_cum, out=np.zeros_like(tp_cum, dtype=np.float32), where=pred_cum > 0
    )
    mask = (recall >= recall_min) & (recall <= recall_max)
    note = None
    if mask.any():
        candidates = np.where(mask)[0]
        best_idx = candidates[np.argmax(precision[mask])]
    else:
        note = "no threshold in recall range; using best precision under recall_max"
        mask2 = recall <= recall_max
        if mask2.any():
            candidates = np.where(mask2)[0]
            best_idx = candidates[np.argmax(precision[mask2])]
        else:
            best_idx = int(np.argmax(precision))
            note = "no threshold under recall_max; using global best precision"
    threshold = float(p_tp[order][best_idx])
    metrics = binary_metrics_from_threshold(p_tp, y_true, threshold)
    if base_counts:
        total = base_counts["tp"] + base_counts["sl"]
        metrics["base_tp_rate"] = base_counts["tp"] / total if total > 0 else 0.0
        metrics["base_sl_rate"] = base_counts["sl"] / total if total > 0 else 0.0
    metrics["note"] = note
    return metrics


def format_classification_report(y_true, y_pred, labels, target_names):
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    loss_fn,
    mixup_fn=None,
    grad_clip=None,
    ema=None,
    steps_per_epoch=None,
    progress=False,
    desc=None,
    acc_window=0,
    progress_steps=None,
):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    acc_available = mixup_fn is None
    amp_enabled = scaler is not None and scaler.is_enabled()
    iterator = loader
    if progress:
        total_steps = progress_steps if progress_steps is not None else steps_per_epoch
        iterator = tqdm(loader, total=total_steps, desc=desc, leave=False)
    window = deque(maxlen=acc_window) if acc_window and mixup_fn is None else None
    for step, (x, y) in enumerate(iterator):
        if steps_per_epoch and step >= steps_per_epoch:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if mixup_fn is not None:
            x, y_mix = mixup_fn(x, y)
            targets = y_mix
        else:
            targets = y
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = loss_fn(logits, targets)
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        if acc_available:
            preds = logits.argmax(dim=1)
            batch_correct = (preds == y).sum().item()
            correct += batch_correct
            if window is not None:
                window.append((batch_correct, x.size(0)))
                if progress:
                    w_correct = sum(c for c, _ in window)
                    w_total = sum(n for _, n in window)
                    acc_win = w_correct / w_total if w_total > 0 else 0.0
                    iterator.set_postfix(acc_win=f"{acc_win:.3f}")
        elif progress and window is None:
            iterator.set_postfix(acc_win="NA")
    avg_loss = total_loss / total if total > 0 else 0.0
    if acc_available:
        acc = correct / total if total > 0 else 0.0
    else:
        acc = None
    return avg_loss, acc


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
    loss_fn,
    num_classes,
    limit_steps=None,
    progress=False,
    desc="Val",
    acc_window=0,
    progress_steps=None,
    collect_threshold=False,
):
    model.eval()
    total_loss = 0.0
    total = 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    p_list = []
    y_list = []
    base_counts = {"tp": 0, "sl": 0}
    iterator = loader
    if progress:
        total_steps = progress_steps if progress_steps is not None else limit_steps
        iterator = tqdm(loader, total=total_steps, desc=desc, leave=False)
    window = deque(maxlen=acc_window) if acc_window else None
    for step, (x, y) in enumerate(iterator):
        if limit_steps and step >= limit_steps:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        preds = logits.argmax(dim=1)
        update_confusion(cm, y, preds, num_classes)
        if window is not None and progress:
            batch_correct = (preds == y).sum().item()
            window.append((batch_correct, x.size(0)))
            w_correct = sum(c for c, _ in window)
            w_total = sum(n for _, n in window)
            acc_win = w_correct / w_total if w_total > 0 else 0.0
            iterator.set_postfix(acc_win=f"{acc_win:.3f}")
        if collect_threshold:
            mask = y != 2
            if mask.any():
                p_tp = torch.sigmoid(logits[:, 1] - logits[:, 0])
                p_list.append(p_tp[mask].detach().cpu().numpy())
                y_sel = y[mask].detach().cpu().numpy()
                y_list.append(y_sel)
                base_counts["tp"] += int((y_sel == 1).sum())
                base_counts["sl"] += int((y_sel == 0).sum())
    metrics = compute_metrics(cm)
    metrics["loss"] = total_loss / total if total > 0 else 0.0
    metrics["samples"] = total
    if collect_threshold:
        if p_list and y_list:
            p_all = np.concatenate(p_list)
            y_all = np.concatenate(y_list)
        else:
            p_all = None
            y_all = None
            base_counts = None
        return metrics, (p_all, y_all, base_counts)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNetV2 on heatmap 3-class labels")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--train-shards", default="")
    parser.add_argument("--val-shards", default="")
    parser.add_argument("--test-shards", default="")
    parser.add_argument("--label-key", default="")
    parser.add_argument("--model", default="efficientnetv2_s")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--drop-rate", type=float, default=0.2)
    parser.add_argument("--drop-path-rate", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--mixup-prob", type=float, default=1.0)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
    parser.add_argument("--class-weights", choices=["auto", "none"], default="none")
    parser.add_argument("--class-weight-max-samples", type=int, default=200000)
    parser.add_argument("--timeout-weight-mult", type=float, default=0.5)
    parser.add_argument("--ema", action="store_true", default=True)
    parser.add_argument("--no-ema", action="store_false", dest="ema")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-batches-per-epoch", type=int, default=400)
    parser.add_argument("--threshold-after-epoch", type=int, default=0)
    parser.add_argument("--threshold-every-epoch", action="store_true", default=True)
    parser.add_argument("--no-threshold-every-epoch", action="store_false", dest="threshold_every_epoch")
    parser.add_argument("--tp-recall-min", type=float, default=0.01)
    parser.add_argument("--tp-recall-max", type=float, default=0.06)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument("--acc-window", type=int, default=50)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    parser.add_argument("--persistent-workers", action="store_true", default=True)
    parser.add_argument("--no-persistent-workers", action="store_false", dest="persistent_workers")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--shuffle-buf", type=int, default=1000)
    parser.add_argument("--loader-timeout", type=int, default=0)
    parser.add_argument("--blur-sigma", type=float, default=0.6)
    parser.add_argument("--resume", default="")
    parser.add_argument("--output-dir", default="./runs/effnetv2_3class")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    train_shards = expand_patterns(args.train_shards) if args.train_shards else default_shards(args.data_dir, "train")
    val_shards = expand_patterns(args.val_shards) if args.val_shards else default_shards(args.data_dir, "val")
    test_shards = expand_patterns(args.test_shards) if args.test_shards else default_shards(args.data_dir, "test")
    if not train_shards:
        raise FileNotFoundError("No training shards found.")

    label_key = args.label_key.strip() if args.label_key else infer_label_key(train_shards)
    ensure_label_key(train_shards, label_key)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Optional cap on batches per epoch (official recommendation for resampled streams)
    steps_per_epoch = None
    if args.max_batches_per_epoch and args.max_batches_per_epoch > 0:
        steps_per_epoch = args.max_batches_per_epoch
    elif args.steps_per_epoch and args.steps_per_epoch > 0:
        steps_per_epoch = args.steps_per_epoch
    epoch_samples = steps_per_epoch * args.batch_size if steps_per_epoch else None
    batch_counts = load_batch_counts(args.data_dir)
    train_steps_cached = get_cached_steps(batch_counts, "train", args.batch_size, num_classes=3)
    val_steps_cached = get_cached_steps(batch_counts, "val", args.batch_size, num_classes=3)
    test_steps_cached = get_cached_steps(batch_counts, "test", args.batch_size, num_classes=3)

    train_loader = make_loader(
        train_shards,
        label_key,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # resampled=True allows an effectively infinite randomized stream
        repeat=steps_per_epoch is not None,
        drop_last=True,
        shuffle_buf=args.shuffle_buf,
        epoch_samples=epoch_samples,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=max(2, args.prefetch_factor),
        timeout=args.loader_timeout,
        blur_sigma=args.blur_sigma,
    )
    val_loader = make_loader(
        val_shards,
        label_key,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=max(2, args.prefetch_factor),
        timeout=args.loader_timeout,
        blur_sigma=args.blur_sigma,
    )
    test_loader = make_loader(
        test_shards,
        label_key,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=max(2, args.prefetch_factor),
        timeout=args.loader_timeout,
        blur_sigma=args.blur_sigma,
    )

    num_classes = 3
    class_weights = None
    if args.class_weights == "auto":
        counts, n = count_labels(train_shards, label_key, num_classes, args.class_weight_max_samples)
        if counts is not None and n > 0:
            print(f"Class counts (sampled={n}): {counts.tolist()}")
            if np.any(counts == 0):
                print("Zero-count class detected; disabling class weights.")
            else:
                class_weights = class_weights_from_counts(counts).to(device)
    if class_weights is None and args.timeout_weight_mult != 1.0:
        class_weights = torch.tensor([1.0, 1.0, args.timeout_weight_mult], dtype=torch.float32).to(device)
    elif class_weights is not None:
        class_weights[2] *= args.timeout_weight_mult
    if class_weights is not None:
        print(f"Loss class weights: {[round(v, 4) for v in class_weights.tolist()]}")

    model = timm.create_model(
        args.model,
        in_chans=2,
        num_classes=num_classes,
        pretrained=False,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")

    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        if Mixup is None:
            print("Mixup/CutMix requested but timm Mixup is unavailable; disabling.")
        else:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup,
                cutmix_alpha=args.cutmix,
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode="batch",
                label_smoothing=args.label_smoothing,
                num_classes=num_classes,
            )

    if mixup_fn is None:
        train_loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=args.label_smoothing
        )
    else:
        train_loss_fn = lambda logits, targets: soft_target_cross_entropy(
            logits, targets, class_weights
        )

    eval_loss_fn = torch.nn.CrossEntropyLoss()

    ema = None
    if args.ema:
        if ModelEmaV2 is None:
            print("EMA requested but timm ModelEmaV2 is unavailable; disabling.")
        else:
            ema = ModelEmaV2(model, decay=args.ema_decay)

    start_epoch = 1
    best_val = float("inf")
    patience_count = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val_loss", best_val)
        if ema is not None and ckpt.get("ema") is not None:
            ema.load_state_dict(ckpt["ema"])

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        base_metrics = None
        threshold_report = None
        base_report = None
        thr_report = None
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            train_loss_fn,
            mixup_fn=mixup_fn,
            grad_clip=args.grad_clip,
            ema=ema,
            steps_per_epoch=steps_per_epoch,
            progress=args.progress,
            desc=f"Train {epoch}",
            acc_window=args.acc_window,
            progress_steps=train_steps_cached if steps_per_epoch is None else None,
        )
        eval_model = ema.module if ema is not None else model
        val_metrics = None
        do_threshold = args.threshold_every_epoch or (
            args.threshold_after_epoch and epoch == args.threshold_after_epoch
        )
        if val_loader is not None:
            if do_threshold:
                val_metrics, thr_data = evaluate(
                    eval_model,
                    val_loader,
                    device,
                    eval_loss_fn,
                    num_classes,
                    progress=args.progress,
                    desc=f"Val {epoch}",
                    acc_window=args.acc_window,
                    progress_steps=val_steps_cached,
                    collect_threshold=True,
                )
                p_tp, y_true, base_counts = thr_data
                base_metrics = binary_metrics_from_threshold(p_tp, y_true, 0.5)
                threshold_report = find_threshold_for_recall_range(
                    p_tp, y_true, args.tp_recall_min, args.tp_recall_max, base_counts
                )
                if base_metrics and base_counts:
                    base_total = base_counts["tp"] + base_counts["sl"]
                    if base_total > 0:
                        base_metrics["base_tp_rate"] = base_counts["tp"] / base_total
                        base_metrics["base_sl_rate"] = base_counts["sl"] / base_total
                if p_tp is not None and y_true is not None:
                    base_pred = (p_tp >= 0.5).astype(np.int64)
                    base_report = format_classification_report(
                        y_true, base_pred, labels=[0, 1], target_names=["SL", "TP"]
                    )
                    if threshold_report:
                        thr = threshold_report["threshold"]
                        thr_pred = (p_tp >= thr).astype(np.int64)
                        thr_report = format_classification_report(
                            y_true, thr_pred, labels=[0, 1], target_names=["SL", "TP"]
                        )
            else:
                val_metrics = evaluate(
                    eval_model,
                    val_loader,
                    device,
                    eval_loss_fn,
                    num_classes,
                    progress=args.progress,
                    desc=f"Val {epoch}",
                    acc_window=args.acc_window,
                    progress_steps=val_steps_cached,
                )
        scheduler.step()

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        if val_metrics:
            epoch_log.update({f"val_{k}": v for k, v in val_metrics.items()})
        if val_metrics and do_threshold:
            if base_metrics:
                epoch_log.update({f"val_base_{k}": v for k, v in base_metrics.items()})
            if threshold_report:
                epoch_log.update({f"val_thr_{k}": v for k, v in threshold_report.items()})
            if base_report:
                epoch_log["val_base_report"] = base_report
            if thr_report:
                epoch_log["val_thr_report"] = thr_report
        history.append(epoch_log)
        acc_str = "NA" if train_acc is None else f"{train_acc:.3f}"
        print(
            f"Epoch {epoch:03d} | loss {train_loss:.4f} | acc {acc_str} | "
            f"val {val_metrics['loss']:.4f} | f1 {val_metrics['macro_f1']:.3f} | "
            f"time {time.time()-t0:.1f}s"
            if val_metrics
            else f"Epoch {epoch:03d} | loss {train_loss:.4f} | acc {acc_str} | time {time.time()-t0:.1f}s"
        )
        if epoch == args.threshold_after_epoch and val_metrics:
            print(f"Val confusion matrix: {val_metrics['confusion_matrix']}")
            print(f"Val per-class recall: {val_metrics['per_class_recall']}")
        if val_metrics and do_threshold and base_metrics:
            print(
                f"Val base@epoch{epoch}: acc={base_metrics['acc']:.4f}, "
                f"TP prec/rec={base_metrics['precision_tp']:.4f}/{base_metrics['recall_tp']:.4f}, "
                f"SL prec/rec={base_metrics['precision_sl']:.4f}/{base_metrics['recall_sl']:.4f}, "
                f"pred TP/SL={base_metrics['pred_tp_rate']:.3f}/{base_metrics['pred_sl_rate']:.3f}"
            )
            if "base_tp_rate" in base_metrics:
                print(
                    f"Val base label ratio (TP/SL): "
                    f"{base_metrics['base_tp_rate']:.3f}/{base_metrics['base_sl_rate']:.3f}"
                )
            if base_report:
                print("Val base classification report (SL/TP):")
                print(base_report)
        if val_metrics and do_threshold and threshold_report:
            note = threshold_report.get("note")
            note_str = f" ({note})" if note else ""
            print(
                f"Val thr@epoch{epoch}: thr={threshold_report['threshold']:.4f}, "
                f"acc={threshold_report['acc']:.4f}, "
                f"TP prec/rec={threshold_report['precision_tp']:.4f}/{threshold_report['recall_tp']:.4f}, "
                f"SL prec/rec={threshold_report['precision_sl']:.4f}/{threshold_report['recall_sl']:.4f}{note_str}"
            )
            if thr_report:
                print("Val threshold classification report (SL/TP):")
                print(thr_report)

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val,
        }
        if ema is not None:
            ckpt["ema"] = ema.state_dict()
        torch.save(ckpt, os.path.join(args.output_dir, "last.pt"))

        improved = val_metrics and val_metrics["loss"] < best_val - 1e-6
        if improved:
            best_val = val_metrics["loss"]
            ckpt["best_val_loss"] = best_val
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
            patience_count = 0
        elif val_metrics:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    test_metrics = None
    if test_loader is not None:
        best_path = os.path.join(args.output_dir, "best.pt")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            if ema is not None and ckpt.get("ema") is not None:
                ema.load_state_dict(ckpt["ema"])
        eval_model = ema.module if ema is not None else model
        test_metrics = evaluate(eval_model, test_loader, device, eval_loss_fn, num_classes)
        print(f"Test | loss {test_metrics['loss']:.4f} | f1 {test_metrics['macro_f1']:.3f}")

    summary = {"best_val_loss": best_val, "test_metrics": test_metrics}
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
