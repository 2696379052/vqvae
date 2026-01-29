"""
Train VQ-VAE on heatmap shards, extract codebook histogram features, and train XGBoost.
Default task: 2class (filters timeout label -1).
"""

import argparse
import glob
import io
import os
import random
import itertools

import numpy as np
import torch
import webdataset as wds
import xgboost as xgb

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from sklearn.metrics import classification_report
except Exception:
    classification_report = None

from models.vqvae import VQVAE


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


def load_npy(x):
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, (bytes, bytearray, memoryview)):
        with io.BytesIO(x) as f:
            arr = np.load(f)
    else:
        arr = np.asarray(x)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def parse_label(x):
    if isinstance(x, (tuple, list)) and len(x) == 1:
        x = x[0]
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8")
    if isinstance(x, np.ndarray):
        x = x.item()
    return int(x)


def count_samples(shards, label_key, max_samples=None):
    ds = wds.WebDataset(shards, shardshuffle=False).to_tuple(label_key).map(parse_label)
    loader = wds.WebLoader(ds, num_workers=0, batch_size=None)
    n = 0
    for _ in loader:
        n += 1
        if max_samples and n >= max_samples:
            break
    return n


def load_sample_cache(data_dir):
    path = os.path.join(data_dir, "sample_count_cache.json")
    if not os.path.exists(path):
        return {}
    try:
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_cache_steps(cache, split, batch_size, num_classes):
    key = f"{split}_bs{batch_size}_nc{num_classes}"
    val = cache.get(key)
    if isinstance(val, (int, float)) and val > 0:
        return int(val)
    return None


def infer_label_key(shards):
    ds = wds.WebDataset(shards, shardshuffle=False)
    sample = next(iter(ds))
    candidates = [
        "label_2c_1.0_30t.cls",
        "label_2c_1.0.cls",
        "label_1.0_30t.cls",
        "label_1.0.cls",
    ]
    for key in candidates:
        if key in sample:
            return key
    cls_keys = sorted([k for k in sample.keys() if k.endswith(".cls")])
    if cls_keys:
        return cls_keys[0]
    raise KeyError("No .cls label key found in dataset sample.")


def ensure_label_key(shards, label_key):
    ds = wds.WebDataset(shards, shardshuffle=False)
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
    shuffle_buf=1000,
    num_workers=4,
    repeat=False,
    drop_last=False,
    epoch_samples=None,
):
    if not shards:
        return None
    shardshuffle = 0 if repeat else (100 if shuffle else 0)
    dataset = wds.WebDataset(shards, resampled=repeat, shardshuffle=shardshuffle)
    if shuffle:
        dataset = dataset.shuffle(shuffle_buf)
    dataset = dataset.to_tuple("input.npy", label_key).map_tuple(load_npy, parse_label)
    if repeat and epoch_samples:
        dataset = dataset.with_epoch(epoch_samples)
    dataset = dataset.batched(batch_size, partial=not drop_last)
    return wds.WebLoader(dataset, num_workers=num_workers, batch_size=None)


def estimate_var(loader, num_batches):
    mean = 0.0
    m2 = 0.0
    count = 0
    for i, (x, _) in enumerate(loader):
        if num_batches and i >= num_batches:
            break
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            continue
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = x.size
        delta = batch_mean - mean
        total = count + batch_count
        mean = mean + delta * batch_count / total
        m2 = m2 + batch_var * batch_count + delta * delta * count * batch_count / total
        count = total
    if count == 0:
        return 1.0
    return max(m2 / count, 1e-6)


def eval_vqvae(model, loader, device, steps, x_train_var):
    if loader is None or steps <= 0:
        return None, None
    model.eval()
    total_recon = 0.0
    total_ppl = 0.0
    total = 0
    with torch.inference_mode():
        for x, _ in itertools.islice(loader, steps):
            x = torch.from_numpy(np.asarray(x)).float().to(device)
            embedding_loss, x_hat, perplexity = model(x)
            if x_hat.shape != x.shape:
                h = min(x_hat.shape[2], x.shape[2])
                w = min(x_hat.shape[3], x.shape[3])
                x_hat = x_hat[:, :, :h, :w]
                x = x[:, :, :h, :w]
            recon_loss = torch.mean((x_hat - x) ** 2) / x_train_var
            total_recon += recon_loss.item() * x.size(0)
            total_ppl += perplexity.item() * x.size(0)
            total += x.size(0)
    if total == 0:
        return None, None
    return total_recon / total, total_ppl / total


def train_vqvae(
    model,
    loader,
    device,
    epochs,
    steps_per_epoch,
    learning_rate,
    x_train_var,
    log_interval,
    save_path,
    progress,
    val_loader,
    val_steps,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    best_val = None
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_ppl = 0.0
        total = 0
        stream = itertools.islice(loader, steps_per_epoch)
        if progress and tqdm is not None:
            stream = tqdm(stream, total=steps_per_epoch, desc=f"Train {epoch}", leave=False)
        for step, (x, _) in enumerate(stream):
            x = torch.from_numpy(np.asarray(x)).float().to(device)
            optimizer.zero_grad()
            embedding_loss, x_hat, perplexity = model(x)
            if x_hat.shape != x.shape:
                h = min(x_hat.shape[2], x.shape[2])
                w = min(x_hat.shape[3], x.shape[3])
                x_hat = x_hat[:, :, :h, :w]
                x = x[:, :, :h, :w]
            recon_loss = torch.mean((x_hat - x) ** 2) / x_train_var
            loss = recon_loss + embedding_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_recon += recon_loss.item() * x.size(0)
            total_ppl += perplexity.item() * x.size(0)
            total += x.size(0)
            if log_interval and global_step % log_interval == 0:
                print(
                    f"Step {global_step} | recon {recon_loss.item():.6f} | "
                    f"loss {loss.item():.6f} | ppl {perplexity.item():.4f}"
                )
            global_step += 1
        if total > 0:
            train_recon = total_recon / total
            train_loss = total_loss / total
            train_ppl = total_ppl / total
            val_recon, val_ppl = eval_vqvae(model, val_loader, device, val_steps, x_train_var)
            if val_recon is not None:
                print(
                    f"Epoch {epoch:03d} | train recon {train_recon:.6f} "
                    f"loss {train_loss:.6f} ppl {train_ppl:.4f} | "
                    f"val recon {val_recon:.6f} ppl {val_ppl:.4f}"
                )
                if best_val is None or val_recon < best_val:
                    best_val = val_recon
                    if save_path:
                        save_dir = os.path.dirname(save_path)
                        if save_dir:
                            os.makedirs(save_dir, exist_ok=True)
                        torch.save({"model": model.state_dict(), "val_recon": best_val}, save_path)
            else:
                print(
                    f"Epoch {epoch:03d} | "
                    f"recon {train_recon:.6f} | "
                    f"loss {train_loss:.6f} | "
                    f"ppl {train_ppl:.4f}"
                )
    if save_path and best_val is None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save({"model": model.state_dict()}, save_path)


def encode_codebook_hist(
    loader,
    model,
    device,
    n_embeddings,
    task,
    normalize_hist,
    max_samples,
    progress,
    desc,
    total,
):
    features = []
    labels = []
    model.eval()
    pbar = None
    if progress and tqdm is not None:
        pbar = tqdm(total=total, desc=desc, unit="sample")
    with torch.inference_mode():
        for x, y in loader:
            x = np.asarray(x)
            y = np.asarray(y)
            if task == "2class":
                keep = y != -1
            else:
                keep = np.ones_like(y, dtype=bool)
            if not np.any(keep):
                continue
            x = x[keep]
            y = y[keep]
            x_tensor = torch.from_numpy(x).float().to(device)
            _, _, _, indices = model(x_tensor, return_indices=True)
            b = x_tensor.size(0)
            indices = indices.view(b, -1).cpu().numpy()
            for i in range(b):
                hist = np.bincount(indices[i], minlength=n_embeddings).astype(np.float32)
                if normalize_hist:
                    total = hist.sum()
                    if total > 0:
                        hist /= total
                features.append(hist)
                labels.append(int(y[i]))
            if pbar is not None:
                pbar.update(b)
            if max_samples and len(labels) >= max_samples:
                break
    if pbar is not None:
        pbar.close()
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def accuracy(y_true, y_pred, exclude_label=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if exclude_label is not None:
        mask = y_true != exclude_label
        if mask.sum() == 0:
            return float("nan")
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    return float((y_true == y_pred).mean())


def classification_report_text(y_true, y_pred, labels, target_names):
    if classification_report is None:
        return None
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )


def find_threshold_for_recall_range(p_tp, y_true, recall_min, recall_max):
    if p_tp is None or y_true is None:
        return None
    pos = (y_true == 1).astype(np.int64)
    total_pos = int(pos.sum())
    if total_pos == 0:
        return None
    order = np.argsort(-p_tp)
    pos_sorted = pos[order]
    tp_cum = np.cumsum(pos_sorted)
    pred_cum = np.arange(1, len(pos_sorted) + 1)
    recall = tp_cum / total_pos
    precision = np.divide(
        tp_cum, pred_cum, out=np.zeros_like(tp_cum, dtype=np.float32), where=pred_cum > 0
    )
    mask = (recall >= recall_min) & (recall <= recall_max)
    if mask.any():
        candidates = np.where(mask)[0]
        best_idx = candidates[np.argmax(precision[mask])]
    else:
        mask2 = recall <= recall_max
        if mask2.any():
            candidates = np.where(mask2)[0]
            best_idx = candidates[np.argmax(precision[mask2])]
        else:
            best_idx = int(np.argmax(precision))
    return float(p_tp[order][best_idx])


def tp_precision_recall(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def train_xgboost(X_train, y_train, X_val, y_val, params, rounds):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    return xgb.train(params, dtrain, num_boost_round=rounds, evals=[(dtrain, "train"), (dval, "val")])


def main():
    parser = argparse.ArgumentParser(description="VQ-VAE + XGBoost on heatmap shards")
    data = parser.add_argument_group("data")
    data.add_argument("--data-dir", default="./data", help="分片数据根目录")
    data.add_argument("--train-shards", default="", help="训练分片 glob（逗号分隔）")
    data.add_argument("--val-shards", default="", help="验证分片 glob（逗号分隔）")
    data.add_argument("--test-shards", default="", help="测试分片 glob（逗号分隔）")
    data.add_argument("--label-key", default="", help="标签字段名，例如 label_2c_1.0.cls")
    data.add_argument("--task", choices=["2class", "3class"], default="2class", help="标签类型")

    io = parser.add_argument_group("io")
    io.add_argument("--feature-out-dir", default="features", help="特征保存目录（.npz）")
    io.add_argument("--vqvae-out", default="results/vqvae_heatmap.pth", help="最优 VQ-VAE 模型路径")
    io.add_argument("--xgb-out", default="results/xgb_model.json", help="XGBoost 模型路径")

    loader = parser.add_argument_group("loader")
    loader.add_argument("--batch-size", type=int, default=32, help="批大小")
    loader.add_argument("--num-workers", type=int, default=0, help="WebDataset 工作进程数（Windows 建议 0）")
    loader.add_argument("--shuffle-buf", type=int, default=1000, help="shuffle 缓冲区大小")
    loader.add_argument("--seed", type=int, default=42, help="随机种子")

    vq = parser.add_argument_group("vqvae-train")
    vq.add_argument("--epochs", type=int, default=10, help="VQ-VAE 训练轮数")
    vq.add_argument("--steps-per-epoch", type=int, default=600, help="每轮的 batch 数（0=使用缓存/统计）")
    vq.add_argument("--count-samples", action="store_true", default=False, help="当 steps-per-epoch <= 0 时统计训练集样本数")
    vq.add_argument("--val-steps", type=int, default=400, help="每轮验证评估的 batch 数（0=使用缓存）")
    vq.add_argument("--progress", action="store_true", default=True, help="显示进度条")
    vq.add_argument("--no-progress", action="store_false", dest="progress", help="关闭进度条")
    vq.add_argument("--val-random", action="store_true", default=True, help="验证集随机抽取（不放回）")
    vq.add_argument("--no-val-random", action="store_false", dest="val_random", help="关闭验证集随机抽样")
    vq.add_argument("--val-shuffle-buf", type=int, default=1000, help="验证集 shuffle 缓冲区大小")
    vq.add_argument("--learning-rate", type=float, default=3e-4, help="学习率")
    vq.add_argument("--log-interval", type=int, default=50, help="step 日志间隔")
    vq.add_argument("--var-batches", type=int, default=50, help="估计方差的 batch 数")
    vq.add_argument("--n-hiddens", type=int, default=128, help="编码器/解码器隐藏维度")
    vq.add_argument("--n-residual-hiddens", type=int, default=32, help="残差块隐藏维度")
    vq.add_argument("--n-residual-layers", type=int, default=2, help="残差层数")
    vq.add_argument("--embedding-dim", type=int, default=64, help="码本向量维度")
    vq.add_argument("--n-embeddings", type=int, default=128, help="码本大小")
    vq.add_argument("--beta", type=float, default=0.25, help="承诺损失系数")
    vq.add_argument("--input-channels", type=int, default=2, help="输入通道数")

    enc = parser.add_argument_group("encoding")
    enc.add_argument("--max-train-samples", type=int, default=4000, help="训练集最大编码样本数")
    enc.add_argument("--max-val-samples", type=int, default=1000, help="验证集最大编码样本数")
    enc.add_argument("--max-test-samples", type=int, default=1000, help="测试集最大编码样本数")
    enc.add_argument("--no-normalize-hist", action="store_false", dest="normalize_hist", help="关闭码本直方图归一化")
    parser.set_defaults(normalize_hist=True)

    xg = parser.add_argument_group("xgboost")
    xg.add_argument("--xgb-max-depth", type=int, default=6, help="树深")
    xg.add_argument("--xgb-eta", type=float, default=0.1, help="学习率")
    xg.add_argument("--xgb-subsample", type=float, default=0.8, help="行采样率")
    xg.add_argument("--xgb-colsample", type=float, default=0.8, help="列采样率")
    xg.add_argument("--xgb-rounds", type=int, default=300, help="训练轮数")
    xg.add_argument("--exclude-timeout-eval", action="store_true", default=True, help="评估时排除超时样本")
    xg.add_argument("--include-timeout-eval", action="store_false", dest="exclude_timeout_eval", help="评估时包含超时样本")
    xg.add_argument("--xgb-only", action="store_true", default=False, help="仅训练 XGBoost（从 .npz 读取特征）")
    xg.add_argument("--tp-recall-min", type=float, default=0.1, help="止盈召回率下限（阈值搜索）")
    xg.add_argument("--tp-recall-max", type=float, default=0.8, help="止盈召回率上限（阈值搜索）")

    args = parser.parse_args()

    if args.xgb_only:
        train_npz = os.path.join(args.feature_out_dir, "train_features.npz")
        val_npz = os.path.join(args.feature_out_dir, "val_features.npz")
        test_npz = os.path.join(args.feature_out_dir, "test_features.npz")
        for path in (train_npz, val_npz, test_npz):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing feature file: {path}")
        train_data = np.load(train_npz)
        val_data = np.load(val_npz)
        test_data = np.load(test_npz)
        X_train, y_train = train_data["X"], train_data["y"]
        X_val, y_val = val_data["X"], val_data["y"]
        X_test, y_test = test_data["X"], test_data["y"]

        if args.task == "2class":
            uniq = np.unique(y_train)
            if uniq.size < 2:
                raise ValueError(
                    f"2class training requires both labels 0/1, got {uniq.tolist()}."
                )
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": args.xgb_max_depth,
                "eta": args.xgb_eta,
                "subsample": args.xgb_subsample,
                "colsample_bytree": args.xgb_colsample,
                "base_score": 0.5,
            }
            eval_exclude = -1
        else:
            params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "max_depth": args.xgb_max_depth,
                "eta": args.xgb_eta,
                "subsample": args.xgb_subsample,
                "colsample_bytree": args.xgb_colsample,
            }
            eval_exclude = 2 if args.exclude_timeout_eval else None

        print(f"Training XGBoost (samples: train={len(y_train)}, val={len(y_val)})...")
        booster = train_xgboost(X_train, y_train, X_val, y_val, params, args.xgb_rounds)
        os.makedirs(os.path.dirname(args.xgb_out) or ".", exist_ok=True)
        booster.save_model(args.xgb_out)

        val_pred = booster.predict(xgb.DMatrix(X_val))
        if args.task == "2class":
            val_pred_label = (val_pred >= 0.5).astype(np.int64)
        else:
            val_pred_label = val_pred.argmax(axis=1)
        val_acc = accuracy(y_val, val_pred_label, exclude_label=eval_exclude)

        test_pred = booster.predict(xgb.DMatrix(X_test))
        if args.task == "2class":
            test_pred_label = (test_pred >= 0.5).astype(np.int64)
        else:
            test_pred_label = test_pred.argmax(axis=1)
        test_acc = accuracy(y_test, test_pred_label, exclude_label=eval_exclude)

        print("Val Acc (exclude timeout):", val_acc)
        print("Test Acc (exclude timeout):", test_acc)

        if args.task == "2class":
            if classification_report is not None:
                print("Val 分类报告（默认阈值）:")
                print(classification_report_text(y_val, val_pred_label, [0, 1], ["SL", "TP"]))
                print("Test 分类报告（默认阈值）:")
                print(classification_report_text(y_test, test_pred_label, [0, 1], ["SL", "TP"]))
            thr = find_threshold_for_recall_range(val_pred, y_val, args.tp_recall_min, args.tp_recall_max)
            if thr is not None:
                val_thr_pred = (val_pred >= thr).astype(np.int64)
                test_thr_pred = (test_pred >= thr).astype(np.int64)
                prec, rec = tp_precision_recall(y_val, val_thr_pred)
                print(f"Val 阈值={thr:.4f} | 止盈 Precision={prec:.4f} Recall={rec:.4f}")
                if classification_report is not None:
                    print("Val 分类报告（阈值后）:")
                    print(classification_report_text(y_val, val_thr_pred, [0, 1], ["SL", "TP"]))
                prec_t, rec_t = tp_precision_recall(y_test, test_thr_pred)
                print(f"Test 阈值={thr:.4f} | 止盈 Precision={prec_t:.4f} Recall={rec_t:.4f}")
                if classification_report is not None:
                    print("Test 分类报告（阈值后）:")
                    print(classification_report_text(y_test, test_thr_pred, [0, 1], ["SL", "TP"]))
        return

    train_shards = expand_patterns(args.train_shards) if args.train_shards else default_shards(args.data_dir, "train")
    val_shards = expand_patterns(args.val_shards) if args.val_shards else default_shards(args.data_dir, "val")
    test_shards = expand_patterns(args.test_shards) if args.test_shards else default_shards(args.data_dir, "test")
    if not train_shards:
        raise FileNotFoundError("No training shards found.")

    label_key = args.label_key.strip() if args.label_key else infer_label_key(train_shards)
    ensure_label_key(train_shards, label_key)

    if os.name == "nt" and args.num_workers > 0:
        print("Windows detected: forcing num_workers=0 to avoid WebDataset pickling issues.")
        args.num_workers = 0

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = load_sample_cache(args.data_dir)
    num_classes = 2 if args.task == "2class" else 3

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch <= 0:
        cached = get_cache_steps(cache, "train", args.batch_size, num_classes)
        if cached:
            steps_per_epoch = cached
            print(f"使用缓存：steps_per_epoch={steps_per_epoch}")
        elif args.count_samples:
            train_count = count_samples(train_shards, label_key)
            print(f"Samples | train={train_count}")
            steps_per_epoch = max(1, (train_count + args.batch_size - 1) // args.batch_size)
        else:
            raise ValueError("steps_per_epoch 未指定，且无缓存；请设置 --steps-per-epoch 或 --count-samples。")

    epoch_samples = steps_per_epoch * args.batch_size
    train_loader = make_loader(
        train_shards,
        label_key,
        args.batch_size,
        shuffle=args.shuffle_buf > 0,
        shuffle_buf=args.shuffle_buf,
        num_workers=args.num_workers,
        repeat=True,
        drop_last=True,
        epoch_samples=epoch_samples,
    )
    var_loader = make_loader(
        train_shards,
        label_key,
        args.batch_size,
        shuffle=False,
        num_workers=0,
        repeat=False,
    )
    x_train_var = estimate_var(var_loader, args.var_batches)
    print("Estimated train variance:", x_train_var)

    val_loader = make_loader(
        val_shards,
        label_key,
        args.batch_size,
        shuffle=args.val_random,
        shuffle_buf=args.val_shuffle_buf,
        num_workers=args.num_workers,
        repeat=False,
        drop_last=False,
    )

    model = VQVAE(
        args.n_hiddens,
        args.n_residual_hiddens,
        args.n_residual_layers,
        args.n_embeddings,
        args.embedding_dim,
        args.beta,
        input_channels=args.input_channels,
        output_channels=args.input_channels,
    ).to(device)
    train_vqvae(
        model,
        train_loader,
        device,
        args.epochs,
        steps_per_epoch,
        args.learning_rate,
        x_train_var,
        args.log_interval,
        args.vqvae_out,
        args.progress,
        val_loader,
        get_cache_steps(cache, "val", args.batch_size, num_classes) if args.val_steps <= 0 else args.val_steps,
    )

    os.makedirs(args.feature_out_dir, exist_ok=True)
    val_loader = make_loader(
        val_shards,
        label_key,
        args.batch_size,
        shuffle=args.val_random,
        shuffle_buf=args.val_shuffle_buf,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        test_shards,
        label_key,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    max_train = args.max_train_samples
    max_val = args.max_val_samples
    max_test = args.max_test_samples
    max_train = max_train if max_train > 0 else None
    max_val = max_val if max_val > 0 else None
    max_test = max_test if max_test > 0 else None
    total_hint = max_train
    if total_hint is None:
        cached = get_cache_steps(cache, "train", args.batch_size, num_classes)
        if cached:
            total_hint = cached * args.batch_size
    X_train, y_train = encode_codebook_hist(
        make_loader(train_shards, label_key, args.batch_size, shuffle=False, num_workers=args.num_workers),
        model,
        device,
        args.n_embeddings,
        args.task,
        args.normalize_hist,
        max_train,
        args.progress,
        "Encode train",
        total_hint,
    )
    total_hint = max_val
    if total_hint is None:
        cached = get_cache_steps(cache, "val", args.batch_size, num_classes)
        if cached:
            total_hint = cached * args.batch_size
    X_val, y_val = encode_codebook_hist(
        val_loader,
        model,
        device,
        args.n_embeddings,
        args.task,
        args.normalize_hist,
        max_val,
        args.progress,
        "Encode val",
        total_hint,
    )
    total_hint = max_test
    if total_hint is None:
        cached = get_cache_steps(cache, "test", args.batch_size, num_classes)
        if cached:
            total_hint = cached * args.batch_size
    X_test, y_test = encode_codebook_hist(
        test_loader,
        model,
        device,
        args.n_embeddings,
        args.task,
        args.normalize_hist,
        max_test,
        args.progress,
        "Encode test",
        total_hint,
    )

    np.savez(os.path.join(args.feature_out_dir, "train_features.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(args.feature_out_dir, "val_features.npz"), X=X_val, y=y_val)
    np.savez(os.path.join(args.feature_out_dir, "test_features.npz"), X=X_test, y=y_test)

    if args.task == "2class":
        uniq = np.unique(y_train)
        if uniq.size < 2:
            raise ValueError(
                f"2class training requires both labels 0/1, got {uniq.tolist()}. "
                "Reduce filtering or increase samples."
            )
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": args.xgb_max_depth,
            "eta": args.xgb_eta,
            "subsample": args.xgb_subsample,
            "colsample_bytree": args.xgb_colsample,
            "base_score": 0.5,
        }
        eval_exclude = -1
    else:
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": args.xgb_max_depth,
            "eta": args.xgb_eta,
            "subsample": args.xgb_subsample,
            "colsample_bytree": args.xgb_colsample,
        }
        eval_exclude = 2 if args.exclude_timeout_eval else None

    print(f"Training XGBoost (samples: train={len(y_train)}, val={len(y_val)})...")
    booster = train_xgboost(X_train, y_train, X_val, y_val, params, args.xgb_rounds)
    os.makedirs(os.path.dirname(args.xgb_out) or ".", exist_ok=True)
    booster.save_model(args.xgb_out)

    val_pred = booster.predict(xgb.DMatrix(X_val))
    if args.task == "2class":
        val_pred_label = (val_pred >= 0.5).astype(np.int64)
    else:
        val_pred_label = val_pred.argmax(axis=1)
    val_acc = accuracy(y_val, val_pred_label, exclude_label=eval_exclude)

    test_pred = booster.predict(xgb.DMatrix(X_test))
    if args.task == "2class":
        test_pred_label = (test_pred >= 0.5).astype(np.int64)
    else:
        test_pred_label = test_pred.argmax(axis=1)
    test_acc = accuracy(y_test, test_pred_label, exclude_label=eval_exclude)

    print("Val Acc (exclude timeout):", val_acc)
    print("Test Acc (exclude timeout):", test_acc)

    if args.task == "2class":
        if classification_report is not None:
            print("Val 分类报告（默认阈值）:")
            print(classification_report_text(y_val, val_pred_label, [0, 1], ["SL", "TP"]))
            print("Test 分类报告（默认阈值）:")
            print(classification_report_text(y_test, test_pred_label, [0, 1], ["SL", "TP"]))
        thr = find_threshold_for_recall_range(val_pred, y_val, args.tp_recall_min, args.tp_recall_max)
        if thr is not None:
            val_thr_pred = (val_pred >= thr).astype(np.int64)
            test_thr_pred = (test_pred >= thr).astype(np.int64)
            prec, rec = tp_precision_recall(y_val, val_thr_pred)
            print(f"Val 阈值={thr:.4f} | 止盈 Precision={prec:.4f} Recall={rec:.4f}")
            if classification_report is not None:
                print("Val 分类报告（阈值后）:")
                print(classification_report_text(y_val, val_thr_pred, [0, 1], ["SL", "TP"]))
            prec_t, rec_t = tp_precision_recall(y_test, test_thr_pred)
            print(f"Test 阈值={thr:.4f} | 止盈 Precision={prec_t:.4f} Recall={rec_t:.4f}")
            if classification_report is not None:
                print("Test 分类报告（阈值后）:")
                print(classification_report_text(y_test, test_thr_pred, [0, 1], ["SL", "TP"]))


if __name__ == "__main__":
    main()
