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


def encode_codebook_hist(loader, model, device, n_embeddings, task, normalize_hist, max_samples):
    features = []
    labels = []
    model.eval()
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
            if max_samples and len(labels) >= max_samples:
                break
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


def train_xgboost(X_train, y_train, X_val, y_val, params, rounds):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    return xgb.train(params, dtrain, num_boost_round=rounds, evals=[(dtrain, "train"), (dval, "val")])


def main():
    parser = argparse.ArgumentParser(description="VQ-VAE + XGBoost on heatmap shards")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--train-shards", default="")
    parser.add_argument("--val-shards", default="")
    parser.add_argument("--test-shards", default="")
    parser.add_argument("--label-key", default="")
    parser.add_argument("--task", choices=["2class", "3class"], default="2class")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle-buf", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=400)
    parser.add_argument("--count-samples", action="store_true", default=False)
    parser.add_argument("--val-steps", type=int, default=200)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--var-batches", type=int, default=50)

    parser.add_argument("--n-hiddens", type=int, default=128)
    parser.add_argument("--n-residual-hiddens", type=int, default=32)
    parser.add_argument("--n-residual-layers", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--n-embeddings", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--input-channels", type=int, default=2)

    parser.add_argument("--feature-out-dir", default="features")
    parser.add_argument("--vqvae-out", default="results/vqvae_heatmap.pth")
    parser.add_argument("--xgb-out", default="results/xgb_model.json")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-normalize-hist", action="store_false", dest="normalize_hist")
    parser.set_defaults(normalize_hist=True)

    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-eta", type=float, default=0.1)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample", type=float, default=0.8)
    parser.add_argument("--xgb-rounds", type=int, default=300)
    parser.add_argument("--exclude-timeout-eval", action="store_true", default=True)
    parser.add_argument("--include-timeout-eval", action="store_false", dest="exclude_timeout_eval")

    args = parser.parse_args()

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

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch <= 0:
        if not args.count_samples:
            raise ValueError("steps_per_epoch is required unless --count-samples is set.")
        train_count = count_samples(train_shards, label_key)
        print(f"Samples | train={train_count}")
        steps_per_epoch = max(1, (train_count + args.batch_size - 1) // args.batch_size)

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
        shuffle=False,
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
        args.val_steps,
    )

    os.makedirs(args.feature_out_dir, exist_ok=True)
    val_loader = make_loader(val_shards, label_key, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_shards, label_key, args.batch_size, shuffle=False, num_workers=args.num_workers)

    max_samples = args.max_samples if args.max_samples > 0 else None
    X_train, y_train = encode_codebook_hist(
        make_loader(train_shards, label_key, args.batch_size, shuffle=False, num_workers=args.num_workers),
        model,
        device,
        args.n_embeddings,
        args.task,
        args.normalize_hist,
        max_samples,
    )
    X_val, y_val = encode_codebook_hist(
        val_loader,
        model,
        device,
        args.n_embeddings,
        args.task,
        args.normalize_hist,
        max_samples,
    )
    X_test, y_test = encode_codebook_hist(
        test_loader,
        model,
        device,
        args.n_embeddings,
        args.task,
        args.normalize_hist,
        max_samples,
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


if __name__ == "__main__":
    main()
