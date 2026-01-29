# Heatmap WebDataset 读取要点（VQ-VAE）

## 1. 数据集结构
- 分片文件：`data/train-*.tar.gz`, `data/val-*.tar.gz`, `data/test-*.tar.gz`
- 当前为按天分片，例如：`train-GC_2025-01-17.tar.gz`
- 每个样本包含：
  - `input.npy`：热图张量，形状 `(2, 101, 160)`，dtype 通常为 `float16`
  - 标签字段（若需要监督）：
    - 二分类：`label_2c_{ratio}.cls`
    - 三分类：`label_3c_{ratio}.cls` 或 `label_3c_{name}.cls`
      - 例：`label_3c_1.0_30t.cls`

## 2. 归一化与预处理一致性
- 数据集已在构建阶段做过归一化（默认 `log1p` + gamma）。
- VQ-VAE 训练时建议只做轻量增强（如高斯模糊），不要重复归一化。

## 3. WebDataset 读取示例（PyTorch）
```python
import io
import numpy as np
import torch
import webdataset as wds

def load_npy(x):
    if isinstance(x, (bytes, bytearray, memoryview)):
        return np.load(io.BytesIO(x))
    return x

train_shards = "data/train-*.tar.gz"
ds = (
    wds.WebDataset(train_shards, shardshuffle=100)
      .shuffle(10000)
      .to_tuple("input.npy")
      .map_tuple(load_npy)
)

loader = wds.WebLoader(ds, batch_size=64, num_workers=4)
for x in loader:
    x = torch.from_numpy(x).float()  # (B, 2, 101, 160)
```

## 4. 标签键名检查（如需监督）
```python
import webdataset as wds

sample = next(iter(wds.WebDataset("data/train-*.tar.gz")))
print(sample.keys())
```

## 5. 注意事项
- `input.npy` 为 `float16`，训练前一般转换为 `float32`。
- 形状固定 `(2, 101, 160)`：2 通道（买/卖），101 档（±50 tick），160 时间步（15秒K线）。
- Windows 下建议使用相对路径（如 `data/train-*.tar.gz`），避免路径解析问题。
