#!/usr/bin/env python3
"""
增量训练脚本 - 基于新模型结构 113->64->32->16->1

功能：
1. 加载已有模型继续训练（增量学习）
2. 支持从头训练（--retrain）
3. 自动从目录收集新数据

用法:
    python3 do_train_incremental.py [数据目录] [--epochs N] [--lr RATE] [--retrain] [--help]
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
torch.set_num_threads(1)

import struct
import json
import random
import glob
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============= 模型配置 =============
INPUT_DIM = 113
HIDDEN1 = 64
HIDDEN2 = 32
HIDDEN3 = 16
OUTPUT_DIM = 1

# ============= 训练配置 =============
DEFAULT_MODEL_PATH = "/Users/chenmao/wiase/models/value_nn.bin"
DEFAULT_DATA_DIR = "/Users/chenmao/wiase/Logfiles"
MAX_SAMPLES = 50000
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.0001  # 增量学习用较小学习率

class ValueNN(nn.Module):
    """模型结构: 113 -> 64 -> 32 -> 16 -> 1"""
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN2, HIDDEN3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN3, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.net(x)

def load_model_from_bin(model, bin_path):
    """从bin文件加载模型权重"""
    with open(bin_path, 'rb') as f:
        # magic
        magic = struct.unpack('i', f.read(4))[0]
        if magic != 1:
            raise ValueError(f"Invalid magic number: {magic}")

        # mean, std (不用于推理，仅记录)
        mean = np.frombuffer(f.read(INPUT_DIM * 4), dtype=np.float32)
        std = np.frombuffer(f.read(INPUT_DIM * 4), dtype=np.float32)

        # w1: (64, 113)
        w1 = np.frombuffer(f.read(HIDDEN1 * INPUT_DIM * 4), dtype=np.float32)
        b1 = np.frombuffer(f.read(HIDDEN1 * 4), dtype=np.float32)

        # w2: (32, 64)
        w2 = np.frombuffer(f.read(HIDDEN2 * HIDDEN1 * 4), dtype=np.float32)
        b2 = np.frombuffer(f.read(HIDDEN2 * 4), dtype=np.float32)

        # w3: (16, 32)
        w3 = np.frombuffer(f.read(HIDDEN3 * HIDDEN2 * 4), dtype=np.float32)
        b3 = np.frombuffer(f.read(HIDDEN3 * 4), dtype=np.float32)

        # w4: (1, 16)
        w4 = np.frombuffer(f.read(OUTPUT_DIM * HIDDEN3 * 4), dtype=np.float32)
        b4 = np.frombuffer(f.read(OUTPUT_DIM * 4), dtype=np.float32)

    # 转换为PyTorch格式
    state_dict = model.state_dict()
    state_dict['net.0.weight'] = torch.from_numpy(w1.reshape(HIDDEN1, INPUT_DIM).copy())
    state_dict['net.0.bias'] = torch.from_numpy(b1.copy())
    state_dict['net.3.weight'] = torch.from_numpy(w2.reshape(HIDDEN2, HIDDEN1).copy())
    state_dict['net.3.bias'] = torch.from_numpy(b2.copy())
    state_dict['net.6.weight'] = torch.from_numpy(w3.reshape(HIDDEN3, HIDDEN2).copy())
    state_dict['net.6.bias'] = torch.from_numpy(b3.copy())
    state_dict['net.9.weight'] = torch.from_numpy(w4.reshape(OUTPUT_DIM, HIDDEN3).copy())
    state_dict['net.9.bias'] = torch.from_numpy(b4.copy())

    model.load_state_dict(state_dict)
    return model

def save_model_to_bin(model, bin_path):
    """保存模型为bin格式"""
    state_dict = model.state_dict()

    with open(bin_path, 'wb') as f:
        # magic
        f.write(struct.pack('i', 1))

        # mean, std (全零/全一，不用于推理)
        mean = np.zeros(INPUT_DIM, dtype=np.float32)
        std = np.ones(INPUT_DIM, dtype=np.float32)
        f.write(struct.pack(f'{INPUT_DIM}f', *mean))
        f.write(struct.pack(f'{INPUT_DIM}f', *std))

        # w1: (64, 113) -> flatten
        w1 = state_dict['net.0.weight'].numpy().flatten()
        b1 = state_dict['net.0.bias'].numpy()
        f.write(struct.pack(f'{HIDDEN1 * INPUT_DIM}f', *w1))
        f.write(struct.pack(f'{HIDDEN1}f', *b1))

        # w2: (32, 64) -> flatten
        w2 = state_dict['net.3.weight'].numpy().flatten()
        b2 = state_dict['net.3.bias'].numpy()
        f.write(struct.pack(f'{HIDDEN2 * HIDDEN1}f', *w2))
        f.write(struct.pack(f'{HIDDEN2}f', *b2))

        # w3: (16, 32) -> flatten
        w3 = state_dict['net.6.weight'].numpy().flatten()
        b3 = state_dict['net.6.bias'].numpy()
        f.write(struct.pack(f'{HIDDEN3 * HIDDEN2}f', *w3))
        f.write(struct.pack(f'{HIDDEN3}f', *b3))

        # w4: (1, 16) -> flatten
        w4 = state_dict['net.9.weight'].numpy().flatten()
        b4 = state_dict['net.9.bias'].numpy()
        f.write(struct.pack(f'{OUTPUT_DIM * HIDDEN3}f', *w4))
        f.write(struct.pack(f'{OUTPUT_DIM}f', *b4))

def load_data(data_dir, max_samples=MAX_SAMPLES):
    """从目录加载所有json数据文件"""
    print(f"Loading data from {data_dir}...")

    # 匹配 rl_data_*.json 和 data_*.json 文件
    json_files = glob.glob(os.path.join(data_dir, "rl_data_*.json"))
    json_files.extend(glob.glob(os.path.join(data_dir, "data_*.json")))
    print(f"Found {len(json_files)} data files")

    all_lines = []
    for fp in json_files:
        with open(fp, 'r') as f:
            for line in f:
                all_lines.append(line)

    if len(all_lines) > max_samples:
        random.shuffle(all_lines)
        all_lines = all_lines[:max_samples]
        print(f"Sampled {max_samples} from {len(all_lines)} total lines")

    features, values = [], []
    for line in all_lines:
        try:
            sample = json.loads(line.strip())
            if len(sample.get('f', [])) == INPUT_DIM:
                features.append(sample['f'])
                values.append(sample['v'])
        except:
            continue

    print(f"Loaded {len(features)} valid samples (113-dim)")
    return features, values

def train(data_dir, model_path, epochs, lr, retrain):
    """执行训练"""
    features, values = load_data(data_dir)

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
    print(f"Data shape: X={X.shape}, y={y.shape}")

    train_ds = TensorDataset(X, y)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = ValueNN()

    # 增量学习：加载已有模型
    if not retrain and os.path.exists(model_path):
        print(f"[Incremental] Loading existing model from {model_path}...")
        model = load_model_from_bin(model, model_path)
        print("[Incremental] Model loaded, will continue training...")
    else:
        if retrain:
            print("[Retrain] Training from scratch...")
        else:
            print("[New] No existing model, training from scratch...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"Training: epochs={epochs}, lr={lr}, batch_size={BATCH_SIZE}")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

    # 保存模型
    print(f"Saving model to {model_path}...")
    save_model_to_bin(model, model_path)
    print("Done!")

if __name__ == "__main__":
    import sys

    data_dir = DEFAULT_DATA_DIR
    model_path = DEFAULT_MODEL_PATH
    epochs = EPOCHS
    lr = LEARNING_RATE
    retrain = False

    # 解析参数
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--help" or args[i] == "-h":
            print(__doc__)
            print("\nExamples:")
            print("  python3 do_train_incremental.py                                    # 默认参数")
            print("  python3 do_train_incremental.py /path/to/data                      # 指定数据目录")
            print("  python3 do_train_incremental.py --epochs 30 --lr 0.0005           # 指定训练参数")
            print("  python3 do_train_incremental.py --retrain                         # 从头训练")
            sys.exit(0)
        elif args[i] == "--epochs" and i + 1 < len(args):
            epochs = int(args[i + 1])
            i += 2
        elif args[i] == "--lr" and i + 1 < len(args):
            lr = float(args[i + 1])
            i += 2
        elif args[i] == "--retrain":
            retrain = True
            i += 1
        elif args[i] == "--model" and i + 1 < len(args):
            model_path = args[i + 1]
            i += 2
        elif not args[i].startswith("-"):
            data_dir = args[i]
            i += 1
        else:
            i += 1

    print("=" * 60)
    print("Incremental Training for Value NN (113->64->32->16->1)")
    print("=" * 60)
    print(f"  Data directory: {data_dir}")
    print(f"  Model path:     {model_path}")
    print(f"  Epochs:         {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Retrain:        {retrain}")
    print("=" * 60)

    train(data_dir, model_path, epochs, lr, retrain)