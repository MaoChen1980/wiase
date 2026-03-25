#!/usr/bin/env python3
"""113维训练脚本 - NN-based Behavior Selection"""
import os
import struct
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

INPUT_DIM = 113
HIDDEN1 = 128
HIDDEN2 = 64
HIDDEN3 = 32
OUTPUT_DIM = 1
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 0.001

class ValueNN(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN2, HIDDEN3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN3, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.net(x)

class SoccerDataset(Dataset):
    def __init__(self, data_file):
        self.features = []
        self.values = []

        with open(data_file, 'r') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    if len(sample['f']) == INPUT_DIM:
                        self.features.append(sample['f'])
                        self.values.append(sample['v'])
                except:
                    continue

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.values = torch.tensor(self.values, dtype=torch.float32).unsqueeze(1)

        print(f"Loaded {len(self.features)} samples, dim={len(self.features[0])}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.values[idx]

def train(data_file, output_prefix="value_nn_113d"):
    dataset = SoccerDataset(data_file)

    if len(dataset) == 0:
        print("No valid samples found!")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = ValueNN(INPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for features, values in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, values)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, values in val_loader:
                outputs = model(features)
                loss = criterion(outputs, values)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        print(f"Epoch {epoch+1}/{EPOCHS}: Train={train_loss:.4f}, Val={val_loss:.4f}")

    pth_path = f"models/{output_prefix}.pth"
    bin_path = f"models/{output_prefix}.bin"
    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), pth_path)
    print(f"Saved to {pth_path}")

    save_binary_model(model, bin_path)
    print(f"Saved to {bin_path}")

def save_binary_model(model, bin_path):
    state_dict = torch.load(bin_path.replace('.bin', '.pth'), map_location='cpu')

    with open(bin_path, 'wb') as f:
        f.write(struct.pack('i', 1))

        # mean, std
        mean = np.zeros(INPUT_DIM, dtype=np.float32)
        std = np.ones(INPUT_DIM, dtype=np.float32)
        f.write(struct.pack(f'{INPUT_DIM}f', *mean))
        f.write(struct.pack(f'{INPUT_DIM}f', *std))

        # w1: (INPUT_DIM, HIDDEN1)
        w1 = state_dict['net.0.weight'].numpy().flatten()
        b1 = state_dict['net.0.bias'].numpy()
        f.write(struct.pack(f'{INPUT_DIM * HIDDEN1}f', *w1))
        f.write(struct.pack(f'{HIDDEN1}f', *b1))

        # w2: (HIDDEN1, HIDDEN2)
        w2 = state_dict['net.3.weight'].numpy().flatten()
        b2 = state_dict['net.3.bias'].numpy()
        f.write(struct.pack(f'{HIDDEN1 * HIDDEN2}f', *w2))
        f.write(struct.pack(f'{HIDDEN2}f', *b2))

        # w3: (HIDDEN2, HIDDEN3)
        w3 = state_dict['net.6.weight'].numpy().flatten()
        b3 = state_dict['net.6.bias'].numpy()
        f.write(struct.pack(f'{HIDDEN2 * HIDDEN3}f', *w3))
        f.write(struct.pack(f'{HIDDEN3}f', *b3))

if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else "Logfiles/data_latest.json"

    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        sys.exit(1)

    train(data_file, "value_nn_113d")
