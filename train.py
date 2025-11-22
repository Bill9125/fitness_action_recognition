import torch
import os
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
import argparse
from models import ResNet32, BiLSTMModel
from tools import set_seed, f1_score, compute_f1_score, write_results
from test import test_model_with_path_tracking
from dataset import *
import json, random
import math


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path, num_epochs=100, patience=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_f1 = 0.0  # 用來儲存最佳 F1-score
    patience_counter = 0

    # **存放訓練過程的數據**
    train_losses = []
    train_f1_scores = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        y_true, y_pred = [], []

        for inputs, labels, indices in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        train_f1 = f1_score(y_true, y_pred)
        val_f1 = compute_f1_score(model, valid_loader)

        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        scheduler.step()
        # **紀錄數據**
        train_losses.append(avg_loss)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

        # 根據 F1-score 儲存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print("✅ Model Saved (Best F1-score)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early Stopping Triggered")
                break

    # **繪製 Loss 和 F1-score**
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label="Train Loss", color='blue', marker='o')
    plt.plot(epochs, train_f1_scores, label="Train F1-score", color='green', marker='s')
    plt.plot(epochs, val_f1_scores, label="Validation F1-score", color='red', marker='d')

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Training Loss & F1-score per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")  # 儲存高解析度圖片

# ----------------------
# Validation Function
# ----------------------
def validate_model(model, valid_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels, indices in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(valid_loader)

def get_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs: int,
    max_epochs: int,
    min_lr_ratio: float = 0.0,
):
    """
    warmup_epochs:   線性 warmup epoch 數（從 0 -> base_lr）
    max_epochs:      總訓練 epoch 數
    min_lr_ratio:    最小 lr / base_lr，比方 0.0 就是衰到 0
    """
    assert warmup_epochs < max_epochs

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # 線性 warmup: 0 -> 1
            return float(current_epoch + 1) / float(warmup_epochs)
        # cosine decay: 1 -> min_lr_ratio
        progress = (current_epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ----------------------
# (6) Main Execution
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--GT_class',type=int)
    parser.add_argument('--model', type=str, default='BiLSTM', choices=['Resnet32', 'BiLSTM'], help='Model type to use for training')
    parser.add_argument('--data',type=str)
    args = parser.parse_args()
    GT_class = args.GT_class
    model_type = args.model
    data_file = args.data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = {0: 'wrists_bending_backward', 1: 'tilting_to_the_left', 2: 'tilting_to_the_right', 3: 'elbows_flaring', 4: 'scapular_protraction'}
    
    data_path = os.path.join(os.getcwd(), 'data', data_file, 'data_wrist.json')
    save_dir = os.path.join(os.getcwd(), 'models', 'benchpress', model_type, data_file, 'test_random_seed_wrist', class_names[GT_class])
    os.makedirs(save_dir, exist_ok=True)
    print(f'read {data_path} as dataset ...')
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    best_f1 = -1
    best_seed = None
    best_model_path = ""

    all_f1_scores = []
    all_avg_times = []
    all_acc = []
    # 打亂所有 keys
    all_keys = list(map(str, data.keys()))
    # random.shuffle(all_keys)

    # # 切成六份
    # num_folds = 6
    # fold_size = len(all_keys) // num_folds
    # folds = [all_keys[i*fold_size:(i+1)*fold_size] for i in range(num_folds)]

    # # 如果不能整除，把剩下的分配到前幾份
    # remainder = len(all_keys) % num_folds
    # for i in range(remainder):
    #     folds[i].append(all_keys[num_folds*fold_size + i])
    
    # # 生成六組 train/test
    # datasets = []
    # for i in range(num_folds):
    #     test_keys = folds[i]
    #     train_keys = [k for j, f in enumerate(folds) if j != i for k in f]

    #     test_data = {str(k): data[str(k)] for k in test_keys}
    #     train_data = {str(k): data[str(k)] for k in train_keys}
    #     datasets.append((train_data, test_data))

    n_total = len(all_keys)
    split_idx = int(n_total * 0.9)
    seeds = [42, 2023, 7, 88, 100, 999]
    datasets = []
    for se in seeds:
        set_seed(se)
        perm = torch.randperm(n_total).tolist() 
        shuffled_keys = [all_keys[i] for i in perm]
        train_keys = shuffled_keys[:split_idx]
        test_keys = shuffled_keys[split_idx:]

        test_data = {str(k): data[str(k)] for k in test_keys}
        train_data = {str(k): data[str(k)] for k in train_keys}
        datasets.append((train_data, test_data))
        
    for i, (train_data, test_data) in enumerate(datasets):
        train_valid_dataset = Dataset_Benchpress(train_data, GT_class)
        test_dataset = Dataset_Benchpress(test_data, GT_class)
        all_indices = list(range(len(train_valid_dataset)))
        random.shuffle(all_indices)
        
        category_ratio = train_valid_dataset.get_ratio()
        P_ratio = category_ratio[1]
        input_dim = train_valid_dataset.dim
        print('input_dim',input_dim)
        print(f'Category : {category_ratio}')
        
        train_size = int(0.95 * len(train_valid_dataset))
        valid_size = int(len(train_valid_dataset)) - train_size
        test_size = int(len(test_dataset))
        print(f'train_size : {train_size}, valid_size : {valid_size}, test_size : {test_size}')
        
        train_indices = all_indices[:train_size]
        valid_indices = all_indices[train_size:]
        
        train_dataset = ResnetSubset(train_valid_dataset, train_indices, transform=True)
        valid_dataset = ResnetSubset(train_valid_dataset, valid_indices, transform=False)
        test_dataset = ResnetSubset(test_dataset, list(range(test_size)), transform=False)
        
        train_labels = [train_valid_dataset.labels[i] for i in train_indices]

        # 建立 Weighted Sampler
        class_weights = [1.0 / sum(np.array(train_labels) == i) for i in range(2)]
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                num_samples=len(train_dataset),
                                replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
        valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # 訓練與測試
        if model_type == 'BiLSTM':
            model = BiLSTMModel(input_dim).to(device)
        elif model_type == 'Resnet32':
            model = ResNet32(input_dim).to(device)
        class_counts = torch.tensor([P_ratio, 1 - P_ratio])
        criterion = CrossEntropyLoss(weight=(1.0 / class_counts).to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = get_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=5,
            max_epochs=100,
            min_lr_ratio=0.01,
        )

        save_path = os.path.join(save_dir, f"{model_type}_model_seed{i}.pth")
        txt_dir = os.path.join(save_dir, f"{model_type}_train_results_seed{i}_results")
        fig_path = os.path.join(txt_dir, f"{model_type}_train_results_seed{i}.png")
        os.makedirs(txt_dir, exist_ok=True)

        train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path)

        avg_loss, f1, acc, avg_time_per_sample = test_model_with_path_tracking(
            model, test_loader, criterion, txt_dir, save_path, title=class_names[GT_class]
        )

        print(f"Seed {i} Test F1: {f1:.4f}")
        all_f1_scores.append(f1)
        all_avg_times.append(avg_time_per_sample)
        all_acc.append(acc)

        if f1 > best_f1:
            best_f1 = f1
            best_seed = i
            best_model_path = save_path

    # write_results(model, input_dim, category_ratio, num_folds, all_f1_scores, all_avg_times, all_acc, best_f1, best_seed, best_model_path, save_dir)
    write_results(model, input_dim, category_ratio, seeds, all_f1_scores, all_avg_times, all_acc, best_f1, best_seed, best_model_path, save_dir)