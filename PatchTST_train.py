import torch
import os, json
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import PatchTSTClassifier
from sklearn.metrics import f1_score
from tools import *
import argparse
from PatchTST_test import test_model_with_path_tracking
import math

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path, num_epochs=150, patience=8):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)  # [B, num_classes]
            preds = (probs > 0.5).int()     # [B, num_classes]
            y_true.extend(labels.cpu().numpy().tolist())    # labels shape: (batch, num_classes)
            y_pred.extend(preds.tolist())                   # preds shape: (batch, num_classes)

        avg_loss = total_loss / len(train_loader)
        train_f1 = f1_score(y_true, y_pred, average='macro')
        val_f1 = compute_f1_score(model, valid_loader)

        scheduler.step()
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

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
    
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', type=str)
    parser.add_argument('--split_training', type=bool)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of subset workers for DataLoader')
    args = parser.parse_args()
    seeds = [42, 2023, 7, 88, 100, 999]
    
    from dataset import *
    
    if args.sport == 'deadlift':
        data_path = os.path.join(os.getcwd(), 'data', 'deadlift', '2D_traindata_Final')
        full_dataset = Dataset_TST_Deadlift(data_path)
        save_dir = f'./models/deadlift/TST_Deadlift/15'
        num_classes = 4
        input_len = 110
        
    elif args.sport == 'benchpress':
        data_path = os.path.join(os.getcwd(), 'data', args.sport, 'no_wrist', 'data.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        save_dir = './models/benchpress/TST_Benchpress/system_use/random_seed_no_wrist'
        num_classes = 4
        input_len = 100
        
        if args.split_training:
            # 打亂所有 keys
            all_keys = list(map(str, all_data.keys()))
            random.shuffle(all_keys)

            # 切成六份
            num_folds = 6
            fold_size = len(all_keys) // num_folds
            folds = [all_keys[i*fold_size:(i+1)*fold_size] for i in range(num_folds)]
            
            # 如果不能整除，把剩下的分配到前幾份
            remainder = len(all_keys) % num_folds
            for i in range(remainder):
                folds[i].append(all_keys[num_folds*fold_size + i])
                
            # 生成六組 train/test
            datasets = []
            for i in range(num_folds):
                test_keys = folds[i]
                train_keys = [k for j, f in enumerate(folds) if j != i for k in f]

                test_data = {str(k): all_data[str(k)] for k in test_keys}
                train_data = {str(k): all_data[str(k)] for k in train_keys}
                datasets.append((train_data, test_data))
        else:
            # 打亂所有 keys
            all_keys = list(map(str, all_data.keys()))
            datasets = []
            for se in seeds:
                set_seed(se)
                random.shuffle(all_keys)
                
                train_len = int(len(all_keys) * 0.95)
                train_keys = all_keys[:train_len]
                test_keys = all_keys[train_len:]

                train_data = {str(k): all_data[str(k)] for k in train_keys}
                test_data = {str(k): all_data[str(k)] for k in test_keys}
                datasets.append((train_data, test_data))
            num_folds = len(seeds)
            
    best_f1 = -1
    best_seed = None
    best_model_path = ""

    all_f1_scores = []
    cost_times = []
    accuracies = []

    # for i, se in enumerate(seeds):
    for i, (train_data, test_data) in enumerate(datasets):
        # set_seed(se)

        # 分割資料
        # gen = torch.Generator().manual_seed(se)  # 為每個seed創建獨立生成器
        # train_size = int(0.9 * len(full_dataset))
        # valid_size = int(0.05 * len(full_dataset))
        # test_size = int(len(full_dataset)) - train_size - valid_size
        # train_indices, valid_indices, test_indices = random_split(
        #     range(len(full_dataset)), [train_size, valid_size, test_size],
        #     generator=gen
        # )
        train_valid_dataset = Dataset_Benchpress(train_data)
        test_dataset = Dataset_Benchpress(test_data)
        all_indices = list(range(len(train_valid_dataset)))
        random.shuffle(all_indices)
        
        input_dim = train_valid_dataset.dim
        print('Input dimention',input_dim)
        
        train_size = int(0.9 * len(train_valid_dataset))
        valid_size = int(len(train_valid_dataset)) - train_size
        test_size = int(len(test_dataset))
        print(f'train_size : {train_size}, valid_size : {valid_size}, test_size : {test_size}')
        train_indices = all_indices[:train_size]
        valid_indices = all_indices[train_size:]
        test_indices = list(range(len(test_dataset)))
        
        train_dataset = Datasubset(train_valid_dataset, train_indices, transform=True)
        valid_dataset = Datasubset(train_valid_dataset, valid_indices, transform=False)
        test_dataset = Datasubset(test_dataset, test_indices, transform=False)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        # 訓練與測試
        model = PatchTSTClassifier(input_dim, num_classes, input_len).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0003)
        criterion = torch.nn.BCEWithLogitsLoss()
        scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=5, max_epochs=100, min_lr_ratio=0.0)

        save_path = os.path.join(save_dir, f"PatchTST_model_fold{i}.pth")
        txt_dir = os.path.join(save_dir, f"PatchTST_model_fold{i}_results")
        fig_path = os.path.join(txt_dir, f"train_results_fold{i}.png")
        os.makedirs(txt_dir, exist_ok=True)

        train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path)

        avg_loss, f1, avg_time_per_sample, accuracy = test_model_with_path_tracking(
            model, test_loader, criterion, txt_dir, save_path, num_classes
        )
        print(f"Fold {i} Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, cost {avg_time_per_sample} sec")
        all_f1_scores.append(f1)
        cost_times.append(avg_time_per_sample)
        accuracies.append(accuracy)

        if f1 > best_f1:
            best_f1 = f1
            best_seed = i
            best_model_path = save_path

    write_result(model, num_folds, all_f1_scores, accuracies, cost_times, save_dir, best_f1, best_seed, best_model_path)