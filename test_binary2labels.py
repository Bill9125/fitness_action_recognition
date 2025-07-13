import torch
import time
from tools import set_seed
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split
import argparse
from models import ResNet32, BiLSTMModel
from torch.nn import CrossEntropyLoss
from dataset import *
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def test_model_with_path_tracking(model, test_loader, criterion, save_path, class_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    model.eval()
    
    total_loss, total_time = 0.0, 0.0  
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels, indices in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return y_true, y_pred

def dataset_setup(data_path, GT_class):
    full_dataset = Dataset_Benchpress(data_path, GT_class)
    category_ratio = full_dataset.get_ratio()
    P_ratio = category_ratio[1]

    print('input_dim', full_dataset.dim)
    
    print(f'Category : {category_ratio}')
    train_size = int(0.75 * len(full_dataset))
    valid_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size
    # 分割資料
    gen = torch.Generator().manual_seed(se)  # 為每個seed創建獨立生成器

    train_indices, valid_indices, test_indices = random_split(
        range(len(full_dataset)), [train_size, valid_size, test_size],
        generator=gen
    )
    test_dataset  = ResnetSubset(full_dataset, test_indices, transform=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    return test_loader, P_ratio, full_dataset.dim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str)
    args = parser.parse_args()
    data = args.data
    all_true = []
    all_preds = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    class_names = {0: 'tilting_to_the_left', 1: 'tilting_to_the_right', 2: 'elbows_flaring', 3: 'scapular_protraction'}
    data_path = os.path.join(os.getcwd(), 'data', data, 'bench_press_multilabel_cut4.csv')
    
    seeds = [42, 2023, 7, 88, 100, 999]
    all_preds = []
    all_gts = []

    for model_type in ['BiLSTM', 'Resnet32']:
        for se in seeds:
            set_seed(se)
            per_model_binary_preds = []  # 每一個模型的 binary 預測（0/1）

            for GT_class, class_name in class_names.items():
                # 取得 test loader
                test_loader, P_ratio, input_dim = dataset_setup(data_path, GT_class)
                class_counts = torch.tensor([P_ratio, 1 - P_ratio])
                criterion = CrossEntropyLoss(weight=(1.0 / class_counts).to(device))
                if model_type == 'BiLSTM':
                    model = BiLSTMModel(input_dim).to(device)
                elif model_type == 'Resnet32':
                    model = ResNet32(input_dim).to(device)
                save_dir = os.path.join(os.getcwd(), 'models', 'benchpress', model_type, data)
                save_path = os.path.join(save_dir, class_name, f"{model_type}_model_seed{se}.pth")
                y_true, y_scores = test_model_with_path_tracking(model, test_loader, criterion, save_path, class_name)

            # 第一次 loop 儲存 ground truth（應該是 multi-hot vector）
            if len(all_gts) == 0:
                all_gts = [[label] for label in y_true]
            else:
                for i, label in enumerate(y_true):
                    all_gts[i].append(label)

            y_pred = [1 if score > 0.5 else 0 for score in y_scores]
            if len(per_model_binary_preds) == 0:
                per_model_binary_preds = [[v] for v in y_pred]
            else:
                for i, v in enumerate(y_pred):
                    per_model_binary_preds[i].append(v)
        
        # 最後得到 per_model_binary_preds: (num_samples, 4)
        all_preds = per_model_binary_preds  # shape: [num_samples, 4]
        all_gts = [list(v) for v in all_gts]  # shape: [num_samples, 4]

        # 計算 multilabel confusion matrix
        mcm = multilabel_confusion_matrix(all_gts, all_preds)

        # 建立儲存目錄
        save_dir = os.path.join(os.getcwd(), 'models', 'benchpress', model_type, data)
        os.makedirs(save_dir, exist_ok=True)

        # 畫出每一類別的 confusion matrix 並儲存
        for i, cm in enumerate(mcm):
            plt.figure(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
            plt.title(f"Confusion Matrix: {class_names[i]} (Seed {se})")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            fig_path = os.path.join(save_dir, f"confusion_matrix_class{i}_{class_names[i]}_seed{se}.png")
            plt.savefig(fig_path)
            plt.close()
            print(f"Saved heatmap to {fig_path}")