import torch
import time
from sklearn.metrics import f1_score
from tools import multilabel_confusion_matrix_mix, plot_custom_confusion_matrix, set_seed
import os, json
from torch.utils.data import DataLoader, random_split
import argparse
from models import ResNet32, BiLSTMModel
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from dataset.dataset import *
import numpy as np

def test_model_with_path_tracking(model, test_loader, criterion, save_path, title = 'Confusion Matrix'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    avg_loss = total_loss / len(test_loader)
    avg_time_per_sample = total_time / len(y_true)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred) 

    return y_true, y_pred, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Resnet32', choices=['Resnet32', 'BiLSTM'], help='Model type to use for training')
    parser.add_argument('--data',type=str)
    parser.add_argument('--output_dir',type=str)
    args = parser.parse_args()
    model_type = args.model
    data_file = args.data
    output_dir = args.output_dir

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    dir = os.path.join(os.getcwd(), 'models', 'benchpress', model_type, 'BP_data_new_skeleton', 'test_assigned_20')
    class_names = {0: 'tilting_to_the_left', 1: 'tilting_to_the_right', 2: 'elbows_flaring', 3: 'scapular_protraction'}
    
    results_dir = os.path.join(dir, output_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    data_path = os.path.join(os.getcwd(), 'data', data_file, 'data.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_f1 = {v : [] for v in class_names.values()}
    seeds = [42, 2023, 7, 88, 100, 999]

    for se in seeds:
        print(f'Testing seed: {se}')
        gen = torch.Generator().manual_seed(se)  # 為每個seed創建獨立生成器
        set_seed(se)
        y_ts = {i: [] for i in range(len(class_names))}
        y_ps = {i: [] for i in range(len(class_names))}
        random_keys = random.sample(list(map(int, data.keys())), 20)
        test_data = {str(k): data[str(k)] for k in random_keys}
        train_data = {str(k): data[str(k)] for k in data if int(k) not in random_keys}
        for GT_class, class_name in class_names.items():
            model_path = os.path.join(dir, class_name, f"{model_type}_model_seed{se}.pth")
            
            # 讀取 dataset
            full_train_dataset = Dataset_Benchpress(train_data, GT_class)
            test_dataset = Dataset_Benchpress(test_data, GT_class)
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
            
            category_ratio = full_train_dataset.get_ratio()
            P_ratio = category_ratio[1]
            input_dim = full_train_dataset.dim

            # 測試
            if model_type == 'BiLSTM':
                model = BiLSTMModel(input_dim).to(device)
            elif model_type == 'Resnet32':
                model = ResNet32(input_dim).to(device)
            class_counts = torch.tensor([P_ratio, 1 - P_ratio])
            criterion = CrossEntropyLoss(weight=(1.0 / class_counts).to(device))

            y_true, y_pred, f1 = test_model_with_path_tracking(
                model, test_loader, criterion, model_path, title=class_names[GT_class]
            )
            y_ts[GT_class].extend([y.astype(int) for y in y_true])
            y_ps[GT_class].extend([y.astype(int) for y in y_pred])
            all_f1[class_name].append((se, f'{f1:.4f}'))
            
        y_ts = np.array(list(y_ts.values())).T
        y_ps = np.array(list(y_ps.values())).T
        
        cm = multilabel_confusion_matrix_mix(y_ts, y_ps, len(class_names))
        print('Finished calculating confusion matrix for seed:', se)
        plot_custom_confusion_matrix(cm, ['right'] + list(class_names.values()), f"{results_dir}/confusion_matrix_mix_{se}.png", f1)
        
    best_f1 = -1
    for class_name, tups in all_f1.items():
        f1s = []
        for tup in tups:
            f1s.append(float(tup[1]))
        f1s = np.array(f1s)
        all_f1[class_name].append(f'{np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
        
    with open(f"{results_dir}/test_summary.txt", 'w', encoding="utf-8") as f:
        f.write(f"Model: {model_type}\n")
        f.write(f"Training Data: BP_data_new_skeleton\n")
        f.write(f"Testing Data: {data_file}\n")
        f.write("="*80 + "\n")
        f.write("F1 Score Results\n")
        f.write("="*80 + "\n\n")
        
        for class_name, scores in all_f1.items():
            f.write(f"Class: {class_name.replace('_', ' ').title()}\n")
            f.write("-" * 50 + "\n")
            
            # 寫入各個 seed 的結果
            for seed, score in scores[:-1]:  # 排除最後的統計值
                f.write(f"  Seed {seed:4d}: {score}\n")
            
            # 寫入平均值和標準差
            f.write(f"  Average:    {scores[-1]}\n\n")