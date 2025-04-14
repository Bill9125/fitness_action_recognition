from torch.utils.data import Dataset
import os, glob
import torch

class Dataset_25f(Dataset):
    def __init__(self, dataset, GT_class):
        self.sample_paths = []   
        self.features = []       
        self.labels = []    
        counter = 0     
        
        valid_categories = {'Category_1', 'Category_2', 'Category_3', 'Category_4', 'Category_5'}
        categories = [path for path in glob.glob(os.path.join(dataset, '*')) if os.path.basename(path) in valid_categories]
        
        T_recordings = os.listdir(os.path.join(dataset, f'Category_{GT_class}'))
        
        for category in categories:
            recordings = glob.glob(os.path.join(category, '*'))
            for recording in recordings:
                # 判斷此影片是否在正確的分類
                if os.path.basename(category) != f'Category_{GT_class}':
                    if os.path.basename(recording) in T_recordings:
                        print(f'{recording} is overlapped with True recording.')
                        continue
                        
                delta_path = os.path.join(recording, 'filtered_delta_norm')
                delta2_path = os.path.join(recording, 'filtered_delta2_norm')
                square_path = os.path.join(recording, 'filtered_delta_square_norm')
                zscore_path = os.path.join(recording, 'filtered_zscore_norm')
                orin_path = os.path.join(recording, 'filtered_norm')
                
                if not all(map(os.path.exists, [delta_path, delta2_path, zscore_path, square_path, orin_path])):
                    print(f"Missing data in {recording}")
                    continue
                
                deltas = glob.glob(os.path.join(delta_path, '*.txt'))
                delta2s = glob.glob(os.path.join(delta2_path, '*.txt'))
                squares = glob.glob(os.path.join(square_path, '*.txt'))
                zscores = glob.glob(os.path.join(zscore_path, '*.txt'))
                orins = glob.glob(os.path.join(orin_path, '*.txt'))
                
                data_per_ind = list(self.fetch(zip(deltas, delta2s, zscores, squares, orins)))
                self.features.extend(data_per_ind)
                self.labels.extend([1 if f'Category_{GT_class}' in category else 0] * len(data_per_ind))
                for _ in range(len(data_per_ind)):
                    counter += 1
                
            print(f'{category} have {counter} samples')
            counter = 0
    
    def fetch(self, uds):
        """
        处理数据并返回特征和对应的文件路径
        """
        data_per_ind = []
        
        # 對每一下做處理
        for ud in uds:
            parsed_data = []
            
            for file in ud:
                with open(file, 'r') as f:
                    lines = f.read().strip().split('\n')
                    parsed_data.append([list(map(float, line.split(','))) for line in lines])
            self.sample_paths.append(file)
            
            for num in zip(*parsed_data):
                # 將 num 裡的數據變成 25*1 
                frame_data = [item for sublist in num for item in sublist]
                data_per_ind.append(frame_data)
                
                if len(data_per_ind) == 110:  # 达到110帧时返回
                    yield data_per_ind
                    data_per_ind = []

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
        torch.tensor(self.features[idx], dtype=torch.float32),
        torch.tensor(self.labels[idx], dtype=torch.long),
        idx  # 把原始 full_dataset 的 index 傳出
    )
    
    def get_sample_path(self, idx):
        return self.sample_paths[idx]
    
class Dataset_19f(Dataset):
    def __init__(self, dataset, GT_class):
        self.sample_paths = []   
        self.features = []       
        self.labels = []    
        counter = 0     
        
        valid_categories = {'Category_1', 'Category_2', 'Category_3', 'Category_4', 'Category_5'}
        categories = [path for path in glob.glob(os.path.join(dataset, '*')) if os.path.basename(path) in valid_categories]
        
        T_recordings = os.listdir(os.path.join(dataset, f'Category_{GT_class}'))
        
        for category in categories:
            recordings = glob.glob(os.path.join(category, '*'))
            for recording in recordings:
                # 判斷此影片是否在正確的分類
                if os.path.basename(category) != f'Category_{GT_class}':
                    if os.path.basename(recording) in T_recordings:
                        print(f'{recording} is overlapped with True recording.')
                        continue
                        
                delta_path = os.path.join(recording, 'filtered_delta_norm')
                delta2_path = os.path.join(recording, 'filtered_delta2_norm')
                square_path = os.path.join(recording, 'filtered_delta_square_norm')
                zscore_path = os.path.join(recording, 'filtered_zscore_norm')
                orin_path = os.path.join(recording, 'filtered_norm')
                
                if not all(map(os.path.exists, [delta_path, delta2_path, zscore_path, square_path, orin_path])):
                    print(f"Missing data in {recording}")
                    continue
                
                deltas = glob.glob(os.path.join(delta_path, '*.txt'))
                delta2s = glob.glob(os.path.join(delta2_path, '*.txt'))
                squares = glob.glob(os.path.join(square_path, '*.txt'))
                zscores = glob.glob(os.path.join(zscore_path, '*.txt'))
                orins = glob.glob(os.path.join(orin_path, '*.txt'))
                
                data_per_ind = list(self.fetch(zip(deltas, delta2s, zscores, squares, orins)))
                self.features.extend(data_per_ind)
                self.labels.extend([1 if f'Category_{GT_class}' in category else 0] * len(data_per_ind))
                for _ in range(len(data_per_ind)):
                    counter += 1
                
            print(f'{category} have {counter} samples')
            counter = 0
    
    def fetch(self, uds):
        """
        处理数据并返回特征和对应的文件路径
        """
        data_per_ind = []
        
        # 對每一下做處理
        for ud in uds:
            parsed_data = []
            
            for file in ud:
                with open(file, 'r') as f:
                    lines = f.read().strip().split('\n')
                    parsed_data.append([list(map(float, line.split(','))) for line in lines])
            self.sample_paths.append(file)
            
            for num in zip(*parsed_data):
                # 將 num 裡的數據變成 25*1 
                frame_data = [item for sublist in num for item in sublist]
                data_per_ind.append(frame_data)
                
                if len(data_per_ind) == 110:  # 达到110帧时返回
                    yield data_per_ind
                    data_per_ind = []

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
        torch.tensor(self.features[idx], dtype=torch.float32),
        torch.tensor(self.labels[idx], dtype=torch.long),
        idx  # 把原始 full_dataset 的 index 傳出
    )
    
    def get_sample_path(self, idx):
        return self.sample_paths[idx]