from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler


class MultiDataLoader(Dataset):
    def __init__(self, xlsx_path, npy_dir, categorical_columns=None, numeric_columns=None,
                 target_columns=['s_Label', 'i_Label', 'all_Label'], key_column='病理号', fillna_strategy='random',
                 train=True, test_size=0.3, random_state=42, batch_size=256):

        self.batch_size = batch_size
        self.data = pd.read_excel(xlsx_path)


        for col in self.data.columns:
            if self.data[col].isnull().sum() > 0:
                if fillna_strategy == 'random':
                    self.data[col] = self.data[col].apply(
                        lambda x: np.random.choice(self.data[col].dropna().values) if pd.isnull(x) else x)
                elif fillna_strategy == 'mean':
                    self.data[col] = self.data[col].fillna(self.data[col].mean())
                elif fillna_strategy == 'zero':
                    self.data[col] = self.data[col].fillna(0)

        assert self.data.isnull().sum().sum() == 0, "数据仍有缺失值，需检查清理流程！"

        train_data, val_data = train_test_split(
            self.data, test_size=test_size, random_state=random_state, stratify=self.data[target_columns[0]]
        )
        self.data = train_data if train else val_data

        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.target_columns = target_columns
        self.key_column = key_column

        self.npy_dir = os.path.abspath(npy_dir)
        self.npy_files = {os.path.splitext(f)[0]: os.path.join(self.npy_dir, f) for f in os.listdir(self.npy_dir) if
                          f.endswith('.npy')}
        self.data['npy_file'] = self.data[self.key_column].astype(str).map(self.npy_files)

        if self.data['npy_file'].isnull().any():
            missing_keys = self.data[self.data['npy_file'].isnull()][self.key_column].tolist()
            raise ValueError(f"以下样本没有对应的 .npy 文件，请检查数据一致性！缺失的样本: {missing_keys}")

        self.label_encoders = {}
        for target_column in self.target_columns:
            le = LabelEncoder()
            self.data[target_column] = le.fit_transform(self.data[target_column])
            self.label_encoders[target_column] = le

        if self.numeric_columns:
            self.data[self.numeric_columns] = self.data[self.numeric_columns].apply(pd.to_numeric, errors='coerce')
            self.scaler = StandardScaler()
            self.data[self.numeric_columns] = self.scaler.fit_transform(self.data[self.numeric_columns])

        if self.categorical_columns:
            self.cat_encoders = {}
            for col in self.categorical_columns:
                encoder = LabelEncoder()
                self.data[col] = encoder.fit_transform(self.data[col].astype(str))
                self.cat_encoders[col] = encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if self.numeric_columns:
            numeric_input = torch.tensor(row[self.numeric_columns].astype(float).values, dtype=torch.float32)
        else:
            numeric_input = torch.empty(0, dtype=torch.float32)

        if self.categorical_columns:
            categorical_input = torch.tensor(row[self.categorical_columns].astype(int).values, dtype=torch.long)
        else:
            categorical_input = torch.empty(0, dtype=torch.long)

        labels = {}
        for target_column in self.target_columns:
            labels[target_column] = torch.tensor(row[target_column], dtype=torch.long)

        npy_path = row['npy_file']
        npy_data = np.load(npy_path)
        npy_input = torch.tensor(npy_data, dtype=torch.float32)
        patient_id = str(row[self.key_column])

        return categorical_input, numeric_input, npy_input, labels, patient_id


    def get_dataloader(self, shuffle=True):

        return DataLoader(self, batch_size=self.batch_size, shuffle=shuffle)
