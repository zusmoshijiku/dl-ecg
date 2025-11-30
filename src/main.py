import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import random
import os
import torch
import torch.nn as nn
from models.rnn import CRNN_Model
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from neurokit2 import ecg
import neurokit2 as nk
import numpy as np
import stumpy
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import torch.nn.functional as F

class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 15, stride: int = 1):
        super(ResBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return F.relu(x)



class CNN_Feature_Extractor_12Lead(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 4):
        super(CNN_Feature_Extractor_12Lead, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.res_block1 = ResBlock1D(64, 64)
        self.res_block2 = ResBlock1D(64, 128, stride=2)
        self.res_block3 = ResBlock1D(128, 256, stride=2)
        self.res_block4 = ResBlock1D(256, 512, stride=2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        return x
    

class CRNN_Model(nn.Module):
    def __init__(self, n_channels_cnn: int = 12, 
                 rnn_hidden_size: int = 128, 
                 rnn_num_layers: int = 2,
                 num_classes: int = 4,
                 bidirectional: bool = True):
        super(CRNN_Model, self).__init__()
        self.cnn_features = CNN_Feature_Extractor_12Lead(in_channels=n_channels_cnn)
        cnn_output_features = 512 
        self.rnn = nn.LSTM(input_size=cnn_output_features, 
                           hidden_size=rnn_hidden_size, 
                           num_layers=rnn_num_layers,
                           batch_first=True, 
                           dropout=0.3,
                           bidirectional=bidirectional)
        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(rnn_hidden_size * num_directions, num_classes)

    def forward(self, x):
        x = self.cnn_features(x)
        x = x.permute(0, 2, 1)
        out, (hn, cn) = self.rnn(x)
        last_output = out[:, -1, :]
        logits = self.fc(last_output)
        return logits
    


SAMPLING_RATE = 1000

class ECGDataset(Dataset):
    def __init__(self, data_folder, class_folders, files_per_class=200, mp_window = 1000):
        self.samples = []
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.mp_window = mp_window
        for folder, label in class_folders.items():
            files = glob.glob(os.path.join(data_folder, folder, '*.parquet.gzip'))
            if len(files) >= files_per_class:
                files = random.sample(files, files_per_class)
            else:
                files = random.choices(files, k=files_per_class)

            for f in files:
                try:
                    df = pd.read_parquet(f, engine='fastparquet')
                except Exception as e:
                    print(f"Failed to read {f}: {e}")
                    continue

                # ensure required lead columns exist
                if not set(self.leads).issubset(df.columns):
                    print(f"Missing leads in {f}, skipping")
                    continue
                df_leads = df[self.leads].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

                # shape -> (12, time)
                signal = df_leads.values.T

                matrix_profiles = []
                for i in range(signal.shape[0]):
                    mp = stumpy.stump(signal[i].astype(np.float64), m=self.mp_window)[:, 0].astype(np.float32)
                    pad_width = signal.shape[1] - len(mp)
                    padded_mp = np.pad(mp, (0, pad_width), 'constant', constant_values=0)
                    padded_mp[np.isinf(padded_mp)] = 1e9
                    matrix_profiles.append(padded_mp)
                matrix_profiles_np = np.array(matrix_profiles, dtype=np.float32)
                combined_signal = np.concatenate((signal, matrix_profiles_np), axis=0)
                # self.samples.append((torch.tensor(combined_signal, dtype=torch.float32), label, os.path.basename(f)))
                self.samples.append((torch.tensor(matrix_profiles_np, dtype=torch.float32), label, os.path.basename(f)))
                # self.samples.append((torch.tensor(signal, dtype=torch.float32), label, os.path.basename(f)))
    def process_file(self, f, label):
        try:
            df = pd.read_parquet(f, engine='fastparquet')
        except Exception as e:
            print(f"Failed to read {f}: {e}")
            return None
        
        if not set(self.leads).issubset(df.columns):
            print(f"Missing leads in {f}, skipping")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        signal, label, ecg_id = self.samples[idx]
        return signal, label, ecg_id

class_folders = {
    'arritmia': 0,
    'block': 1,
    'fibrilation': 2,
    'normal': 3
}
data_folder = 'data'
dataset = ECGDataset(data_folder, class_folders, files_per_class=10)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='min', checkpoint_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_score = np.inf if mode == 'min' else -np.inf
        self.early_stop = False

    def __call__(self, current_score, model):
        is_better = False
        if self.mode == 'min':
            is_better = current_score < (self.best_score - self.min_delta)
        else:
            is_better = current_score > (self.best_score + self.min_delta)

        if is_better:
            self.best_score = current_score
            self.counter = 0
            print(f"Mejora detectada. Guardando modelo en {self.checkpoint_path}")
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            print(f"Sin mejora. Contador de paciencia: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("--- EARLY STOPPING ACTIVADO ---")


indices = list(range(len(dataset)))
labels_arr = [dataset.samples[i][1] for i in indices]
train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels_arr, random_state=42)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CRNN_Model(
    n_channels_cnn=12,
    rnn_hidden_size=128, 
    rnn_num_layers=2,    
    num_classes=4,
    bidirectional=True
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

early_stopper = EarlyStopping(patience=10, mode='min', checkpoint_path='crnn_stumpy.pth')

num_epochs = 100

print(f"--- Iniciando entrenamiento de CRNN en {device} ---")

for epoch in range(num_epochs):
    
    model.train()
    total_train_loss = 0.0
    total_train = 0
    correct_train = 0
    
    for signals, labels, ids in train_loader:
        signals = signals.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(signals) 
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item() * signals.size(0)
        total_train += signals.size(0)
        preds = logits.argmax(dim=1)
        correct_train += (preds == labels).sum().item()
        
    train_acc = correct_train / total_train if total_train else 0.0
    avg_train_loss = total_train_loss / total_train if total_train else 0.0

    model.eval()
    total_val_loss = 0.0
    total_val = 0
    correct_val = 0
    
    with torch.inference_mode():
        for signals, labels, ids in val_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            logits = model(signals)
            
            loss = criterion(logits, labels)
            total_val_loss += loss.item() * signals.size(0)
            
            preds = logits.argmax(dim=1)
            total_val += signals.size(0)
            correct_val += (preds == labels).sum().item()
            
    val_acc = correct_val / total_val if total_val else 0.0
    avg_val_loss = total_val_loss / total_val if total_val else 0.0

    print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}')
    early_stopper(avg_val_loss, model)
    
    if early_stopper.early_stop:
        print("Deteniendo el entrenamiento anticipadamente.")
        break

print("--- Entrenamiento Finalizado ---")

print(f"Cargando el mejor modelo desde {early_stopper.checkpoint_path} (Mejor Val Loss: {early_stopper.best_score:.6f})")
model.load_state_dict(torch.load(early_stopper.checkpoint_path))