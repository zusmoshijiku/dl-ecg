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

import matplotlib.pyplot as plt

SAMPLING_RATE = 1000

class ECGDataset(Dataset):
    def __init__(self, data_folder, class_folders, files_per_class=200):
        self.samples = []
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        for folder, label in class_folders.items():
            files = glob.glob(os.path.join(data_folder, folder, '*.parquet.gzip'))
            # enforce exact files_per_class per class (downsample or upsample with replacement)
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

                # convert lead columns to numeric, coerce non-numeric to NaN, then fill and cast
                df_leads = df[self.leads].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

                # shape -> (12, time)
                signal = df_leads.values.T
                self.samples.append((torch.tensor(signal, dtype=torch.float32), label, os.path.basename(f)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        signal, label, ecg_id = self.samples[idx]
        return signal, label, ecg_id

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

# Usage example
class_folders = {
    'arritmia': 0,
    'block': 1,
    'fibrilation': 2,
    'normal': 3
}
data_folder = 'data'
dataset = ECGDataset(data_folder, class_folders, files_per_class=1970)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

print('Datos cargados')

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

early_stopper = EarlyStopping(patience=10, mode='min', checkpoint_path='crnn_best_model.pth')

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