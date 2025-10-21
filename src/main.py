import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
from neurokit2 import ecg
import neurokit2 as nk
import numpy as np
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

SAMPLING_RATE = 1000
SIZE = 8

class ECGDataset(Dataset):
    def __init__(self, data_folder, class_folders, files_per_class=200):
        self.samples = []
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        for folder, label in class_folders.items():
            files = glob.glob(os.path.join(data_folder, folder, '*.parquet.gzip'))
            if len(files) > files_per_class:
                files = random.sample(files, files_per_class)
            for f in files:
                try:
                    df = pd.read_parquet(f, engine='fastparquet')
                except Exception as e:
                    print(f"Failed to read {f}: {e}")
                    continue
                signal = df[self.leads].values.T  # shape: (12, time)
                self.samples.append((torch.tensor(signal, dtype=torch.float32), label, os.path.basename(f)))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        signal, label, ecg_id = self.samples[idx]
        return signal, label, ecg_id

def toFeature(signal: pd.core.frame.DataFrame):
    F = []
    for lead in ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
        clean = nk.ecg_clean(signal[lead], sampling_rate=SAMPLING_RATE)
        _, rpeaks = nk.ecg_peaks(clean, sampling_rate=SAMPLING_RATE)
        rpeak_indices = rpeaks['ECG_R_Peaks']
        # Only proceed if enough R-peaks are found
        if np.sum(rpeak_indices) < 2:
            F += [np.nan, np.nan, np.nan, np.nan]
            continue
        try:
            _, waves_peak = nk.ecg_delineate(clean, rpeaks, sampling_rate=SAMPLING_RATE, method="peak")
            mean_r = np.mean([clean[i] if not np.isnan(i) else 0 for i in rpeaks['ECG_R_Peaks']]) if np.any(rpeaks['ECG_R_Peaks']) else np.nan
            mean_p = np.mean([clean[i] if not np.isnan(i) else 0 for i in waves_peak['ECG_P_Peaks']]) if 'ECG_P_Peaks' in waves_peak else np.nan
            mean_q = np.mean([clean[i] if not np.isnan(i) else 0 for i in waves_peak['ECG_Q_Peaks']]) if 'ECG_Q_Peaks' in waves_peak else np.nan
            mean_s = np.mean([clean[i] if not np.isnan(i) else 0 for i in waves_peak['ECG_S_Peaks']]) if 'ECG_S_Peaks' in waves_peak else np.nan
        except Exception:
            mean_r = mean_p = mean_q = mean_s = np.nan
        F += [mean_r, mean_p, mean_q, mean_s]
    return np.array(F)

def extract_features(idx):
    signal, label, ecg_id = dataset[idx]
    df = pd.DataFrame(signal.numpy().T, columns=dataset.leads)
    feat = toFeature(df)
    return feat, label

def main():
    # Usage example
    class_folders = {
        'arritmia': 0,
        'block': 1,
        'fibrilation': 2,
        'normal': 3
    }
    data_folder = 'data'
    dataset = ECGDataset(data_folder, class_folders, files_per_class=200)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    # Apply toFeature to all signals in the dataset
    results = Parallel(n_jobs=SIZE)(delayed(extract_features)(i) for i in range(len(dataset)))

    features, labels = zip(*results)
    features = np.vstack(features)
    labels = np.array(labels)

    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)

    # PCA

    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_imputed)

    # KNN
    
    X_train, X_test, y_train, y_test = train_test_split(features_pca, labels, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=4, n_jobs=SIZE)
    knn.fit(X_train, y_train)
    print("KNN accuracy:", knn.score(X_test, y_test))

if __name__ == "__main__":
    main()