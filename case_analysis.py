import os
import pandas as pd
import numpy as np
from scipy import interpolate, stats
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Configuration
CASE_DIR = 'data/CASE/data/interpolated'
PHYS_DIR = os.path.join(CASE_DIR, 'physiological')
ANN_DIR = os.path.join(CASE_DIR, 'annotations')
MODEL_PATH = 'best_lstm.pth'  

TARGET_FS = 32
ORIG_FS = 1000
WINDOW_SIZE = 160  # 5 seconds at 32 Hz
STRIDE = 32  # 1 second stride
BATCH_SIZE = 32
DEVICE = torch.device('mps' if torch.backends.mps.is_available() 
                      else 'cuda' if torch.cuda.is_available() 
                      else 'cpu')

def get_available_subjects():
    """Get list of available subject IDs"""
    if not os.path.exists(PHYS_DIR):
        raise FileNotFoundError(f"Physiological data directory not found: {PHYS_DIR}")
    
    files = os.listdir(PHYS_DIR)
    subs = []
    for f in files:
        m = re.match(r'^sub_(\d+)\.csv$', f)
        if m:
            subs.append(int(m.group(1)))
    
    if not subs:
        raise FileNotFoundError("No subject files found in the specified directory")
    
    subs = sorted(subs)
    return [str(s) for s in subs]

def load_subject_data(sid):
    """Load physiological and annotation data for a subject"""
    physio_fp = os.path.join(PHYS_DIR, f'sub_{sid}.csv')
    annot_fp = os.path.join(ANN_DIR, f'sub_{sid}.csv')
    
    if not os.path.exists(physio_fp):
        raise FileNotFoundError(f"Physiological file not found: {physio_fp}")
    if not os.path.exists(annot_fp):
        raise FileNotFoundError(f"Annotation file not found: {annot_fp}")
    
    df_phys = pd.read_csv(physio_fp)
    df_ann = pd.read_csv(annot_fp)
    
    # Verify data alignment
    if len(df_phys) != len(df_ann):
        print(f"Warning: Length mismatch for subject {sid}: physio={len(df_phys)}, annot={len(df_ann)}")
        min_len = min(len(df_phys), len(df_ann))
        df_phys = df_phys.iloc[:min_len]
        df_ann = df_ann.iloc[:min_len]
    
    return df_phys, df_ann

def downsample_df(df, target_fs=32, orig_fs=1000):
    """Downsample dataframe to target frequency"""
    # Identify columns to preserve
    preserve_cols = ['daqtime', 'jstime', 'video']
    drop_cols = [c for c in preserve_cols if c in df.columns]
    
    # Get data columns
    data_cols = [c for c in df.columns if c not in preserve_cols]
    arr = df[data_cols].values
    
    orig_len, nchan = arr.shape
    new_len = int(orig_len * target_fs / orig_fs)
    
    # Create time vectors
    t_orig = np.linspace(0, 1, orig_len)
    t_new = np.linspace(0, 1, new_len)
    
    # Interpolate each channel
    out = np.zeros((new_len, nchan))
    for i in range(nchan):
        f = interpolate.interp1d(t_orig, arr[:, i], 
                                kind='linear', 
                                fill_value='extrapolate')
        out[:, i] = f(t_new)
    
    return pd.DataFrame(out, columns=data_cols)

def create_windows(X, y, window_size=160, stride=32):
    """Create sliding windows from time series data"""
    Xw, yw = [], []
    for start in range(0, len(X) - window_size + 1, stride):
        win = X[start:start + window_size]
        lbls = y[start:start + window_size]
        
        # Use majority voting for window label
        try:
            mode_result = stats.mode(lbls, keepdims=False)
            if hasattr(mode_result, 'mode'):
                mode = mode_result.mode
            else:
                mode = mode_result[0]  # For older scipy versions
        except:
            # Fallback: use most common value
            unique, counts = np.unique(lbls, return_counts=True)
            mode = unique[np.argmax(counts)]
        
        Xw.append(win)
        yw.append(mode)
    
    return np.array(Xw), np.array(yw)

def create_emotion_labels(df_ann, method='valence_binary'):
    """Create discrete emotion labels from valence and arousal values
    
    Args:
        df_ann: DataFrame with valence and arousal columns
        method: str, method for creating labels
            - 'valence_binary': Binary classification based on valence (>5 = positive, <=5 = negative)
            - 'arousal_binary': Binary classification based on arousal (>5 = high, <=5 = low)
            - 'quadrant': 4-class classification based on valence-arousal quadrants
            - 'wesad_mapping': Map to 4 WESAD-like classes for transfer learning
            - 'valence_continuous': Use continuous valence values
            - 'arousal_continuous': Use continuous arousal values
    
    Returns:
        labels: numpy array of labels
    """
    if method == 'valence_binary':
        labels = (df_ann['valence'] > 5).astype(int)
    elif method == 'arousal_binary':
        labels = (df_ann['arousal'] > 5).astype(int)
    elif method == 'quadrant':
        # 4 emotion quadrants: 0=low_val_low_ar, 1=low_val_high_ar, 2=high_val_low_ar, 3=high_val_high_ar
        high_val = df_ann['valence'] > 5
        high_ar = df_ann['arousal'] > 5
        labels = high_val.astype(int) * 2 + high_ar.astype(int)
    elif method == 'wesad_mapping':
        # Map CASE emotions to WESAD-like categories for transfer learning
        # WESAD: 0=Baseline, 1=Stress, 2=Amusement, 3=Meditation
        # CASE mapping based on valence-arousal:
        high_val = df_ann['valence'] > 6  # More stringent thresholds
        low_val = df_ann['valence'] < 4
        high_ar = df_ann['arousal'] > 6
        low_ar = df_ann['arousal'] < 4
        
        labels = np.full(len(df_ann), 0)  # Default to baseline
        labels[(low_val) & (high_ar)] = 1   # Stress-like (negative + high arousal)
        labels[(high_val) & (high_ar)] = 2  # Amusement-like (positive + high arousal)
        labels[(high_val) & (low_ar)] = 3   # Meditation-like (positive + low arousal)
        # Neutral/baseline states remain as 0
        
    elif method == 'wesad_mapping_balanced':
        # More balanced mapping using quartiles for better class distribution
        val_q25 = np.percentile(df_ann['valence'], 25)  # ~4.0
        val_q75 = np.percentile(df_ann['valence'], 75)  # ~6.6
        ar_q25 = np.percentile(df_ann['arousal'], 25)   # ~5.0  
        ar_q75 = np.percentile(df_ann['arousal'], 75)   # ~6.1
        
        print(f"  Valence quartiles: Q25={val_q25:.2f}, Q75={val_q75:.2f}")
        print(f"  Arousal quartiles: Q25={ar_q25:.2f}, Q75={ar_q75:.2f}")
        
        # Create 4 balanced classes based on quartiles
        low_val = df_ann['valence'] <= val_q25
        high_val = df_ann['valence'] >= val_q75
        low_ar = df_ann['arousal'] <= ar_q25
        high_ar = df_ann['arousal'] >= ar_q75
        
        labels = np.full(len(df_ann), 0)  # Default: baseline (middle range)
        labels[(low_val) & (high_ar)] = 1   # Stress (low valence + high arousal)
        labels[(high_val) & (high_ar)] = 2  # Amusement (high valence + high arousal)  
        labels[(high_val) & (low_ar)] = 3   # Meditation (high valence + low arousal)
        
        # Override middle values back to baseline
        mid_val = (df_ann['valence'] > val_q25) & (df_ann['valence'] < val_q75)
        mid_ar = (df_ann['arousal'] > ar_q25) & (df_ann['arousal'] < ar_q75)
        labels[mid_val | mid_ar] = 0
    elif method == 'valence_continuous':
        labels = df_ann['valence'].values
    elif method == 'arousal_continuous':
        labels = df_ann['arousal'].values
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return labels.values if hasattr(labels, 'values') else labels

class PhysioLSTM(nn.Module):
    """LSTM model for physiological signals - matches models_experiments.py"""
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=0.2 if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        
        return self.classifier(last_hidden)

class TSData(Dataset):
    """Time series dataset"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): 
        return len(self.y)
    def __getitem__(self, i): 
        return self.X[i], self.y[i]

def evaluate_transfer_learning(model, test_loader, class_names):
    """Evaluate transfer learning performance"""
    model.eval()
    all_preds, all_trues = [], []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            _, predicted = torch.max(out.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_trues.extend(yb.cpu().numpy())
    
    acc = accuracy_score(all_trues, all_preds)
    
    print(f"\nTransfer Learning Results:")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Classification Report:")
    print(classification_report(all_trues, all_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_trues, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - WESAD Model on CASE Data')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_transfer.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return acc, all_preds, all_trues

def select_wesad_compatible_features(df_phys):
    """Select features from CASE that best match WESAD wrist sensors
    
    WESAD wrist sensors: ACC (3D), BVP, EDA, TEMP = 6 features total
    CASE features: ecg, bvp, gsr, rsp, skt, emg_zygo, emg_coru, emg_trap
    
    Mapping:
    - BVP: bvp (exact match)
    - EDA: gsr (galvanic skin response, similar to electrodermal activity)
    - TEMP: skt (skin temperature)
    - ACC (3D): Use ECG + 2 EMG channels as motion-related proxies
    """
    
    # Select 6 features that best approximate WESAD wrist sensors
    selected_features = ['ecg', 'bvp', 'gsr', 'skt', 'emg_zygo', 'emg_coru']
    
    if not all(feat in df_phys.columns for feat in selected_features):
        available = [f for f in selected_features if f in df_phys.columns]
        print(f"Warning: Not all features available. Using: {available}")
        selected_features = available
    
    return df_phys[selected_features]

def load_multiple_subjects(subject_ids, method='wesad_mapping'):
    """Load and process multiple subjects"""
    all_X, all_y = [], []
    
    for sid in subject_ids:
        try:
            print(f"Processing subject {sid}...")
            df_phys, df_ann = load_subject_data(sid)
            
            # Select WESAD-compatible features
            df_phys_selected = select_wesad_compatible_features(df_phys)
            
            # Downsample
            df_phys_down = downsample_df(df_phys_selected, target_fs=TARGET_FS, orig_fs=ORIG_FS)
            df_ann_down = downsample_df(df_ann, target_fs=TARGET_FS, orig_fs=1000//50)
            
            # Create labels
            emotion_labels = create_emotion_labels(df_ann_down, method=method)
            
            # Create windows
            X, y = create_windows(df_phys_down.values, emotion_labels)
            
            all_X.append(X)
            all_y.append(y)
            
            print(f"  Subject {sid}: {X.shape[0]} windows, {X.shape[2]} features, label dist: {np.bincount(y)}")
            
        except Exception as e:
            print(f"  Error processing subject {sid}: {e}")
            continue
    
    if not all_X:
        raise ValueError("No subjects processed successfully")
    
    # Combine all subjects
    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)
    
    return X_combined, y_combined

def fine_tune_model(model, train_loader, val_loader, model_name, freeze_lstm=False, lr=1e-5, epochs=20):
    """Fine-tune pre-trained model on CASE data"""
    model.to(DEVICE)
    
    # Option to freeze LSTM layers and only train classifier
    if freeze_lstm:
        print("🔒 Freezing LSTM layers, only training classifier")
        for param in model.lstm.parameters():
            param.requires_grad = False
    else:
        print("🔓 Training all layers")
    
    # Use lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_accs = [], []
    
    print(f"\nFine-tuning {model_name} on CASE data...")
    print(f"Learning rate: {lr}, Epochs: {epochs}, Freeze LSTM: {freeze_lstm}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                _, predicted = torch.max(out.data, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        
        val_acc = correct / total
        val_accs.append(val_acc)
        scheduler.step(1 - val_acc)
        
        print(f"Epoch {epoch+1:2d}/{epochs}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'finetuned_{model_name.lower()}.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:  # Shorter patience for fine-tuning
                print(f"Early stopping for {model_name}!")
                break
    
    return best_val_acc, train_losses, val_accs

def prepare_case_train_data(subject_ids, method='wesad_mapping_balanced', test_size=0.3):
    """Prepare CASE data for training with train/val/test splits"""
    print(f"Preparing CASE training data from {len(subject_ids)} subjects...")
    
    all_X, all_y = [], []
    
    for sid in subject_ids:
        try:
            print(f"  Loading subject {sid}...")
            df_phys, df_ann = load_subject_data(sid)
            
            # Select WESAD-compatible features
            df_phys_selected = select_wesad_compatible_features(df_phys)
            
            # Downsample
            df_phys_down = downsample_df(df_phys_selected, target_fs=TARGET_FS, orig_fs=ORIG_FS)
            df_ann_down = downsample_df(df_ann, target_fs=TARGET_FS, orig_fs=1000//50)
            
            # Create labels
            emotion_labels = create_emotion_labels(df_ann_down, method=method)
            
            # Create windows
            X, y = create_windows(df_phys_down.values, emotion_labels)
            
            if X.shape[0] > 0:  # Only add if we have valid windows
                all_X.append(X)
                all_y.append(y)
                print(f"    {X.shape[0]} windows, label dist: {np.bincount(y)}")
            
        except Exception as e:
            print(f"    Error processing subject {sid}: {e}")
            continue
    
    if not all_X:
        raise ValueError("No subjects processed successfully")
    
    # Combine all subjects
    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)
    
    print(f"\nCombined dataset: {X_combined.shape}")
    print(f"Label distribution: {np.bincount(y_combined)}")
    
    # Split into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_combined, y_combined, test_size=test_size, random_state=42, stratify=y_combined
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Create data loaders
    train_dataset = TSData(X_train, y_train)
    val_dataset = TSData(X_val, y_val)
    test_dataset = TSData(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print(f"using {DEVICE}")
    subs = get_available_subjects()
    print(f"Found {len(subs)} subjects:", subs[:10], "..." if len(subs) > 10 else "")

    # === OPTION 1: Single Subject Analysis ===
    print("\n" + "="*60)
    print("SINGLE SUBJECT ANALYSIS")
    print("="*60)
    
    # Load data for subject 1
    df_phys, df_ann = load_subject_data('1')
    print("Physiological data columns:", df_phys.columns.tolist())
    print("Annotation data columns:", df_ann.columns.tolist())
    print(df_ann.describe())

    # Select WESAD-compatible features (6 features)
    df_phys_selected = select_wesad_compatible_features(df_phys)
    print(f"Selected features for WESAD compatibility: {df_phys_selected.columns.tolist()}")

    # Downsample physiological data
    df_phys_down = downsample_df(df_phys_selected, target_fs=TARGET_FS, orig_fs=ORIG_FS)
    
    # Downsample annotation data to match physiological data length
    df_ann_down = downsample_df(df_ann, target_fs=TARGET_FS, orig_fs=1000//50)
    
    # Test different labeling methods
    methods_to_test = [
        ('valence_binary', ['Negative', 'Positive']),
        ('quadrant', ['Low-Val-Low-Ar', 'Low-Val-High-Ar', 'High-Val-Low-Ar', 'High-Val-High-Ar']),
        ('wesad_mapping', ['Baseline', 'Stress', 'Amusement', 'Meditation']),
        ('wesad_mapping_balanced', ['Baseline', 'Stress', 'Amusement', 'Meditation'])
    ]
    
    for method, class_names in methods_to_test:
        print(f"\n--- Testing method: {method} ---")
        
        # Create emotion labels
        emotion_labels = create_emotion_labels(df_ann_down, method=method)
        print(f"Label distribution: {np.bincount(emotion_labels)}")
        
        # Create windows
        X, y = create_windows(df_phys_down.values, emotion_labels)
        print(f"Windows shape: {X.shape}, Labels shape: {y.shape}")
        
        if X.shape[0] == 0:
            print("No windows created, skipping...")
            continue
        
        # Load pre-trained LSTM model
        input_dim = X.shape[2]  # Should now be 6 to match WESAD
        num_classes = len(class_names)
        
        if method == 'wesad_mapping' and num_classes == 4:
            # For 4-class transfer learning
            model = PhysioLSTM(input_dim=input_dim, num_classes=4)
            
            if os.path.exists(MODEL_PATH):
                print(f"Loading pre-trained model from {MODEL_PATH}")
                try:
                    # Load the state dict
                    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                    
                    # Check input dimension compatibility
                    if input_dim == 6:
                        print("✅ Input dimensions match WESAD (6 features)")
                    else:
                        print(f"⚠️  Input dimension mismatch. CASE: {input_dim}, WESAD: 6")
                        continue
                    
                    model.load_state_dict(state_dict)
                    model.to(DEVICE)
                    
                    # Create test dataset
                    test_dataset = TSData(X, y)
                    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                    
                    # Evaluate
                    acc, preds, trues = evaluate_transfer_learning(model, test_loader, class_names)
                    
                except Exception as e:
                    print(f"Error loading model: {e}")
                    print("Model architecture mismatch or file corruption")
            else:
                print(f"Model file not found: {MODEL_PATH}")
        elif (method == 'wesad_mapping_balanced' and num_classes == 4):
            # For 4-class transfer learning with balanced labels
            model = PhysioLSTM(input_dim=input_dim, num_classes=4)
            
            if os.path.exists(MODEL_PATH):
                print(f"Loading pre-trained model from {MODEL_PATH}")
                try:
                    # Load the state dict
                    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                    
                    # Check input dimension compatibility
                    if input_dim == 6:
                        print("✅ Input dimensions match WESAD (6 features)")
                    else:
                        print(f"⚠️  Input dimension mismatch. CASE: {input_dim}, WESAD: 6")
                        continue
                    
                    model.load_state_dict(state_dict)
                    model.to(DEVICE)
                    
                    # Create test dataset
                    test_dataset = TSData(X, y)
                    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                    
                    # Evaluate
                    acc, preds, trues = evaluate_transfer_learning(model, test_loader, class_names)
                    
                except Exception as e:
                    print(f"Error loading model: {e}")
                    print("Model architecture mismatch or file corruption")
            else:
                print(f"Model file not found: {MODEL_PATH}")
        else:
            print(f"Skipping model evaluation for {method} (not 4-class)")

    # === OPTION 2: Multi-Subject Analysis ===
    print("\n" + "="*60)
    print("MULTI-SUBJECT ANALYSIS")
    print("="*60)
    
    # Use first 5 subjects for faster testing
    test_subjects = subs[:5]
    
    try:
        X_multi, y_multi = load_multiple_subjects(test_subjects, method='wesad_mapping_balanced')
        print(f"\nCombined data: {X_multi.shape}, Labels: {y_multi.shape}")
        print(f"Overall label distribution: {np.bincount(y_multi)}")
        
        # Load and evaluate model on multi-subject data
        if os.path.exists(MODEL_PATH):
            input_dim = X_multi.shape[2]
            model = PhysioLSTM(input_dim=input_dim, num_classes=4)
            
            try:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                model.to(DEVICE)
                
                # Create test dataset
                test_dataset = TSData(X_multi, y_multi)
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                
                # Evaluate
                class_names = ['Baseline', 'Stress', 'Amusement', 'Meditation']
                acc, preds, trues = evaluate_transfer_learning(model, test_loader, class_names)
                
                print(f"\n🎯 Multi-subject transfer learning accuracy: {acc*100:.2f}%")
                
            except Exception as e:
                print(f"Error in multi-subject evaluation: {e}")
        
    except Exception as e:
        print(f"Error in multi-subject processing: {e}")

    # === OPTION 3: Fine-tuning Analysis ===
    print("\n" + "="*60)
    print("FINE-TUNING ANALYSIS")
    print("="*60)
    
    # Use more subjects for fine-tuning (better training data)
    training_subjects = subs[:10]  # Use first 10 subjects
    
    try:
        # Prepare training data
        train_loader, val_loader, test_loader = prepare_case_train_data(
            training_subjects, 
            method='wesad_mapping_balanced'
        )
        
        # Test different fine-tuning strategies
        finetune_strategies = [
            ("classifier_only", True, 1e-4),   # Freeze LSTM, train classifier only
            ("full_model_slow", False, 1e-5), # Train all layers with slow LR
            ("full_model_fast", False, 1e-4), # Train all layers with faster LR
        ]
        
        finetune_results = {}
        
        for strategy_name, freeze_lstm, learning_rate in finetune_strategies:
            print(f"\n--- Fine-tuning Strategy: {strategy_name} ---")
            
            if os.path.exists(MODEL_PATH):
                # Load pre-trained model
                model = PhysioLSTM(input_dim=6, num_classes=4)
                
                try:
                    print(f"Loading pre-trained WESAD model...")
                    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                    model.load_state_dict(state_dict)
                    
                    # Fine-tune the model
                    best_val_acc, train_losses, val_accs = fine_tune_model(
                        model, train_loader, val_loader, 
                        f"lstm_{strategy_name}", 
                        freeze_lstm=freeze_lstm, 
                        lr=learning_rate,
                        epochs=15
                    )
                    
                    # Test fine-tuned model
                    print(f"\nTesting fine-tuned model...")
                    model.load_state_dict(torch.load(f'finetuned_lstm_{strategy_name}.pth'))
                    
                    class_names = ['Baseline', 'Stress', 'Amusement', 'Meditation']
                    test_acc, test_preds, test_trues = evaluate_transfer_learning(
                        model, test_loader, class_names
                    )
                    
                    finetune_results[strategy_name] = {
                        'val_acc': best_val_acc,
                        'test_acc': test_acc,
                        'train_losses': train_losses,
                        'val_accs': val_accs
                    }
                    
                    print(f"🎯 {strategy_name} - Test Accuracy: {test_acc*100:.2f}%")
                    
                except Exception as e:
                    print(f"Error in fine-tuning {strategy_name}: {e}")
                    continue
            else:
                print(f"Pre-trained model not found: {MODEL_PATH}")
                break
        
        # Compare results
        if finetune_results:
            print(f"\n" + "="*60)
            print("FINE-TUNING RESULTS COMPARISON")
            print("="*60)
            
            for strategy, results in finetune_results.items():
                print(f"{strategy:<20}: Test Acc = {results['test_acc']:.4f} ({results['test_acc']*100:.2f}%)")
            
            # Find best strategy
            best_strategy = max(finetune_results.keys(), key=lambda k: finetune_results[k]['test_acc'])
            best_acc = finetune_results[best_strategy]['test_acc']
            print(f"\n🏆 Best Fine-tuning Strategy: {best_strategy} with {best_acc*100:.2f}% accuracy")
            
            # Plot comparison
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 2, 1)
            for strategy, results in finetune_results.items():
                if 'train_losses' in results:
                    plt.plot(results['train_losses'], label=f'{strategy} (Loss)')
            plt.title('Fine-tuning Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            for strategy, results in finetune_results.items():
                if 'val_accs' in results:
                    plt.plot(results['val_accs'], label=f'{strategy} (Val Acc)')
            plt.title('Fine-tuning Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('finetuning_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📊 Fine-tuning comparison saved to: finetuning_comparison.png")
        
    except Exception as e:
        print(f"Error in fine-tuning analysis: {e}")

    print(f"\n✅ CASE dataset analysis completed!")
    print(f"📊 Results saved to: confusion_matrix_transfer.png, finetuning_comparison.png")
