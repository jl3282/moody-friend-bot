import os
import pickle
import numpy as np
from scipy import interpolate, stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR    = 'data/WESAD/'  # path to WESAD PKL folders
# DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('mps' if torch.backends.mps.is_available() 
                      else 'cuda' if torch.cuda.is_available() 
                      else 'cpu')
BATCH_SIZE  = 64
LR          = 1e-4
EPOCHS      = 50
PATIENCE    = 10
# Window: 5s @32Hz = 160 samples, stride 1s = 32
WINDOW_SIZE = 160
STRIDE      = 32

print(f"Using device: {DEVICE}")

# -----------------------------
# WESAD Preprocessing (Fixed to match notebook)
# -----------------------------
def list_subjects(data_dir):
    subs = [d for d in os.listdir(data_dir) if d.startswith('S')]
    return sorted(subs, key=lambda x: int(x[1:]))


def load_subject_data(data_dir, subject_id):
    pkl_path = os.path.join(data_dir, subject_id, f"{subject_id}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Data file not found: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def extract_wrist_data(subject_data):
    """Extract wrist data - matches notebook implementation"""
    wrist_dict = subject_data['signal']['wrist']
    labels = subject_data['label']
    
    # Step 1: Find the wrist sensor with the longest time axis (usually ACC)
    max_len = 0
    for sensor_data in wrist_dict.values():
        max_len = max(max_len, len(sensor_data))

    # Step 2: Resample each wrist sensor to the reference length (max_len)
    wrist_signals = []
    for sensor_name, sensor_data in wrist_dict.items():
        # Ensure shape is (T, D)
        if len(sensor_data.shape) == 1:
            sensor_data = sensor_data.reshape(-1, 1)
        T, D = sensor_data.shape
        resampled = np.zeros((max_len, D))
        for d in range(D):
            f = interpolate.interp1d(np.linspace(0, 1, T), sensor_data[:, d], kind='linear', fill_value="extrapolate")
            resampled[:, d] = f(np.linspace(0, 1, max_len))
        wrist_signals.append(resampled)

    # Step 3: Concatenate all wrist signals horizontally → shape: (max_len, num_features)
    combined_signals = np.concatenate(wrist_signals, axis=1)

    # Step 4: Resample labels to match wrist data length
    label_interp = interpolate.interp1d(
        np.linspace(0, 1, len(labels)), labels, kind='nearest', fill_value="extrapolate"
    )
    resampled_labels = label_interp(np.linspace(0, 1, max_len)).astype(int)

    return combined_signals, resampled_labels


# Load and preprocess WESAD data
print("Loading WESAD data...")
subjects = list_subjects(DATA_DIR)
all_X, all_y = [], []

for sid in subjects:
    print(f"Loading {sid}...")
    sd = load_subject_data(DATA_DIR, sid)
    Xw, yw = extract_wrist_data(sd)
    mask = np.isin(yw, [1,2,3,4])  # keep only baseline, stress, amusement, meditation
    all_X.append(Xw[mask])
    all_y.append(yw[mask])

# Stack
X = np.vstack(all_X)
y = np.hstack(all_y)
print("All wrist data shape:", X.shape, y.shape)

# Normalize
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Create sliding windows
def create_windows(X, y, window_size, stride):
    Xw, yw = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        win = X[i:i+window_size]
        lbl_window = y[i:i+window_size]
        # Use scipy.stats.mode correctly for different versions
        try:
            mode_result = stats.mode(lbl_window, axis=None, keepdims=True)
            lbl = mode_result.mode[0]
        except:
            # Fallback for older scipy versions
            mode_result = stats.mode(lbl_window)
            lbl = mode_result.mode[0] if hasattr(mode_result.mode, '__len__') else mode_result.mode
        
        Xw.append(win)
        yw.append(lbl-1)  # shift labels to 0–3
    return np.array(Xw), np.array(yw)

Xw, yw = create_windows(X_norm, y, WINDOW_SIZE, STRIDE)
print("Windowed data shape:", Xw.shape, yw.shape)

# -----------------------------
# PyTorch Models (Custom implementations)
# -----------------------------

class Physio1DCNN(nn.Module):
    """Simple 1D CNN for physiological signals"""
    def __init__(self, input_dim, num_classes, sequence_length):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            # First conv block
            nn.Conv1d(input_dim, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Second conv block
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Third conv block
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        features = self.feature_extractor(x)
        return self.classifier(features)


class PhysioTransformer(nn.Module):
    """Transformer model for physiological signals"""
    def __init__(self, input_dim, num_classes, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        seq_len = x.shape[1]
        
        x = self.input_projection(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        encoded = self.transformer(x)
        pooled = encoded.mean(dim=1)  # Global average pooling
        
        return self.classifier(pooled)


class PhysioLSTM(nn.Module):
    """LSTM model for physiological signals"""
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


# Dataset & split
class TSData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): 
        return len(self.y)
    def __getitem__(self, i): 
        return self.X[i], self.y[i]

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(Xw, yw, test_size=0.3, random_state=42, stratify=yw)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

train_ds = TSData(X_train, y_train)
val_ds = TSData(X_val, y_val)
test_ds = TSData(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# -----------------------------
# Training/Evaluation Functions
# -----------------------------
def train_model(model, train_loader, val_loader, model_name):
    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_accs = [], []
    
    print(f"\nTraining {model_name}...")
    
    for epoch in range(EPOCHS):
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
        scheduler.step(1 - val_acc)  # Schedule based on accuracy
        
        print(f"Epoch {epoch+1:2d}/{EPOCHS}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name.lower()}.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping for {model_name}!")
                break
    
    return best_val_acc, train_losses, val_accs


def test_model(model, test_loader, model_name):
    # Load best model
    model.load_state_dict(torch.load(f'best_{model_name.lower()}.pth'))
    model.to(DEVICE)
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
    
    print(f"\n{model_name} Test Results:")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    # Classification report
    target_names = ['Baseline', 'Stress', 'Amusement', 'Meditation']
    print(classification_report(all_trues, all_preds, target_names=target_names))
    
    return acc, all_preds, all_trues


# -----------------------------
# Run Experiments
# -----------------------------
input_dim = Xw.shape[2]  # Number of features (should be 6 for WESAD wrist)
num_classes = 4
sequence_length = Xw.shape[1]

print(f"Input dimensions: {input_dim} features, {sequence_length} time steps")

# Models to compare
models_to_test = [
    ("1D_CNN", Physio1DCNN(input_dim, num_classes, sequence_length)),
    ("Transformer", PhysioTransformer(input_dim, num_classes)),
    ("LSTM", PhysioLSTM(input_dim, num_classes))
]

results = {}

for model_name, model in models_to_test:
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print('='*50)
    
    try:
        # Train
        best_val_acc, train_losses, val_accs = train_model(model, train_loader, val_loader, model_name)
        
        # Test
        test_acc, test_preds, test_trues = test_model(model, test_loader, model_name)
        
        results[model_name] = {
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'train_losses': train_losses,
            'val_accs': val_accs
        }
        
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        results[model_name] = {'val_acc': 0, 'test_acc': 0}

# Final Results
print(f"\n{'='*80}")
print("FINAL RESULTS SUMMARY")
print('='*80)
for model_name, result in results.items():
    if result['test_acc'] > 0:
        print(f"{model_name:<15}: Test Acc = {result['test_acc']:.4f} ({result['test_acc']*100:.2f}%)")

# Find best model
best_model = max(results.keys(), key=lambda k: results[k]['test_acc'])
best_acc = results[best_model]['test_acc']
print(f"\n🏆 Best Model: {best_model} with {best_acc*100:.2f}% accuracy")

# Plot training curves
if len(results) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for model_name, result in results.items():
        if 'train_losses' in result and len(result['train_losses']) > 0:
            ax1.plot(result['train_losses'], label=f'{model_name} (Train Loss)')
            ax2.plot(result['val_accs'], label=f'{model_name} (Val Acc)')
    
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

print(f"\n✅ Transfer learning comparison completed!")
print(f"📊 Training curves saved to: training_results.png")
