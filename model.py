import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# --- LOAD DATA ---
df = pd.read_csv("synthetic_ev_dataset.csv")

# --- PREPROCESSING ---
X = df.drop(columns=["energy_consumed_Wh", "segment_id"])
y = df["energy_consumed_Wh"].values

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

X_processed = preprocessor.fit_transform(X)
input_dim = X_processed.shape[1]

X_tensor = torch.tensor(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

train_size = int(0.7 * len(X_tensor))
val_size = int(0.15 * len(X_tensor))
test_size = len(X_tensor) - train_size - val_size

dataset = TensorDataset(X_tensor, y_tensor)
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# --- MODEL ---
class EnergyMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = EnergyMLP(input_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- TRAINING ---
def train_model(model, train_loader, val_loader, epochs=25):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_loss += criterion(preds, yb).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

train_model(model, train_loader, val_loader)

# --- TESTING ---
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb)
        y_true.extend(yb.numpy())
        y_pred.extend(preds.numpy())

mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"\nTest MAE: {mae:.2f} Wh | R² Score: {r2:.3f}")

torch.save(model.state_dict(), "energy_predictor_model.pth")
print("✅ Model saved as energy_predictor_model.pth")
