import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load data and model
df = pd.read_csv("synthetic_ev_dataset.csv")
X = df.drop(columns=["energy_consumed_Wh", "segment_id"])
y = df["energy_consumed_Wh"].values

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
X_processed = preprocessor.fit_transform(X)

# Model definition (same as before)
class EnergyMLP(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# Load trained weights
input_dim = X_processed.shape[1]
model = EnergyMLP(input_dim)
model.load_state_dict(torch.load("energy_predictor_model.pth", map_location="cpu"))
model.eval()

# Predict
X_tensor = torch.tensor(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed, dtype=torch.float32)
with torch.no_grad():
    y_pred = model(X_tensor).squeeze().numpy()

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.4, label="Predictions")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="Perfect Fit")
plt.xlabel("Actual Energy (Wh)")
plt.ylabel("Predicted Energy (Wh)")
plt.title("Predicted vs Actual Energy Consumption")
plt.legend()
plt.grid(True)
plt.show()
