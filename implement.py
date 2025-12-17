import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- 1. Load your trained model ---
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

# Load dataset to reconstruct preprocessors
df = pd.read_csv("synthetic_ev_dataset.csv")
X = df.drop(columns=["energy_consumed_Wh", "segment_id"])
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Recreate the same preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
preprocessor.fit(X)  # fit on the full dataset

input_dim = preprocessor.transform(X[:1]).shape[1]
model = EnergyMLP(input_dim)
model.load_state_dict(torch.load("energy_predictor_model.pth", map_location="cpu"))
model.eval()

# --- 2. Define a test input sample ---
sample_input = pd.DataFrame([{
    "segment_length_m": 1200,
    "avg_speed_kmh": 55,
    "speed_limit_kmh": 60,
    "road_gradient_deg": 2,
    "traffic_density": 0.4,
    "vehicle_mass_kg": 1600,
    "start_SOC_%": 80,
    "ambient_temp_C": 30,
    "accel_mps2": 0.6,
    "regen_efficiency": 0.85,
    "weather_condition": "sunny",
    "road_type": "city",
    "num_stops": 3,
    "vehicle_speed_var": 8,
    "payload_weight_kg": 200,
    "wind_speed_mps": 3,
    "lane_changes": 1,
    "tire_pressure_Pa": 220000
}])

# --- 3. Transform and predict ---
x_processed = preprocessor.transform(sample_input)
x_tensor = torch.tensor(x_processed.toarray() if hasattr(x_processed, "toarray") else x_processed, dtype=torch.float32)

with torch.no_grad():
    prediction = model(x_tensor).item()

print(f"⚡ Predicted Energy Consumption: {prediction:.2f} Wh for the segment")
