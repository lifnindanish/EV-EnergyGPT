from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import json

app = Flask(__name__)
CORS(app)

# --- Load Model and Preprocessor ---
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
preprocessor.fit(X)

input_dim = preprocessor.transform(X[:1]).shape[1]
model = EnergyMLP(input_dim)
model.load_state_dict(torch.load("energy_predictor_model.pth", map_location="cpu"))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Create DataFrame from input
        sample_input = pd.DataFrame([{
            "segment_length_m": float(data.get("segment_length_m", 1200)),
            "avg_speed_kmh": float(data.get("avg_speed_kmh", 55)),
            "speed_limit_kmh": float(data.get("speed_limit_kmh", 60)),
            "road_gradient_deg": float(data.get("road_gradient_deg", 2)),
            "traffic_density": float(data.get("traffic_density", 0.4)),
            "vehicle_mass_kg": float(data.get("vehicle_mass_kg", 1600)),
            "start_SOC_%": float(data.get("start_SOC_%", 80)),
            "ambient_temp_C": float(data.get("ambient_temp_C", 30)),
            "accel_mps2": float(data.get("accel_mps2", 0.6)),
            "regen_efficiency": float(data.get("regen_efficiency", 0.85)),
            "weather_condition": data.get("weather_condition", "sunny"),
            "road_type": data.get("road_type", "city"),
            "num_stops": int(data.get("num_stops", 3)),
            "vehicle_speed_var": float(data.get("vehicle_speed_var", 8)),
            "payload_weight_kg": float(data.get("payload_weight_kg", 200)),
            "wind_speed_mps": float(data.get("wind_speed_mps", 3)),
            "lane_changes": int(data.get("lane_changes", 1)),
            "tire_pressure_Pa": float(data.get("tire_pressure_Pa", 220000))
        }])
        
        # Transform and predict
        x_processed = preprocessor.transform(sample_input)
        x_tensor = torch.tensor(
            x_processed.toarray() if hasattr(x_processed, "toarray") else x_processed, 
            dtype=torch.float32
        )
        
        with torch.no_grad():
            prediction = model(x_tensor).item()
        
        return jsonify({
            'success': True,
            'energy_consumed_Wh': round(prediction, 2),
            'energy_consumed_kWh': round(prediction / 1000, 4)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/visualization-data', methods=['GET'])
def get_visualization_data():
    try:
        # Load dataset for visualization
        df = pd.read_csv("synthetic_ev_dataset.csv")
        
        # Prepare data for various visualizations
        viz_data = {
            'energy_by_road_type': df.groupby('road_type')['energy_consumed_Wh'].mean().to_dict(),
            'energy_by_weather': df.groupby('weather_condition')['energy_consumed_Wh'].mean().to_dict(),
            'speed_vs_energy': {
                'speed': df['avg_speed_kmh'].tolist()[:100],
                'energy': df['energy_consumed_Wh'].tolist()[:100]
            },
            'temperature_vs_energy': {
                'temperature': df['ambient_temp_C'].tolist()[:100],
                'energy': df['energy_consumed_Wh'].tolist()[:100]
            }
        }
        
        return jsonify({
            'success': True,
            'data': viz_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
