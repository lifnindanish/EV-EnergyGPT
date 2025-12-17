# EV Energy Consumption Predictor - Web UI

A modern web application for predicting electric vehicle energy consumption using machine learning.

## Features

- **📊 Visualization Tab**: Interactive charts showing energy consumption patterns
  - Energy consumption by road type
  - Energy consumption by weather condition
  - Speed vs energy consumption scatter plot
  - Temperature vs energy consumption line chart

- **🤖 ML Model Tab**: Real-time energy prediction with 18 input parameters
  - Distance & Speed parameters
  - Road conditions
  - Vehicle parameters
  - Environmental conditions

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure you have the following files in the project directory:
   - `synthetic_ev_dataset.csv` (training dataset)
   - `energy_predictor_model.pth` (trained model)

2. Start the Flask server:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Visualization Tab
- Click on the "📊 Visualization" tab to view interactive charts
- Charts are automatically loaded from the dataset
- Explore different patterns in energy consumption

### ML Model Tab
- Click on the "🤖 ML Model" tab
- Fill in the input parameters (default values are pre-filled)
- Click "🔮 Predict Energy Consumption"
- View the predicted energy consumption in Wh and kWh

## Input Parameters

The model accepts the following 18 parameters:

**Distance & Speed:**
- Segment Length (m)
- Average Speed (km/h)
- Speed Limit (km/h)
- Vehicle Speed Variance

**Road Conditions:**
- Road Gradient (degrees)
- Traffic Density (0-1)
- Road Type (city/highway/rural)
- Number of Stops
- Lane Changes

**Vehicle Parameters:**
- Vehicle Mass (kg)
- Payload Weight (kg)
- Start State of Charge (%)
- Acceleration (m/s²)
- Regenerative Efficiency (0-1)
- Tire Pressure (Pa)

**Environmental Conditions:**
- Ambient Temperature (°C)
- Weather Condition (sunny/rainy/cloudy/snowy)
- Wind Speed (m/s)

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Framework**: PyTorch
- **Data Processing**: Pandas, Scikit-learn
- **Visualization**: Chart.js

## API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Predict energy consumption
- `GET /visualization-data` - Get visualization data

## Notes

- The application runs on port 5000 by default
- Make sure the model file and dataset are in the same directory as app.py
- The web UI features a modern gradient design with responsive layout
