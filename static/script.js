// Tab switching functionality
function openTab(evt, tabName) {
    // Hide all tab contents
    const tabContents = document.getElementsByClassName('tab-content');
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove('active');
    }

    // Remove active class from all tab buttons
    const tabButtons = document.getElementsByClassName('tab-button');
    for (let i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove('active');
    }

    // Show the selected tab and mark button as active
    document.getElementById(tabName).classList.add('active');
    evt.currentTarget.classList.add('active');

    // Load visualization data if visualization tab is opened
    if (tabName === 'visualization') {
        loadVisualizationData();
    }
}

// Chart instances
let roadTypeChart, weatherChart, speedEnergyChart, tempEnergyChart;

// Load and display visualization data
async function loadVisualizationData() {
    try {
        const response = await fetch('/visualization-data');
        const result = await response.json();

        if (result.success) {
            const data = result.data;

            // Destroy existing charts if they exist
            if (roadTypeChart) roadTypeChart.destroy();
            if (weatherChart) weatherChart.destroy();
            if (speedEnergyChart) speedEnergyChart.destroy();
            if (tempEnergyChart) tempEnergyChart.destroy();

            // Energy by Road Type - Bar Chart
            const roadTypeCtx = document.getElementById('roadTypeChart').getContext('2d');
            roadTypeChart = new Chart(roadTypeCtx, {
                type: 'bar',
                data: {
                    labels: Object.keys(data.energy_by_road_type),
                    datasets: [{
                        label: 'Average Energy (Wh)',
                        data: Object.values(data.energy_by_road_type),
                        backgroundColor: [
                            'rgba(102, 126, 234, 0.8)',
                            'rgba(118, 75, 162, 0.8)',
                            'rgba(237, 100, 166, 0.8)'
                        ],
                        borderColor: [
                            'rgba(102, 126, 234, 1)',
                            'rgba(118, 75, 162, 1)',
                            'rgba(237, 100, 166, 1)'
                        ],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Energy (Wh)'
                            }
                        }
                    }
                }
            });

            // Energy by Weather - Doughnut Chart
            const weatherCtx = document.getElementById('weatherChart').getContext('2d');
            weatherChart = new Chart(weatherCtx, {
                type: 'doughnut',
                data: {
                    labels: Object.keys(data.energy_by_weather),
                    datasets: [{
                        label: 'Average Energy (Wh)',
                        data: Object.values(data.energy_by_weather),
                        backgroundColor: [
                            'rgba(255, 206, 86, 0.8)',
                            'rgba(54, 162, 235, 0.8)',
                            'rgba(153, 102, 255, 0.8)',
                            'rgba(201, 203, 207, 0.8)'
                        ],
                        borderColor: [
                            'rgba(255, 206, 86, 1)',
                            'rgba(54, 162, 235, 1)',
                            'rgba(153, 102, 255, 1)',
                            'rgba(201, 203, 207, 1)'
                        ],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });

            // Speed vs Energy - Scatter Chart
            const speedEnergyCtx = document.getElementById('speedEnergyChart').getContext('2d');
            const speedEnergyData = data.speed_vs_energy.speed.map((speed, index) => ({
                x: speed,
                y: data.speed_vs_energy.energy[index]
            }));

            speedEnergyChart = new Chart(speedEnergyCtx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Speed vs Energy',
                        data: speedEnergyData,
                        backgroundColor: 'rgba(102, 126, 234, 0.6)',
                        borderColor: 'rgba(102, 126, 234, 1)',
                        pointRadius: 5,
                        pointHoverRadius: 7
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Speed (km/h)'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Energy (Wh)'
                            }
                        }
                    }
                }
            });

            // Temperature vs Energy - Line Chart
            const tempEnergyCtx = document.getElementById('tempEnergyChart').getContext('2d');
            tempEnergyChart = new Chart(tempEnergyCtx, {
                type: 'line',
                data: {
                    labels: data.temperature_vs_energy.temperature,
                    datasets: [{
                        label: 'Energy Consumption',
                        data: data.temperature_vs_energy.energy,
                        backgroundColor: 'rgba(237, 100, 166, 0.2)',
                        borderColor: 'rgba(237, 100, 166, 1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Temperature (°C)'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Energy (Wh)'
                            }
                        }
                    }
                }
            });
        }
    } catch (error) {
        console.error('Error loading visualization data:', error);
    }
}

// Handle form submission for ML prediction
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    // Hide previous results and errors
    document.getElementById('results').classList.add('hidden');
    document.getElementById('error').classList.add('hidden');

    // Collect form data
    const formData = new FormData(e.target);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }

    try {
        // Show loading state
        const predictButton = e.target.querySelector('.predict-button');
        const originalText = predictButton.textContent;
        predictButton.textContent = '⏳ Predicting...';
        predictButton.disabled = true;

        // Make prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        // Restore button state
        predictButton.textContent = originalText;
        predictButton.disabled = false;

        if (result.success) {
            // Display results
            document.getElementById('energyWh').textContent = result.energy_consumed_Wh.toFixed(2);
            document.getElementById('energyKWh').textContent = result.energy_consumed_kWh.toFixed(4);
            document.getElementById('results').classList.remove('hidden');

            // Scroll to results
            document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            // Display error
            document.getElementById('error').textContent = `Error: ${result.error}`;
            document.getElementById('error').classList.remove('hidden');
        }
    } catch (error) {
        // Restore button state
        const predictButton = e.target.querySelector('.predict-button');
        predictButton.textContent = '🔮 Predict Energy Consumption';
        predictButton.disabled = false;

        // Display error
        document.getElementById('error').textContent = `Error: ${error.message}`;
        document.getElementById('error').classList.remove('hidden');
    }
});

// Load visualization data on page load if visualization tab is active
window.addEventListener('load', () => {
    const visualizationTab = document.getElementById('visualization');
    if (visualizationTab.classList.contains('active')) {
        loadVisualizationData();
    }
});
