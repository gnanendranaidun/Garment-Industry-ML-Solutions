// Dashboard initialization
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initMetrics();
    initCharts();
    initPredictions();
    initOptimization();
    initSimulation();
    initDefectAnalysis();
    
    // Set up auto-refresh
    setInterval(refreshData, 300000); // Refresh every 5 minutes
});

// Metrics initialization and update
function initMetrics() {
    updateMetrics();
}

function updateMetrics() {
    fetch('/api/quality-metrics')
        .then(response => response.json())
        .then(data => {
            updateMetricDisplay('total-production', data.total_units);
            updateMetricDisplay('quality-score', data.quality_score);
            updateMetricDisplay('defect-rate', data.defect_rate);
            updateMetricDisplay('efficiency', 1 - data.defect_rate);
        })
        .catch(error => console.error('Error updating metrics:', error));
}

function updateMetricDisplay(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = typeof value === 'number' ? 
            (value * 100).toFixed(1) + '%' : 
            value.toLocaleString();
    }
}

// Charts initialization and update
function initCharts() {
    createProductionChart();
    createQualityChart();
}

function createProductionChart() {
    fetch('/api/production-data')
        .then(response => response.json())
        .then(data => {
            const trace = {
                x: data.map(d => d.date),
                y: data.map(d => d.total_units),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Production',
                line: {
                    color: '#007bff',
                    width: 2
                }
            };
            
            const layout = {
                title: 'Daily Production Volume',
                xaxis: { title: 'Date' },
                yaxis: { title: 'Units' },
                margin: { t: 30, r: 20, b: 40, l: 60 }
            };
            
            Plotly.newPlot('production-chart', [trace], layout);
        })
        .catch(error => console.error('Error creating production chart:', error));
}

function createQualityChart() {
    fetch('/api/quality-trends')
        .then(response => response.json())
        .then(data => {
            const trace = {
                x: data.map(d => d.date),
                y: data.map(d => d.mean),
                type: 'scatter',
                mode: 'lines',
                name: 'Quality Score',
                line: {
                    color: '#28a745',
                    width: 2
                },
                error_y: {
                    type: 'data',
                    array: data.map(d => d.std),
                    visible: true,
                    color: '#28a745',
                    thickness: 1
                }
            };
            
            const layout = {
                title: 'Quality Score Trends',
                xaxis: { title: 'Date' },
                yaxis: { title: 'Quality Score' },
                margin: { t: 30, r: 20, b: 40, l: 60 }
            };
            
            Plotly.newPlot('quality-chart', [trace], layout);
        })
        .catch(error => console.error('Error creating quality chart:', error));
}

// ML Predictions
function initPredictions() {
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', handlePrediction);
    }
}

function handlePrediction(event) {
    event.preventDefault();
    
    const formData = {
        temperature: parseFloat(document.getElementById('temperature').value),
        pressure: parseFloat(document.getElementById('pressure').value),
        speed: parseFloat(document.getElementById('speed').value),
        humidity: parseFloat(document.getElementById('humidity').value)
    };
    
    fetch('/api/predictions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        displayPredictionResults(data);
    })
    .catch(error => console.error('Error getting predictions:', error));
}

function displayPredictionResults(data) {
    const resultsDiv = document.getElementById('prediction-results');
    if (resultsDiv) {
        resultsDiv.innerHTML = `
            <div class="alert alert-info fade-in">
                <h6>Prediction Results:</h6>
                <p>Production Prediction: ${data.production_prediction.toFixed(2)} units</p>
                <p>Quality Prediction: ${(data.quality_prediction * 100).toFixed(1)}%</p>
            </div>
        `;
    }
}

// Production Optimization
function initOptimization() {
    updateOptimization();
}

function updateOptimization() {
    fetch('/api/optimization')
        .then(response => response.json())
        .then(data => {
            createOptimizationChart(data);
            displayOptimalParameters(data);
        })
        .catch(error => console.error('Error updating optimization:', error));
}

function createOptimizationChart(data) {
    const products = Object.keys(data);
    const traces = [
        {
            x: products,
            y: products.map(p => data[p].temperature),
            type: 'bar',
            name: 'Temperature',
            marker: { color: '#007bff' }
        },
        {
            x: products,
            y: products.map(p => data[p].pressure),
            type: 'bar',
            name: 'Pressure',
            marker: { color: '#28a745' }
        },
        {
            x: products,
            y: products.map(p => data[p].speed),
            type: 'bar',
            name: 'Speed',
            marker: { color: '#ffc107' }
        }
    ];
    
    const layout = {
        title: 'Optimal Parameters by Product',
        barmode: 'group',
        xaxis: { title: 'Product' },
        yaxis: { title: 'Value' },
        margin: { t: 30, r: 20, b: 40, l: 60 }
    };
    
    Plotly.newPlot('optimization-chart', traces, layout);
}

function displayOptimalParameters(data) {
    const paramsDiv = document.getElementById('optimal-parameters');
    if (paramsDiv) {
        paramsDiv.innerHTML = '<h6>Optimal Parameters:</h6>';
        Object.entries(data).forEach(([product, params]) => {
            paramsDiv.innerHTML += `
                <div class="card mb-2 fade-in">
                    <div class="card-body">
                        <h6>Product ${product}</h6>
                        <p>Temperature: ${params.temperature.toFixed(1)}°C</p>
                        <p>Pressure: ${params.pressure.toFixed(1)} bar</p>
                        <p>Speed: ${params.speed.toFixed(1)} units/min</p>
                        <p>Confidence: ${(params.confidence * 100).toFixed(1)}%</p>
                    </div>
                </div>
            `;
        });
    }
}

// Production Simulation
function initSimulation() {
    const startButton = document.getElementById('start-simulation');
    if (startButton) {
        startButton.addEventListener('click', runSimulation);
    }
}

function runSimulation() {
    const duration = parseInt(document.getElementById('sim-duration').value);
    const speed = parseInt(document.getElementById('sim-speed').value);
    
    // Generate simulation data
    const timePoints = Array.from({length: duration * 60}, (_, i) => i);
    const production = timePoints.map(t => speed * t);
    const quality = timePoints.map(t => 0.95 - (t / (duration * 60)) * 0.1);
    
    const traces = [
        {
            x: timePoints,
            y: production,
            type: 'scatter',
            mode: 'lines',
            name: 'Production',
            line: { color: '#007bff', width: 2 }
        },
        {
            x: timePoints,
            y: quality,
            type: 'scatter',
            mode: 'lines',
            name: 'Quality',
            line: { color: '#28a745', width: 2 }
        }
    ];
    
    const layout = {
        title: 'Production Simulation',
        xaxis: { title: 'Time (minutes)' },
        yaxis: { title: 'Value' },
        margin: { t: 30, r: 20, b: 40, l: 60 }
    };
    
    Plotly.newPlot('simulation-results', traces, layout);
}

// Defect Analysis
function initDefectAnalysis() {
    updateDefectAnalysis();
}

function updateDefectAnalysis() {
    fetch('/api/defect-analysis')
        .then(response => response.json())
        .then(data => {
            createDefectChart(data);
            displayDefectAnalysis(data);
        })
        .catch(error => console.error('Error updating defect analysis:', error));
}

function createDefectChart(data) {
    const defectTypes = Object.entries(data.defect_types);
    const trace = {
        values: defectTypes.map(d => d[1]),
        labels: defectTypes.map(d => d[0]),
        type: 'pie',
        name: 'Defect Types',
        marker: {
            colors: ['#007bff', '#28a745', '#ffc107', '#dc3545', '#6c757d']
        }
    };
    
    const layout = {
        title: 'Defect Types Distribution',
        margin: { t: 30, r: 20, b: 40, l: 60 }
    };
    
    Plotly.newPlot('defect-chart', [trace], layout);
}

function displayDefectAnalysis(data) {
    const analysisDiv = document.getElementById('defect-analysis');
    if (analysisDiv) {
        analysisDiv.innerHTML = '<h6>Defect Analysis:</h6>';
        
        Object.entries(data.defect_by_parameter).forEach(([param, values]) => {
            const total = Object.values(values).reduce((a, b) => a + b, 0);
            const avg = total / Object.keys(values).length;
            
            analysisDiv.innerHTML += `
                <div class="card mb-2 fade-in">
                    <div class="card-body">
                        <h6>${param.charAt(0).toUpperCase() + param.slice(1)} Analysis</h6>
                        <p>Total Defects: ${total}</p>
                        <p>Average Defects: ${avg.toFixed(1)}</p>
                    </div>
                </div>
            `;
        });
    }
}

// Data refresh function
function refreshData() {
    updateMetrics();
    createProductionChart();
    createQualityChart();
    updateOptimization();
    updateDefectAnalysis();
} 