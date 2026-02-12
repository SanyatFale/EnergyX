"""Neural forecasting models as LangChain tools."""

import json
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from langchain.tools import tool

#implemented a simple version for init purposess
class SimpleNBEATS(nn.Module):
    """Simplified N-BEATS architecture for small datasets.""" 
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_simple_neural_model(
    y_train: np.ndarray,
    n_lags: int,
    hidden_size: int,
    epochs: int,
    lr: float,
) -> nn.Module:
    """Train a simple neural forecasting model.
    
    Args:
        y_train: Training data
        n_lags: Number of lags
        hidden_size: Hidden layer size
        epochs: Training epochs
        lr: Learning rate
        
    Returns:
        Trained model
    """
    # Create lagged features
    X, y = [], []
    for i in range(n_lags, len(y_train)):
        X.append(y_train[i-n_lags:i])
        y.append(y_train[i])
    
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).unsqueeze(1)
    
    # Initialize model
    model = SimpleNBEATS(n_lags, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model


def forecast_neural_recursive(
    model: nn.Module,
    last_values: np.ndarray,
    horizon: int,
) -> List[float]:
    """Generate recursive forecasts with neural model."""
    predictions = []
    current = torch.FloatTensor(last_values).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(horizon):
            pred = model(current).item()
            predictions.append(float(pred))
            # Shift window
            current = torch.cat([current[:, 1:], torch.FloatTensor([[pred]])], dim=1)
    
    return predictions


@tool
def train_nbeats(
    train_data: str,
    horizon: int,
    n_lags: int = 24,
    hidden_size: int = 64,
    epochs: int = 50,
    learning_rate: float = 0.001,
    **kwargs
) -> str:
    """Train a simplified N-BEATS forecaster.
    
    Args:
        train_data: JSON string of training data (list of floats)
        horizon: Forecast horizon
        n_lags: Number of lagged features (default: 24)
        hidden_size: Hidden layer size (default: 64)
        epochs: Training epochs (default: 50)
        learning_rate: Learning rate (default: 0.001)
        
    Returns:
        JSON string with predictions and metadata
    """
    y_train = np.array(json.loads(train_data))
    
    try:
        # Train model
        model = train_simple_neural_model(
            y_train, n_lags, hidden_size, epochs, learning_rate
        )
        
        # Generate forecasts
        last_values = y_train[-n_lags:]
        predictions = forecast_neural_recursive(model, last_values, horizon)
        
        result = {
            "model_name": "N-BEATS",
            "predictions": predictions,
            "metadata": {
                "n_lags": n_lags,
                "hidden_size": hidden_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
            }
        }
    except Exception as e:
        result = {
            "model_name": "N-BEATS",
            "predictions": [float(y_train[-1])] * horizon,
            "metadata": {
                "error": str(e),
                "fallback": "naive",
            }
        }
    
    return json.dumps(result)

