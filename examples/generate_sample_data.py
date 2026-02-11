"""Generate sample time series datasets for testing."""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_seasonal_sales_data(
    n_points: int = 365,
    seasonal_period: int = 7,
    trend_slope: float = 0.1,
    noise_level: float = 5.0,
    output_path: str = "data/raw/seasonal_sales.csv",
):
    """Generate synthetic seasonal sales data.
    
    Args:
        n_points: Number of data points
        seasonal_period: Seasonal period (e.g., 7 for weekly)
        trend_slope: Linear trend slope
        noise_level: Standard deviation of noise
        output_path: Output file path
    """
    
    # Generate time index
    dates = pd.date_range(start="2022-01-01", periods=n_points, freq="D")
    
    # Generate components
    t = np.arange(n_points)
    
    # Trend
    trend = trend_slope * t
    
    # Seasonality
    seasonal = 20 * np.sin(2 * np.pi * t / seasonal_period)
    
    # Noise
    noise = np.random.normal(0, noise_level, n_points)
    
    # Combine
    sales = 100 + trend + seasonal + noise
    
    # Add some outliers
    outlier_indices = np.random.choice(n_points, size=int(n_points * 0.02), replace=False)
    sales[outlier_indices] += np.random.normal(0, 30, len(outlier_indices))
    
    # Create DataFrame
    df = pd.DataFrame({
        "date": dates,
        "sales": sales,
        "day_of_week": dates.dayofweek,
        "month": dates.month,
    })
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated seasonal sales data: {output_path}")
    print(f"  - {n_points} observations")
    print(f"  - Seasonal period: {seasonal_period}")
    print(f"  - Date range: {dates[0]} to {dates[-1]}")
    
    return df


def generate_anomaly_data(
    n_points: int = 500,
    anomaly_rate: float = 0.05,
    output_path: str = "data/raw/anomaly_data.csv",
):
    """Generate synthetic data with anomalies.
    
    Args:
        n_points: Number of data points
        anomaly_rate: Proportion of anomalies
        output_path: Output file path
    """
    
    # Generate time index
    dates = pd.date_range(start="2023-01-01", periods=n_points, freq="H")
    
    # Generate normal data
    values = np.random.normal(50, 10, n_points)
    
    # Add anomalies
    n_anomalies = int(n_points * anomaly_rate)
    anomaly_indices = np.random.choice(n_points, size=n_anomalies, replace=False)
    
    # Anomalies are extreme values
    values[anomaly_indices] += np.random.choice([-1, 1], n_anomalies) * np.random.uniform(30, 50, n_anomalies)
    
    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": dates,
        "value": values,
        "hour": dates.hour,
    })
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated anomaly data: {output_path}")
    print(f"  - {n_points} observations")
    print(f"  - {n_anomalies} anomalies ({anomaly_rate*100:.1f}%)")
    
    return df


def generate_trend_data(
    n_points: int = 200,
    output_path: str = "data/raw/trend_data.csv",
):
    """Generate data with strong trend and no seasonality.
    
    Args:
        n_points: Number of data points
        output_path: Output file path
    """
    
    # Generate time index
    dates = pd.date_range(start="2023-01-01", periods=n_points, freq="W")
    
    # Generate exponential trend
    t = np.arange(n_points)
    trend = 100 * np.exp(0.01 * t)
    
    # Add noise
    noise = np.random.normal(0, trend * 0.05)
    
    values = trend + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        "date": dates,
        "metric": values,
        "week": dates.isocalendar().week,
    })
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated trend data: {output_path}")
    print(f"  - {n_points} observations")
    print(f"  - Exponential trend")
    
    return df


if __name__ == "__main__":
    # Generate all sample datasets
    print("Generating sample datasets...\n")
    
    generate_seasonal_sales_data()
    print()
    
    generate_anomaly_data()
    print()
    
    generate_trend_data()
    print()
    
    print("✓ All sample datasets generated!")

