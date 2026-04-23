"""fetch_weather_forecast — reads temperature from the IDEAL sensor data.

Weather context comes from the temperature_room / temperature_probe sensors
already captured in the HistoricStore.  No external weather API required.
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def fetch_weather_forecast(home_id: str = "unknown", location: str = "") -> str:
    """Return recent indoor temperature readings from the IDEAL sensor data.

    Weather context is sourced from the household's own temperature sensors
    (temperature_room, temperature_probe) stored in the HistoricStore.

    Args:
        home_id:  IDEAL home identifier (e.g. "home96").
        location: Ignored — kept for backward compatibility.

    Returns:
        JSON with latest_temp_c, sensor_count, source="ideal_sensors".
    """
    try:
        from energyx.data.historic_store import get_historic_store
        from datetime import datetime, timedelta

        store = get_historic_store()
        end   = datetime.utcnow()
        start = end - timedelta(hours=24)
        df = store.read_ticks(home_id, start=start, end=end)

        temp_rows = df[df["sensor_type"].isin(["temperature_room", "temperature_probe"])]
        if temp_rows.empty:
            return json.dumps({
                "source": "ideal_sensors",
                "home_id": home_id,
                "note": "No temperature readings found.",
                "latest_temp_c": None,
                "sensor_count": 0,
            })

        latest_temps = temp_rows.groupby("sensor_id")["value"].last().dropna()
        mean_temp = round(float(latest_temps.mean()), 1)
        return json.dumps({
            "source": "ideal_sensors",
            "home_id": home_id,
            "latest_temp_c": mean_temp,
            "sensor_count": int(len(latest_temps)),
            "per_sensor": {k: round(float(v), 1) for k, v in latest_temps.items()},
        })
    except Exception as e:
        logger.warning(f"fetch_weather_forecast (IDEAL) failed: {e}")
        return json.dumps({
            "source": "ideal_sensors",
            "home_id": home_id,
            "error": str(e),
            "latest_temp_c": None,
        })
