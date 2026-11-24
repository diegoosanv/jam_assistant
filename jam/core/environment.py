import json
import urllib.request


def get_location_and_weather(timeout: float = 3.0):
    try:
        # 1. Obtener ubicación basada en IP
        ip_req = urllib.request.Request(
            "http://ip-api.com/json/",
            headers={"User-Agent": "JAM-Assistant/1.1"},
        )
        with urllib.request.urlopen(ip_req, timeout=timeout) as response:
            loc_data = json.loads(response.read().decode("utf-8"))
            lat = loc_data.get("lat", 0)
            lon = loc_data.get("lon", 0)
            city = loc_data.get("city", "Unknown")

        # 2. Obtener clima (Open-Meteo es gratis y sin API Key)
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current_weather=true"
        )
        w_req = urllib.request.Request(
            weather_url,
            headers={"User-Agent": "JAM-Assistant/1.1"},
        )
        with urllib.request.urlopen(w_req, timeout=timeout) as response:
            w_data = json.loads(response.read().decode("utf-8"))
            temp = w_data.get("current_weather", {}).get("temperature", "--")

        return {"city": city, "temp": temp, "online": True}
    except Exception:
        return {"city": "OFFLINE", "temp": "--", "online": False}
