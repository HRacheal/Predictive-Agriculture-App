import requests

# URL of your FastAPI endpoint
url = "http://127.0.0.1:8000/predict"

# Example JSON input using numeric labels (matches your dataset)
data = {
    "Crop_Type": 9,      # numeric label from your dataset
    "Soil_Type": 2,      # numeric label from your dataset
    "Soil_pH": 6.5,
    "Temperature": 26.0,
    "Humidity": 70.0,
    "Wind_Speed": 5.0,
    "N": 80.0,
    "P": 40.0,
    "K": 60.0,
    "Soil_Quality": 7.0,
    "month": 1,
    "year": 2026
}

try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # raise error if status != 200

    # Print JSON response
    try:
        result = response.json()
        print("Response from backend:")
        print(result)
    except ValueError:
        print("Response is not JSON:")
        print(response.text)

except requests.exceptions.HTTPError as e:
    print(f"HTTP error occurred: {e}")
    print("Response content:", response.text)
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
