import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "features": [0.0,0.0,0.0,0.0,1.0,16.0,14.9,8.365206834418355,5.71042701737487,4.867534450455582]
}

response = requests.post(url, json=data)
print("Prediction:", response.json())

