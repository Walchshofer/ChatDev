import requests

server_url = "http://localhost:5000/api/v1/model"

response = requests.get(server_url)

if response.status_code == 200:
    print("Successfully got the model name:", response.json()['result'])
else:
    print(f"Failed to get the model name. Status code: {response.status_code}")


