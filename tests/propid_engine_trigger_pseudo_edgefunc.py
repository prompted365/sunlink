import requests
import json

# Define the URL and the payload
url = "http://localhost:8000/process"
payload = {
    "property_id": "26002082-e74a-47fb-96f8-e0484acec0d6"
}

# Set the headers to specify the content type
headers = {
    "Content-Type": "application/json"
}

# Make the POST request
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Print the response status code and response content
print("Status Code:", response.status_code)
print("Response Content:", response.text)