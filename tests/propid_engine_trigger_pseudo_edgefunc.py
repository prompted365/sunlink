import requests
import json

# Define the URL and the payload
url = "http://localhost:8000/process"
payload = {
    "property_id": "8844ccd4-518c-457d-8886-120e0f32c90c"
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