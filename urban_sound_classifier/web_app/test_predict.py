import requests
import sys

try:
    response = requests.post('http://localhost:5000/predict', files={'file': open('test.wav', 'rb')})
    print(f'Status code: {response.status_code}')
    print(f'Response: {response.text}')
except Exception as e:
    print(f'Error: {e}')