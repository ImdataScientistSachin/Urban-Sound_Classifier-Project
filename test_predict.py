#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the /predict endpoint of the Urban Sound Classifier web app.
"""

import requests
import os
import sys
import json

def test_predict_endpoint():
    # URL of the predict endpoint
    url = 'http://localhost:5000/predict'
    
    # Path to test audio file
    test_file_path = os.path.join('test_audio_samples', 'dog_bark.wav')
    
    # Check if the file exists
    if not os.path.exists(test_file_path):
        print(f"Error: Test file not found at {test_file_path}")
        return
    
    # Open the file in binary mode
    with open(test_file_path, 'rb') as f:
        # Create a dictionary with the file
        files = {'file': (os.path.basename(test_file_path), f, 'audio/wav')}
        
        try:
            # Send the POST request
            print(f"Sending request to {url} with file {test_file_path}...")
            response = requests.post(url, files=files)
            
            # Print the status code
            print(f"Status code: {response.status_code}")
            
            # Try to parse the response as JSON
            try:
                json_response = response.json()
                print("Response:")
                print(json_response)
            except ValueError:
                print("Response is not in JSON format:")
                print(response.text)
        
        except requests.exceptions.RequestException as e:
            print(f"Error sending request: {e}")

if __name__ == "__main__":
    test_predict_endpoint()