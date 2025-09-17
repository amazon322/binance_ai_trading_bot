#!/usr/bin/env python3
"""
Script to download and optimize static assets for better performance
"""
import os
import requests
import gzip
from pathlib import Path

def download_file(url, local_path, compress=False):
    """Download a file and optionally compress it"""
    print(f"Downloading {url} to {local_path}")
    
    response = requests.get(url)
    response.raise_for_status()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    if compress:
        # Compress the file
        with gzip.open(f"{local_path}.gz", 'wb') as f:
            f.write(response.content)
        print(f"Compressed version saved as {local_path}.gz")
    
    # Save original file
    with open(local_path, 'wb') as f:
        f.write(response.content)
    
    print(f"Downloaded {len(response.content)} bytes")

def main():
    """Download all required static assets"""
    
    # Define assets to download
    assets = [
        {
            'url': 'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css',
            'path': 'static/css/bootstrap.min.css'
        },
        {
            'url': 'https://code.jquery.com/jquery-3.5.1.slim.min.js',
            'path': 'static/js/jquery-3.5.1.slim.min.js'
        },
        {
            'url': 'https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js',
            'path': 'static/js/bootstrap.bundle.min.js'
        }
    ]
    
    print("Downloading static assets for better performance...")
    
    for asset in assets:
        try:
            download_file(asset['url'], asset['path'], compress=True)
        except Exception as e:
            print(f"Error downloading {asset['url']}: {e}")
    
    print("Asset download completed!")

if __name__ == "__main__":
    main()