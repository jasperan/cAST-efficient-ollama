"""
Sample code file for vectorization demonstration.
Contains various code patterns for testing retrieval.
"""

import requests
import re
from datetime import datetime

class DataValidator:
    """Validates user input data."""
    
    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_username(self, username: str) -> bool:
        """Validate username length and characters."""
        return len(username) >= 3 and username.isalnum()

class DatabaseManager:
    """Manages database connections."""
    
    def connect_with_retry(self, max_retries: int = 3) -> bool:
        """Connect to database with retry logic."""
        for attempt in range(max_retries):
            try:
                # Connection logic here
                print("Connected successfully")
                return True
            except Exception as e:
                print(f"Connection failed: {e}")
                if attempt == max_retries - 1:
                    raise
                continue
        return False

def process_api_response(response: dict) -> list:
    """Process API response and extract data."""
    try:
        data = response.get('data', [])
        return [item for item in data if item.get('valid')]
    except KeyError:
        raise ValueError("Invalid response format")

def make_http_request(url: str) -> dict:
    """Make HTTP GET request to external API."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def process_string(input_str: str) -> str:
    """Process and format string output."""
    return input_str.strip().upper()

def sorting_algorithm(items: list) -> list:
    """Simple sorting algorithm implementation."""
    return sorted(items)

# More functions can be added as needed
