import os
import json
import sqlite3
import datetime
from typing import Dict, Any, Optional

# Database configuration
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'metrics.db')

def ensure_database():
    """Ensure the database exists and has the required tables"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        request_id TEXT,
        file_name TEXT,
        processing_time REAL,
        prediction TEXT,
        confidence REAL,
        error TEXT,
        metadata TEXT
    )
    """)
    
    conn.commit()
    conn.close()

def log_to_database(table: str, data: Dict[str, Any]):
    """Log data to the specified database table"""
    ensure_database()
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    if table == 'prediction_metrics':
        # Extract fields from data
        timestamp = data.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        request_id = data.get('request_id', '')
        file_name = data.get('file_name', '')
        processing_time = data.get('processing_time', 0.0)
        prediction = data.get('prediction', '')
        confidence = data.get('confidence', 0.0)
        error = data.get('error', None)
        
        # Store remaining data as JSON metadata
        metadata = {k: v for k, v in data.items() if k not in [
            'timestamp', 'request_id', 'file_name', 'processing_time', 'prediction', 'confidence', 'error'
        ]}
        
        # Insert into database
        cursor.execute("""
        INSERT INTO prediction_metrics 
        (timestamp, request_id, file_name, processing_time, prediction, confidence, error, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, request_id, file_name, processing_time, prediction, confidence, error, json.dumps(metadata)
        ))
    else:
        # Generic logging for other tables
        # Convert all data to JSON string
        cursor.execute(f"INSERT INTO {table} (data) VALUES (?)", (json.dumps(data),))
    
    conn.commit()
    conn.close()
    
    return True

def query_metrics(table: str, filters: Optional[Dict[str, Any]] = None, limit: int = 100):
    """Query metrics from the database with optional filters"""
    ensure_database()
    
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    cursor = conn.cursor()
    
    query = f"SELECT * FROM {table}"
    params = []
    
    # Add filters if provided
    if filters:
        conditions = []
        for key, value in filters.items():
            conditions.append(f"{key} = ?")
            params.append(value)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
    
    # Add limit
    query += f" ORDER BY timestamp DESC LIMIT {limit}"
    
    cursor.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return results