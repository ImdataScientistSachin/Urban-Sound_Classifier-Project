import os
import json
import yaml
import csv
import pickle
import logging
import shutil
from typing import Dict, List, Any, Union, Optional

class FileUtils:
    """
    Utility class for file operations.
    
    This class provides static methods for common file operations
    such as reading/writing various file formats, creating directories,
    and managing file paths.
    """
    
    @staticmethod
    def ensure_dir(directory: str) -> str:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            directory (str): Directory path
            
        Returns:
            str: The directory path
            
        Raises:
            OSError: If directory creation fails
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return directory
    
    @staticmethod
    def read_json(file_path: str) -> Dict[str, Any]:
        """
        Read a JSON file.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            Dict[str, Any]: Parsed JSON data
            
        Raises:
            FileNotFoundError: If the file does not exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def write_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
        """
        Write data to a JSON file.
        
        Args:
            data (Dict[str, Any]): Data to write
            file_path (str): Path to the output file
            indent (int): Indentation level for pretty printing
            
        Raises:
            TypeError: If the data is not JSON serializable
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
    
    @staticmethod
    def read_yaml(file_path: str) -> Dict[str, Any]:
        """
        Read a YAML file.
        
        Args:
            file_path (str): Path to the YAML file
            
        Returns:
            Dict[str, Any]: Parsed YAML data
            
        Raises:
            FileNotFoundError: If the file does not exist
            yaml.YAMLError: If the file is not valid YAML
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def write_yaml(data: Dict[str, Any], file_path: str) -> None:
        """
        Write data to a YAML file.
        
        Args:
            data (Dict[str, Any]): Data to write
            file_path (str): Path to the output file
            
        Raises:
            TypeError: If the data is not YAML serializable
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    @staticmethod
    def read_csv(file_path: str, delimiter: str = ',', has_header: bool = True) -> List[Dict[str, str]]:
        """
        Read a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            delimiter (str): Field delimiter
            has_header (bool): Whether the CSV file has a header row
            
        Returns:
            List[Dict[str, str]]: List of dictionaries, one per row
            
        Raises:
            FileNotFoundError: If the file does not exist
            csv.Error: If the file is not valid CSV
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', newline='') as f:
            if has_header:
                reader = csv.DictReader(f, delimiter=delimiter)
                return list(reader)
            else:
                reader = csv.reader(f, delimiter=delimiter)
                rows = list(reader)
                # Create dictionaries with column indices as keys
                return [{str(i): value for i, value in enumerate(row)} for row in rows]
    
    @staticmethod
    def write_csv(data: List[Dict[str, Any]], file_path: str, fieldnames: Optional[List[str]] = None, delimiter: str = ',') -> None:
        """
        Write data to a CSV file.
        
        Args:
            data (List[Dict[str, Any]]): List of dictionaries to write
            file_path (str): Path to the output file
            fieldnames (Optional[List[str]]): List of field names (columns)
            delimiter (str): Field delimiter
            
        Raises:
            ValueError: If data is empty and fieldnames is not provided
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # If fieldnames is not provided, use the keys from the first dictionary
        if fieldnames is None:
            if not data:
                raise ValueError("Data is empty and fieldnames is not provided")
            fieldnames = list(data[0].keys())
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(data)
    
    @staticmethod
    def save_pickle(data: Any, file_path: str) -> None:
        """
        Save data to a pickle file.
        
        Args:
            data (Any): Data to save
            file_path (str): Path to the output file
            
        Raises:
            pickle.PickleError: If pickling fails
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_pickle(file_path: str) -> Any:
        """
        Load data from a pickle file.
        
        Args:
            file_path (str): Path to the pickle file
            
        Returns:
            Any: Unpickled data
            
        Raises:
            FileNotFoundError: If the file does not exist
            pickle.UnpicklingError: If unpickling fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def copy_file(src_path: str, dst_path: str, overwrite: bool = False) -> None:
        """
        Copy a file from source to destination.
        
        Args:
            src_path (str): Source file path
            dst_path (str): Destination file path
            overwrite (bool): Whether to overwrite existing files
            
        Raises:
            FileNotFoundError: If the source file does not exist
            FileExistsError: If the destination file exists and overwrite is False
            shutil.Error: If copying fails
        """
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source file not found: {src_path}")
            
        if os.path.exists(dst_path) and not overwrite:
            raise FileExistsError(f"Destination file already exists: {dst_path}")
            
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        
        shutil.copy2(src_path, dst_path)
    
    @staticmethod
    def move_file(src_path: str, dst_path: str, overwrite: bool = False) -> None:
        """
        Move a file from source to destination.
        
        Args:
            src_path (str): Source file path
            dst_path (str): Destination file path
            overwrite (bool): Whether to overwrite existing files
            
        Raises:
            FileNotFoundError: If the source file does not exist
            FileExistsError: If the destination file exists and overwrite is False
            shutil.Error: If moving fails
        """
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source file not found: {src_path}")
            
        if os.path.exists(dst_path) and not overwrite:
            raise FileExistsError(f"Destination file already exists: {dst_path}")
            
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        
        shutil.move(src_path, dst_path)
    
    @staticmethod
    def list_files(directory: str, pattern: Optional[str] = None, recursive: bool = False) -> List[str]:
        """
        List files in a directory, optionally matching a pattern.
        
        Args:
            directory (str): Directory path
            pattern (Optional[str]): Glob pattern for matching files
            recursive (bool): Whether to search recursively
            
        Returns:
            List[str]: List of file paths
            
        Raises:
            FileNotFoundError: If the directory does not exist
        """
        import glob
        
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Not a directory: {directory}")
            
        if pattern is None:
            pattern = '*'
            
        if recursive:
            return glob.glob(os.path.join(directory, '**', pattern), recursive=True)
        else:
            return glob.glob(os.path.join(directory, pattern))
    
    @staticmethod
    def get_file_size(file_path: str, unit: str = 'bytes') -> float:
        """
        Get the size of a file.
        
        Args:
            file_path (str): Path to the file
            unit (str): Unit for the size ('bytes', 'KB', 'MB', 'GB')
            
        Returns:
            float: File size in the specified unit
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the unit is not valid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        size_bytes = os.path.getsize(file_path)
        
        if unit.lower() == 'bytes':
            return size_bytes
        elif unit.lower() == 'kb':
            return size_bytes / 1024
        elif unit.lower() == 'mb':
            return size_bytes / (1024 * 1024)
        elif unit.lower() == 'gb':
            return size_bytes / (1024 * 1024 * 1024)
        else:
            raise ValueError(f"Invalid unit: {unit}. Valid units are 'bytes', 'KB', 'MB', 'GB'.")
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """
        Get the extension of a file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            str: File extension (without the dot)
        """
        return os.path.splitext(file_path)[1][1:]
    
    @staticmethod
    def get_file_name(file_path: str, with_extension: bool = True) -> str:
        """
        Get the name of a file.
        
        Args:
            file_path (str): Path to the file
            with_extension (bool): Whether to include the extension
            
        Returns:
            str: File name
        """
        if with_extension:
            return os.path.basename(file_path)
        else:
            return os.path.splitext(os.path.basename(file_path))[0]