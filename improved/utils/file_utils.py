import os
import shutil
import tempfile
import json
import yaml
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, BinaryIO, TextIO, Tuple
from datetime import datetime

from improved.utils.exceptions import FileOperationError

logger = logging.getLogger(__name__)


class FileUtils:
    """
    Utility class for file operations with robust error handling.
    
    This class provides methods for common file operations like reading, writing,
    copying, and deleting files with proper error handling and logging.
    """
    
    @staticmethod
    def ensure_directory(directory_path: Union[str, Path]) -> Path:
        """
        Ensure that a directory exists, creating it if necessary.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            Path: Path object for the directory
            
        Raises:
            FileOperationError: If directory creation fails
        """
        try:
            path = Path(directory_path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {path}")
            return path
        except Exception as e:
            error_msg = f"Failed to create directory {directory_path}: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e
    
    @staticmethod
    def safe_write_file(file_path: Union[str, Path], 
                       content: Union[str, bytes], 
                       mode: str = 'w',
                       encoding: Optional[str] = 'utf-8',
                       use_temp_file: bool = True) -> Path:
        """
        Safely write content to a file using a temporary file if specified.
        
        Args:
            file_path: Path to the file
            content: Content to write (string or bytes)
            mode: File mode ('w' for text, 'wb' for binary)
            encoding: File encoding (for text mode)
            use_temp_file: Whether to use a temporary file for atomic writes
            
        Returns:
            Path: Path object for the written file
            
        Raises:
            FileOperationError: If file writing fails
        """
        path = Path(file_path)
        
        # Ensure parent directory exists
        FileUtils.ensure_directory(path.parent)
        
        try:
            if use_temp_file:
                # Write to temporary file first
                fd, temp_path = tempfile.mkstemp(dir=path.parent)
                try:
                    with os.fdopen(fd, mode, encoding=encoding if 'b' not in mode else None) as f:
                        f.write(content)
                    # Rename temporary file to target file (atomic operation)
                    shutil.move(temp_path, path)
                except Exception as e:
                    # Clean up temporary file if writing fails
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                    raise e
            else:
                # Direct write to file
                with open(path, mode, encoding=encoding if 'b' not in mode else None) as f:
                    f.write(content)
            
            logger.debug(f"Successfully wrote to file: {path}")
            return path
        except Exception as e:
            error_msg = f"Failed to write to file {file_path}: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e
    
    @staticmethod
    def safe_read_file(file_path: Union[str, Path], 
                      mode: str = 'r',
                      encoding: Optional[str] = 'utf-8',
                      default: Any = None) -> Union[str, bytes, Any]:
        """
        Safely read content from a file with error handling.
        
        Args:
            file_path: Path to the file
            mode: File mode ('r' for text, 'rb' for binary)
            encoding: File encoding (for text mode)
            default: Default value to return if file doesn't exist or reading fails
            
        Returns:
            Union[str, bytes, Any]: File content or default value
            
        Raises:
            FileOperationError: If file reading fails and no default is provided
        """
        path = Path(file_path)
        
        if not path.exists():
            if default is not None:
                logger.warning(f"File not found: {path}, returning default value")
                return default
            error_msg = f"File not found: {path}"
            logger.error(error_msg)
            raise FileOperationError(error_msg)
        
        try:
            with open(path, mode, encoding=encoding if 'b' not in mode else None) as f:
                content = f.read()
            logger.debug(f"Successfully read file: {path}")
            return content
        except Exception as e:
            if default is not None:
                logger.warning(f"Failed to read file {file_path}: {str(e)}, returning default value")
                return default
            error_msg = f"Failed to read file {file_path}: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e
    
    @staticmethod
    def load_json(file_path: Union[str, Path], default: Any = None) -> Any:
        """
        Load JSON data from a file.
        
        Args:
            file_path: Path to the JSON file
            default: Default value to return if file doesn't exist or loading fails
            
        Returns:
            Any: Parsed JSON data or default value
            
        Raises:
            FileOperationError: If JSON loading fails and no default is provided
        """
        try:
            content = FileUtils.safe_read_file(file_path, default=None)
            if content is None:
                if default is not None:
                    return default
                error_msg = f"Failed to read JSON file: {file_path}"
                logger.error(error_msg)
                raise FileOperationError(error_msg)
            
            data = json.loads(content)
            logger.debug(f"Successfully loaded JSON from: {file_path}")
            return data
        except json.JSONDecodeError as e:
            if default is not None:
                logger.warning(f"Invalid JSON in {file_path}: {str(e)}, returning default value")
                return default
            error_msg = f"Invalid JSON in {file_path}: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e
        except Exception as e:
            if default is not None:
                logger.warning(f"Failed to load JSON from {file_path}: {str(e)}, returning default value")
                return default
            error_msg = f"Failed to load JSON from {file_path}: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e
    
    @staticmethod
    def save_json(file_path: Union[str, Path], 
                 data: Any, 
                 indent: int = 4,
                 ensure_ascii: bool = False) -> Path:
        """
        Save data as JSON to a file.
        
        Args:
            file_path: Path to the JSON file
            data: Data to save as JSON
            indent: Indentation level for pretty-printing
            ensure_ascii: Whether to escape non-ASCII characters
            
        Returns:
            Path: Path object for the written file
            
        Raises:
            FileOperationError: If JSON saving fails
        """
        try:
            json_str = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
            return FileUtils.safe_write_file(file_path, json_str)
        except Exception as e:
            error_msg = f"Failed to save JSON to {file_path}: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e
    
    @staticmethod
    def load_yaml(file_path: Union[str, Path], default: Any = None) -> Any:
        """
        Load YAML data from a file.
        
        Args:
            file_path: Path to the YAML file
            default: Default value to return if file doesn't exist or loading fails
            
        Returns:
            Any: Parsed YAML data or default value
            
        Raises:
            FileOperationError: If YAML loading fails and no default is provided
        """
        try:
            content = FileUtils.safe_read_file(file_path, default=None)
            if content is None:
                if default is not None:
                    return default
                error_msg = f"Failed to read YAML file: {file_path}"
                logger.error(error_msg)
                raise FileOperationError(error_msg)
            
            data = yaml.safe_load(content)
            logger.debug(f"Successfully loaded YAML from: {file_path}")
            return data
        except yaml.YAMLError as e:
            if default is not None:
                logger.warning(f"Invalid YAML in {file_path}: {str(e)}, returning default value")
                return default
            error_msg = f"Invalid YAML in {file_path}: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e
        except Exception as e:
            if default is not None:
                logger.warning(f"Failed to load YAML from {file_path}: {str(e)}, returning default value")
                return default
            error_msg = f"Failed to load YAML from {file_path}: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e
    
    @staticmethod
    def save_yaml(file_path: Union[str, Path], data: Any) -> Path:
        """
        Save data as YAML to a file.
        
        Args:
            file_path: Path to the YAML file
            data: Data to save as YAML
            
        Returns:
            Path: Path object for the written file
            
        Raises:
            FileOperationError: If YAML saving fails
        """
        try:
            yaml_str = yaml.dump(data, default_flow_style=False)
            return FileUtils.safe_write_file(file_path, yaml_str)
        except Exception as e:
            error_msg = f"Failed to save YAML to {file_path}: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e
    
    @staticmethod
    def safe_copy_file(source_path: Union[str, Path], 
                      dest_path: Union[str, Path],
                      overwrite: bool = False) -> Path:
        """
        Safely copy a file with error handling.
        
        Args:
            source_path: Path to the source file
            dest_path: Path to the destination file
            overwrite: Whether to overwrite existing destination file
            
        Returns:
            Path: Path object for the destination file
            
        Raises:
            FileOperationError: If file copying fails
        """
        src_path = Path(source_path)
        dst_path = Path(dest_path)
        
        if not src_path.exists():
            error_msg = f"Source file not found: {src_path}"
            logger.error(error_msg)
            raise FileOperationError(error_msg)
        
        if dst_path.exists() and not overwrite:
            error_msg = f"Destination file already exists: {dst_path}"
            logger.error(error_msg)
            raise FileOperationError(error_msg)
        
        # Ensure parent directory exists
        FileUtils.ensure_directory(dst_path.parent)
        
        try:
            # Use temporary file for atomic copy
            fd, temp_path = tempfile.mkstemp(dir=dst_path.parent)
            os.close(fd)
            
            try:
                shutil.copy2(src_path, temp_path)
                shutil.move(temp_path, dst_path)
            except Exception as e:
                # Clean up temporary file if copying fails
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise e
            
            logger.debug(f"Successfully copied file: {src_path} -> {dst_path}")
            return dst_path
        except Exception as e:
            error_msg = f"Failed to copy file {src_path} to {dst_path}: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e
    
    @staticmethod
    def safe_delete_file(file_path: Union[str, Path], missing_ok: bool = True) -> bool:
        """
        Safely delete a file with error handling.
        
        Args:
            file_path: Path to the file to delete
            missing_ok: Whether to ignore missing files
            
        Returns:
            bool: True if file was deleted, False if file didn't exist and missing_ok is True
            
        Raises:
            FileOperationError: If file deletion fails or file doesn't exist and missing_ok is False
        """
        path = Path(file_path)
        
        if not path.exists():
            if missing_ok:
                logger.debug(f"File not found for deletion (ignored): {path}")
                return False
            error_msg = f"File not found for deletion: {path}"
            logger.error(error_msg)
            raise FileOperationError(error_msg)
        
        try:
            path.unlink()
            logger.debug(f"Successfully deleted file: {path}")
            return True
        except Exception as e:
            error_msg = f"Failed to delete file {file_path}: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e
    
    @staticmethod
    def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5', buffer_size: int = 65536) -> str:
        """
        Calculate hash of a file.
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256')
            buffer_size: Size of buffer for reading file in chunks
            
        Returns:
            str: Hexadecimal hash digest
            
        Raises:
            FileOperationError: If file hashing fails
        """
        path = Path(file_path)
        
        if not path.exists():
            error_msg = f"File not found for hashing: {path}"
            logger.error(error_msg)
            raise FileOperationError(error_msg)
        
        try:
            hash_obj = getattr(hashlib, algorithm)()
            
            with open(path, 'rb') as f:
                while True:
                    data = f.read(buffer_size)
                    if not data:
                        break
                    hash_obj.update(data)
            
            hash_digest = hash_obj.hexdigest()
            logger.debug(f"Calculated {algorithm} hash for {path}: {hash_digest}")
            return hash_digest
        except Exception as e:
            error_msg = f"Failed to calculate hash for {file_path}: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e
    
    @staticmethod
    def list_files(directory_path: Union[str, Path], 
                  pattern: str = '*', 
                  recursive: bool = False) -> List[Path]:
        """
        List files in a directory matching a pattern.
        
        Args:
            directory_path: Path to the directory
            pattern: Glob pattern for matching files
            recursive: Whether to search recursively
            
        Returns:
            List[Path]: List of matching file paths
            
        Raises:
            FileOperationError: If directory listing fails
        """
        path = Path(directory_path)
        
        if not path.exists():
            error_msg = f"Directory not found: {path}"
            logger.error(error_msg)
            raise FileOperationError(error_msg)
        
        if not path.is_dir():
            error_msg = f"Path is not a directory: {path}"
            logger.error(error_msg)
            raise FileOperationError(error_msg)
        
        try:
            if recursive:
                files = list(path.glob(f'**/{pattern}'))
            else:
                files = list(path.glob(pattern))
            
            # Filter out directories
            files = [f for f in files if f.is_file()]
            
            logger.debug(f"Found {len(files)} files matching '{pattern}' in {path}")
            return files
        except Exception as e:
            error_msg = f"Failed to list files in {directory_path}: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e
    
    @staticmethod
    def create_timestamped_directory(base_dir: Union[str, Path], prefix: str = '') -> Path:
        """
        Create a directory with a timestamp in its name.
        
        Args:
            base_dir: Base directory path
            prefix: Prefix for the directory name
            
        Returns:
            Path: Path object for the created directory
            
        Raises:
            FileOperationError: If directory creation fails
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_name = f"{prefix}_{timestamp}" if prefix else timestamp
        dir_path = Path(base_dir) / dir_name
        
        return FileUtils.ensure_directory(dir_path)
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """
        Get size of a file in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            int: File size in bytes
            
        Raises:
            FileOperationError: If file size retrieval fails
        """
        path = Path(file_path)
        
        if not path.exists():
            error_msg = f"File not found: {path}"
            logger.error(error_msg)
            raise FileOperationError(error_msg)
        
        try:
            size = path.stat().st_size
            logger.debug(f"File size of {path}: {size} bytes")
            return size
        except Exception as e:
            error_msg = f"Failed to get file size for {file_path}: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e
    
    @staticmethod
    def get_file_modification_time(file_path: Union[str, Path]) -> datetime:
        """
        Get last modification time of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            datetime: Last modification time
            
        Raises:
            FileOperationError: If modification time retrieval fails
        """
        path = Path(file_path)
        
        if not path.exists():
            error_msg = f"File not found: {path}"
            logger.error(error_msg)
            raise FileOperationError(error_msg)
        
        try:
            mtime = path.stat().st_mtime
            dt = datetime.fromtimestamp(mtime)
            logger.debug(f"Last modification time of {path}: {dt}")
            return dt
        except Exception as e:
            error_msg = f"Failed to get modification time for {file_path}: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e