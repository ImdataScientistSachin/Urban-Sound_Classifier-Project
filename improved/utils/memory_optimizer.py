import gc
import os
import time
import logging
import psutil
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any, Callable, List, Tuple, Union
from contextlib import contextmanager

class MemoryOptimizer:
    """
    Utility class for memory optimization during processing.
    
    This class provides methods to monitor and optimize memory usage during
    resource-intensive operations like feature extraction and model training.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the MemoryOptimizer.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance for logging memory information
        """
        self.logger = logger or logging.getLogger(__name__)
        self.process = psutil.Process(os.getpid())
        self.memory_logs = []
        self.peak_memory = 0
        self.initial_memory = self.get_memory_usage()
        
        # Log initial memory state
        self.logger.info(f"Initial memory usage: {self.format_bytes(self.initial_memory)}")
    
    def get_memory_usage(self) -> int:
        """
        Get current memory usage of the process.
        
        Returns:
            int: Memory usage in bytes
        """
        try:
            return self.process.memory_info().rss
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return 0
    
    def log_memory_usage(self, tag: str = "") -> int:
        """
        Log current memory usage and return it.
        
        Args:
            tag (str): Optional tag to identify the memory usage point
            
        Returns:
            int: Current memory usage in bytes
        """
        current_memory = self.get_memory_usage()
        delta = current_memory - self.initial_memory
        
        # Update peak memory
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        
        # Log memory usage
        message = f"Memory usage{' (' + tag + ')' if tag else ''}: {self.format_bytes(current_memory)}"
        message += f" (Δ: {'+' if delta >= 0 else ''}{self.format_bytes(delta)})"
        
        self.logger.debug(message)
        
        # Store memory log
        self.memory_logs.append({
            'timestamp': time.time(),
            'tag': tag,
            'memory_bytes': current_memory,
            'delta_bytes': delta
        })
        
        return current_memory
    
    def clear_memory(self, force_gc: bool = True) -> int:
        """
        Clear unused memory and run garbage collection.
        
        Args:
            force_gc (bool): Whether to force garbage collection
            
        Returns:
            int: Memory freed in bytes
        """
        before = self.get_memory_usage()
        
        # Clear TensorFlow memory if available
        try:
            if hasattr(tf, 'keras'):
                tf.keras.backend.clear_session()
                self.logger.debug("Cleared TensorFlow Keras session")
        except Exception as e:
            self.logger.warning(f"Failed to clear TensorFlow session: {e}")
        
        # Run garbage collection
        if force_gc:
            gc.collect()
        
        # Get memory after cleanup
        after = self.get_memory_usage()
        freed = before - after
        
        self.logger.debug(f"Memory cleanup: freed {self.format_bytes(freed)}")
        
        return freed
    
    def estimate_batch_size(self, 
                           sample_size_bytes: int, 
                           target_memory_usage: float = 0.7, 
                           overhead_factor: float = 1.5) -> int:
        """
        Estimate optimal batch size based on available memory.
        
        Args:
            sample_size_bytes (int): Size of a single sample in bytes
            target_memory_usage (float): Target memory usage as a fraction of available memory
            overhead_factor (float): Factor to account for processing overhead
            
        Returns:
            int: Estimated optimal batch size
        """
        try:
            # Get available memory
            available_memory = psutil.virtual_memory().available
            
            # Calculate memory available for batch processing
            usable_memory = available_memory * target_memory_usage
            
            # Account for overhead
            effective_sample_size = sample_size_bytes * overhead_factor
            
            # Calculate batch size
            batch_size = max(1, int(usable_memory / effective_sample_size))
            
            self.logger.info(f"Estimated batch size: {batch_size} (sample size: {self.format_bytes(sample_size_bytes)}, "
                           f"available memory: {self.format_bytes(available_memory)})")
            
            return batch_size
        except Exception as e:
            self.logger.warning(f"Failed to estimate batch size: {e}. Using default batch size of 16.")
            return 16
    
    def is_memory_critical(self, threshold: float = 0.9) -> bool:
        """
        Check if memory usage is at a critical level.
        
        Args:
            threshold (float): Memory usage threshold as a fraction of total memory
            
        Returns:
            bool: True if memory usage is critical, False otherwise
        """
        try:
            memory_percent = psutil.virtual_memory().percent / 100.0
            is_critical = memory_percent >= threshold
            
            if is_critical:
                self.logger.warning(f"Memory usage critical: {memory_percent:.1%} (threshold: {threshold:.1%})")
            
            return is_critical
        except Exception as e:
            self.logger.warning(f"Failed to check memory status: {e}")
            return False
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of memory usage statistics.
        
        Returns:
            Dict[str, Any]: Memory usage summary
        """
        current_memory = self.get_memory_usage()
        
        return {
            'current_bytes': current_memory,
            'current_formatted': self.format_bytes(current_memory),
            'peak_bytes': self.peak_memory,
            'peak_formatted': self.format_bytes(self.peak_memory),
            'initial_bytes': self.initial_memory,
            'initial_formatted': self.format_bytes(self.initial_memory),
            'delta_bytes': current_memory - self.initial_memory,
            'delta_formatted': self.format_bytes(current_memory - self.initial_memory),
            'system_total': self.format_bytes(psutil.virtual_memory().total),
            'system_available': self.format_bytes(psutil.virtual_memory().available),
            'system_percent': f"{psutil.virtual_memory().percent}%"
        }
    
    def log_memory_summary(self) -> None:
        """
        Log a summary of memory usage statistics.
        """
        summary = self.get_memory_summary()
        
        self.logger.info(f"Memory Usage Summary:")
        self.logger.info(f"  Current: {summary['current_formatted']}")
        self.logger.info(f"  Peak: {summary['peak_formatted']}")
        self.logger.info(f"  Change: {summary['delta_formatted']}")
        self.logger.info(f"  System: {summary['system_percent']} used, {summary['system_available']} available")
    
    @staticmethod
    def format_bytes(bytes_value: int) -> str:
        """
        Format bytes value to human-readable string.
        
        Args:
            bytes_value (int): Bytes value to format
            
        Returns:
            str: Formatted string
        """
        if bytes_value < 0:
            return f"-{MemoryOptimizer.format_bytes(-bytes_value)}"
            
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        unit_index = 0
        value = float(bytes_value)
        
        while value >= 1024 and unit_index < len(units) - 1:
            value /= 1024
            unit_index += 1
        
        return f"{value:.2f} {units[unit_index]}"
    
    @contextmanager
    def memory_monitor(self, operation_name: str, log_interval: float = 5.0) -> None:
        """
        Context manager to monitor memory usage during an operation.
        
        Args:
            operation_name (str): Name of the operation being monitored
            log_interval (float): Interval in seconds between memory logs
            
        Yields:
            None
        """
        start_time = time.time()
        start_memory = self.get_memory_usage()
        last_log_time = start_time
        
        self.logger.info(f"Starting {operation_name} (initial memory: {self.format_bytes(start_memory)})")
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.get_memory_usage()
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.logger.info(f"Completed {operation_name} in {duration:.2f}s")
            self.logger.info(f"Memory change: {self.format_bytes(memory_delta)} "
                           f"({self.format_bytes(start_memory)} → {self.format_bytes(end_memory)})")
    
    def optimize_numpy_arrays(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        """
        Optimize memory usage of numpy arrays by converting to appropriate dtypes.
        
        Args:
            arrays (List[np.ndarray]): List of numpy arrays to optimize
            
        Returns:
            List[np.ndarray]: Optimized numpy arrays
        """
        optimized = []
        total_saved = 0
        
        for i, arr in enumerate(arrays):
            original_size = arr.nbytes
            
            # Skip small arrays
            if original_size < 1024:  # Less than 1KB
                optimized.append(arr)
                continue
            
            # Optimize based on data type and range
            try:
                if arr.dtype == np.float64:
                    # Convert float64 to float32
                    arr_opt = arr.astype(np.float32)
                    optimized.append(arr_opt)
                    saved = original_size - arr_opt.nbytes
                    total_saved += saved
                elif arr.dtype == np.int64:
                    # Check value range to determine optimal int type
                    min_val, max_val = np.min(arr), np.max(arr)
                    
                    if min_val >= 0:
                        if max_val < 256:
                            arr_opt = arr.astype(np.uint8)
                        elif max_val < 65536:
                            arr_opt = arr.astype(np.uint16)
                        else:
                            arr_opt = arr.astype(np.uint32)
                    else:
                        if min_val >= -128 and max_val < 128:
                            arr_opt = arr.astype(np.int8)
                        elif min_val >= -32768 and max_val < 32768:
                            arr_opt = arr.astype(np.int16)
                        else:
                            arr_opt = arr.astype(np.int32)
                    
                    optimized.append(arr_opt)
                    saved = original_size - arr_opt.nbytes
                    total_saved += saved
                else:
                    # Keep original array for other types
                    optimized.append(arr)
            except Exception as e:
                self.logger.warning(f"Failed to optimize array {i}: {e}")
                optimized.append(arr)
        
        if total_saved > 0:
            self.logger.debug(f"Optimized arrays: saved {self.format_bytes(total_saved)}")
        
        return optimized
    
    def enable_tensorflow_memory_growth(self) -> bool:
        """
        Enable memory growth for TensorFlow GPU to prevent allocating all GPU memory at once.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"Enabled memory growth for {len(gpus)} GPU(s)")
                return True
            else:
                self.logger.debug("No GPUs available for TensorFlow")
                return False
        except Exception as e:
            self.logger.warning(f"Failed to configure TensorFlow GPU memory growth: {e}")
            return False
    
    def limit_tensorflow_memory(self, memory_limit_mb: int) -> bool:
        """
        Limit TensorFlow GPU memory usage.
        
        Args:
            memory_limit_mb (int): Memory limit in MB
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=memory_limit_mb
                        )]
                    )
                self.logger.info(f"Limited TensorFlow GPU memory to {memory_limit_mb}MB for {len(gpus)} GPU(s)")
                return True
            else:
                self.logger.debug("No GPUs available for TensorFlow")
                return False
        except Exception as e:
            self.logger.warning(f"Failed to limit TensorFlow GPU memory: {e}")
            return False

# Create a singleton instance
memory_optimizer = MemoryOptimizer()