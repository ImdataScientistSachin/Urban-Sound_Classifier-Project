import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

from ..config.config_manager import ConfigManager
from ..feature_extraction import FeatureExtractor, MelSpectrogramExtractor, MFCCExtractor

class AudioDataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for audio classification tasks.
    
    This class handles loading audio files, extracting features, and generating
    batches of data for training, with support for data augmentation.
    
    Attributes:
        config (ConfigManager): Configuration manager instance
        file_paths (List[str]): List of audio file paths
        labels (np.ndarray): Array of labels
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data between epochs
        feature_extractor (FeatureExtractor): Feature extractor instance
        augment (bool): Whether to apply data augmentation
        indices (np.ndarray): Array of indices for shuffling
    """
    
    def __init__(self, 
                config: ConfigManager,
                file_paths: List[str],
                labels: np.ndarray,
                batch_size: int = 32,
                shuffle: bool = True,
                feature_type: str = 'mel_spectrogram',
                augment: bool = False):
        """
        Initialize the AudioDataGenerator.
        
        Args:
            config (ConfigManager): Configuration manager instance
            file_paths (List[str]): List of audio file paths
            labels (np.ndarray): Array of labels
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle data between epochs
            feature_type (str): Type of features to extract ('mel_spectrogram' or 'mfcc')
            augment (bool): Whether to apply data augmentation
        """
        self.config = config
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        
        # Initialize feature extractor
        if feature_type == 'mel_spectrogram':
            self.feature_extractor = MelSpectrogramExtractor(config)
        elif feature_type == 'mfcc':
            self.feature_extractor = MFCCExtractor(config)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        
        # Initialize indices
        self.indices = np.arange(len(self.file_paths))
        
        # Shuffle data if requested
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self) -> int:
        """
        Get the number of batches per epoch.
        
        Returns:
            int: Number of batches per epoch
        """
        return int(np.ceil(len(self.file_paths) / self.batch_size))
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of data.
        
        Args:
            index (int): Batch index
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Batch of features and labels
        """
        # Get batch indices
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Get batch file paths
        batch_file_paths = [self.file_paths[i] for i in batch_indices]
        
        # Get batch labels
        batch_labels = self.labels[batch_indices]
        
        # Extract features for batch
        batch_features = []
        for file_path in batch_file_paths:
            try:
                # Extract features
                features = self.feature_extractor.extract_features_from_file(file_path)
                
                # Apply augmentation if requested
                if self.augment:
                    features = self._augment_features(features)
                
                batch_features.append(features)
            except Exception as e:
                print(f"Error extracting features from {file_path}: {e}")
                # Use zeros as fallback
                shape = self.feature_extractor.get_output_shape()
                batch_features.append(np.zeros(shape))
        
        # Convert to numpy arrays
        batch_features = np.array(batch_features)
        
        return batch_features, batch_labels
    
    def on_epoch_end(self) -> None:
        """
        Called at the end of each epoch.
        
        Shuffles the data if shuffle is True.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _augment_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to features.
        
        Args:
            features (np.ndarray): Features to augment
            
        Returns:
            np.ndarray: Augmented features
        """
        # Get augmentation parameters from config
        time_shift_prob = self.config.get('AUGMENTATION.time_shift_prob', 0.5)
        pitch_shift_prob = self.config.get('AUGMENTATION.pitch_shift_prob', 0.5)
        noise_prob = self.config.get('AUGMENTATION.noise_prob', 0.5)
        mask_prob = self.config.get('AUGMENTATION.mask_prob', 0.5)
        
        # Apply time shifting
        if np.random.random() < time_shift_prob:
            max_shift = self.config.get('AUGMENTATION.max_time_shift', 10)
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift > 0:
                features = np.pad(features, ((0, 0), (shift, 0)), mode='constant')[:, :-shift]
            elif shift < 0:
                features = np.pad(features, ((0, 0), (0, -shift)), mode='constant')[:, -shift:]
        
        # Apply frequency/pitch shifting
        if np.random.random() < pitch_shift_prob:
            max_shift = self.config.get('AUGMENTATION.max_freq_shift', 5)
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift > 0:
                features = np.pad(features, ((shift, 0), (0, 0)), mode='constant')[:-shift, :]
            elif shift < 0:
                features = np.pad(features, ((0, -shift), (0, 0)), mode='constant')[-shift:, :]
        
        # Apply noise
        if np.random.random() < noise_prob:
            noise_level = self.config.get('AUGMENTATION.noise_level', 0.005)
            noise = np.random.normal(0, noise_level, features.shape)
            features = features + noise
        
        # Apply masking (time or frequency)
        if np.random.random() < mask_prob:
            mask_type = np.random.choice(['time', 'freq'])
            
            if mask_type == 'time':
                # Time masking
                max_mask_width = self.config.get('AUGMENTATION.max_time_mask_width', 10)
                mask_width = np.random.randint(1, max_mask_width + 1)
                mask_start = np.random.randint(0, features.shape[1] - mask_width + 1)
                features[:, mask_start:mask_start + mask_width] = 0
            else:
                # Frequency masking
                max_mask_height = self.config.get('AUGMENTATION.max_freq_mask_height', 5)
                mask_height = np.random.randint(1, max_mask_height + 1)
                mask_start = np.random.randint(0, features.shape[0] - mask_height + 1)
                features[mask_start:mask_start + mask_height, :] = 0
        
        return features


class DataLoader:
    """
    Class for loading and preparing audio data for training.
    
    This class handles loading audio data from CSV files or directories,
    splitting data into train/validation/test sets, and creating data generators.
    
    Attributes:
        config (ConfigManager): Configuration manager instance
        file_paths (List[str]): List of audio file paths
        labels (np.ndarray): Array of labels
        label_map (Dict[str, int]): Mapping from label strings to indices
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the DataLoader.
        
        Args:
            config (ConfigManager): Configuration manager instance
        """
        self.config = config
        self.file_paths = []
        self.labels = []
        self.label_map = {}
    
    def load_from_csv(self, csv_path: str, file_column: str = 'file_path', label_column: str = 'label') -> None:
        """
        Load data from a CSV file.
        
        Args:
            csv_path (str): Path to the CSV file
            file_column (str): Name of the column containing file paths
            label_column (str): Name of the column containing labels
            
        Raises:
            FileNotFoundError: If the CSV file does not exist
            ValueError: If the CSV file does not contain the specified columns
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load CSV file
        df = pd.read_csv(csv_path)
        
        # Check if columns exist
        if file_column not in df.columns:
            raise ValueError(f"CSV file does not contain column: {file_column}")
        if label_column not in df.columns:
            raise ValueError(f"CSV file does not contain column: {label_column}")
        
        # Get file paths and labels
        self.file_paths = df[file_column].tolist()
        
        # Create label map if it doesn't exist
        if not self.label_map:
            unique_labels = df[label_column].unique()
            self.label_map = {label: i for i, label in enumerate(sorted(unique_labels))}
        
        # Convert labels to indices
        self.labels = np.array([self.label_map[label] for label in df[label_column]])
        
        # Convert labels to one-hot encoding if needed
        if self.config.get('TRAINING.one_hot_labels', True):
            self.labels = tf.keras.utils.to_categorical(self.labels, num_classes=len(self.label_map))
    
    def load_from_directory(self, data_dir: str, valid_extensions: List[str] = None) -> None:
        """
        Load data from a directory structure.
        
        The directory structure should be:
        data_dir/
            class1/
                file1.wav
                file2.wav
                ...
            class2/
                file1.wav
                file2.wav
                ...
            ...
        
        Args:
            data_dir (str): Path to the data directory
            valid_extensions (List[str], optional): List of valid file extensions.
                If None, use default extensions from config.
            
        Raises:
            FileNotFoundError: If the data directory does not exist
            ValueError: If no valid files are found
        """
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Get valid extensions from config if not provided
        if valid_extensions is None:
            valid_extensions = self.config.get('AUDIO.valid_extensions', ['.wav', '.mp3', '.ogg', '.flac'])
        
        # Initialize lists for file paths and labels
        file_paths = []
        labels = []
        
        # Get class directories
        class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        # Create label map
        self.label_map = {label: i for i, label in enumerate(sorted(class_dirs))}
        
        # Load files from each class directory
        for class_dir in class_dirs:
            class_path = os.path.join(data_dir, class_dir)
            class_label = self.label_map[class_dir]
            
            # Get files in class directory
            for root, _, files in os.walk(class_path):
                for file in files:
                    # Check if file has valid extension
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        file_path = os.path.join(root, file)
                        file_paths.append(file_path)
                        labels.append(class_label)
        
        if not file_paths:
            raise ValueError(f"No valid files found in {data_dir}")
        
        # Set file paths and labels
        self.file_paths = file_paths
        self.labels = np.array(labels)
        
        # Convert labels to one-hot encoding if needed
        if self.config.get('TRAINING.one_hot_labels', True):
            self.labels = tf.keras.utils.to_categorical(self.labels, num_classes=len(self.label_map))
    
    def split_data(self, 
                  val_split: float = 0.2, 
                  test_split: float = 0.1, 
                  stratify: bool = True,
                  random_state: int = None) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str], np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            val_split (float): Fraction of data to use for validation
            test_split (float): Fraction of data to use for testing
            stratify (bool): Whether to stratify the split based on labels
            random_state (int, optional): Random state for reproducibility
            
        Returns:
            Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str], np.ndarray]:
                Train file paths, train labels, validation file paths, validation labels,
                test file paths, test labels
            
        Raises:
            ValueError: If no data is loaded or if split fractions are invalid
        """
        if not self.file_paths or len(self.file_paths) == 0:
            raise ValueError("No data loaded. Call load_from_csv() or load_from_directory() first.")
        
        if val_split + test_split >= 1.0:
            raise ValueError("Sum of validation and test splits must be less than 1.0")
        
        # Import train_test_split from sklearn
        from sklearn.model_selection import train_test_split
        
        # Convert one-hot encoded labels back to indices for stratification if needed
        stratify_labels = None
        if stratify:
            if len(self.labels.shape) > 1 and self.labels.shape[1] > 1:
                # One-hot encoded labels
                stratify_labels = np.argmax(self.labels, axis=1)
            else:
                # Index labels
                stratify_labels = self.labels
        
        # First split: train+val vs test
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            self.file_paths, self.labels,
            test_size=test_split,
            random_state=random_state,
            stratify=stratify_labels
        )
        
        # Update stratify labels for the second split
        if stratify and stratify_labels is not None:
            if len(self.labels.shape) > 1 and self.labels.shape[1] > 1:
                stratify_labels = np.argmax(train_val_labels, axis=1)
            else:
                stratify_labels = train_val_labels
        
        # Second split: train vs val
        val_size = val_split / (1 - test_split)  # Adjust val_size relative to train+val
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files, train_val_labels,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify_labels
        )
        
        return train_files, train_labels, val_files, val_labels, test_files, test_labels
    
    def create_generators(self, 
                         train_files: List[str], 
                         train_labels: np.ndarray, 
                         val_files: List[str], 
                         val_labels: np.ndarray, 
                         test_files: List[str], 
                         test_labels: np.ndarray,
                         batch_size: int = None,
                         feature_type: str = None,
                         augment_train: bool = None) -> Tuple[AudioDataGenerator, AudioDataGenerator, AudioDataGenerator]:
        """
        Create data generators for train, validation, and test sets.
        
        Args:
            train_files (List[str]): Training file paths
            train_labels (np.ndarray): Training labels
            val_files (List[str]): Validation file paths
            val_labels (np.ndarray): Validation labels
            test_files (List[str]): Test file paths
            test_labels (np.ndarray): Test labels
            batch_size (int, optional): Batch size. If None, use value from config.
            feature_type (str, optional): Type of features to extract.
                If None, use value from config.
            augment_train (bool, optional): Whether to apply data augmentation to training data.
                If None, use value from config.
            
        Returns:
            Tuple[AudioDataGenerator, AudioDataGenerator, AudioDataGenerator]:
                Train, validation, and test data generators
        """
        # Get parameters from config if not provided
        if batch_size is None:
            batch_size = self.config.get('TRAINING.batch_size', 32)
        
        if feature_type is None:
            feature_type = self.config.get('FEATURES.type', 'mel_spectrogram')
        
        if augment_train is None:
            augment_train = self.config.get('TRAINING.use_augmentation', False)
        
        # Create generators
        train_generator = AudioDataGenerator(
            config=self.config,
            file_paths=train_files,
            labels=train_labels,
            batch_size=batch_size,
            shuffle=True,
            feature_type=feature_type,
            augment=augment_train
        )
        
        val_generator = AudioDataGenerator(
            config=self.config,
            file_paths=val_files,
            labels=val_labels,
            batch_size=batch_size,
            shuffle=False,
            feature_type=feature_type,
            augment=False
        )
        
        test_generator = AudioDataGenerator(
            config=self.config,
            file_paths=test_files,
            labels=test_labels,
            batch_size=batch_size,
            shuffle=False,
            feature_type=feature_type,
            augment=False
        )
        
        return train_generator, val_generator, test_generator
    
    def get_class_weights(self) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets.
        
        Returns:
            Dict[int, float]: Dictionary mapping class indices to weights
            
        Raises:
            ValueError: If no data is loaded
        """
        if not self.file_paths or len(self.file_paths) == 0:
            raise ValueError("No data loaded. Call load_from_csv() or load_from_directory() first.")
        
        # Import compute_class_weight from sklearn
        from sklearn.utils.class_weight import compute_class_weight
        
        # Convert one-hot encoded labels to indices if needed
        if len(self.labels.shape) > 1 and self.labels.shape[1] > 1:
            y = np.argmax(self.labels, axis=1)
        else:
            y = self.labels
        
        # Compute class weights
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        # Create dictionary mapping class indices to weights
        class_weights = {i: w for i, w in zip(classes, weights)}
        
        return class_weights
    
    def get_label_map(self) -> Dict[str, int]:
        """
        Get the mapping from label strings to indices.
        
        Returns:
            Dict[str, int]: Label map
        """
        return self.label_map
    
    def get_inverse_label_map(self) -> Dict[int, str]:
        """
        Get the mapping from indices to label strings.
        
        Returns:
            Dict[int, str]: Inverse label map
        """
        return {v: k for k, v in self.label_map.items()}
    
    def save_label_map(self, output_path: str) -> None:
        """
        Save the label map to a file.
        
        Args:
            output_path (str): Path to save the label map
        """
        import json
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save label map as JSON
        with open(output_path, 'w') as f:
            json.dump(self.label_map, f, indent=4)
    
    @staticmethod
    def load_label_map(input_path: str) -> Dict[str, int]:
        """
        Load a label map from a file.
        
        Args:
            input_path (str): Path to the label map file
            
        Returns:
            Dict[str, int]: Label map
            
        Raises:
            FileNotFoundError: If the label map file does not exist
        """
        import json
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Label map file not found: {input_path}")
        
        # Load label map from JSON
        with open(input_path, 'r') as f:
            label_map = json.load(f)
        
        # Convert string keys to integers if needed
        if all(k.isdigit() for k in label_map.keys()):
            label_map = {int(k): v for k, v in label_map.items()}
        
        return label_map