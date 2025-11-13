#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Urban Sound Classifier - Feature Extraction Runner

This script provides a command-line interface for extracting features from audio files
for the Urban Sound Classifier. It supports various feature types, batch processing,
and caching options.

Usage:
    python run_feature_extraction.py --input INPUT --output OUTPUT [options]

Example:
    python run_feature_extraction.py --input datasets/raw --output datasets/features \
        --feature-type mel_spectrogram --normalize --cache
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm

from urban_sound_classifier.config.config_manager import ConfigManager
from urban_sound_classifier.feature_extraction import MelSpectrogramExtractor, MFCCExtractor
from urban_sound_classifier.utils.file import FileUtils
from urban_sound_classifier.utils.audio import AudioUtils


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Urban Sound Classifier Feature Extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input audio file or directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to output directory for extracted features'
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (JSON or YAML)'
    )
    
    # Feature extraction options
    parser.add_argument(
        '--feature-type',
        type=str,
        choices=['mel_spectrogram', 'mfcc', 'combined'],
        default='mel_spectrogram',
        help='Type of features to extract'
    )
    
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize extracted features'
    )
    
    parser.add_argument(
        '--cache',
        action='store_true',
        help='Cache extracted features to disk'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['npy', 'npz', 'pickle'],
        default='npy',
        help='Output format for extracted features'
    )
    
    parser.add_argument(
        '--metadata',
        action='store_true',
        help='Save metadata along with features'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing files'
    )
    
    # Audio preprocessing options
    parser.add_argument(
        '--sample-rate',
        type=int,
        help='Target sample rate for audio preprocessing'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        help='Target duration for audio preprocessing (in seconds)'
    )
    
    parser.add_argument(
        '--mono',
        action='store_true',
        help='Convert audio to mono'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default='info',
        help='Logging level'
    )
    
    return parser.parse_args()


def setup_logging(log_level):
    """
    Set up logging configuration.
    
    Args:
        log_level (str): Logging level (debug, info, warning, error, critical)
    """
    # Map string log level to logging constants
    level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    
    numeric_level = level_map.get(log_level.lower(), logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_feature_extractor(feature_type, config):
    """
    Create a feature extractor based on the specified type.
    
    Args:
        feature_type (str): Type of features to extract
        config (ConfigManager): Configuration manager
    
    Returns:
        object: Feature extractor
    """
    if feature_type == 'mel_spectrogram':
        return MelSpectrogramExtractor(config)
    
    elif feature_type == 'mfcc':
        return MFCCExtractor(config)
    
    elif feature_type == 'combined':
        # Create a combined feature extractor that returns both mel spectrogram and MFCC
        class CombinedExtractor:
            def __init__(self, config):
                self.mel_extractor = MelSpectrogramExtractor(config)
                self.mfcc_extractor = MFCCExtractor(config)
            
            def extract(self, audio_file):
                mel_features = self.mel_extractor.extract(audio_file)
                mfcc_features = self.mfcc_extractor.extract(audio_file)
                return {'mel_spectrogram': mel_features, 'mfcc': mfcc_features}
        
        return CombinedExtractor(config)
    
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def save_features(features, output_path, format_type, metadata=None):
    """
    Save extracted features to disk.
    
    Args:
        features: Extracted features
        output_path (str): Path to save features
        format_type (str): Output format ('npy', 'npz', or 'pickle')
        metadata (dict, optional): Metadata to save along with features
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if format_type == 'npy':
        # Save as NumPy array
        np.save(output_path, features)
        
        # Save metadata separately if provided
        if metadata:
            metadata_path = os.path.splitext(output_path)[0] + '_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    elif format_type == 'npz':
        # Save as compressed NumPy archive
        if metadata:
            np.savez_compressed(output_path, features=features, **metadata)
        else:
            np.savez_compressed(output_path, features=features)
    
    elif format_type == 'pickle':
        # Save as pickle file
        import pickle
        with open(output_path, 'wb') as f:
            if metadata:
                pickle.dump({'features': features, 'metadata': metadata}, f)
            else:
                pickle.dump(features, f)
    
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def process_audio_file(audio_file, feature_extractor, output_dir, args, config):
    """
    Process a single audio file and extract features.
    
    Args:
        audio_file (str): Path to audio file
        feature_extractor: Feature extractor
        output_dir (str): Output directory
        args: Command-line arguments
        config: Configuration manager
    
    Returns:
        tuple: (features, metadata)
    """
    try:
        # Preprocess audio if needed
        if args.sample_rate or args.duration or args.mono:
            # Create temporary file for preprocessed audio
            temp_dir = os.path.join(output_dir, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Get preprocessing parameters
            sample_rate = args.sample_rate or config.get('AUDIO.sample_rate', 22050)
            duration = args.duration or config.get('AUDIO.duration', None)
            mono = args.mono or config.get('AUDIO.mono', True)
            
            # Preprocess audio
            preprocessed_file = AudioUtils.preprocess_audio(
                audio_file,
                output_dir=temp_dir,
                sample_rate=sample_rate,
                duration=duration,
                mono=mono
            )
            
            # Extract features from preprocessed audio
            features = feature_extractor.extract_features_from_file(preprocessed_file)
            
            # Remove temporary file
            os.remove(preprocessed_file)
        else:
            # Extract features directly
            features = feature_extractor.extract_features_from_file(audio_file)
        
        # Normalize features if requested
        if args.normalize:
            if isinstance(features, dict):
                # Normalize each feature type separately
                for key in features:
                    features[key] = (features[key] - np.mean(features[key])) / (np.std(features[key]) + 1e-8)
            else:
                # Normalize single feature type
                features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        # Create metadata
        metadata = None
        if args.metadata:
            metadata = {
                'file_path': audio_file,
                'sample_rate': AudioUtils.get_sample_rate(audio_file),
                'duration': AudioUtils.get_duration(audio_file),
                'channels': AudioUtils.get_channels(audio_file),
                'feature_type': args.feature_type,
                'normalized': args.normalize,
                'timestamp': datetime.now().isoformat()
            }
        
        return features, metadata
    
    except Exception as e:
        logging.error(f"Error processing {audio_file}: {e}")
        return None, None


def main():
    """
    Main entry point.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = ConfigManager()
    if args.config:
        if not os.path.exists(args.config):
            logging.error(f'Configuration file {args.config} not found')
            sys.exit(1)
        config.load(args.config)
    
    # Override configuration with command-line arguments
    config_updates = {
        'FEATURE_EXTRACTION': {
            'feature_type': args.feature_type,
            'normalize': args.normalize,
            'cache': args.cache
        },
        'AUDIO': {}
    }
    
    if args.sample_rate:
        config_updates['AUDIO']['sample_rate'] = args.sample_rate
    
    if args.duration:
        config_updates['AUDIO']['duration'] = args.duration
    
    if args.mono:
        config_updates['AUDIO']['mono'] = args.mono
    
    config.update(config_updates)
    
    # Check if input exists
    if not os.path.exists(args.input):
        logging.error(f'Input path {args.input} not found')
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Create feature extractor
    feature_extractor = create_feature_extractor(args.feature_type, config)
    
    # Process input
    if os.path.isfile(args.input):
        # Process single file
        logging.info(f"Processing file: {args.input}")
        
        # Extract features
        features, metadata = process_audio_file(args.input, feature_extractor, args.output, args, config)
        
        if features is not None:
            # Generate output path
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            output_path = os.path.join(args.output, f"{base_name}.{args.format}")
            
            # Save features
            save_features(features, output_path, args.format, metadata)
            logging.info(f"Features saved to {output_path}")
    
    elif os.path.isdir(args.input):
        # Process directory
        logging.info(f"Processing directory: {args.input}")
        
        # Get all audio files in directory
        audio_files = []
        for ext in config.get('AUDIO.valid_extensions', ['.wav', '.mp3', '.ogg', '.flac']):
            audio_files.extend(FileUtils.list_files(args.input, ext))
        
        if not audio_files:
            logging.error(f"No audio files found in {args.input}")
            sys.exit(1)
        
        logging.info(f"Found {len(audio_files)} audio files")
        
        # Process files in batches
        batch_size = args.batch_size
        for i in range(0, len(audio_files), batch_size):
            batch_files = audio_files[i:i+batch_size]
            
            # Process each file in batch
            for audio_file in tqdm(batch_files, desc=f"Batch {i//batch_size + 1}/{(len(audio_files)-1)//batch_size + 1}"):
                # Extract features
                features, metadata = process_audio_file(audio_file, feature_extractor, args.output, args, config)
                
                if features is not None:
                    # Generate output path
                    rel_path = os.path.relpath(audio_file, args.input)
                    base_name = os.path.splitext(rel_path)[0]
                    output_path = os.path.join(args.output, f"{base_name}.{args.format}")
                    
                    # Save features
                    save_features(features, output_path, args.format, metadata)
        
        logging.info(f"All features saved to {args.output}")
    
    # Save configuration
    config_path = os.path.join(args.output, 'extraction_config.json')
    with open(config_path, 'w') as f:
        json.dump(config.get_all(), f, indent=2)
    logging.info(f"Configuration saved to {config_path}")
    
    print(f"\nFeature extraction completed successfully. All outputs saved to {args.output}")


if __name__ == '__main__':
    main()