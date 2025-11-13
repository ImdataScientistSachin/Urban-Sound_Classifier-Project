#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Urban Sound Classifier - Data Augmentation Runner

This script provides a command-line interface for augmenting audio files for the
Urban Sound Classifier. It supports various augmentation techniques, batch processing,
and customizable output formats.

Usage:
    python run_augmentation.py --input INPUT --output OUTPUT [options]

Example:
    python run_augmentation.py --input datasets/raw --output datasets/augmented \
        --techniques pitch_shift time_stretch add_noise --factor 3
"""

import os
import sys
import argparse
import logging
import json
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm

from urban_sound_classifier.config.config_manager import ConfigManager
from urban_sound_classifier.utils.file import FileUtils
from urban_sound_classifier.utils.audio import AudioUtils


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Urban Sound Classifier Data Augmentation',
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
        help='Path to output directory for augmented audio'
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (JSON or YAML)'
    )
    
    # Augmentation options
    parser.add_argument(
        '--techniques',
        type=str,
        nargs='+',
        choices=[
            'pitch_shift', 'time_stretch', 'add_noise', 'add_background',
            'reverb', 'eq', 'compression', 'normalize', 'random_gain',
            'polarity_inversion', 'clip_distortion', 'all'
        ],
        default=['pitch_shift', 'time_stretch', 'add_noise'],
        help='Augmentation techniques to apply'
    )
    
    parser.add_argument(
        '--factor',
        type=int,
        default=2,
        help='Augmentation factor (number of augmented versions per file)'
    )
    
    parser.add_argument(
        '--preserve-original',
        action='store_true',
        help='Copy original files to output directory'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing files'
    )
    
    # Technique-specific parameters
    parser.add_argument(
        '--pitch-shift-range',
        type=float,
        nargs=2,
        default=[-3.0, 3.0],
        help='Range for pitch shifting in semitones (min max)'
    )
    
    parser.add_argument(
        '--time-stretch-range',
        type=float,
        nargs=2,
        default=[0.8, 1.2],
        help='Range for time stretching (min max)'
    )
    
    parser.add_argument(
        '--noise-range',
        type=float,
        nargs=2,
        default=[0.001, 0.01],
        help='Range for noise amplitude (min max)'
    )
    
    parser.add_argument(
        '--background-dir',
        type=str,
        help='Directory containing background audio files'
    )
    
    parser.add_argument(
        '--background-range',
        type=float,
        nargs=2,
        default=[0.1, 0.3],
        help='Range for background mix ratio (min max)'
    )
    
    # Output options
    parser.add_argument(
        '--format',
        type=str,
        choices=['wav', 'mp3', 'ogg', 'flac'],
        default='wav',
        help='Output audio format'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=22050,
        help='Output sample rate'
    )
    
    parser.add_argument(
        '--metadata',
        action='store_true',
        help='Save metadata for augmented files'
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


def apply_augmentation(audio_file, output_dir, args, config):
    """
    Apply augmentation to a single audio file.
    
    Args:
        audio_file (str): Path to audio file
        output_dir (str): Output directory
        args: Command-line arguments
        config: Configuration manager
    
    Returns:
        list: Paths to augmented files
    """
    try:
        # Load audio file
        y, sr = AudioUtils.load_audio(audio_file, sr=args.sample_rate)
        
        # Create output directory structure
        rel_path = os.path.relpath(audio_file, args.input)
        output_subdir = os.path.dirname(rel_path)
        full_output_dir = os.path.join(output_dir, output_subdir)
        os.makedirs(full_output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        
        # Copy original file if requested
        augmented_files = []
        if args.preserve_original:
            original_output_path = os.path.join(full_output_dir, f"{base_name}.{args.format}")
            AudioUtils.save_audio(y, sr, original_output_path, format=args.format)
            augmented_files.append(original_output_path)
            
            # Save metadata if requested
            if args.metadata:
                metadata = {
                    'original_file': audio_file,
                    'augmentation': 'none',
                    'parameters': {},
                    'timestamp': datetime.now().isoformat()
                }
                metadata_path = os.path.splitext(original_output_path)[0] + '_metadata.json'
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        # Determine which techniques to apply
        techniques = args.techniques
        if 'all' in techniques:
            techniques = [
                'pitch_shift', 'time_stretch', 'add_noise', 'add_background',
                'reverb', 'eq', 'compression', 'normalize', 'random_gain',
                'polarity_inversion', 'clip_distortion'
            ]
        
        # Apply augmentation multiple times
        for i in range(args.factor):
            # Randomly select techniques to apply for this instance
            num_techniques = random.randint(1, min(3, len(techniques)))
            selected_techniques = random.sample(techniques, num_techniques)
            
            # Start with original audio
            aug_y = np.copy(y)
            aug_params = {}
            
            # Apply selected techniques
            for technique in selected_techniques:
                if technique == 'pitch_shift':
                    # Apply pitch shifting
                    pitch_shift = random.uniform(args.pitch_shift_range[0], args.pitch_shift_range[1])
                    aug_y = AudioUtils.pitch_shift(aug_y, sr, pitch_shift)
                    aug_params['pitch_shift'] = pitch_shift
                
                elif technique == 'time_stretch':
                    # Apply time stretching
                    rate = random.uniform(args.time_stretch_range[0], args.time_stretch_range[1])
                    aug_y = AudioUtils.time_stretch(aug_y, rate)
                    aug_params['time_stretch'] = rate
                
                elif technique == 'add_noise':
                    # Add noise
                    noise_amp = random.uniform(args.noise_range[0], args.noise_range[1])
                    aug_y = AudioUtils.add_noise(aug_y, noise_amp)
                    aug_params['noise_amplitude'] = noise_amp
                
                elif technique == 'add_background' and args.background_dir:
                    # Add background noise from a random file
                    background_files = FileUtils.list_files(args.background_dir)
                    if background_files:
                        background_file = random.choice(background_files)
                        mix_ratio = random.uniform(args.background_range[0], args.background_range[1])
                        aug_y = AudioUtils.add_background(aug_y, sr, background_file, mix_ratio)
                        aug_params['background_file'] = os.path.basename(background_file)
                        aug_params['background_ratio'] = mix_ratio
                
                elif technique == 'reverb':
                    # Apply reverb
                    reverberance = random.uniform(0, 100)
                    damping = random.uniform(0, 100)
                    room_scale = random.uniform(0, 100)
                    aug_y = AudioUtils.apply_reverb(aug_y, sr, reverberance, damping, room_scale)
                    aug_params['reverb'] = {
                        'reverberance': reverberance,
                        'damping': damping,
                        'room_scale': room_scale
                    }
                
                elif technique == 'eq':
                    # Apply random EQ
                    freq = random.uniform(100, 5000)
                    q = random.uniform(0.5, 5.0)
                    gain = random.uniform(-12, 12)
                    aug_y = AudioUtils.apply_eq(aug_y, sr, freq, q, gain)
                    aug_params['eq'] = {
                        'frequency': freq,
                        'q': q,
                        'gain': gain
                    }
                
                elif technique == 'compression':
                    # Apply compression
                    threshold = random.uniform(-30, -10)
                    ratio = random.uniform(2, 8)
                    aug_y = AudioUtils.apply_compression(aug_y, sr, threshold, ratio)
                    aug_params['compression'] = {
                        'threshold': threshold,
                        'ratio': ratio
                    }
                
                elif technique == 'normalize':
                    # Normalize audio
                    aug_y = AudioUtils.normalize(aug_y)
                    aug_params['normalize'] = True
                
                elif technique == 'random_gain':
                    # Apply random gain
                    gain_db = random.uniform(-6, 6)
                    aug_y = AudioUtils.apply_gain(aug_y, gain_db)
                    aug_params['gain_db'] = gain_db
                
                elif technique == 'polarity_inversion':
                    # Invert polarity
                    aug_y = -aug_y
                    aug_params['polarity_inversion'] = True
                
                elif technique == 'clip_distortion':
                    # Apply clipping distortion
                    threshold = random.uniform(0.5, 0.9)
                    aug_y = AudioUtils.apply_clipping(aug_y, threshold)
                    aug_params['clip_threshold'] = threshold
            
            # Save augmented audio
            aug_output_path = os.path.join(
                full_output_dir,
                f"{base_name}_aug_{i+1}.{args.format}"
            )
            AudioUtils.save_audio(aug_y, sr, aug_output_path, format=args.format)
            augmented_files.append(aug_output_path)
            
            # Save metadata if requested
            if args.metadata:
                metadata = {
                    'original_file': audio_file,
                    'augmentation': 'multiple',
                    'techniques': selected_techniques,
                    'parameters': aug_params,
                    'timestamp': datetime.now().isoformat()
                }
                metadata_path = os.path.splitext(aug_output_path)[0] + '_metadata.json'
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        return augmented_files
    
    except Exception as e:
        logging.error(f"Error augmenting {audio_file}: {e}")
        return []


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
    
    # Check if input exists
    if not os.path.exists(args.input):
        logging.error(f'Input path {args.input} not found')
        sys.exit(1)
    
    # Check if background directory exists if specified
    if 'add_background' in args.techniques and args.background_dir and not os.path.exists(args.background_dir):
        logging.error(f'Background directory {args.background_dir} not found')
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Process input
    if os.path.isfile(args.input):
        # Process single file
        logging.info(f"Augmenting file: {args.input}")
        
        # Apply augmentation
        augmented_files = apply_augmentation(args.input, args.output, args, config)
        
        logging.info(f"Created {len(augmented_files)} augmented files")
    
    elif os.path.isdir(args.input):
        # Process directory
        logging.info(f"Augmenting directory: {args.input}")
        
        # Get all audio files in directory
        audio_files = []
        for ext in ['.wav', '.mp3', '.ogg', '.flac']:
            audio_files.extend(FileUtils.list_files(args.input, ext))
        
        if not audio_files:
            logging.error(f"No audio files found in {args.input}")
            sys.exit(1)
        
        logging.info(f"Found {len(audio_files)} audio files")
        
        # Process files in batches
        total_augmented = 0
        batch_size = args.batch_size
        for i in range(0, len(audio_files), batch_size):
            batch_files = audio_files[i:i+batch_size]
            
            # Process each file in batch
            for audio_file in tqdm(batch_files, desc=f"Batch {i//batch_size + 1}/{(len(audio_files)-1)//batch_size + 1}"):
                # Apply augmentation
                augmented_files = apply_augmentation(audio_file, args.output, args, config)
                total_augmented += len(augmented_files)
        
        logging.info(f"Created {total_augmented} augmented files")
    
    # Save configuration
    config_path = os.path.join(args.output, 'augmentation_config.json')
    with open(config_path, 'w') as f:
        # Create augmentation config
        aug_config = {
            'input': args.input,
            'output': args.output,
            'techniques': args.techniques,
            'factor': args.factor,
            'preserve_original': args.preserve_original,
            'format': args.format,
            'sample_rate': args.sample_rate,
            'parameters': {
                'pitch_shift_range': args.pitch_shift_range,
                'time_stretch_range': args.time_stretch_range,
                'noise_range': args.noise_range,
                'background_range': args.background_range
            },
            'timestamp': datetime.now().isoformat()
        }
        json.dump(aug_config, f, indent=2)
    logging.info(f"Configuration saved to {config_path}")
    
    print(f"\nAugmentation completed successfully. All outputs saved to {args.output}")


if __name__ == '__main__':
    main()