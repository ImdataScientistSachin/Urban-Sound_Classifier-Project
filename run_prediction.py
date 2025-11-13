#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Urban Sound Classifier - Prediction Runner

This script provides a simple way to run predictions on audio files using the
Urban Sound Classifier. It can process individual files or entire directories,
and output the results to the console or save them to a file.

Usage:
    python run_prediction.py --model MODEL --input INPUT [--output OUTPUT] [--top-k TOP_K]

Example:
    python run_prediction.py --model models/urban_sound_model.h5 --input audio/siren.wav --top-k 3
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

from urban_sound_classifier.config.config_manager import ConfigManager
from urban_sound_classifier.models.prediction import Predictor
from urban_sound_classifier.utils.file import FileUtils


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Urban Sound Classifier Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (JSON or YAML)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input audio file or directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to output file or directory'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of top predictions to return'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default='info',
        help='Logging level'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'text'],
        default='json',
        help='Output format (when saving to file)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate and save spectrograms for each prediction'
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


def save_results(results, output_path, format_type, visualize=False):
    """
    Save prediction results to a file.
    
    Args:
        results (dict): Prediction results
        output_path (str): Path to output file
        format_type (str): Output format ('json', 'csv', or 'text')
        visualize (bool): Whether to generate and save spectrograms
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if format_type == 'json':
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    elif format_type == 'csv':
        # Save as CSV
        import pandas as pd
        
        # Flatten results for CSV format
        rows = []
        for file_path, predictions in results.items():
            for pred in predictions:
                rows.append({
                    'file_path': file_path,
                    'label': pred['label'],
                    'probability': pred['probability']
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    
    elif format_type == 'text':
        # Save as plain text
        with open(output_path, 'w') as f:
            for file_path, predictions in results.items():
                f.write(f"\nPredictions for {os.path.basename(file_path)}:\n")
                for i, pred in enumerate(predictions):
                    f.write(f"{i+1}. {pred['label']}: {pred['probability']:.4f} ({pred['probability']*100:.2f}%)\n")
                f.write("\n" + "-"*50 + "\n")
    
    # Generate and save spectrograms if requested
    if visualize:
        try:
            import matplotlib.pyplot as plt
            from urban_sound_classifier.feature_extraction import MelSpectrogramExtractor
            
            # Create spectrograms directory
            spectrograms_dir = os.path.join(os.path.dirname(output_path), 'spectrograms')
            os.makedirs(spectrograms_dir, exist_ok=True)
            
            # Initialize feature extractor
            config = ConfigManager()
            feature_extractor = MelSpectrogramExtractor(config)
            
            # Generate spectrograms for each file
            for file_path in results.keys():
                try:
                    # Extract features
                    features = feature_extractor.extract_features_from_file(file_path)
                    
                    # Generate spectrogram image
                    plt.figure(figsize=(10, 4))
                    plt.imshow(features, aspect='auto', origin='lower', cmap='viridis')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title(f'Mel Spectrogram - {os.path.basename(file_path)}')
                    plt.xlabel('Time')
                    plt.ylabel('Mel Frequency')
                    plt.tight_layout()
                    
                    # Save image
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    spectrogram_path = os.path.join(spectrograms_dir, f"{base_name}_spectrogram.png")
                    plt.savefig(spectrogram_path, dpi=100)
                    plt.close()
                    
                    logging.info(f"Saved spectrogram to {spectrogram_path}")
                except Exception as e:
                    logging.error(f"Error generating spectrogram for {file_path}: {e}")
        except ImportError:
            logging.error("Could not import matplotlib. Spectrograms not generated.")


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
    
    # Check if model exists
    if not os.path.exists(args.model):
        logging.error(f'Model file {args.model} not found')
        sys.exit(1)
    
    # Check if input exists
    if not os.path.exists(args.input):
        logging.error(f'Input path {args.input} not found')
        sys.exit(1)
    
    # Initialize predictor
    predictor = Predictor(config)
    predictor.load_models([args.model])
    
    # Process input
    all_results = {}
    
    if os.path.isfile(args.input):
        # Process single file
        logging.info(f"Processing file: {args.input}")
        results = predictor.predict(args.input, top_k=args.top_k)
        all_results[args.input] = results
        
        # Print results
        print(f"\nPredictions for {os.path.basename(args.input)}:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['label']}: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
    
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
        
        # Process each file
        for audio_file in audio_files:
            logging.info(f"Processing file: {audio_file}")
            results = predictor.predict(audio_file, top_k=args.top_k)
            all_results[audio_file] = results
            
            # Print results
            print(f"\nPredictions for {os.path.basename(audio_file)}:")
            for i, result in enumerate(results):
                print(f"{i+1}. {result['label']}: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
    
    # Save results if output is specified
    if args.output:
        # If output is a directory, create a timestamped file
        if os.path.isdir(args.output) or args.output.endswith('/'):
            os.makedirs(args.output, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if args.format == 'json':
                output_path = os.path.join(args.output, f"predictions_{timestamp}.json")
            elif args.format == 'csv':
                output_path = os.path.join(args.output, f"predictions_{timestamp}.csv")
            else:  # text
                output_path = os.path.join(args.output, f"predictions_{timestamp}.txt")
        else:
            output_path = args.output
        
        # Save results
        save_results(all_results, output_path, args.format, args.visualize)
        logging.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()