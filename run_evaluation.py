#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Urban Sound Classifier - Evaluation Runner

This script provides a command-line interface for evaluating trained Urban Sound Classifier models.
It supports various evaluation metrics, confusion matrix visualization, and detailed reporting.

Usage:
    python run_evaluation.py --model MODEL --data DATA [options]

Example:
    python run_evaluation.py --model models/urban_sound_model.h5 --data datasets/test \
        --output evaluation_results --report --plot
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

from urban_sound_classifier.config.config_manager import ConfigManager
from urban_sound_classifier.evaluation import Evaluator
from urban_sound_classifier.training import DataLoader
from urban_sound_classifier.models.loaders import ModelLoader
from urban_sound_classifier.utils.file import FileUtils


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Urban Sound Classifier Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model or directory containing models'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to test data directory or CSV file'
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (JSON or YAML)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='evaluation_results',
        help='Path to output directory for evaluation results'
    )
    
    # Evaluation options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate detailed classification report'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots (confusion matrix, ROC curve)'
    )
    
    parser.add_argument(
        '--html-report',
        action='store_true',
        help='Generate HTML evaluation report'
    )
    
    parser.add_argument(
        '--per-file',
        action='store_true',
        help='Evaluate and report metrics for each file individually'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=1,
        help='Consider top-k predictions for accuracy calculation'
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


def evaluate_model(model, test_data, config, args, output_dir):
    """
    Evaluate a single model.
    
    Args:
        model: Loaded model
        test_data: Test data (generator or tuple)
        config: Configuration manager
        args: Command-line arguments
        output_dir: Output directory for results
    
    Returns:
        dict: Evaluation results
    """
    # Create evaluator
    evaluator = Evaluator(config)
    
    # Evaluate model
    if isinstance(test_data, tuple):
        # Unpack test data
        X_test, y_test = test_data
        
        # Make predictions and evaluate
        results = evaluator.predict_and_evaluate(
            model,
            X_test,
            y_test,
            class_names=config.get('DATA.class_names', None),
            top_k=args.top_k
        )
    else:
        # Evaluate using generator
        results = evaluator.evaluate_model(model, test_data)
    
    # Save evaluation results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        results_dict = {}
        for key, value in results.items():
            if hasattr(value, 'item'):
                results_dict[key] = value.item()
            else:
                results_dict[key] = value
        json.dump(results_dict, f, indent=2)
    logging.info(f"Evaluation results saved to {results_path}")
    
    # Generate plots if requested
    if args.plot:
        # Plot confusion matrix
        if 'confusion_matrix' in results:
            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            evaluator.plot_confusion_matrix(
                results['confusion_matrix'],
                class_names=config.get('DATA.class_names', None),
                output_path=cm_path
            )
            logging.info(f"Confusion matrix saved to {cm_path}")
        
        # Plot ROC curve
        if 'y_true' in results and 'y_pred_proba' in results:
            roc_path = os.path.join(output_dir, 'roc_curve.png')
            evaluator.plot_roc_curve(
                results['y_true'],
                results['y_pred_proba'],
                class_names=config.get('DATA.class_names', None),
                output_path=roc_path
            )
            logging.info(f"ROC curve saved to {roc_path}")
    
    # Generate HTML report if requested
    if args.html_report:
        html_path = os.path.join(output_dir, 'evaluation_report.html')
        evaluator.generate_evaluation_report(
            results,
            output_path=html_path,
            model_name=os.path.basename(args.model),
            include_plots=args.plot
        )
        logging.info(f"HTML evaluation report saved to {html_path}")
    
    return results


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
        logging.error(f'Model path {args.model} not found')
        sys.exit(1)
    
    # Check if data exists
    if not os.path.exists(args.data):
        logging.error(f'Data path {args.data} not found')
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Create a timestamped subdirectory for this evaluation run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output, f"eval_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config.get_all(), f, indent=2)
    logging.info(f"Configuration saved to {config_path}")
    
    # Load data
    data_loader = DataLoader(config)
    
    if os.path.isfile(args.data) and args.data.endswith('.csv'):
        # Load from CSV file
        _, _, test_data = data_loader.load_from_csv(
            args.data,
            validation_split=0,
            test_split=1.0  # Use all data for testing
        )
    else:
        # Load from directory
        _, _, test_data = data_loader.load_from_directory(
            args.data,
            validation_split=0,
            test_split=1.0  # Use all data for testing
        )
    
    # Create test generator
    test_generator = data_loader.create_generator(
        test_data,
        batch_size=args.batch_size,
        augment=False,
        shuffle=False  # Don't shuffle for evaluation
    )
    
    # Load model(s)
    model_loader = ModelLoader(config)
    
    if os.path.isdir(args.model):
        # Load all models in directory
        model_files = FileUtils.list_files(args.model, '.h5')
        if not model_files:
            logging.error(f"No model files found in {args.model}")
            sys.exit(1)
        
        # Evaluate each model
        all_results = {}
        for model_file in model_files:
            model_name = os.path.basename(model_file)
            logging.info(f"Evaluating model: {model_name}")
            
            # Load model
            model = model_loader.load(model_file)
            
            # Create model-specific output directory
            model_dir = os.path.join(run_dir, os.path.splitext(model_name)[0])
            os.makedirs(model_dir, exist_ok=True)
            
            # Evaluate model
            results = evaluate_model(model, test_generator, config, args, model_dir)
            all_results[model_name] = results
            
            # Print evaluation results
            print(f"\nEvaluation Results for {model_name}:")
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"{metric}: {value:.4f}")
        
        # Save comparative results
        comparative_path = os.path.join(run_dir, 'comparative_results.json')
        with open(comparative_path, 'w') as f:
            # Extract accuracy for each model
            comparative = {}
            for model_name, results in all_results.items():
                comparative[model_name] = {
                    'accuracy': results.get('accuracy', 0),
                    'f1_score': results.get('f1_score', 0),
                    'precision': results.get('precision', 0),
                    'recall': results.get('recall', 0)
                }
            json.dump(comparative, f, indent=2)
        logging.info(f"Comparative results saved to {comparative_path}")
        
        # Print comparative results
        print("\nComparative Results:")
        print("-" * 80)
        print(f"{'Model':<40} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 80)
        for model_name, metrics in comparative.items():
            print(f"{model_name:<40} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f}")
    
    else:
        # Load single model
        logging.info(f"Evaluating model: {args.model}")
        model = model_loader.load(args.model)
        
        # Evaluate model
        results = evaluate_model(model, test_generator, config, args, run_dir)
        
        # Print evaluation results
        print("\nEvaluation Results:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
    
    print(f"\nEvaluation completed successfully. All outputs saved to {run_dir}")


if __name__ == '__main__':
    main()