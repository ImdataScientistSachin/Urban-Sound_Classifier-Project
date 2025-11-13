#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Urban Sound Classifier - Training Runner

This script provides a command-line interface for training the Urban Sound Classifier model.
It supports various configuration options, including model architecture, training parameters,
and data augmentation settings.

Usage:
    python run_training.py --data DATA_DIR --output OUTPUT_DIR [options]

Example:
    python run_training.py --data datasets/urbansound8k --output models/trained \
        --epochs 50 --batch-size 32 --augment --model double_unet
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

from urban_sound_classifier.config.config_manager import ConfigManager
from urban_sound_classifier.training import Trainer, DataLoader
from urban_sound_classifier.models.architectures import DoubleUNet
from urban_sound_classifier.utils.file import FileUtils


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Urban Sound Classifier Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to dataset directory or CSV file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to output directory for trained model and logs'
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (JSON or YAML)'
    )
    
    # Model architecture
    parser.add_argument(
        '--model',
        type=str,
        choices=['double_unet', 'simple_cnn', 'resnet', 'custom'],
        default='double_unet',
        help='Model architecture to use'
    )
    
    parser.add_argument(
        '--custom-model',
        type=str,
        help='Path to custom model definition (Python file)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Initial learning rate'
    )
    
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['adam', 'sgd', 'rmsprop', 'adagrad'],
        default='adam',
        help='Optimizer to use'
    )
    
    parser.add_argument(
        '--loss',
        type=str,
        choices=['categorical_crossentropy', 'binary_crossentropy', 'sparse_categorical_crossentropy'],
        default='categorical_crossentropy',
        help='Loss function to use'
    )
    
    # Data splitting
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Fraction of data to use for validation'
    )
    
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.1,
        help='Fraction of data to use for testing'
    )
    
    # Data augmentation
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Enable data augmentation'
    )
    
    parser.add_argument(
        '--augmentation-factor',
        type=float,
        default=0.5,
        help='Probability of applying each augmentation'
    )
    
    # Feature extraction
    parser.add_argument(
        '--feature-type',
        type=str,
        choices=['mel_spectrogram', 'mfcc', 'combined'],
        default='mel_spectrogram',
        help='Type of features to extract'
    )
    
    # Callbacks
    parser.add_argument(
        '--early-stopping',
        action='store_true',
        help='Enable early stopping'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Patience for early stopping'
    )
    
    parser.add_argument(
        '--lr-scheduler',
        action='store_true',
        help='Enable learning rate scheduler'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default='info',
        help='Logging level'
    )
    
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Enable TensorBoard logging'
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


def create_model(model_type, config, input_shape, num_classes):
    """
    Create a model based on the specified architecture.
    
    Args:
        model_type (str): Type of model to create
        config (ConfigManager): Configuration manager
        input_shape (tuple): Input shape for the model
        num_classes (int): Number of output classes
    
    Returns:
        tf.keras.Model: Created model
    """
    if model_type == 'double_unet':
        return DoubleUNet(input_shape=input_shape, num_classes=num_classes).build()
    
    elif model_type == 'simple_cnn':
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
        
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    elif model_type == 'resnet':
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        from tensorflow.keras.models import Model
        
        # Load pre-trained ResNet50 without top layers
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        return model
    
    elif model_type == 'custom':
        # Load custom model from a file
        import os
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        # Get custom model path from arguments or use default
        custom_model_path = args.get('custom_model_path', None)
        
        if not custom_model_path:
            raise ValueError("Custom model path must be provided when model_type is 'custom'")
        
        if not os.path.exists(custom_model_path):
            raise FileNotFoundError(f"Custom model file not found: {custom_model_path}")
            
        # Load the custom model
        try:
            # Define custom objects dictionary for any custom layers
            from custom_model_loader import AttentionGate
            custom_objects = {
                'AttentionGate': AttentionGate
            }
            
            # Load the model with custom objects
            custom_model = load_model(custom_model_path, custom_objects=custom_objects)
            print(f"Custom model loaded successfully from {custom_model_path}")
            
            # Check if the model output matches the required number of classes
            output_shape = custom_model.output_shape
            if output_shape[-1] != num_classes:
                print(f"Warning: Custom model output shape {output_shape} doesn't match required number of classes {num_classes}")
                
                # Replace the last layer to match the required number of classes
                x = custom_model.layers[-2].output
                predictions = Dense(num_classes, activation='softmax')(x)
                custom_model = Model(inputs=custom_model.input, outputs=predictions)
                print(f"Modified custom model to output {num_classes} classes")
            
            return custom_model
        except Exception as e:
            raise ValueError(f"Failed to load custom model: {str(e)}")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
        'TRAINING': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'optimizer': args.optimizer,
            'loss': args.loss,
            'validation_split': args.validation_split,
            'test_split': args.test_split,
            'early_stopping': args.early_stopping,
            'patience': args.patience,
            'lr_scheduler': args.lr_scheduler,
            'tensorboard': args.tensorboard
        },
        'FEATURE_EXTRACTION': {
            'feature_type': args.feature_type
        },
        'DATA_AUGMENTATION': {
            'enabled': args.augment,
            'augmentation_factor': args.augmentation_factor
        }
    }
    config.update(config_updates)
    
    # Check if data path exists
    if not os.path.exists(args.data):
        logging.error(f'Data path {args.data} not found')
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Create a timestamped subdirectory for this training run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output, f"run_{timestamp}")
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
        train_data, val_data, test_data = data_loader.load_from_csv(
            args.data,
            validation_split=args.validation_split,
            test_split=args.test_split
        )
    else:
        # Load from directory
        train_data, val_data, test_data = data_loader.load_from_directory(
            args.data,
            validation_split=args.validation_split,
            test_split=args.test_split
        )
    
    # Create data generators
    train_generator = data_loader.create_generator(
        train_data,
        batch_size=args.batch_size,
        augment=args.augment
    )
    
    val_generator = data_loader.create_generator(
        val_data,
        batch_size=args.batch_size,
        augment=False
    )
    
    # Determine input shape and number of classes
    sample_batch = next(iter(train_generator))
    input_shape = sample_batch[0].shape[1:]
    num_classes = sample_batch[1].shape[1]
    
    logging.info(f"Input shape: {input_shape}, Number of classes: {num_classes}")
    
    # Create model
    model = create_model(args.model, config, input_shape, num_classes)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Configure callbacks
    callbacks = []
    
    # ModelCheckpoint callback
    model_checkpoint_path = os.path.join(run_dir, 'checkpoints')
    os.makedirs(model_checkpoint_path, exist_ok=True)
    callbacks.append(trainer.create_model_checkpoint_callback(
        os.path.join(model_checkpoint_path, 'model_{epoch:02d}_{val_accuracy:.4f}.h5'),
        monitor='val_accuracy'
    ))
    
    # Early stopping callback
    if args.early_stopping:
        callbacks.append(trainer.create_early_stopping_callback(
            patience=args.patience,
            monitor='val_loss'
        ))
    
    # Learning rate scheduler callback
    if args.lr_scheduler:
        callbacks.append(trainer.create_reduce_lr_callback(
            patience=args.patience // 2,
            monitor='val_loss'
        ))
    
    # TensorBoard callback
    if args.tensorboard:
        tensorboard_path = os.path.join(run_dir, 'logs')
        os.makedirs(tensorboard_path, exist_ok=True)
        callbacks.append(trainer.create_tensorboard_callback(tensorboard_path))
    
    # CSV Logger callback
    csv_log_path = os.path.join(run_dir, 'training_log.csv')
    callbacks.append(trainer.create_csv_logger_callback(csv_log_path))
    
    # Compile model
    trainer.compile_model(
        model,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        loss=args.loss,
        metrics=['accuracy']
    )
    
    # Train model
    history = trainer.train_with_generator(
        model,
        train_generator,
        val_generator,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Save final model
    final_model_path = os.path.join(run_dir, 'final_model.h5')
    model.save(final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(run_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        history_dict = {}
        for key, value in history.history.items():
            history_dict[key] = [float(v) for v in value]
        json.dump(history_dict, f, indent=2)
    logging.info(f"Training history saved to {history_path}")
    
    # Evaluate model on test data if available
    if test_data:
        test_generator = data_loader.create_generator(
            test_data,
            batch_size=args.batch_size,
            augment=False
        )
        
        # Evaluate model
        evaluation = trainer.evaluate_model(model, test_generator)
        
        # Save evaluation results
        evaluation_path = os.path.join(run_dir, 'evaluation_results.json')
        with open(evaluation_path, 'w') as f:
            # Convert numpy values to Python types for JSON serialization
            evaluation_dict = {}
            for key, value in evaluation.items():
                evaluation_dict[key] = float(value)
            json.dump(evaluation_dict, f, indent=2)
        logging.info(f"Evaluation results saved to {evaluation_path}")
        
        # Print evaluation results
        print("\nEvaluation Results:")
        for metric, value in evaluation.items():
            print(f"{metric}: {value:.4f}")
    
    print(f"\nTraining completed successfully. All outputs saved to {run_dir}")


if __name__ == '__main__':
    main()