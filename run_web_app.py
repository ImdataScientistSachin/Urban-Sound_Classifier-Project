#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Urban Sound Classifier - Web Application Runner

This script provides a simple way to start the Urban Sound Classifier web application.
It handles configuration loading and sets up the web server with the specified parameters.

Usage:
    python run_web_app.py [--config CONFIG] [--host HOST] [--port PORT] [--debug] [--workers WORKERS]

Example:
    python run_web_app.py --host 0.0.0.0 --port 5000 --workers 4
"""

import os
import sys
import argparse
import logging
import platform

from urban_sound_classifier.config.config_manager import ConfigManager
from urban_sound_classifier.web_app import WebApp, create_app


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Urban Sound Classifier Web Application',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (JSON or YAML)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to run the web server on'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to run the web server on'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of worker processes for the server'
    )
    
    parser.add_argument(
        '--threads',
        type=int,
        default=1,
        help='Number of threads per worker process'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default='info',
        help='Logging level'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=120,
        help='Request timeout in seconds'
    )
    
    parser.add_argument(
        '--max-upload-size',
        type=int,
        default=16,
        help='Maximum upload file size in MB'
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
    
    # Update config with command-line arguments
    config.update({
        'WEB.host': args.host,
        'WEB.port': args.port,
        'WEB.debug': args.debug,
        'WEB.request_timeout': args.timeout,
        'WEB.max_upload_size': args.max_upload_size if hasattr(args, 'max_upload_size') else 16,
        'PATHS.models_dir': os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
    })
    
    # Log the models directory path for debugging
    logging.info(f"Models directory set to: {config.get('PATHS.models_dir')}")
    
    # Verify models directory exists
    models_dir = config.get('PATHS.models_dir')
    if not os.path.exists(models_dir):
        logging.warning(f"Models directory does not exist: {models_dir}")
        os.makedirs(models_dir, exist_ok=True)
        logging.info(f"Created models directory: {models_dir}")
    else:
        # List available models
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5') or f.endswith('.tflite') or os.path.isdir(os.path.join(models_dir, f))]
        if model_files:
            logging.info(f"Found {len(model_files)} model files in {models_dir}: {', '.join(model_files)}")
        else:
            logging.warning(f"No model files found in {models_dir}")
    
    # Create upload directory if it doesn't exist
    upload_dir = config.get('PATHS.upload_dir', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    # Create and initialize web app
    logging.info(f"Starting Urban Sound Classifier web application on {args.host}:{args.port}")
    app = create_app(config)
    
    # Determine which WSGI server to use based on platform
    system = platform.system().lower()
    
    if system == 'windows':
        # Use Waitress on Windows
        try:
            from waitress import serve
            logging.info(f"Running with Waitress server on {args.host}:{args.port} with {args.threads} threads")
            serve(app, host=args.host, port=args.port, threads=args.threads)
        except ImportError:
            logging.warning("Waitress not installed. Installing it is recommended for production use on Windows.")
            logging.warning("Run: pip install waitress")
            logging.info("Falling back to Flask's development server")
            app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        # Use Gunicorn on Linux/macOS
        try:
            import gunicorn.app.base
            
            class StandaloneApplication(gunicorn.app.base.BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()
                
                def load_config(self):
                    for key, value in self.options.items():
                        if key in self.cfg.settings and value is not None:
                            self.cfg.set(key.lower(), value)
                
                def load(self):
                    return self.application
            
            options = {
                'bind': f"{args.host}:{args.port}",
                'workers': args.workers,
                'threads': args.threads,
                'worker_class': 'gthread',
                'timeout': args.timeout,
                'loglevel': args.log_level,
            }
            
            logging.info(f"Running with Gunicorn server on {args.host}:{args.port} with {args.workers} workers")
            StandaloneApplication(app, options).run()
        except ImportError:
            logging.warning("Gunicorn not installed. Installing it is recommended for production use on Linux/macOS.")
            logging.warning("Run: pip install gunicorn")
            logging.info("Falling back to Flask's development server")
            app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()