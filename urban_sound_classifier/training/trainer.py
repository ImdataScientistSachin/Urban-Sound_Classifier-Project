import os
import time
import logging
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

from ..config.config_manager import ConfigManager
from ..models.loaders.model_loader import ModelLoader
from ..utils.file import FileUtils

class Trainer:
    """
    Class for training and validating audio classification models.
    
    This class handles the training pipeline, including data loading,
    model training, validation, and saving trained models.
    
    Attributes:
        config (ConfigManager): Configuration manager instance
        model_loader (ModelLoader): Model loader instance
        model (tf.keras.Model): The model being trained
        callbacks (List[tf.keras.callbacks.Callback]): List of training callbacks
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the Trainer.
        
        Args:
            config (ConfigManager): Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize model loader
        self.model_loader = ModelLoader(config)
        
        # Initialize model and callbacks as None
        self.model = None
        self.callbacks = None
    
    def setup_model(self, model: tf.keras.Model = None) -> tf.keras.Model:
        """
        Set up the model for training.
        
        Args:
            model (tf.keras.Model, optional): Pre-initialized model.
                If None, a model will be loaded or created based on the configuration.
            
        Returns:
            tf.keras.Model: The model ready for training
        """
        if model is not None:
            self.model = model
        else:
            # Load model from path if specified in config
            model_path = self.config.get('TRAINING.model_path', None)
            if model_path and os.path.exists(model_path):
                self.model = self.model_loader.load_model(model_path)
            else:
                # Create a new model based on architecture specified in config
                architecture = self.config.get('MODEL.architecture', 'double_unet')
                
                if architecture.lower() == 'double_unet':
                    from ..models.architectures.double_unet import DoubleUNet
                    self.model = DoubleUNet(self.config).build()
                else:
                    raise ValueError(f"Unsupported model architecture: {architecture}")
        
        # Compile the model
        optimizer_name = self.config.get('TRAINING.optimizer', 'adam').lower()
        learning_rate = self.config.get('TRAINING.learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            momentum = self.config.get('TRAINING.momentum', 0.9)
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Get loss function
        loss_name = self.config.get('TRAINING.loss', 'categorical_crossentropy').lower()
        if loss_name == 'categorical_crossentropy':
            loss = tf.keras.losses.CategoricalCrossentropy()
        elif loss_name == 'sparse_categorical_crossentropy':
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
        elif loss_name == 'binary_crossentropy':
            loss = tf.keras.losses.BinaryCrossentropy()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
        
        # Get metrics
        metrics = ['accuracy']
        if self.config.get('TRAINING.use_f1_score', False):
            # Add F1 score metric if requested
            from tensorflow.keras.metrics import Precision, Recall
            
            class F1Score(tf.keras.metrics.Metric):
                def __init__(self, name='f1_score', **kwargs):
                    super().__init__(name=name, **kwargs)
                    self.precision = Precision()
                    self.recall = Recall()
                
                def update_state(self, y_true, y_pred, sample_weight=None):
                    self.precision.update_state(y_true, y_pred, sample_weight)
                    self.recall.update_state(y_true, y_pred, sample_weight)
                
                def result(self):
                    p = self.precision.result()
                    r = self.recall.result()
                    return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
                
                def reset_state(self):
                    self.precision.reset_state()
                    self.recall.reset_state()
            
            metrics.append(F1Score())
        
        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        self.logger.info(f"Model compiled with optimizer={optimizer_name}, loss={loss_name}")
        return self.model
    
    def setup_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Set up callbacks for training.
        
        Returns:
            List[tf.keras.callbacks.Callback]: List of callbacks
        """
        callbacks = []
        
        # Model checkpoint callback
        checkpoint_dir = self.config.get('PATHS.checkpoints_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"model_{time.strftime('%Y%m%d_%H%M%S')}.h5"
        )
        
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping callback
        if self.config.get('TRAINING.use_early_stopping', True):
            patience = self.config.get('TRAINING.early_stopping_patience', 10)
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Learning rate scheduler
        if self.config.get('TRAINING.use_lr_scheduler', False):
            scheduler_type = self.config.get('TRAINING.lr_scheduler_type', 'reduce_on_plateau')
            
            if scheduler_type == 'reduce_on_plateau':
                factor = self.config.get('TRAINING.lr_reduction_factor', 0.5)
                patience = self.config.get('TRAINING.lr_patience', 5)
                min_lr = self.config.get('TRAINING.min_lr', 1e-6)
                
                lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=factor,
                    patience=patience,
                    min_lr=min_lr,
                    verbose=1
                )
                callbacks.append(lr_scheduler)
            
            elif scheduler_type == 'step_decay':
                initial_lr = self.config.get('TRAINING.learning_rate', 0.001)
                drop_rate = self.config.get('TRAINING.lr_drop_rate', 0.5)
                epochs_drop = self.config.get('TRAINING.lr_epochs_drop', 10)
                
                def step_decay(epoch):
                    return initial_lr * (drop_rate ** (epoch // epochs_drop))
                
                lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay, verbose=1)
                callbacks.append(lr_scheduler)
        
        # TensorBoard callback
        if self.config.get('TRAINING.use_tensorboard', False):
            log_dir = self.config.get('PATHS.tensorboard_dir', 'logs/tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
            callbacks.append(tensorboard)
        
        # CSV Logger
        if self.config.get('TRAINING.use_csv_logger', True):
            log_dir = self.config.get('PATHS.logs_dir', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            csv_path = os.path.join(
                log_dir,
                f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            csv_logger = tf.keras.callbacks.CSVLogger(
                csv_path,
                separator=',',
                append=False
            )
            callbacks.append(csv_logger)
        
        self.callbacks = callbacks
        return callbacks
    
    def train(self, 
              x_train: np.ndarray, 
              y_train: np.ndarray, 
              x_val: np.ndarray = None, 
              y_val: np.ndarray = None,
              batch_size: int = None,
              epochs: int = None,
              callbacks: List[tf.keras.callbacks.Callback] = None,
              class_weights: Dict[int, float] = None) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            x_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            x_val (np.ndarray, optional): Validation features
            y_val (np.ndarray, optional): Validation labels
            batch_size (int, optional): Batch size. If None, use value from config.
            epochs (int, optional): Number of epochs. If None, use value from config.
            callbacks (List[tf.keras.callbacks.Callback], optional): List of callbacks.
                If None, use callbacks from setup_callbacks().
            class_weights (Dict[int, float], optional): Class weights for imbalanced datasets.
            
        Returns:
            tf.keras.callbacks.History: Training history
            
        Raises:
            ValueError: If model is not set up
        """
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model() first.")
        
        # Get batch size and epochs from config if not provided
        if batch_size is None:
            batch_size = self.config.get('TRAINING.batch_size', 32)
        
        if epochs is None:
            epochs = self.config.get('TRAINING.epochs', 100)
        
        # Set up callbacks if not provided
        if callbacks is None:
            if self.callbacks is None:
                callbacks = self.setup_callbacks()
            else:
                callbacks = self.callbacks
        
        # Calculate class weights if not provided and enabled in config
        if class_weights is None and self.config.get('TRAINING.use_class_weights', False):
            from sklearn.utils.class_weight import compute_class_weight
            
            # Convert one-hot encoded labels to class indices if needed
            if len(y_train.shape) > 1 and y_train.shape[1] > 1:
                y_indices = np.argmax(y_train, axis=1)
            else:
                y_indices = y_train
            
            # Compute class weights
            classes = np.unique(y_indices)
            weights = compute_class_weight('balanced', classes=classes, y=y_indices)
            class_weights = {i: w for i, w in zip(classes, weights)}
            
            self.logger.info(f"Computed class weights: {class_weights}")
        
        # Train the model
        self.logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        
        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val) if x_val is not None and y_val is not None else None,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        self.logger.info("Training completed")
        return history
    
    def train_with_generator(self,
                            train_generator: tf.keras.utils.Sequence,
                            validation_generator: tf.keras.utils.Sequence = None,
                            epochs: int = None,
                            callbacks: List[tf.keras.callbacks.Callback] = None,
                            class_weights: Dict[int, float] = None) -> tf.keras.callbacks.History:
        """
        Train the model using data generators.
        
        Args:
            train_generator (tf.keras.utils.Sequence): Training data generator
            validation_generator (tf.keras.utils.Sequence, optional): Validation data generator
            epochs (int, optional): Number of epochs. If None, use value from config.
            callbacks (List[tf.keras.callbacks.Callback], optional): List of callbacks.
                If None, use callbacks from setup_callbacks().
            class_weights (Dict[int, float], optional): Class weights for imbalanced datasets.
            
        Returns:
            tf.keras.callbacks.History: Training history
            
        Raises:
            ValueError: If model is not set up
        """
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model() first.")
        
        # Get epochs from config if not provided
        if epochs is None:
            epochs = self.config.get('TRAINING.epochs', 100)
        
        # Set up callbacks if not provided
        if callbacks is None:
            if self.callbacks is None:
                callbacks = self.setup_callbacks()
            else:
                callbacks = self.callbacks
        
        # Train the model
        self.logger.info(f"Starting training with generator for {epochs} epochs")
        
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        self.logger.info("Training completed")
        return history
    
    def evaluate(self, 
                x_test: np.ndarray, 
                y_test: np.ndarray,
                batch_size: int = None) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            x_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            batch_size (int, optional): Batch size. If None, use value from config.
            
        Returns:
            Dict[str, float]: Evaluation metrics
            
        Raises:
            ValueError: If model is not set up
        """
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model() first.")
        
        # Get batch size from config if not provided
        if batch_size is None:
            batch_size = self.config.get('TRAINING.batch_size', 32)
        
        # Evaluate the model
        self.logger.info("Evaluating model on test data")
        
        results = self.model.evaluate(
            x_test, y_test,
            batch_size=batch_size,
            verbose=1
        )
        
        # Create metrics dictionary
        metrics_dict = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics_dict[metric_name] = results[i]
        
        self.logger.info(f"Evaluation results: {metrics_dict}")
        return metrics_dict
    
    def evaluate_with_generator(self, 
                              test_generator: tf.keras.utils.Sequence) -> Dict[str, float]:
        """
        Evaluate the model using a data generator.
        
        Args:
            test_generator (tf.keras.utils.Sequence): Test data generator
            
        Returns:
            Dict[str, float]: Evaluation metrics
            
        Raises:
            ValueError: If model is not set up
        """
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model() first.")
        
        # Evaluate the model
        self.logger.info("Evaluating model with generator")
        
        results = self.model.evaluate(
            test_generator,
            verbose=1
        )
        
        # Create metrics dictionary
        metrics_dict = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics_dict[metric_name] = results[i]
        
        self.logger.info(f"Evaluation results: {metrics_dict}")
        return metrics_dict
    
    def save_model(self, output_path: str = None, format: str = 'h5') -> str:
        """
        Save the trained model.
        
        Args:
            output_path (str, optional): Path to save the model.
                If None, use default path from config.
            format (str): Format to save the model ('h5', 'savedmodel', or 'tflite')
            
        Returns:
            str: Path to the saved model
            
        Raises:
            ValueError: If model is not set up
        """
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model() first.")
        
        # Get output path from config if not provided
        if output_path is None:
            models_dir = self.config.get('PATHS.models_dir', 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            output_path = os.path.join(
                models_dir,
                f"model_{time.strftime('%Y%m%d_%H%M%S')}.{format}"
            )
        
        # Save the model
        return self.model_loader.save_model(self.model, output_path, format=format)
    
    def save_training_config(self, output_path: str = None) -> str:
        """
        Save the training configuration.
        
        Args:
            output_path (str, optional): Path to save the configuration.
                If None, use default path from config.
            
        Returns:
            str: Path to the saved configuration
        """
        # Get output path from config if not provided
        if output_path is None:
            config_dir = self.config.get('PATHS.config_dir', 'config')
            os.makedirs(config_dir, exist_ok=True)
            
            output_path = os.path.join(
                config_dir,
                f"training_config_{time.strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        # Save the configuration
        self.config.save(output_path)
        self.logger.info(f"Saved training configuration to {output_path}")
        
        return output_path