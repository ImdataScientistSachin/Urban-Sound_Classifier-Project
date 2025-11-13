import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

from ..config.config_manager import ConfigManager
from ..models.prediction import Predictor
from ..training.data_generator import AudioDataGenerator
from ..utils.file import FileUtils

class Evaluator:
    """
    Class for evaluating audio classification models.
    
    This class handles model evaluation, including metrics calculation,
    confusion matrix generation, and visualization of results.
    
    Attributes:
        config (ConfigManager): Configuration manager instance
        predictor (Predictor): Predictor instance
        class_labels (List[str]): List of class labels
    """
    
    def __init__(self, config: ConfigManager, predictor: Predictor = None):
        """
        Initialize the Evaluator.
        
        Args:
            config (ConfigManager): Configuration manager instance
            predictor (Predictor, optional): Predictor instance.
                If None, a new Predictor will be created.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize predictor if not provided
        if predictor is None:
            self.predictor = Predictor(config)
        else:
            self.predictor = predictor
        
        # Get class labels
        self.class_labels = self.predictor._load_class_labels()
    
    def evaluate_model(self, 
                      test_generator: AudioDataGenerator,
                      model: tf.keras.Model = None) -> Dict[str, float]:
        """
        Evaluate a model using a test data generator.
        
        Args:
            test_generator (AudioDataGenerator): Test data generator
            model (tf.keras.Model, optional): Model to evaluate.
                If None, use the model from the predictor.
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        # Use model from predictor if not provided
        if model is None:
            if not self.predictor.models:
                raise ValueError("No models loaded in predictor. Load models first.")
            model = self.predictor.models[0]
        
        # Evaluate model
        self.logger.info("Evaluating model on test data")
        
        results = model.evaluate(
            test_generator,
            verbose=1
        )
        
        # Create metrics dictionary
        metrics_dict = {}
        for i, metric_name in enumerate(model.metrics_names):
            metrics_dict[metric_name] = results[i]
        
        self.logger.info(f"Evaluation results: {metrics_dict}")
        return metrics_dict
    
    def predict_and_evaluate(self, 
                           test_files: List[str], 
                           test_labels: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions on test files and evaluate performance.
        
        Args:
            test_files (List[str]): List of test file paths
            test_labels (np.ndarray): Array of test labels
            
        Returns:
            Dict[str, Any]: Dictionary of evaluation metrics and results
            
        Raises:
            ValueError: If no models are loaded in the predictor
        """
        if not self.predictor.models:
            raise ValueError("No models loaded in predictor. Call predictor.load_models() first.")
        
        # Convert one-hot encoded labels to indices if needed
        if len(test_labels.shape) > 1 and test_labels.shape[1] > 1:
            true_labels = np.argmax(test_labels, axis=1)
        else:
            true_labels = test_labels
        
        # Make predictions
        self.logger.info(f"Making predictions on {len(test_files)} test files")
        
        pred_results = self.predictor.predict_batch(test_files)
        pred_labels = np.array([result[0]['index'] for result in pred_results])
        
        # Calculate metrics
        accuracy = np.mean(pred_labels == true_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted'
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Generate classification report
        class_names = [self.class_labels[i] if i < len(self.class_labels) else f"Unknown {i}" 
                      for i in range(max(np.max(true_labels), np.max(pred_labels)) + 1)]
        
        report = classification_report(
            true_labels, pred_labels,
            target_names=class_names,
            output_dict=True
        )
        
        # Compile results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'true_labels': true_labels,
            'pred_labels': pred_labels
        }
        
        self.logger.info(f"Evaluation results: accuracy={accuracy:.4f}, precision={precision:.4f}, "
                       f"recall={recall:.4f}, f1_score={f1:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, 
                             confusion_matrix: np.ndarray, 
                             output_path: str = None,
                             normalize: bool = True,
                             figsize: Tuple[int, int] = (10, 8),
                             cmap: str = 'Blues') -> plt.Figure:
        """
        Plot a confusion matrix.
        
        Args:
            confusion_matrix (np.ndarray): Confusion matrix to plot
            output_path (str, optional): Path to save the plot.
                If None, the plot will not be saved.
            normalize (bool): Whether to normalize the confusion matrix
            figsize (Tuple[int, int]): Figure size
            cmap (str): Colormap for the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Normalize confusion matrix if requested
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            vmin, vmax = 0, 1
        else:
            cm = confusion_matrix
            title = 'Confusion Matrix'
            vmin, vmax = None, None
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap(cmap), vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Count' if not normalize else 'Percentage', rotation=-90, va='bottom')
        
        # Set labels
        num_classes = cm.shape[0]
        class_names = self.class_labels[:num_classes] if len(self.class_labels) >= num_classes else \
                     [str(i) for i in range(num_classes)]
        
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Set axis labels
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # Adjust layout
        fig.tight_layout()
        
        # Save plot if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved confusion matrix plot to {output_path}")
        
        return fig
    
    def plot_metrics(self, 
                    history: tf.keras.callbacks.History, 
                    output_path: str = None,
                    metrics: List[str] = None,
                    figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot training metrics from history.
        
        Args:
            history (tf.keras.callbacks.History): Training history
            output_path (str, optional): Path to save the plot.
                If None, the plot will not be saved.
            metrics (List[str], optional): List of metrics to plot.
                If None, all metrics in history will be plotted.
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Get metrics from history if not provided
        if metrics is None:
            metrics = [m for m in history.history.keys() if not m.startswith('val_')]
        
        # Create figure and axes
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
        if len(metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Plot training metric
            ax.plot(history.history[metric], label=f'Training {metric}')
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                ax.plot(history.history[val_metric], label=f'Validation {metric}')
            
            # Set labels and title
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} vs. Epochs')
            ax.legend()
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-axis label for the bottom plot
        axes[-1].set_xlabel('Epochs')
        
        # Adjust layout
        fig.tight_layout()
        
        # Save plot if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved metrics plot to {output_path}")
        
        return fig
    
    def plot_roc_curve(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray, 
                      output_path: str = None,
                      figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot ROC curve for multi-class classification.
        
        Args:
            y_true (np.ndarray): True labels (one-hot encoded)
            y_pred (np.ndarray): Predicted probabilities
            output_path (str, optional): Path to save the plot.
                If None, the plot will not be saved.
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        from sklearn.metrics import roc_curve, auc
        from itertools import cycle
        
        # Convert one-hot encoded labels to indices if needed
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            n_classes = y_true.shape[1]
            y_true_indices = np.argmax(y_true, axis=1)
            y_true_onehot = y_true
        else:
            n_classes = len(np.unique(y_true))
            y_true_indices = y_true
            # Convert to one-hot encoding
            y_true_onehot = np.zeros((len(y_true), n_classes))
            for i, val in enumerate(y_true):
                y_true_onehot[i, val] = 1
        
        # Ensure y_pred has the right shape
        if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
            # Binary classification
            y_pred_proba = np.column_stack((1 - y_pred, y_pred))
        else:
            # Multi-class classification
            y_pred_proba = y_pred
        
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Plot ROC curves
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot micro-average ROC curve
        ax.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})',
                color='deeppink', linestyle=':', linewidth=4)
        
        # Plot macro-average ROC curve
        ax.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.2f})',
                color='navy', linestyle=':', linewidth=4)
        
        # Plot ROC curves for each class
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
        
        for i, color in zip(range(n_classes), colors):
            if i < len(self.class_labels):
                class_name = self.class_labels[i]
            else:
                class_name = f"Class {i}"
                
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC for {class_name} (AUC = {roc_auc[i]:.2f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Set labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        
        # Add legend
        ax.legend(loc="lower right")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save plot if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved ROC curve plot to {output_path}")
        
        return fig
    
    def generate_evaluation_report(self, 
                                 results: Dict[str, Any], 
                                 output_dir: str,
                                 include_plots: bool = True) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results (Dict[str, Any]): Evaluation results from predict_and_evaluate()
            output_dir (str): Directory to save the report and plots
            include_plots (bool): Whether to include plots in the report
            
        Returns:
            str: Path to the generated report
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots if requested
        if include_plots:
            # Confusion matrix plot
            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            self.plot_confusion_matrix(results['confusion_matrix'], output_path=cm_path)
            
            # If ROC curve data is available
            if 'y_true_proba' in results and 'y_pred_proba' in results:
                roc_path = os.path.join(output_dir, 'roc_curve.png')
                self.plot_roc_curve(results['y_true_proba'], results['y_pred_proba'], output_path=roc_path)
        
        # Create report dataframe
        report_df = pd.DataFrame(results['classification_report']).T
        
        # Add overall metrics
        metrics_df = pd.DataFrame({
            'accuracy': [results['accuracy']],
            'precision': [results['precision']],
            'recall': [results['recall']],
            'f1_score': [results['f1_score']]
        })
        
        # Save report to CSV
        report_path = os.path.join(output_dir, 'classification_report.csv')
        report_df.to_csv(report_path)
        
        metrics_path = os.path.join(output_dir, 'overall_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        
        # Generate HTML report
        html_path = os.path.join(output_dir, 'evaluation_report.html')
        
        with open(html_path, 'w') as f:
            f.write('<html>\n')
            f.write('<head>\n')
            f.write('<title>Model Evaluation Report</title>\n')
            f.write('<style>\n')
            f.write('body { font-family: Arial, sans-serif; margin: 20px; }\n')
            f.write('h1, h2 { color: #333; }\n')
            f.write('table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }\n')
            f.write('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n')
            f.write('th { background-color: #f2f2f2; }\n')
            f.write('tr:nth-child(even) { background-color: #f9f9f9; }\n')
            f.write('.metric-value { font-weight: bold; }\n')
            f.write('</style>\n')
            f.write('</head>\n')
            f.write('<body>\n')
            
            # Title
            f.write('<h1>Model Evaluation Report</h1>\n')
            
            # Overall metrics
            f.write('<h2>Overall Metrics</h2>\n')
            f.write('<table>\n')
            f.write('<tr><th>Metric</th><th>Value</th></tr>\n')
            f.write(f'<tr><td>Accuracy</td><td class="metric-value">{results["accuracy"]:.4f}</td></tr>\n')
            f.write(f'<tr><td>Precision (weighted)</td><td class="metric-value">{results["precision"]:.4f}</td></tr>\n')
            f.write(f'<tr><td>Recall (weighted)</td><td class="metric-value">{results["recall"]:.4f}</td></tr>\n')
            f.write(f'<tr><td>F1 Score (weighted)</td><td class="metric-value">{results["f1_score"]:.4f}</td></tr>\n')
            f.write('</table>\n')
            
            # Per-class metrics
            f.write('<h2>Per-Class Metrics</h2>\n')
            f.write('<table>\n')
            f.write('<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Support</th></tr>\n')
            
            for class_name, metrics in results['classification_report'].items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    f.write(f'<tr>\n')
                    f.write(f'<td>{class_name}</td>\n')
                    f.write(f'<td>{metrics["precision"]:.4f}</td>\n')
                    f.write(f'<td>{metrics["recall"]:.4f}</td>\n')
                    f.write(f'<td>{metrics["f1-score"]:.4f}</td>\n')
                    f.write(f'<td>{metrics["support"]}</td>\n')
                    f.write(f'</tr>\n')
            
            f.write('</table>\n')
            
            # Confusion Matrix
            if include_plots:
                f.write('<h2>Confusion Matrix</h2>\n')
                f.write(f'<img src="confusion_matrix.png" alt="Confusion Matrix" style="max-width: 100%;">\n')
                
                # ROC Curve if available
                if 'y_true_proba' in results and 'y_pred_proba' in results:
                    f.write('<h2>ROC Curve</h2>\n')
                    f.write(f'<img src="roc_curve.png" alt="ROC Curve" style="max-width: 100%;">\n')
            
            f.write('</body>\n')
            f.write('</html>\n')
        
        self.logger.info(f"Generated evaluation report at {html_path}")
        return html_path