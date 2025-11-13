import tensorflow as tf
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.model import load_model, evaluate_model
from src.utils import extract_features
from src.config import MODEL_PATH, CLASS_LABELS

def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the figure
    os.makedirs('evaluation_results', exist_ok=True)
    plt.savefig('evaluation_results/confusion_matrix.png')
    plt.close()

def evaluate_on_test_data(model, test_data_path):
    """
    Evaluate the model on test data.
    
    Args:
        model: Loaded TensorFlow model
        test_data_path: Path to test data CSV file
    """
    # Load test data
    print(f"Loading test data from {test_data_path}...")
    test_data = pd.read_csv(test_data_path)
    
    # Process a subset of test data (for demonstration)
    num_samples = min(100, len(test_data))
    test_data = test_data.sample(num_samples, random_state=42)
    
    # Initialize arrays for features and labels
    all_features = []
    all_labels = []
    
    # Process each test file
    for idx, row in test_data.iterrows():
        try:
            # Get file path
            fold_num = row['fold']
            file_path = os.path.join('data', 'UrbanSound8K', f'fold{fold_num}', row['slice_file_name'])
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            
            # Extract features
            features = extract_features(file_path)
            all_features.append(features)
            
            # Get label
            label = row['class']
            all_labels.append(label)
            
            print(f"Processed {idx+1}/{num_samples}: {file_path}")
        except Exception as e:
            print(f"Error processing file {row['slice_file_name']}: {e}")
    
    # Convert lists to numpy arrays
    X_test = np.array(all_features)
    y_test = np.array(all_labels)
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_LABELS)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Convert to one-hot encoding
    y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=len(CLASS_LABELS))
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test_onehot)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Make predictions
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    plot_confusion_matrix(cm, CLASS_LABELS)
    
    # Generate classification report
    report = classification_report(y_test_encoded, y_pred, target_names=CLASS_LABELS)
    print("\nClassification Report:")
    print(report)
    
    # Save report to file
    with open('evaluation_results/classification_report.txt', 'w') as f:
        f.write(report)
    
    print("\nEvaluation complete. Results saved to 'evaluation_results' directory.")

def main():
    # Load the model
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    # Compile the model with metrics
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Path to test data
    test_data_path = os.path.join('data', 'UrbanSound8K', 'UrbanSound8K.csv')
    
    # Evaluate on test data
    evaluate_on_test_data(model, test_data_path)

if __name__ == "__main__":
    main()