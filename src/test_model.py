import os
import sys
import tensorflow as tf
import numpy as np

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_model, get_model_summary
from src.config import MODEL_PATH, CLASS_LABELS

def test_model_loading():
    """
    Test if the model can be loaded successfully and print its summary.
    """
    try:
        print(f"Attempting to load model from: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
        
        # Print model summary
        summary = get_model_summary(model)
        print("\nModel Summary:")
        print(summary)
        
        # Print model input and output shapes
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        print(f"\nInput shape: {input_shape}")
        print(f"Output shape: {output_shape}")
        print(f"Number of classes: {output_shape[-1]}")
        
        # Verify that the number of output classes matches our class labels
        assert output_shape[-1] == len(CLASS_LABELS), \
            f"Model has {output_shape[-1]} output classes, but we have {len(CLASS_LABELS)} class labels"
        
        print("\nClass labels:")
        for i, label in enumerate(CLASS_LABELS):
            print(f"{i}: {label}")
        
        print("\nAll tests passed!")
        return True
    except Exception as e:
        print(f"Error testing model: {e}")
        return False

def test_random_prediction():
    """
    Test the model with random input data to verify prediction functionality.
    """
    try:
        model = load_model(MODEL_PATH)
        
        # Create random input data based on the model's input shape
        # Note: This is just for testing the prediction pipeline, not for actual classification
        input_shape = model.input_shape
        
        # Remove batch dimension (None) if present
        if input_shape[0] is None:
            input_shape = input_shape[1:]
        
        # Create random input
        random_input = np.random.random((1,) + input_shape)
        
        # Make prediction
        prediction = model.predict(random_input)
        
        # Get the predicted class
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = CLASS_LABELS[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx]
        
        print(f"\nRandom input prediction test:")
        print(f"Predicted class: {predicted_class} (index {predicted_class_idx})")
        print(f"Confidence: {confidence:.4f}")
        
        return True
    except Exception as e:
        print(f"Error testing prediction: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Urban Sound Classifier Model ===")
    
    # Test model loading
    if test_model_loading():
        # If model loads successfully, test prediction
        test_random_prediction()
    
    print("\nTests completed.")