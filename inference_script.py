import os
import argparse
import numpy as np
from PIL import Image
from tensorflow import keras
from azureml.core.model import Model
import pickle 

def main():
    print("Running inferece script...")
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Micro Organism Image Classification")
    parser.add_argument("--model_name", type=str, help="Name of the registered model")
    parser.add_argument("--label_encoder_name", type=str, help="Name of the registered label encoder")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    # Retrieve the path to the model file using the model name
    # model_path =  Model.get_model_path(model_name='trained_model.h5', version=1)
    model_path = Model.get_model_path(args.model_name)
    
    # Load the Keras model
    model = keras.models.load_model(model_path)
    
    # Retrieve the path to the label encoder file using the label encoder name
    label_encoder_path = args.label_encoder_name
    
    # Load the label encoder
    with open(label_encoder_path, 'rb') as label_encoder_file:
        label_encoder = pickle.load(label_encoder_file)
    
    try:
        # Convert the raw data (image bytes) to a NumPy array
        img = Image.open(args.image_path)
        img = img.resize((128, 128))  # Adjust the size as needed
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = img_array.reshape((1, 128, 128, 3))
        
        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)

        # Map the class index to class name using the loaded label encoder
        predicted_class_name = label_encoder.classes_[predicted_class_index]
        
        # Print prediction details
        print("Predicted class index:", predicted_class_index)
        print("Predicted class name:", predicted_class_name)
        print("Input image path:", args.image_path)

        # Return the predicted class and actual class as a dictionary
        result = {"predicted_class": predicted_class_name, "actual_class": "Amoeba"}
        print(result)

    except Exception as e:
        error = str(e)
        print("Error:", error)
    
if __name__ == "__main__":
    main()
    