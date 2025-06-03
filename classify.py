import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model once
vgg_model = load_model("models/vgg16_bathroom_classifier.h5")
class_names = ["Modern", "Western"]

def classify_image(image):
    """Returns class name and confidence for given image."""
    image_resized = image.resize((224, 224))
    image_array = img_to_array(image_resized)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = vgg_model.predict(image_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return class_names[predicted_index], float(confidence)
