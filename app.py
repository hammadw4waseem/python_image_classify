from flask import Flask, jsonify, request

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

from PIL import Image
import io
import numpy as np

app = Flask(__name__)
model = VGG16(weights='imagenet')





@app.route('/classify_animal', methods=['POST'])
def classify_animal():
    # Get the image from the request
    image = request.files['image'].read()

    print(image);
    
    # Preprocess the image
    image = preprocess_image(image)
    
    # Use the VGG16 model to classify the image
    predictions = model.predict(image)
    
    # Get the top prediction
    top_prediction = decode_predictions(predictions, top=1)[0][0]
    
    # Get the label of the top prediction
    label = top_prediction[1]


    # Get the accuracy of the top prediction
    accuracy = float(top_prediction[2])
    
    prediction = np.argmax(predictions)

    if 'cat' in label:
        animal_category = 'cat'
        animal_breed = label
    elif prediction in [282, 283, 284]:
        animal_category = 'cat'
        animal_breed = label
    elif prediction in range(151, 269):
        animal_category = 'dog'
        animal_breed = label
    else:
        animal_category = 'unknown'
        animal_breed = 'unknown'
    
    # Create a dictionary with the response
    response_dict = {'category': animal_category, 'breed': animal_breed, 'accuracy':accuracy }
    
    # Return the response as JSON
    return jsonify(response_dict)

    
@app.route('/classify_image', methods=['POST'])
def classify_image():
    # Get the image from the request
    image = request.files['image'].read()
    
    # Preprocess the image
    image = preprocess_image(image)
    
    # Use the ResNet50 model to classify the image
    predictions = model.predict(image)
    
    # Get the top 5 predictions
    top_predictions = decode_predictions(predictions, top=5)[0]
    
    # Convert the numpy array to a list
    top_predictions_list = [(pred[1], float(pred[2]), pred[0]) for pred in top_predictions]
    
    # Return the top predictions as JSON
    return jsonify(top_predictions_list)


def preprocess_image(image_bytes):
    # Open the image using PIL
    image = Image.open(io.BytesIO(image_bytes))
    
    # Resize the image to (224, 224)
    image = image.resize((224, 224))
    
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Expand the dimensions of the array to create a batch of size 1
    image_array = np.expand_dims(image_array, axis=0)
    
    # Preprocess the image using the preprocess_input function from the ResNet50 library
    image_preprocessed = preprocess_input(image_array)
    
    return image_preprocessed

if __name__ == '__main__':
    app.run(debug=True)
