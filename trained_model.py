
# Import necessary packages
import numpy as np
from keras.models import load_model
import cv2 

# This class retrieves a trained model and predicts a given component.
class TrainedModel(object):
    def __init__(self):

        # Initialise the image sizes 
        self._image_size = (28,28)
        self._tensor_size = (1,28,28,1)

        # Load model from file 
        self._loaded_model = load_model('90_accuracy.h5')

        # List of component names: diodes, resistor, inductor,capacitor,ground, voltage
        self._class_names = np.array( ['d','r','i','c','g','v'])

        # Initialise the image to predict
        self._img_to_predict = []

    # Get the model architecture
    def get_model_architecture(self):
        return self._loaded_model.summary()

    # Predict a single image 
    def predict_image(self,img):
       

        # Resize the given image to (28,28) for the model 
        _img_resize = cv2.resize(img, self._image_size )

        # Convert into a tensor for the model
        self._img_to_predict = np.reshape(_img_resize,self._tensor_size )

        # Predict the image
        _predictions = self._loaded_model.predict(self._img_to_predict)

        # Convert predictions to percentages
        _confidence_scores =np.array(_predictions) * (100)

        # Create a dictionary of the confidence scores and class names
        _combined_results = dict(zip(_confidence_scores[0,:],self._class_names ))
        
        # Sort this with highest confidence first
        _sorted_results = sorted(_combined_results.items(), key=lambda kv: kv[0],reverse=True)

        return _sorted_results


