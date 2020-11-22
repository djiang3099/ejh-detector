# AMME4710 - EJH Detector 2020
# Circuit Digitaliser 

# 470205127
# 470355499
# 470425954

# Main file for the object classifier

# Import necessary packages
from trained_model import TrainedModel
from direction_identification import DirectionIdentification
import cv2

# This function gets a trained model, predicts a given image and returns 
# the prediction with a rotation index
def component_classifier(img):
    # Create model object
    my_model = TrainedModel()

    # Get model architecture
    # my_model.get_model_architecture()

    # Predict the circuit element
    results = my_model.predict_image(img)

    # Get the most confidence results label
    first_result = results[0][1]
    # print(first_result)

    # Create direction finder object
    my_direction_identifier = DirectionIdentification()

    # Preprocess the image
    my_direction_identifier.preprocess_image(img)

    # Go through each component and return the rotation orientation
    if(first_result=='c'):
        print('Capacitor detected')
        rot_idx = my_direction_identifier.find_capacitor_direction()
    elif(first_result=='i'):
        print('Inductor detected')
        rot_idx = my_direction_identifier.find_inductor_direction()
    elif(first_result=='r'):
        print('Resistor detected')
        rot_idx = my_direction_identifier.find_resistor_direction()
    elif(first_result=='d'):
        print('Diode detected')
        rot_idx = my_direction_identifier.find_diode_direction()
    elif(first_result=='v'):
        print('Power detected')
        rot_idx = my_direction_identifier.find_power_supply_direction()
    elif(first_result == 'g'):
        print("Ground detected")
        rot_idx = my_direction_identifier.find_ground_direction()

    return [first_result, rot_idx]


if __name__ == '__main__':
    
    # Test file path
    filepath = 'test2/094.png' # diode
    # filepath = 'test2/3.jpeg' # diode

    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    [element,rot_idx] = component_classifier(gray)
    print(element)
    print(rot_idx)

    

