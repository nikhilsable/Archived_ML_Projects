import numpy as np
import cv2
import matplotlib.pyplot as plt

from cone_stain_classifier_model import scale_pixel_values
from cone_stain_classifier_model import load_label_encoder_and_inverse_transform


def initialize_configs():
    config_dict = {
        "IMG_SIZE": 256,
        'feature_extractor_model_name': f"\\\\TBD\\us\\shared\\us018601\\CS_DAV\\NSAB_Project_Sources\\cone_stain_classifier\\models\\feature_extractor_model",
        'classifier_model_name': f"\\\\TBD.com\\us\\shared\\us018601\\CS_DAV\\NSAB_Project_Sources\\cone_stain_classifier\\models\\classifier_model",
        'label_encoder_filename': f"\\\\TBD.com\\us\\shared\\us018601\\CS_DAV\\NSAB_Project_Sources\\cone_stain_classifier\\models\\label_encoder_filename.pkl",

    }

    return config_dict

def pre_process_image(img_path, config_dict):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
    img = cv2.resize(img, (config_dict['IMG_SIZE'] , config_dict['IMG_SIZE']))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

def convert_images_to_array(images):
    array_images = np.array(images)

    return array_images

def load_feature_extractor_model(config_dict):
    from tensorflow.keras.models import load_model

    try:
        model = load_model(config_dict['feature_extractor_model_name'] + '.h5', compile=False)
    except:
        print("Couldn't retrieve model...")

    return model

def load_classifier_model(config_dict):
    import joblib

    try:
        model = joblib.load(config_dict['classifier_model_name'] + '.joblib')
    
    except:
        print("Couldn't retrieve model...")

    return model

def get_prediction(x_test, VGG_model, RF_model, config_dict):
        #make a predition using the test image
        img = x_test
        # print("Input Image is : ")
        #plt.imshow(img)
        # plt.show()

        input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
        input_img_feature=VGG_model.predict(input_img)
        input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
        prediction_RF = RF_model.predict(input_img_features)[0] 
        prediction_RF = load_label_encoder_and_inverse_transform([prediction_RF], config_dict)  #Reverse the label encoder to original name
        print("Is there central cone stain in the image?: ", prediction_RF[0])
        # print("The actual label for this image is: ", test_labels[n])
        #For prediction probability -> RF_model.predict_proba()
        # For le classes --> le.classes_

        return prediction_RF

def main(img_path):

    #Create configuration dictionary
    config_dict = initialize_configs()

    #process image
    img = pre_process_image(img_path, config_dict)

    #convert to array
    test_images = convert_images_to_array(img)

    #normalize/scale pixels in image
    x_test = scale_pixel_values(test_images)

    # Retrieve feature extractor model and RF model
    VGG_model = load_feature_extractor_model(config_dict)
    RF_model = load_classifier_model(config_dict)

    #get prediction
    prediction = get_prediction(x_test, VGG_model, RF_model, config_dict)

    return prediction


if __name__ == "__main__":
    main(img_path)