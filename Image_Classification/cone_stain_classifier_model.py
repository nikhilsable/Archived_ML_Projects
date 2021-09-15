import data_maker
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from keras.applications.vgg16 import VGG16
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# from sklearn.ensemble import AdaBoostClassifier


def model_configs():

    config_dict = {
        "feature_extractor_model_name": "\\\\TBD.com\\us\\shared\\TBD\\CS_DAV\\NSAB_Project_Sources\\cone_stain_classifier\\models\\feature_extractor_model",
        "classifier_model_name": "\\\\TBD.com\\us\\shared\\TBD\\CS_DAV\\NSAB_Project_Sources\\cone_stain_classifier\\models\\classifier_model",
        "label_encoder_filename": "\\\\TBD.com\\us\\shared\\TBD\\CS_DAV\\NSAB_Project_Sources\\cone_stain_classifier\\models\\label_encoder_filename.pkl",
        "IMG_SIZE": 256,
        "training": 1,
    }

    return config_dict


def save_label_encoder(le, config_dict):
    # Save label encoder
    from pickle import dump

    dump(le, open(config_dict["label_encoder_filename"], "wb"))
    print("********** Label Encoder Saved *************")


def load_label_encoder_and_transform(labels, config_dict):
    from pickle import load

    le = load(open(config_dict["label_encoder_filename"], "rb"))
    print("********** label encoder Retrieved ***********")

    labels_encoded = le.transform(labels)

    print("********** labels encoded ***********")

    return labels_encoded


def load_label_encoder_and_inverse_transform(encoded_labels, config_dict):
    from pickle import load

    le = load(open(config_dict["label_encoder_filename"], "rb"))
    print("********** label encoder Retrieved ***********")

    decoded_labels = le.inverse_transform(encoded_labels)

    print("********** labels decoded ***********")

    return decoded_labels


def scale_pixel_values(x):
    # Normalize/scale pixel values
    scaled_pixel_values = x / 255.0

    return scaled_pixel_values


def build_classifier_model(config_dict, train_images, train_labels, test_images, test_labels):

    if config_dict["training"] == 1:
        # Encode labels/pre-proccessing

        # Encode labels from text to integers.
        le = preprocessing.LabelEncoder()
        le.fit(train_labels)

        # save label encode serialized for future
        save_label_encoder(le, config_dict)

        train_labels_encoded = load_label_encoder_and_transform(train_labels, config_dict)
        test_labels_encoded = load_label_encoder_and_transform(test_labels, config_dict)

        # Split data into test and train datasets (already split but assigning to meaningful convention)
        x_train, y_train, x_test, _ = (
            train_images,
            train_labels_encoded,
            test_images,
            test_labels_encoded,
        )  # (_ = y_test)

        # Normalize/scale pixel values
        x_train, x_test = scale_pixel_values(x_train), scale_pixel_values(x_test)

        # Load model wothout classifier/fully connected layers
        VGG_model = VGG16(
            weights="imagenet", include_top=False, input_shape=(config_dict["IMG_SIZE"], config_dict["IMG_SIZE"], 3)
        )

        # Make loaded layers as non-trainable (because we want pre-trained weights)
        for layer in VGG_model.layers:
            layer.trainable = False

        print(VGG_model.summary())  # Trainable parameters should be 0

        # Use features from convolutional network for RF classifier
        feature_extractor = VGG_model.predict(x_train)

        features = feature_extractor.reshape(feature_extractor.shape[0], -1)

        X_for_RF = features  # This is our X input to RF

        # RANDOM FOREST
        RF_model = RandomForestClassifier(n_estimators=50, random_state=42)

        # ADABoost Classifier
        # RF_model = AdaBoostClassifier(n_estimators = 50, random_state = 42)

        # Train the model on training data
        RF_model.fit(X_for_RF, y_train)  # For sklearn no one hot encoding

        # Send test data through same feature extractor process
        X_test_feature = VGG_model.predict(x_test)
        X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

        # Now predict using the trained RF model.
        prediction_RF = RF_model.predict(X_test_features)
        # Inverse le transform to get original label back.
        prediction_RF = le.inverse_transform(prediction_RF)

        # Print overall accuracy and class precision
        print("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))
        print("Precision = ", metrics.precision_score(test_labels, prediction_RF, average=None))

        # Confusion Matrix - verify accuracy of each class
        cm = confusion_matrix(test_labels, prediction_RF)
        # print(cm)
        sns.heatmap(cm, annot=True)
        plt.show()

        # save VGG model
        VGG_model.save(config_dict["feature_extractor_model_name"] + ".h5")
        print("VGG model saved...")
        # save classifier model
        joblib.dump(RF_model, config_dict["classifier_model_name"] + ".joblib")
        print("RF classifier model saved...")

        # To check specific examples
        # img = x_test[0]
        # plt.imshow(img)
        # plt.show()

        # input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
        # input_img_feature=VGG_model.predict(input_img)
        # input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
        # prediction_RF = RF_model.predict(input_img_features)[0]
        # prediction_RF = le.inverse_transform([prediction_RF])  #Reverse the label encoder to original name
        # print("The prediction for this image is: ", prediction_RF)
        # print("The actual label for this image is: ", test_labels[0])

    else:
        pass


def main():

    config_dict = model_configs()
    train_images, train_labels, test_images, test_labels = data_maker.main()
    build_classifier_model(config_dict, train_images, train_labels, test_images, test_labels)


if __name__ == "__main__":
    main()
