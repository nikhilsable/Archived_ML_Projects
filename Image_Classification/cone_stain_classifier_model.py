import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import data_maker


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


def visualize_cnn_filters(cnn_model):
    # Iterate thru all the layers of the model
    for layer in cnn_model.layers:
        if "conv" in layer.name:
            weights, bias = layer.get_weights()

            # normalize filter values between  0 and 1 for visualization
            f_min, f_max = weights.min(), weights.max()
            filters = (weights - f_min) / (f_max - f_min)
            print(filters.shape[3])
            filter_cnt = 1

            # plotting all the filters
            for i in range(filters.shape[3]):
                # fig = plt.figure(figsize=(20, 20))
                # get the filters
                filt = filters[:, :, :, i]
                # plotting each of the channel, color image RGB channels
                for j in range(filters.shape[0]):
                    ax = plt.subplot(filters.shape[3], filters.shape[0], filter_cnt)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.imshow(filt[:, :, j])
                    filter_cnt += 1
            plt.show()


def visualize_cnn_activation_maps(input_img, cnn_model):
    # Define a new Model, Input= image
    # Output= intermediate representations for all layers in the
    # previous model after the first.
    successive_outputs = [layer.output for layer in cnn_model.layers[1:]]
    # visualization_model = Model(img_input, successive_outputs)
    visualization_model = Model(inputs=cnn_model.input, outputs=successive_outputs)
    # Get intermediate representations.
    successive_feature_maps = visualization_model.predict(input_img)
    # Retrieve are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in cnn_model.layers]
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        print(feature_map.shape)
        if len(feature_map.shape) == 4:

            # Plot Feature maps for the conv / maxpool layers, not the fully-connected layers
            n_features = feature_map.shape[-1]  # number of features in the feature map
            size = feature_map.shape[1]  # feature map shape (1, size, size, n_features)

            # Image Matrix
            display_grid = np.zeros((size, size * n_features))

            # Postprocess the feature to be visually palatable
            for i in range(n_features):
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype("uint8")
                # Tile each filter into a horizontal grid
                display_grid[:, i * size : (i + 1) * size] = x
            # Display the grid
            scale = 20.0 / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect="auto", cmap="viridis")
            # plt.savefig(f"{layer_name}_activation_map.png",dpi=600)


def plot_saliency_map(input_img, input_image_label, model):

    raw_img = input_img
    input_img = np.expand_dims(input_img, 0)

    images = tf.Variable(input_img, dtype=float)

    with tf.GradientTape() as tape:
        pred = model(images, training=False)
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]

    grads = tape.gradient(loss, images)
    # grads.shape
    dgrad_abs = tf.math.abs(grads)
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
    ## normalize to range between 0 and 1
    arr_min, arr_max = np.min(dgrad_max_), np.max(dgrad_max_)
    grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.title(input_image_label)
    axes[0].imshow(raw_img)
    i = axes[1].imshow(grad_eval, cmap="jet", alpha=0.8)
    fig.colorbar(i)


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

    else:
        pass


def main():
    config_dict = model_configs()
    train_images, train_labels, test_images, test_labels = data_maker.main()
    build_classifier_model(config_dict, train_images, train_labels, test_images, test_labels)


if __name__ == "__main__":
    main()
