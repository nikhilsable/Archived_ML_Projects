import sys

import sklearn
import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
import pandas as pd


import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import seaborn as sns

# to make this notebook's output stable across runs
np.random.seed(42)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification_neural_net"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def plot_confusion_matrix(matrix):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)


def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(17, 6)):
    y_score = clf.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic per class')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.4f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()


# Get Dataset
mnist = keras.datasets.mnist

# Splitting into train test sets
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# test_digit
test_digit = X_train_full[-1]
test_digit_target = y_train_full[-1]

# scaling features
scaler = StandardScaler()

X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# X_train_scaled = scaler.fit_transform(X_train.flatten().astype(np.float64).reshape(-1, 1))
# X_valid_scaled = scaler.fit_transform(X_valid.flatten().astype(np.float64).reshape(-1, 1))
# X_test_scaled = scaler.transform(X_test.flatten().astype(np.float64).reshape(-1, 1))

# Build Neural Net
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[X_train_full.shape[1], X_train_full.shape[2]]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
#model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(len(np.unique(y_train_full.flatten())), activation='softmax')) #unique classes in target

# Compile Neural Net
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# fit/train the model
history = model.fit(X_train, y_train, epochs = 30, validation_data =(X_valid, y_valid))

# Did the model learn / how well did it learn
history_df = pd.DataFrame(history.history)
history_df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


# Accuracy score on training data

print("Training Accuracy Score with One v/s Rest Classifier.....")
print(cross_val_score(ovr_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))

# Accuracy score on testing data
print("Testing Accuracy Score with One v/s Rest Classifier.....")
print(cross_val_score(ovr_clf, X_test_scaled, y_test, cv=3, scoring="accuracy"))

# Make one prediction
print("Making one prediction to test....")
print("Actual Target = " + str(y[-1]) + " and Predicted Value = " + str(ovr_clf.predict([scaler.transform([test_digit]).flatten()]).ravel()))

# total estimators
len(ovr_clf.estimators_)

# Check which class it was most confident of
ovr_clf.decision_function([scaler.transform([test_digit]).flatten()])

# Create and Plot confusion Matrix
y_train_pred = cross_val_predict(ovr_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

plt.matshow(conf_mx, cmap=plt.cm.gray)
save_fig("confusion_matrix_plot_with_ovr_classifier", tight_layout=False)
plt.show()

# Plot Multiclass ROC (a ROC plot for each estimator/class)
plot_multiclass_roc(ovr_clf, X_train_scaled, y_train, n_classes=len(ovr_clf.estimators_), figsize=(16, 10))
save_fig("ROC_plot_per_class", tight_layout=False)
plt.show()