# -*- coding: utf-8 -*-
"""multi_step_lstm_v1.ipynb

"""
import os
import pandas as pd
# from tensorflow.keras.models import load_model
from numpy import hstack
from numpy import array
import numpy as np
import tensorflow as tf
from tensorflow import keras

#For plotting
# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def scale_and_save_scaler(df, model_name):
	from sklearn.preprocessing import MinMaxScaler
	min_max_scaler = MinMaxScaler()

	for col in df.columns:  # go through all of the columns
		df[col] = min_max_scaler.fit_transform(df[col].values.reshape(-1, 1))

	scaler_filename = "scaler_"+model_name+".gz" #scaler pickle file

	# Save scaler
	import joblib
	joblib.dump(min_max_scaler, scaler_filename)

	print("********** Scaler Saved *************")

	return df

def use_training_scaler(df, model_name):
	import joblib

	scaler_filename = "scaler_" + model_name + ".gz"  # scaler pickle file
	min_max_scaler = joblib.load(scaler_filename)

	for col in df.columns:  # go through all of the columns
		df[col] = min_max_scaler.inverse_transform(df[col].values.reshape(-1, 1))

	print("********** Scaler Retrieved ***********")

	return df

def split_sequences(df, n_steps, model_name, training = 0):
	from numpy import array

	#fit trans if training, else just transform
	if training==1:
		df = scale_and_save_scaler(df, model_name)
	else:
		df = use_training_scaler(df, model_name)

	df.dropna(inplace=True)
	df_extracted = np.array(df.values.flatten())

    # split a multivariate sequence into samples
	X, y = [], []
	for i in range(len(df_extracted)):# find the end of this pattern
		end_ix = i + n_steps
		out_end_ix = end_ix + n_steps + 1
	# check if we are beyond the dataset
		if out_end_ix > len(df_extracted):
			break
        # gather input and output parts of the pattern
		seq_x, seq_y = df_extracted[i:end_ix], df_extracted[end_ix:out_end_ix-1]
		X.append(seq_x)
		# y.append(seq_y)

	# X = np.array(X)
	# y = np.array(y)

	return np.array(X)[..., np.newaxis] #, np.array(y)[..., np.newaxis]

def plot_series(series, y=None, y_pred=None, x_label="Time", y_label="scaled(y)"):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    # plt.hlines(0, 0, 100, linewidth=1)
    # plt.axis([0, n_steps + 1, -1, 1])

def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

def plot_multiple_forecasts(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :, 0])
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "ro-", label="Actual")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "bx-", label="Forecast", markersize=10)
    # plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)

def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    # plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

def load_sensor_dataset():
    # load DataFrame
    df = pd.read_csv(r"test_data.csv")
    df = df.set_index('time')
    df.index = pd.DatetimeIndex(df.index)
    # df.columns = ['left_sensor', 'right_sensor', 'pressure']
    df = df.fillna(method='ffill')

    # For univariate
    df = df[['right_sensor']]
    df = df.tail(7000)
    dataset = df.copy()

    return dataset


def load_gold_prices_dataset():
    # load DataFrame
    df = pd.read_csv('gold_price_data.csv', index_col='Date')
    df.index = pd.DatetimeIndex(df.index)
    df = df.fillna(method='ffill')

    dataset = df.copy()

    return dataset

# dataset = load_sensor_dataset()
dataset = load_gold_prices_dataset()

# lookback = 120  # how long of a preceeding sequence to collect for RNN
lookahead = 60  # how far into the future are we trying to predict?
n_steps = 30
epochs = 5 # set the number of epochs you want the NN to run for

model_name = f"lstm_right_enc_{pd.Timestamp('now').strftime('%Y_%m_%d')}"  # save model and architecture to single file

## here, split away some slice of the future data from the main main_df.
train_size_ix_split = int(.80*len(dataset))
validation_size_ix_split = train_size_ix_split + int(.10*len(dataset))

np.random.seed(42)
series = split_sequences(dataset.copy(), n_steps+lookahead, model_name, training = 1)

X_train = series[:train_size_ix_split, :n_steps]
X_valid = series[train_size_ix_split:validation_size_ix_split, :n_steps]
X_test = series[validation_size_ix_split:, :n_steps]
Y = np.empty((series.shape[0], n_steps, lookahead))
for step_ahead in range(1, lookahead + 1):
    Y[..., step_ahead - 1] = series[..., step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:train_size_ix_split]
Y_valid = Y[train_size_ix_split:validation_size_ix_split]
Y_test = Y[validation_size_ix_split:]

#LSTM model
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(lookahead))
])

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=20,
                    validation_data=(X_valid, Y_valid))

#Plot loss
plot_learning_curves(history.history["loss"], history.history["val_loss"])
# plt.show()

#Predict based on Test data (X_test)
Y_pred = model.predict(X_test)

plot_multiple_forecasts(X_test, Y_test, Y_pred)
plt.show()

model.evaluate(X_valid, Y_valid)

# print(f"train data: {len(train_x)} validation: {len(validation_x)}")

#last sample , last value prediction
untransformed_df = pd.DataFrame(Y_pred[-1][-1])
test_df = use_training_scaler(untransformed_df, model_name)
display(test_df)

test_df.index = pd.date_range(start=(pd.Timestamp(dataset.index.max()) + pd.Timedelta(days=1)), periods = len(test_df), freq='D')
test_df.columns = ['Predicted']
combined_df = dataset.join(test_df, how='outer')
combined_df.plot()
 