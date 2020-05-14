import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os
import pandas as pd

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

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

# ### Generate the Dataset
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)

def chop_time_series(df, n_steps, batch_size = None):
    max_samples = n_steps * int(len(df)/n_steps)
    batch_size = int(len(df)/n_steps) if batch_size == None else batch_size
    truncated_df = df.iloc[:max_samples, :]
    series_ts = truncated_df.values.reshape(batch_size, n_steps, 1)

    return batch_size, series_ts.astype(np.float32)

#Generate Plots
def plot_multiple_forecasts(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :, 0])
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "ro-", label="Actual")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "bx-", label="Forecast")
    #plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)

def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):
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
    plt.hlines(0, 0, 100, linewidth=1)
    #plt.axis([0, n_steps + 1, -1, 1])

def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    #plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

def df_reindexer(df, timesteps='H'):
    new_idx = pd.date_range(start=df.index.min().tz_localize(None), end=df.index.max().tz_localize(None), freq=timesteps, tz='UTC')
    df_reindexed = pd.DataFrame(index=new_idx)
    df_reindexed = df.join(df_reindexed, how='outer')

    return df_reindexed

# %%
np.random.seed(42)

lookforward = 50
n_steps = 50


df = pd.read_csv('data_sources/yr_sensor_data_test.csv', index_col='time')
df.index = pd.to_datetime(df.index.values)

#Reindex to identify missing values
df = df_reindexer(df, timesteps='H')

#handle missing values
df = df.ffill()

#series = generate_time_series(10000, n_steps + lookforward)
batch_size, series= chop_time_series(df,n_steps+lookforward)

train_split = int(0.90*batch_size)
valid_split = int(0.05*batch_size)
test_split= int(0.05*batch_size)

# X_train = series[:48, :n_steps]
# y_train = series[:48, -lookforward]
# X_valid = series[49:50, :n_steps]
# y_valid = series[49:50, -lookforward]
# X_test = series[50:, :n_steps]
# y_test = series[50:, -lookforward]

X_train = series[:100, :n_steps]
y_train = series[:100, -lookforward]
X_valid = series[101:103, :n_steps]
y_valid = series[101:103, -lookforward]
X_test = series[103:, :n_steps]
y_test = series[103:, -lookforward]

# %%
print ('Training Data X Shape : ' + str(X_train.shape))
print ('Training Data y Shape : ' + str(y_train.shape))

#pure linear prediction
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[n_steps, 1]),
    keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))

plot_learning_curves(history.history["loss"], history.history["val_loss"])
save_fig("simple_linear_rnn_loss_plot")
plt.show()

y_pred = model.predict(X_valid)
plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
plt.show()

#Simple RNN
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1])
])

optimizer = keras.optimizers.Adam(lr=0.005)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))

model.evaluate(X_valid, y_valid)
plot_learning_curves(history.history["loss"], history.history["val_loss"])
save_fig("simple_rnn_loss_plot")
plt.show()

y_pred = model.predict(X_valid)
plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
save_fig("simple_rnn_1_step_pred_plot")
plt.show()

#Deep RNN
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))

model.evaluate(X_valid, y_valid)
plot_learning_curves(history.history["loss"], history.history["val_loss"])
save_fig("deep_rnn_loss_plot")
plt.show()

y_pred = model.predict(X_valid)
plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
save_fig("deep_rnn_1_step_pred_plot")
plt.show()


'''Multi step predictions using RNNs'''

np.random.seed(43) # not 42, as it would give the first series in the train set

batch_size, series = chop_time_series(df, n_steps + lookforward)

X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
X = X_new
for step_ahead in range(lookforward):
    y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
    X = np.concatenate([X, y_pred_one], axis=1)

Y_pred = X[:, n_steps:]

plot_multiple_forecasts(X_new, Y_new, Y_pred)
save_fig("rnn_multi_step_forecast_ahead_plot")
plt.show()

mm

#Simple LSTM
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(lookforward))
])

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=20,
                    validation_data=(X_valid, Y_valid))


model.evaluate(X_valid, Y_valid)
plot_learning_curves(history.history["loss"], history.history["val_loss"])
save_fig("lstm_learning_curve_plot")
plt.show()


np.random.seed(43)

series = generate_time_series(1, n_steps + lookforward)
X_new, Y_new = series[:, :n_steps, :], series[:, n_steps:, :]
Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

plot_multiple_forecasts(X_new, Y_new, Y_pred)
save_fig("lstm_pred_plot")
plt.show()

#Conv1D RNN
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",
                        input_shape=[None, 1]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(lookforward))
])

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train[:, 3::2], epochs=20,
                    validation_data=(X_valid, Y_valid[:, 3::2]))

# model.evaluate(X_valid, Y_valid)
# plot_learning_curves(history.history["loss"], history.history["val_loss"])
# save_fig("conv_forecast_ahead_plot")
# plt.show()

np.random.seed(43)

series = generate_time_series(1, n_steps + lookforward)
X_new, Y_new = series[:, :n_steps, :], series[:, n_steps:, :]
Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

plot_multiple_forecasts(X_new, Y_new, Y_pred)
save_fig("conv_oos_forecast_ahead_plot")
plt.show()

# %%
