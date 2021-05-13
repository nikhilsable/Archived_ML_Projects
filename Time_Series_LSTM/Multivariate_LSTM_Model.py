# -*- coding: utf-8 -*-
"""multi_step_lstm_v1.ipynb

"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

#For plotting
# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt


def model_configs(dataset):
    config_dict = {
    'lookahead' : 180,  # how far (timesteps) into the future are we trying to predict?
    'n_steps' : 120,  # No of timesteps in each training/evaluation sample
    'epochs' : 30,  # set the number of epochs you want the NN to run for
    'model_name' : f"multivariate_lstm_model_{pd.Timestamp('now').strftime('%Y_%m_%d')}",  # save model and architecture to single file

    ## here, split away some slice of the future data from the main main_df.
    'train_size_ix_split' : int(.80 * len(dataset)),
    'validation_size_ix_split' : int(.80 * len(dataset)) + int(.10 * len(dataset)),
    'training' : 1,
    'last_train_index':dataset.index.max(),
    'target_col': 0, #Note : Arrange input dataset with target col at 0 or -1
    'scaler_filename' : f"scaler_multivariate_lstm_{pd.Timestamp('now').strftime('%Y_%m_%d')}.pkl"

    }

    return config_dict


def scale_and_save_scaler(df, model_name):
    from sklearn.preprocessing import StandardScaler
    standard_scaler = StandardScaler()

    df[list(df.columns)] = standard_scaler.fit_transform(df)

    # Save scaler
    from pickle import dump
    dump(standard_scaler, open(config_dict['scaler_filename'], 'wb'))
    print("********** Scaler Saved *************")

    return df

def do_inverse_transform(df, config_dict):
    from pickle import load

    standard_scaler = load(open(config_dict['scaler_filename'], 'rb'))
    print("********** Scaler Retrieved ***********")

    df[list(df.columns)] = standard_scaler.inverse_transform(df)

    return df

def load_scaler_and_transform(df, model_name):
    from pickle import load

    standard_scaler = load(open(config_dict['scaler_filename'], 'rb'))
    print("********** Scaler Retrieved ***********")

    df[list(df.columns)] = standard_scaler.transform(df)

    print("********** Data Transformed ***********")

    return df

def split_sequences(df, config_dict):
    from numpy import array

    trainX = []
    trainY = []

    n_future = config_dict['lookahead']  # Number of days we want to predict into the future
    n_past = config_dict['n_steps']  # Number of past days we want to use to predict the future

    for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
        trainX.append(df_for_training_scaled[i - n_past:i, :])
        trainY.append(df_for_training_scaled[i :i + n_future, config_dict['target_col']]) # for target variable

    X, Y = np.array(trainX), np.array(trainY)

    print('trainX shape == {}.'.format(X.shape))
    print('trainY shape == {}.'.format(Y.shape))

    return X, Y

def plot_multiple_forecasts(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    # plot_series(X[0, :, 0])
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

def delhi_climate_data():
    # load DataFrame
    df = pd.read_csv('DailyDelhiClimateTrain.csv', index_col='date')
    df.index = pd.DatetimeIndex(df.index)
    df = df.fillna(method='ffill')

    # df = df[['meantemp']]

    dataset = df.copy()

    return dataset

def pre_process_data(config_dict, dataset):
    X, Y = split_sequences(dataset.copy(), config_dict)

    return X, Y

def build_train_multivariate_lstm_model(config_dict, X, Y):
    from tensorflow.keras.callbacks import EarlyStopping
    # Train LSTM model if training==1 else load model
    if config_dict['training'] == 1:
        np.random.seed(42)
        tf.random.set_seed(42)

        model = keras.models.Sequential([
            keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dropout(0.2),
            # keras.layers.TimeDistributed(keras.layers.Dense(config_dict['lookahead']))
            keras.layers.Dense(Y.shape[1])
        ])

        model.compile(loss="mse", optimizer="adam")

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        history = model.fit(X, Y, epochs=config_dict['epochs'],
                            validation_split=0.1, callbacks=[early_stopping])

        # Plot loss
        plot_learning_curves(history.history["loss"], history.history["val_loss"])
        plt.show()

        print(f"Model Validation Loss --> {history.history['val_loss'][-1]}")

        #save model
        model.save(config_dict['model_name']+'.h5')
        print("Model Saved...")

        return config_dict

def load_trained_model(config_dict):
    from tensorflow.keras.models import load_model

    try:
        model = load_model(config_dict['model_name']+'.h5', compile = False)
    except:
        print("Couldn't retrieve model...")

    return model

def predict_using_last_window(dataset, config_dict):
    model = load_trained_model(config_dict)
    # Last window based prediction
    last_values_from_dataset = dataset[-config_dict['lookahead']:]
    last_values_from_dataset_transformed = load_scaler_and_transform(last_values_from_dataset.copy(), config_dict['model_name'])

    # last_values_from_dataset_transformed =  last_values_from_dataset
    Y_pred_new = model.predict(last_values_from_dataset_transformed.values.reshape(1, last_values_from_dataset_transformed.shape[0], last_values_from_dataset_transformed.shape[1]))
    df_new_transformed = pd.DataFrame(Y_pred_new[-1])

    #Creating temp future df to make it easier to inverse scale/get desired shape
    future_dataset = pd.DataFrame(np.zeros((last_values_from_dataset.shape[0], last_values_from_dataset.shape[1])))
    future_dataset.iloc[:, 0] = df_new_transformed.values
    future_dataset.columns = last_values_from_dataset.columns

    #Perform inverse transform to get back unscaled values
    df_new = do_inverse_transform(future_dataset.copy(), config_dict)
    df_new = df_new[[future_dataset.columns[config_dict['target_col']]]]

    # For future prediction based on last window
    df_new.index = pd.date_range(start=(pd.Timestamp(config_dict['last_train_index']) + pd.Timedelta(days=1)),
                                 periods=len(df_new), freq='D')
    df_new.columns = [f"Predicted {future_dataset.columns[config_dict['target_col']]}"]
    combined_df_new = pd.DataFrame(dataset).join(df_new, how='outer')
    combined_df_new.plot(alpha=0.2, style='8')

    return combined_df_new

def make_test_data_prediction_plots(X_test, Y_test, config_dict):
    model = load_trained_model(config_dict)
    Y_pred_test = model.predict(X_test)

    plot_multiple_forecasts(X_test, Y_test, Y_pred_test)
    plt.show()

    return Y_pred_test

# dataset = load_gold_prices_dataset()
dataset = delhi_climate_data()

#Create model/script configuration
config_dict = model_configs(dataset.copy())

#Scale data
dataset_scaled = scale_and_save_scaler(dataset.copy(), config_dict['model_name'])

# Ensure values are of dtype float for ML
df_for_training_scaled = dataset_scaled.values.astype('float')

#pre-process data/split into sequences 
X, Y = pre_process_data(config_dict, df_for_training_scaled)

#Build LSTM Model
config_dict = build_train_multivariate_lstm_model(config_dict, X, Y)

#Predict based on Test data (X_test)
# Y_pred_test = make_test_data_prediction_plots(X_test, Y_test, config_dict)

#Make prediction and plot
prediction_df = predict_using_last_window(dataset.copy(), config_dict)
prediction_df[['meantemp',  'Predicted meantemp']].plot(style='8', alpha=0.2)