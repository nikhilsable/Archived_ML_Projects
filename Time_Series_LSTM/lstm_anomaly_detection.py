# -*- coding: utf-8 -*-
"""lstm_anomaly_detection_v1.ipynb

"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

#For plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

import plotly as py
import plotly.graph_objects as go


def model_configs(dataset):
    config_dict = {
    'lookahead' : 1,  # how far (timesteps) into the future are we trying to predict?
    'n_steps' : 30,  # No of timesteps in each training/evaluation sample
    'epochs' : 50,  # set the number of epochs you want the NN to run for
    'model_name' : f"encoder_decoder_lstm_model_{pd.Timestamp('now').strftime('%Y_%m_%d')}",  # save model and architecture to single file
    'patience':3, 
    ## here, split away some slice of the future data from the main main_df.
    'train_size_ix_split' : int(.80 * len(dataset)),
    'validation_size_ix_split' : int(.80 * len(dataset)) + int(.10 * len(dataset)),
    'training' :1,
    'last_train_index':dataset.index.max(),
    'target_col': 0, #Note : Arrange input dataset with target col at 0 or -1
    'scaler_filename' : f"scaler_encoder_decoder_lstm_{pd.Timestamp('now').strftime('%Y_%m_%d')}.pkl"

    }

    return config_dict

def scale_and_save_scaler(df, config_dict):
    from sklearn.preprocessing import MinMaxScaler
    minmax_scaler = MinMaxScaler()

    df[list(df.columns)] = minmax_scaler.fit_transform(df)

    # Save scaler
    from pickle import dump
    dump(minmax_scaler, open(config_dict['scaler_filename'], 'wb'))
    print("********** Scaler Saved *************")

    return df

def do_inverse_transform(df, config_dict):
    from pickle import load

    minmax_scaler = load(open(config_dict['scaler_filename'], 'rb'))
    print("********** Scaler Retrieved ***********")

    df[list(df.columns)] = minmax_scaler.inverse_transform(df)

    return df

def load_scaler_and_transform(df, config_dict):
    from pickle import load

    minmax_scaler = load(open(config_dict['scaler_filename'], 'rb'))
    print("********** Scaler Retrieved ***********")

    df[list(df.columns)] = minmax_scaler.transform(df)

    print("********** Data Transformed ***********")

    return df

def split_sequences(df, config_dict):
    trainX = []
    trainY = []

    n_future = config_dict['lookahead']  # Number of days we want to predict into the future
    n_past = config_dict['n_steps']  # Number of past days we want to use to predict the future

    for i in range(n_past, len(df) - n_future + 1):
        trainX.append(df_for_training_scaled[i - n_past:i])
        trainY.append(df_for_training_scaled[i :i + n_future]) # for target variable

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
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

def load_sensor_dataset():
    # load DataFrame
    df = pd.read_csv(r"test_data.csv")
    df = df.set_index('time')
    df.index = pd.DatetimeIndex(df.index)
    df.columns = ['left_sensor', 'right_sensor', 'pressure']
    df = df.fillna(method='ffill')

    # For univariate
    # df = df[['right_sensor']]
    # df = df.tail(7000)

    return df.copy()

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
            keras.layers.LSTM(128, activation='relu', input_shape=(X.shape[1], X.shape[2])),
            keras.layers.Dropout(rate=0.2),

            keras.layers.RepeatVector(X.shape[1]),

            keras.layers.LSTM(128, activation='relu', return_sequences=True),
            keras.layers.Dropout(rate=0.2),
            keras.layers.TimeDistributed(keras.layers.Dense(X.shape[2])),            # keras.layers.TimeDistributed(keras.layers.Dense(config_dict['lookahead']))
        ])

        model.compile(loss="mae", optimizer="adam")

        early_stopping = EarlyStopping(monitor='val_loss', patience=config_dict['patience'], mode='min')

        history = model.fit(X, Y, epochs=config_dict['epochs'],
                            validation_split=0.1, callbacks=early_stopping)

        # Plot loss
        plot_learning_curves(history.history["loss"], history.history["val_loss"])
        plt.show()

        print(f"Model Validation Loss --> {history.history['val_loss'][-1]}")

        #save model
        model.save(config_dict['model_name']+'.h5')
        print("Model Saved...")

        return config_dict

    else:
        model = load_trained_model(config_dict)

        print(f"Existing model loaded --> {config_dict['model_name']+'.h5'}")

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
    last_values_from_dataset = dataset[-config_dict['n_steps']:]
    last_values_from_dataset_transformed = load_scaler_and_transform(last_values_from_dataset.copy(), config_dict)

    # last_values_from_dataset_transformed =  last_values_from_dataset
    Y_pred_new = model.predict(last_values_from_dataset_transformed.values.reshape(1, last_values_from_dataset_transformed.shape[0], last_values_from_dataset_transformed.shape[1]))
    df_new_transformed = pd.DataFrame(Y_pred_new[-1])

    #Creating temp future df to make it easier to inverse scale/get desired shape
    future_dataset = pd.DataFrame(np.zeros((config_dict['lookahead'], last_values_from_dataset.shape[1])))
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
    # combined_df_new.plot(alpha=0.2, style='8')

    return combined_df_new

def make_prediction(X, config_dict):
    model = load_trained_model(config_dict)
    Y_pred = model.predict(X)

    return Y_pred

def to_sequences(X, Y,  n_steps):
    x_values = []
    y_values = []

    for i in range(len(X)-n_steps):
        #print(i)
        x_values.append(X.iloc[i:(i+n_steps)].values)
        y_values.append(Y.iloc[i+n_steps])
        
    return np.array(x_values), np.array(y_values)

def get_mae_threshold_training_set(X,config_dict):
    #Load model and make prediction based on train data
    X_pred_train = make_prediction(X, config_dict)

    # Calculate MAE threshold
    mae_threshold =np.mean(np.abs(X_pred_train - X), axis=1)
    plt.hist(mae_threshold, bins=50)
    plt.xlabel('Train MAE loss')
    plt.ylabel('Number of samples')
    plt.show()
    max_mae_threshold = np.round(0.90*mae_threshold.max(), 3)

    print(f'Reconstruction error threshold (training set): {max_mae_threshold}')

    config_dict.update({'mae_threshold_train':max_mae_threshold})

    return mae_threshold, config_dict

def make_prediction_on_test_data(test_set, config_dict):
    # Transform test data based on train scale object
    test_dataset_transformed = load_scaler_and_transform(test_set.copy(), config_dict)
    # Interpret as float to make life easier for algo
    test_dataset_transformed = test_dataset_transformed.astype('float')
    X, Y = to_sequences(test_dataset_transformed, test_dataset_transformed, n_steps = config_dict['n_steps'])

    #Load model and make prediction based on test data
    X_pred_test = make_prediction(X, config_dict)

    # Calculate MAE threshold
    test_loss = np.mean(np.abs(X_pred_test - X), axis=1)
    plt.hist(test_loss, bins=50)
    plt.xlabel('Test MAE loss')
    plt.ylabel('Number of samples')
    plt.show()
    
    test_score_df = pd.DataFrame(test_set[config_dict['n_steps']:])
    test_score_df['loss'] = test_loss
    test_score_df['threshold'] = config_dict['mae_threshold_train']
    test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
    # test_score_df['Predicted Value'] = prediction_df.values
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_score_df.index, y=test_score_df['loss'], name='Test loss'))
    fig.add_trace(go.Scatter(x=test_score_df.index, y=test_score_df['threshold'], name='Threshold'))
    fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
    fig.show()

    #Plot residuals
    plt.plot(X.flatten())
    plt.plot(X_pred_test.flatten(), alpha=0.2)
    plt.show()
    
    return test_score_df

def  make_anomaly_plots(test_prediction_df, test_set, config_dict):

    if len(test_prediction_df.loc[test_prediction_df['anomaly'] == True]) !=  0:
        test_dataset_transformed = load_scaler_and_transform(test_set.copy(), config_dict)
        anomalies = test_prediction_df.loc[test_prediction_df['anomaly'] == True][[test_prediction_df.columns[config_dict['target_col']]]]
        anomalies.columns = ['Anomaly']
        anomalies = anomalies[[anomalies.columns[config_dict['target_col']]]]
        combined_df = do_inverse_transform(test_dataset_transformed[[test_dataset_transformed.columns[config_dict['target_col']]]], config_dict).join(anomalies, how='outer')
        # combined_df.plot(style='8', alpha = 0.2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df[[combined_df.columns[config_dict['target_col']]]].values.flatten(), name='Target'))
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies.values.flatten(), mode='markers', name='Anomaly'))
        fig.update_layout(showlegend=True, title='Detected anomalies')
        py.offline.plot(fig)

    else:
        test_dataset_transformed = load_scaler_and_transform(test_set.copy(), config_dict)
        combined_df = do_inverse_transform(test_dataset_transformed[[test_dataset_transformed.columns[config_dict['target_col']]]], config_dict)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df[[combined_df.columns[config_dict['target_col']]]].values.flatten(), name='Target'))
        fig.update_layout(showlegend=True, title='No Anomalies Detected ')
        py.offline.plot(fig)


    return fig

#Datasets
raw_dataset = load_gold_prices_dataset()
# raw_dataset = delhi_climate_data()
# raw_dataset = raw_dataset[['meantemp']].copy()
# raw_dataset = load_sensor_dataset() # NOTE : 'minute' freq
# raw_dataset = pd.read_csv('spx.csv', parse_dates=['date'], index_col='date')
# raw_dataset =  raw_dataset[:'2018-02-15']

#Create model/script configuration
config_dict = model_configs(raw_dataset.copy())

#Split into train and test sets
dataset = raw_dataset.copy().iloc[:-(config_dict['n_steps']), :]
test_set = raw_dataset.iloc[-(config_dict['n_steps']):, :]

#Scale data
dataset_scaled = scale_and_save_scaler(dataset.copy(), config_dict)

# Ensure values are of dtype float for ML
df_for_training_scaled = dataset_scaled.astype('float')

X, Y = to_sequences(df_for_training_scaled, df_for_training_scaled, n_steps = config_dict['n_steps'])

#Build LSTM Model
config_dict = build_train_multivariate_lstm_model(config_dict, X, Y)

#Calculate MAE threshold based on training data
mae_threshold,config_dict = get_mae_threshold_training_set(X, config_dict)

#Predict
test_prediction_df = make_prediction_on_test_data(raw_dataset.copy(), config_dict)

#make anomaly plots
fig = make_anomaly_plots(test_prediction_df.copy(), raw_dataset.copy(), config_dict)
