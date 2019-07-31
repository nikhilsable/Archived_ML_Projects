from keras.models import load_model
from numpy import hstack
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from keras.layers.advanced_activations import LeakyReLU

def split_sequences(sequences, n_steps_in, n_steps_out):
    from numpy import array

    X, y = list(), list()
    for i in range(len(sequences) ):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def make_multi_step_lstm(dataset, insteps, outsteps):
    from numpy import array
    import tensorflow as tf
    import numpy as np
    from keras.models import Sequential
    from keras.layers import LSTM
    from keras.layers import Dense
    from keras.models import load_model
    from keras.layers import Dropout
    from keras.callbacks import EarlyStopping
    from keras.layers.advanced_activations import LeakyReLU


    # covert into input/output

    test_dataset = dataset[-(int(.10 * len(dataset))):, :]
    dataset = dataset[:-(int(.10 * len(dataset))), :]

    X, y = split_sequences(dataset, insteps, outsteps)
    test_X, test_y = split_sequences(test_dataset, insteps, outsteps)
    print("Training Data shape for X = " + str(X.shape))
    print("Training Data shape for y = " + str(y.shape))
    print("Testing Data shape for X = " + str(test_X.shape))
    print("Testing Data shape for y = " + str(test_y.shape))

    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    print("....Total features used for prediction = " + str(n_features))

    # define model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(LSTM(40))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    # callbacks
    # patient early stopping
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
    callbacks = [es]
    # fit model
    model.fit(X, y, epochs=100, verbose=1, validation_data=(test_X, test_y), callbacks=callbacks, batch_size=128)
    # demonstrate prediction
    # x_input = array([[1, 2], [1, 2]])
    # x_input = x_input.reshape((1, n_steps_in, n_features))
    model_name = "lstm_right_enc_" + (pd.Timestamp('now')).strftime(
        "%Y_%m_%d") + ".h5"  # save model and architecture to single file
    model.save(model_name)
    print("Saved model : " + model_name + " to disk")
    # yhat = model.predict(x_input, verbose=0)

    return model