import pandas as pd
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

SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 60  # how far into the future are we trying to predict?

from sklearn import preprocessing  # pip install sklearn ... if you don't have it!

def preprocess_df(df):
    from collections import deque
    import numpy as np

    for col in df.columns:  # go through all of the columns
        if col != df.columns[-1]:  # normalize all ... except for the target itself!
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... jic. Those nasty NaNs love to creep in.

    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(
        maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    #random.shuffle(sequential_data)  # shuffle for good measure.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!


df = pd.read_csv("test_data.csv")
df = df.set_index('time')
df.columns = ['left_sensor', 'pressure', 'right_sensor']
df.dropna(inplace=True)

## here, split away some slice of the future data from the main main_df.
times = df.index.values
last_5pct = df.index.values[-int(0.05*len(times))]

validation_main_df = df[(df.index >= last_5pct)]
main_df = df[(df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")

def multi_step_lstm_model(SEQ_LEN, FUTURE_PERIOD_PREDICT, train_x, train_y, validation_x, validation_y,epochs=100, batch_size=64, model_name="lstm"):
    import time
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

    EPOCHS = epochs  # how many passes through our data
    BATCH_SIZE = batch_size  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
    NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"  # a unique name for the model

    # define model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(train_x.shape[1:])))
    #model.add(BatchNormalization())  # normalizes activation outputs, same reason you want to normalize your input data.
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(LSTM(40))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(FUTURE_PERIOD_PREDICT))
    model.compile(optimizer='adam', loss='mse')
    # callbacks
    # patient early stopping
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
    callbacks = [es]
    #tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    #filepath = "LSTM_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    #checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True,
    #                                                      mode='max'))  # saves only the best ones
    # fit model
    model.fit(train_x, train_y, epochs=EPOCHS, verbose=1, validation_data=(validation_x, validation_y), callbacks=callbacks, batch_size=128)
    model_name = "lstm_right_enc" + (pd.Timestamp('now')).strftime(
        "%Y_%m_%d") + ".h5"  # save model and architecture to single file
    # Score model
    score = model.evaluate(validation_x, validation_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Save model
    model.save("models/{}".format(model_name))
    print("Saved model : " + model_name + " to disk")


