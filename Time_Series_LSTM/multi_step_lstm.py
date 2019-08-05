import pandas as pd
from keras.models import load_model
from numpy import hstack
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from keras.layers.advanced_activations import LeakyReLU

def split_sequences(df, n_steps_in, n_steps_out):
    from numpy import array
    from sklearn import preprocessing  # pip install sklearn ... if you don't have it!

    for col in df.columns:  # go through all of the columns
        if col != df.columns[-1]:  # normalize all ... except for the target itself!
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup

    df = df.values

    # split a multivariate sequence into samples
    X, y = list(), list()
    for i in range(len(df)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(df):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = df[i:end_ix, :-1], df[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

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
    model.add(LSTM(100, return_sequences=True, input_shape=(train_x.shape[1:])))
    #model.add(BatchNormalization())  # normalizes activation outputs, same reason you want to normalize your input data.
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(LSTM(80))
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
    model.fit(train_x, train_y, epochs=EPOCHS, verbose=1, validation_data=(validation_x, validation_y), callbacks=callbacks, batch_size=BATCH_SIZE)
    model_name = "lstm_right_enc_" + (pd.Timestamp('now')).strftime(
        "%Y_%m_%d") + ".h5"  # save model and architecture to single file
    # Score model
    #score = model.evaluate(validation_x, validation_y, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    # Save model
    #model.save("./models/{}".format(model_name))
    model.save(format(model_name))
    print("Saved model : " + model_name + " to disk")

    return model


df = pd.read_csv("test_data.csv")
df = df.set_index('time')
df.columns = ['left_sensor', 'pressure', 'right_sensor']
df.dropna(inplace=True)

SEQ_LEN = 120  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 120  # how far into the future are we trying to predict?

## here, split away some slice of the future data from the main main_df.
times = df.index.values
last_5pct = df.index.values[-int(0.05*len(times))]

validation_main_df = df[(df.index >= last_5pct)]
main_df = df[(df.index < last_5pct)]

#train_x, train_y = preprocess_df(main_df, SEQ_LEN)
#validation_x, validation_y = preprocess_df(validation_main_df, SEQ_LEN)

train_x, train_y = split_sequences(main_df, SEQ_LEN,FUTURE_PERIOD_PREDICT)
validation_x, validation_y = split_sequences(validation_main_df, SEQ_LEN,FUTURE_PERIOD_PREDICT)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")

#train_x
#train_y

model = multi_step_lstm_model(SEQ_LEN, FUTURE_PERIOD_PREDICT, train_x, train_y, validation_x, validation_y,epochs=100, batch_size=64, model_name="lstm")


##### TESTING

df_test = df.iloc[:, :]
test_x, test_y = split_sequences(df_test, SEQ_LEN,FUTURE_PERIOD_PREDICT)

yhat = model.predict(test_x, verbose=0)
#print(len(yhat.flatten()))
yhat[:1,:].flatten()


