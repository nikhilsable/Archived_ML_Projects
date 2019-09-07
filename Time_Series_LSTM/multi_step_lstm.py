# -*- coding: utf-8 -*-
"""multi_step_lstm_v1.ipynb

"""

import pandas as pd
from keras.models import load_model
from numpy import hstack
from numpy import array

from keras.layers.advanced_activations import LeakyReLU

def scale_and_save_scaler(df, model_name):
	from sklearn.preprocessing import MinMaxScaler
	min_max_scaler = MinMaxScaler()

	for col in df.columns:  # go through all of the columns
		if col != df.columns[-1]:  # normalize all ... except for the target itself!
			df[col] = min_max_scaler.fit_transform(df[col].values.reshape(-1, 1))

	scaler_filename = "scaler_"+model_name+".pkl" #scaler pickle file

	# Save scaler
	import pickle as pkl
	with open(scaler_filename, "wb") as outfile:
		pkl.dump(min_max_scaler, outfile)

	print("********** Scaler Saved *************")

	return df


def use_training_scaler(df, model_name):
	scaler_filename = "scaler_" + model_name + ".pkl"  # scaler pickle file
	import pickle as pkl
	with open(scaler_filename, "rb") as infile:
		min_max_scaler = pkl.load(infile)
		#X_test_scaled = scaler.transform(X_test)

	for col in df.columns:  # go through all of the columns
		if col != df.columns[-1]:  # normalize all ... except for the target itself!
			df[col] = min_max_scaler.transform(df[col].values.reshape(-1, 1))

	print("********** Scaler Retrieved ***********")

	return df

def split_sequences(df, n_steps_in, n_steps_out, model_name, training = 1):
	from numpy import array

	#fit trans if training, else just transform
	if training==1:
		df = scale_and_save_scaler(df, model_name)
	else:
		df = use_training_scaler(df, model_name)

	df.dropna(inplace=True)
	df = df.values

    # split a multivariate sequence into samples
	X, y = list(), list()
	for i in range(len(df)):# find the end of this pattern
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

def multi_step_lstm_model(lookback, lookahead, train_x, train_y, validation_x, validation_y, batch_size,epochs=10, model_name="lstm"):
	import time
	from numpy import array
	import tensorflow as tf
	import numpy as np
	from keras.models import Sequential
	from keras.layers import LSTM
	from keras.layers import Dense
	from keras.models import load_model
	from keras.layers import Dropout
	from keras.layers import ReLU
	from keras.callbacks import EarlyStopping
	from keras.layers.advanced_activations import LeakyReLU
	from keras.layers import Input

	EPOCHS = epochs #how many passes through our data
	BATCH_SIZE = batch_size  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
	NAME = f"{lookback}-SEQ-{lookahead}-PRED-{int(time.time())}"  # a unique name for the model

	# define model
	model = Sequential()
	#input_x = Input(batch_shape=(BATCH_SIZE, lookback, train_x.shape[2]), name='input')
	model.add(LSTM(200,return_sequences=True, batch_input_shape = (BATCH_SIZE,None, train_x.shape[2]),stateful=True, input_shape=(train_x.shape[1:])))

	#model.add(BatchNormalization())  # normalizes activation outputs, same reason you want to normalize your input data.
	model.add(LeakyReLU(alpha=0.1))
	model.add(Dropout(0.2))
	model.add(LSTM(200, stateful=True))
	model.add(ReLU())
	model.add(Dense(lookahead))
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
#df.columns = ['left_sensor', 'right_sensor', 'pressure']
df = df.fillna(method='ffill')

df = df[['pressure', 'left_sensor', 'right_sensor']]

df = df.tail(10000)

lookback = 120  # how long of a preceeding sequence to collect for RNN
lookahead = 120  # how far into the future are we trying to predict?
epochs = 1 # set the number of epochs you want the NN to run for

model_name = "lstm_right_enc_" + (pd.Timestamp('now')).strftime("%Y_%m_%d") + ".h5"  # save model and architecture to single file

## here, split away some slice of the future data from the main main_df.
times = df.index.values
last_10pct = df.index.values[-int(0.20*len(times))]

validation_main_df = df[(df.index >= last_10pct)]
main_df = df[(df.index < last_10pct)]

#train_x, train_y = preprocess_df(main_df, lookback)
#validation_x, validation_y = preprocess_df(validation_main_df, lookback)

train_x, train_y = split_sequences(main_df, lookback,lookahead, model_name, training=1)
validation_x, validation_y = split_sequences(validation_main_df, lookback,lookahead, model_name, training=1)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")

#train_x
#train_y

def computeHCF(x, y):
    if x > y:
        smaller = y
    else:
        smaller = x
    for i in range(1, smaller+1):
        if((x % i == 0) and (y % i == 0)):
            hcf = i

    return hcf

#For stateful LSTM batch size needs to be divisible by both
# training and test data len, so we try a custom number for batch size
# or let the hcf function decide for us

batch_size= computeHCF(train_x.shape[0], validation_x.shape[0])
#batch_size = 100

model = multi_step_lstm_model(lookback, lookahead, train_x, train_y, validation_x, validation_y,epochs=epochs, batch_size=batch_size, model_name=model_name)


##### TESTING

df_test = df.iloc[lookback:, :]
test_x, test_y = split_sequences(df_test, lookback,lookahead, model_name, training=0) #training = 0 implies
# only transform

yhat = model.predict(test_x, verbose=0, batch_size=batch_size)
print(len(yhat[-1].flatten()))
y_pred = pd.DataFrame(data={'y_pred':yhat[-1].flatten()})


#just plotting stuff
df_final = pd.DataFrame()
df_final = df_final.join(df, how='outer')
df_final = df_final.join(y_pred, how='outer')

df_final.plot()

print ("All done, with a batch size of : " + str(batch_size))
